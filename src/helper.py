import logging
import sys
from collections import OrderedDict

import torch

import src.models.vision_transformer as vit
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _adapt_state_dict(state_dict, model):
    """Adapt checkpoint keys between DDP (module.*) and non-DDP formats."""
    model_keys = set(model.state_dict().keys())
    has_module = any(k.startswith('module.') for k in model_keys)
    ckpt_has_module = any(k.startswith('module.') for k in state_dict.keys())

    if has_module == ckpt_has_module:
        return state_dict

    adapted = OrderedDict()
    for k, v in state_dict.items():
        if has_module and not ckpt_has_module:
            adapted[f'module.{k}'] = v
        elif not has_module and ckpt_has_module:
            adapted[k.replace('module.', '', 1)] = v
        else:
            adapted[k] = v
    return adapted


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    expected_batch_size=None,
    expected_world_size=None,
):
    """Load a training checkpoint.

    If `expected_batch_size` or `expected_world_size` are provided and differ from
    the values saved in the checkpoint, a loud warning is emitted. This matters
    because `ipe = len(loader)` changes with batch-size and world-size, which in
    turn would corrupt the scheduler rewind logic (`start_epoch * ipe`).
    """
    checkpoint = torch.load(r_path, map_location=torch.device('cpu'), weights_only=False)
    epoch = checkpoint['epoch']

    ckpt_bs = checkpoint.get('batch_size')
    ckpt_ws = checkpoint.get('world_size')
    if expected_batch_size is not None and ckpt_bs is not None and ckpt_bs != expected_batch_size:
        logger.warning(
            f'===== BATCH-SIZE MISMATCH =====\n'
            f'  checkpoint saved at batch_size={ckpt_bs}, current config batch_size={expected_batch_size}\n'
            f'  This will misalign the LR/WD/EMA schedulers on resume.\n'
            f'  STRONGLY RECOMMENDED: start a fresh run instead of resuming.\n'
            f'=================================')
    if expected_world_size is not None and ckpt_ws is not None and ckpt_ws != expected_world_size:
        logger.warning(
            f'===== WORLD-SIZE MISMATCH =====\n'
            f'  checkpoint saved at world_size={ckpt_ws}, current world_size={expected_world_size}\n'
            f'  This changes iterations-per-epoch and misaligns schedulers.\n'
            f'=================================')

    pretrained_dict = _adapt_state_dict(checkpoint['encoder'], encoder)
    msg = encoder.load_state_dict(pretrained_dict)
    logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    pretrained_dict = _adapt_state_dict(checkpoint['predictor'], predictor)
    msg = predictor.load_state_dict(pretrained_dict)
    logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

    if target_encoder is not None:
        pretrained_dict = _adapt_state_dict(checkpoint['target_encoder'], target_encoder)
        msg = target_encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained target_encoder from epoch {epoch} with msg: {msg}')

    # Optimizer / scaler state is optional: warm-start checkpoints (e.g. the
    # ImageNet-22K I-JEPA weights converted via convert_meta_to_warmstart.py)
    # intentionally omit them so we start a fresh AdamW/scheduler tailored to
    # the new dataset and batch/world size.
    ckpt_opt = checkpoint.get('opt')
    if ckpt_opt is not None:
        opt.load_state_dict(ckpt_opt)
        logger.info(f'loaded optimizers from epoch {epoch}')
    else:
        logger.info(
            'no optimizer state in checkpoint (warm-start mode): '
            'AdamW will initialize fresh')

    ckpt_scaler = checkpoint.get('scaler')
    if scaler is not None and ckpt_scaler is not None:
        scaler.load_state_dict(ckpt_scaler)

    logger.info(f'read-path: {r_path}')
    del checkpoint

    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_model(
    device,
    patch_size=14,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384,
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
    )
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
    )

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)
    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25,
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0,
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0,
        },
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    scaler = None if use_bfloat16 else torch.amp.GradScaler('cuda')
    return optimizer, scaler, scheduler, wd_scheduler
