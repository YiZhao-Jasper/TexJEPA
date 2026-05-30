"""
TexJEPA specialization pre-training loop.

What changed vs ``src/pretrain.py``
===================================

TexJEPA-N — Context-Target Asymmetric Augmentation
--------------------------------------------------
* Reads a new YAML key ``data.context_noise`` (see ``perturbations_pretrain``).
* Builds a ``ContextNoiseAugment`` object and uses it ONLY on the context
  branch — the target encoder still consumes clean images. The student is
  therefore trained to predict the *clean* latent representation from a
  *noisy* input, which is a direct invariance objective.
* A simple curriculum ramps the noise intensity from 0 → 1 over the first
  ``warmup_epochs`` epochs of fine-tuning so that warm-starting from a
  noise-free checkpoint does not shock the encoder.

TexJEPA-R — Register Tokens & Tightened Mask Strategy
-----------------------------------------------------
* Selects the ViT factory by ``meta.model_name``. Names ending in ``_reg``
  pull from ``src.models.vision_transformer_v2`` and pass through
  ``num_registers``.
* All other behaviour (mask scales / counts / aspect-ratio range) is driven
  by the YAML config — Stage 3 only requires *config* changes for the mask.

Backward compat
===============
If the YAML omits ``context_noise`` and uses one of the original ViT names,
this loop is functionally identical to ``src/pretrain.py``. The only
non-trivial divergence is that this file imports ``init_model_v2`` (defined
locally) so it can build either the legacy ViT or the register-token ViT.
"""

import os
import copy
import logging
import sys
import time
import yaml

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.utils.tensors import repeat_interleave_batch, trunc_normal_
from src.datasets.mimic_cxr import make_mimic_pretrain
from src.helper import load_checkpoint, init_opt
from src.transforms import make_pretrain_transforms
from src.perturbations_pretrain import build_context_noise
from src.vicreg_patch import vicreg_patch_loss

# Original ViT factories
import src.models.vision_transformer as vit_v1
# Register-token ViT factories used by TexJEPA-R.
import src.models.vision_transformer_v2 as vit_v2


log_timings = True
log_freq = 10
checkpoint_freq = 10
_MILESTONE_START = 5
GRAD_CLIP_NORM = 1.0
COLLAPSE_CHECK_FREQ = 50
GRAD_SPIKE_HARD_CAP = 50.0

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _barrier():
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


# ---------------------------------------------------------------------------
# v2 model factory — picks the register variant when name ends with "_reg"
# ---------------------------------------------------------------------------

def init_model_v2(
    device,
    patch_size=14,
    model_name='vit_huge',
    crop_size=224,
    pred_depth=12,
    pred_emb_dim=384,
    num_registers=4,
):
    is_reg = model_name.endswith('_reg')
    if is_reg:
        encoder_factory = vit_v2.__dict__[model_name]
        encoder = encoder_factory(
            img_size=[crop_size],
            patch_size=patch_size,
            num_registers=num_registers,
        )
    else:
        encoder = vit_v1.__dict__[model_name](
            img_size=[crop_size],
            patch_size=patch_size,
        )
    predictor = vit_v1.__dict__['vit_predictor'](
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
    logger.info(f'Built encoder ({model_name}, registers={getattr(encoder, "num_registers", 0)})')
    return encoder, predictor


# ---------------------------------------------------------------------------
# Drift health-check (kept identical to v1 for the same diagnostic behaviour)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _encoder_target_drift(encoder_module, target_encoder):
    enc_sd = (encoder_module.module.state_dict() if hasattr(encoder_module, 'module')
              else encoder_module.state_dict())
    tgt_sd = target_encoder.state_dict()
    candidates = [
        'patch_embed.proj.weight', 'pos_embed',
        'blocks.0.attn.qkv.weight', 'blocks.7.attn.qkv.weight',
        'blocks.15.attn.qkv.weight', 'blocks.23.attn.qkv.weight',
        'blocks.31.attn.qkv.weight',
        'blocks.0.mlp.fc1.weight', 'blocks.15.mlp.fc1.weight',
        'blocks.31.mlp.fc1.weight', 'norm.weight',
    ]
    num, den = 0.0, 0.0
    for k in candidates:
        if k not in enc_sd or k not in tgt_sd:
            continue
        a, b = enc_sd[k].float(), tgt_sd[k].float()
        num += (a - b).pow(2).sum().item()
        den += a.pow(2).sum().item()
    return (num / max(den, 1e-20)) ** 0.5


def _classify_drift(rel_drift, warmup_done):
    if rel_drift < 1e-5:
        return "IDENTICAL"
    if warmup_done and rel_drift < 2e-3:
        return "COLLAPSE-RISK"
    if rel_drift < 5e-3:
        return "early"
    if rel_drift < 3e-2:
        return "healthy"
    return "high-drift"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(args, resume_preempt=False):
    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    patch_size = args['meta'].get('patch_size', args['mask']['patch_size'])
    num_registers = args['meta'].get('num_registers', 4)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    use_zip = args['data'].get('use_zip', True)
    splits = args['data'].get('splits', None)
    context_noise_cfg = args['data'].get('context_noise', None)
    noise_warmup_epochs = int((context_noise_cfg or {}).get('warmup_epochs', 5))

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']
    mask_patch_size = args['mask']['patch_size']
    num_enc_masks = args['mask']['num_enc_masks']
    min_keep = args['mask']['min_keep']
    enc_mask_scale = args['mask']['enc_mask_scale']
    num_pred_masks = args['mask']['num_pred_masks']
    pred_mask_scale = args['mask']['pred_mask_scale']
    aspect_ratio = args['mask']['aspect_ratio']

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- TexJEPA-C: optional VICReg auxiliary loss on student encoder patch tokens.
    # YAML schema (all optional; absent -> disabled):
    #   optimization:
    #     vicreg:
    #       enabled: true
    #       var_weight: 1.0
    #       cov_weight: 0.04
    #       gamma: 1.0
    #       warmup_epochs: 5      # ramp 0 -> full weight over this many ep
    vicreg_cfg = args['optimization'].get('vicreg', None) or {}
    vicreg_enabled = bool(vicreg_cfg.get('enabled', False))
    vicreg_var_w = float(vicreg_cfg.get('var_weight', 1.0))
    vicreg_cov_w = float(vicreg_cfg.get('cov_weight', 0.04))
    vicreg_gamma = float(vicreg_cfg.get('gamma', 1.0))
    vicreg_warmup_epochs = int(vicreg_cfg.get('warmup_epochs', 5))

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            logger.warning(f'Checkpoint not found at {load_path}, starting from scratch')
            load_model = False
            load_path = None

    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss'),
        ('%.5f', 'mask-A'),
        ('%.5f', 'mask-B'),
        ('%d', 'time (ms)'),
    )

    # -- init model (v2 factory; falls back to v1 when name has no "_reg")
    encoder, predictor = init_model_v2(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        num_registers=num_registers,
    )
    target_encoder = copy.deepcopy(encoder)

    n_params_enc = sum(p.numel() for p in encoder.parameters()) / 1e6
    n_params_pred = sum(p.numel() for p in predictor.parameters()) / 1e6
    logger.info(f'Encoder params: {n_params_enc:.1f}M | Predictor params: {n_params_pred:.1f}M')

    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=mask_patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep,
    )

    norm_key = args['data'].get('normalization', 'imagenet')
    if norm_key == 'chest_xray':
        from src.transforms import CHEST_XRAY_NORM
        normalization = CHEST_XRAY_NORM
    else:
        from src.transforms import IMAGENET_NORM
        normalization = IMAGENET_NORM
    logger.info(f'Using normalization: {norm_key}')

    transform = make_pretrain_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter,
        normalization=normalization,
    )

    # -- Stage 2: build context-only noise augmenter
    context_noise = build_context_noise(
        context_noise_cfg,
        mean=normalization[0],
        std=normalization[1],
    )
    if context_noise is not None:
        logger.info(
            'Stage-2 asymmetric noise ENABLED (warmup %d epochs): '
            'p_gauss=%.2f p_poisson=%.2f p_jpeg=%.2f',
            noise_warmup_epochs,
            context_noise.p_gauss,
            context_noise.p_poisson,
            context_noise.p_jpeg,
        )
    else:
        logger.info('Stage-2 asymmetric noise DISABLED.')

    logger.info(f'Loading MIMIC-CXR dataset (splits={splits})')
    _, unsupervised_loader, unsupervised_sampler = make_mimic_pretrain(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        drop_last=True,
        use_zip=use_zip,
        splits=splits,
    )
    ipe = len(unsupervised_loader)
    logger.info(f'Iterations per epoch: {ipe}')

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )

    for p in target_encoder.parameters():
        p.requires_grad = False

    if world_size > 1:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)

    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            expected_batch_size=batch_size,
            expected_world_size=world_size,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        _barrier()
        enc_sd = encoder.module.state_dict() if hasattr(encoder, 'module') else encoder.state_dict()
        pred_sd = predictor.module.state_dict() if hasattr(predictor, 'module') else predictor.state_dict()
        save_dict = {
            'encoder': enc_sd,
            'predictor': pred_sd,
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
            'model_name': model_name,
            'num_registers': num_registers,
        }
        if rank == 0:
            tmp_path = latest_path + '.tmp'
            torch.save(save_dict, tmp_path)
            os.replace(tmp_path, latest_path)
            if epoch >= _MILESTONE_START and (epoch - _MILESTONE_START) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch}'))
            logger.info(f'Checkpoint saved (epoch {epoch})')

    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    logger.info(
        f'Starting training: epochs={num_epochs}, start_epoch={start_epoch}, '
        f'batch={batch_size}x{world_size}={batch_size*world_size}, '
        f'lr={lr}, grad_clip={GRAD_CLIP_NORM}, '
        f'context_noise={"on" if context_noise else "off"}, '
        f'registers={num_registers if model_name.endswith("_reg") else 0}, '
        f'vicreg={"on (var=%.2f cov=%.3f gamma=%.1f warmup=%d)" % (vicreg_var_w, vicreg_cov_w, vicreg_gamma, vicreg_warmup_epochs) if vicreg_enabled else "off"}'
    )

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        logger.info('Epoch %d/%d' % (epoch + 1, num_epochs))
        unsupervised_sampler.set_epoch(epoch)

        # Stage-2 curriculum: ramp noise intensity 0 -> 1 over warmup_epochs
        if context_noise is not None:
            elapsed = max(0, epoch - start_epoch)
            intensity = min(1.0, elapsed / max(1.0, float(noise_warmup_epochs)))
            context_noise.set_intensity(intensity)
            if rank == 0:
                logger.info(
                    '[ep %d] context-noise intensity = %.2f', epoch + 1, intensity)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)

            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target():
                    """Target encoder always sees the CLEAN image."""
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))
                        B = len(h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    """Context encoder sees a (possibly) noised image.

                    Returns ``(z_pred, z_enc)`` where:
                      * z_pred = predictor output, used by the I-JEPA loss
                        (shape (B*M, N_pred, D)).
                      * z_enc  = student encoder output BEFORE the predictor,
                        used by the TexJEPA-C VICReg auxiliary loss on patch tokens
                        (shape (B, N_visible, D)). Returned even when VICReg
                        is disabled — caller decides whether to use it.
                    """
                    if context_noise is not None:
                        imgs_ctx = context_noise(imgs)
                    else:
                        imgs_ctx = imgs
                    z_enc = encoder(imgs_ctx, masks_enc)
                    z_pred = predictor(z_enc, masks_enc, masks_pred)
                    return z_pred, z_enc

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True):
                    h = forward_target()
                    z, z_enc = forward_context()
                    loss_jepa = loss_fn(z, h)

                    if vicreg_enabled:
                        # Curriculum ramp 0 -> 1 over warmup_epochs (matches the
                        # noise ramp; avoids a sudden change in gradient scale
                        # when warm-starting from a noise-only checkpoint).
                        elapsed_v = max(0, epoch - start_epoch)
                        v_intensity = min(
                            1.0, elapsed_v / max(1.0, float(vicreg_warmup_epochs)))
                        vd = vicreg_patch_loss(
                            z_enc,
                            var_weight=vicreg_var_w,
                            cov_weight=vicreg_cov_w,
                            gamma=vicreg_gamma,
                        )
                        # Cast back to amp dtype for the backward, then fold into
                        # the total loss with the ramp applied.
                        loss_vicreg = vd['total'].to(loss_jepa.dtype) * v_intensity
                        loss = loss_jepa + loss_vicreg
                        v_var_val = float(vd['var_loss'].detach())
                        v_cov_val = float(vd['cov_loss'].detach())
                        v_total_val = float(loss_vicreg.detach())
                    else:
                        loss = loss_jepa
                        v_var_val = v_cov_val = v_total_val = 0.0

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    pre_clip_gnorm = torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(predictor.parameters()),
                        GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    pre_clip_gnorm = torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(predictor.parameters()),
                        GRAD_CLIP_NORM)
                    optimizer.step()

                pre_clip_gnorm_val = float(pre_clip_gnorm) if torch.isfinite(pre_clip_gnorm) else float('inf')

                if itr % log_freq == 0:
                    grad_stats = grad_logger(encoder.named_parameters())
                else:
                    grad_stats = None
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

                return (
                    float(loss), float(loss_jepa.detach()),
                    v_var_val, v_cov_val, v_total_val,
                    _new_lr, _new_wd, grad_stats, pre_clip_gnorm_val,
                )

            (loss, loss_jepa_val, v_var_val, v_cov_val, v_total_val,
             _new_lr, _new_wd, grad_stats, pre_clip_gnorm_val), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            if pre_clip_gnorm_val > GRAD_SPIKE_HARD_CAP and rank == 0:
                logger.warning(
                    '[%d, %5d] *** GRAD SPIKE *** pre-clip grad_norm=%.2f '
                    '(> hard cap %.1f)',
                    epoch + 1, itr, pre_clip_gnorm_val, GRAD_SPIKE_HARD_CAP)

            if rank == 0 and (itr % COLLAPSE_CHECK_FREQ == 0):
                warmup_iters = int(warmup * ipe)
                global_iter = epoch * ipe + itr
                warmup_done = global_iter > warmup_iters
                rel_drift = _encoder_target_drift(encoder, target_encoder)
                tag_drift = _classify_drift(rel_drift, warmup_done)
                logger.info(
                    '[%d, %5d] [health] drift=%.3e (%s) pre-clip gnorm=%.3f',
                    epoch + 1, itr, rel_drift, tag_drift, pre_clip_gnorm_val)

            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    mem_mb = (torch.cuda.max_memory_allocated() / 1024. ** 2
                              if torch.cuda.is_available() else 0.)
                    logger.info(
                        '[%d, %5d] loss: %.3f masks: %.1f %.1f '
                        '[wd: %.2e] [lr: %.2e] [mem: %.2e] (%.1f ms)'
                        % (epoch + 1, itr, loss_meter.avg,
                           maskA_meter.avg, maskB_meter.avg,
                           _new_wd, _new_lr, mem_mb, time_meter.avg))
                    if vicreg_enabled:
                        logger.info(
                            '[%d, %5d] vicreg: jepa=%.3f var=%.3f cov=%.3f w_total=%.3f'
                            % (epoch + 1, itr, loss_jepa_val,
                               v_var_val, v_cov_val, v_total_val))
                    if grad_stats is not None:
                        logger.info(
                            '[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                            % (epoch + 1, itr, grad_stats.first_layer,
                               grad_stats.last_layer,
                               grad_stats.min, grad_stats.max))

            log_stats()
            assert not np.isnan(loss), 'loss is nan'

            _smoke_max = int(os.environ.get('TEXJEPA_SMOKE_MAX_ITERS', '0') or '0')
            if _smoke_max > 0 and epoch == start_epoch and itr + 1 >= _smoke_max:
                logger.info(
                    '[SMOKE-TEST] TEXJEPA_SMOKE_MAX_ITERS=%d reached at itr=%d',
                    _smoke_max, itr)
                _barrier()
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
                return

        epoch_time = time.time() - epoch_start
        logger.info(
            'Epoch %d/%d complete | avg loss: %.3f | time: %.1fs'
            % (epoch + 1, num_epochs, loss_meter.avg, epoch_time))
        save_checkpoint(epoch + 1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    _barrier()
    logger.info('Training complete.')


if __name__ == "__main__":
    main()
