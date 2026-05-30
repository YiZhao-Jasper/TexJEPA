"""
Smoke test: verify model builds, forward/backward pass, all imports resolve.
Run without data to validate the engineering is correct.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import torch
import torch.nn.functional as F


def test_encoder_builds():
    """Test all ViT variants build correctly."""
    import src.models.vision_transformer as vit
    print('[1/5] Testing encoder builds...')

    for name, expected_dim in [
        ('vit_tiny', 192), ('vit_small', 384), ('vit_base', 768),
        ('vit_large', 1024), ('vit_huge', 1280),
    ]:
        model = vit.__dict__[name](img_size=[224], patch_size=14)
        assert model.embed_dim == expected_dim, f'{name} embed_dim mismatch'
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'  {name}: embed_dim={model.embed_dim}, params={n_params:.1f}M')

    print('  PASSED')


def test_pretrain_forward():
    """Test full I-JEPA pretrain forward+backward pass with ViT-B/14."""
    from src.helper import init_model
    from src.masks.multiblock import MaskCollator
    from src.masks.utils import apply_masks
    from src.utils.tensors import repeat_interleave_batch
    print('[2/5] Testing pretrain forward pass...')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    encoder, predictor = init_model(
        device=device, patch_size=14, model_name='vit_base',
        crop_size=224, pred_depth=6, pred_emb_dim=384,
    )
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    mask_collator = MaskCollator(
        input_size=224, patch_size=14,
        enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5), nenc=1, npred=4,
        allow_overlap=False, min_keep=10,
    )

    B = 4
    imgs = torch.randn(B, 3, 224, 224, device=device)
    dummy_batch = [(imgs[i], i) for i in range(B)]
    collated, masks_enc, masks_pred = mask_collator(dummy_batch)

    imgs_batch = collated[0].to(device)
    masks_enc = [m.to(device) for m in masks_enc]
    masks_pred = [m.to(device) for m in masks_pred]

    with torch.no_grad():
        h = target_encoder(imgs_batch)
        h = F.layer_norm(h, (h.size(-1),))
        h = apply_masks(h, masks_pred)
        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))

    z = encoder(imgs_batch, masks_enc)
    z = predictor(z, masks_enc, masks_pred)

    loss = F.smooth_l1_loss(z, h)
    loss.backward()

    print(f'  loss={loss.item():.4f}, z.shape={z.shape}, h.shape={h.shape}')
    print('  PASSED')


def test_transforms():
    """Test pretrain transform pipeline."""
    from src.transforms import make_pretrain_transforms, CHEST_XRAY_NORM
    from PIL import Image
    import numpy as np
    print('[3/5] Testing transforms...')

    dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

    t = make_pretrain_transforms(
        crop_size=224, normalization=CHEST_XRAY_NORM)
    out = t(dummy_img)
    assert out.shape == (3, 224, 224), f'Transform shape wrong: {out.shape}'

    print('  PASSED')


def test_mimic_cxr_parsing():
    """Test MIMIC-CXR path parsing and dataset module imports."""
    from src.datasets.mimic_cxr import (
        _parse_image_filenames, MIMIC_ZIP_PREFIXES,
    )
    import tempfile
    print('[4/5] Testing MIMIC-CXR path parsing...')

    assert MIMIC_ZIP_PREFIXES == [f'p{i}' for i in range(10, 20)]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg\n')
        f.write('files/p11/p11000123/s12345678/abcdef01-23456789-abcdef01-23456789-abcdef01.jpg\n')
        f.write('files/p19/p19999999/s99999999/ffffffff-ffffffff-ffffffff-ffffffff-ffffffff.jpg\n')
        f.write('\n')
        fname = f.name

    entries = _parse_image_filenames(fname)
    os.unlink(fname)

    assert len(entries) == 3, f'Expected 3 entries, got {len(entries)}'
    assert entries[0]['zip_prefix'] == 'p10'
    assert entries[1]['zip_prefix'] == 'p11'
    assert entries[2]['zip_prefix'] == 'p19'

    print(f'  Parsed {len(entries)} entries correctly')
    print('  PASSED')


def test_ddp_components():
    """Test distributed components import and work."""
    from src.utils.distributed import AllReduce
    print('[5/5] Testing distributed components...')

    x = torch.tensor([1.0, 2.0, 3.0])
    out = AllReduce.apply(x)
    assert torch.allclose(out, x), 'AllReduce single-process should be identity'

    print('  PASSED')


if __name__ == '__main__':
    print('=' * 50)
    print('  TexJEPA Smoke Test (MIMIC-CXR)')
    print('=' * 50)

    test_encoder_builds()
    test_pretrain_forward()
    test_transforms()
    test_mimic_cxr_parsing()
    test_ddp_components()

    print('=' * 50)
    print('  ALL TESTS PASSED')
    print('=' * 50)
