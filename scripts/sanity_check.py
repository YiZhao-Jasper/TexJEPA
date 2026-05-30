#!/usr/bin/env python3
"""Portable sanity checks for the TexJEPA research release.

This script intentionally uses tiny synthetic tensors where possible. It is a
repository check, not a replacement for full pre-training or downstream
evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.masks.multiblock import MaskCollator  # noqa: E402
from src.perturbations_pretrain import build_context_noise  # noqa: E402
from src.transforms import CHEST_XRAY_NORM  # noqa: E402
from src.vicreg_patch import vicreg_patch_loss  # noqa: E402


EXPECTED_CONFIGS = {
    "ijepa_base_200ep.yaml": {
        "epochs": 200,
        "model_name": "vit_huge",
        "load_checkpoint": False,
    },
    "texjepa_n.yaml": {
        "epochs": 50,
        "model_name": "vit_huge",
        "load_checkpoint": True,
    },
    "texjepa_r.yaml": {
        "epochs": 50,
        "model_name": "vit_huge_reg",
        "load_checkpoint": True,
    },
    "texjepa_c.yaml": {
        "epochs": 50,
        "model_name": "vit_huge",
        "load_checkpoint": True,
    },
}


def check(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)
    print(f"  OK  {message}")


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def check_configs() -> None:
    print("[1/4] Checking public pre-training configs")
    cfg_dir = PROJECT_ROOT / "configs" / "pretrain"
    for filename, expected in EXPECTED_CONFIGS.items():
        cfg = load_yaml(cfg_dir / filename)
        check(cfg["optimization"]["epochs"] == expected["epochs"],
              f"{filename}: epoch count")
        check(cfg["meta"]["model_name"] == expected["model_name"],
              f"{filename}: model name")
        check(cfg["meta"]["load_checkpoint"] is expected["load_checkpoint"],
              f"{filename}: checkpoint loading policy")
        check(cfg["data"]["root_path"] == "data/mimic-cxr-384",
              f"{filename}: portable data root")
        check(cfg["data"]["normalization"] == "chest_xray",
              f"{filename}: chest X-ray normalization")

    tex_n = load_yaml(cfg_dir / "texjepa_n.yaml")
    tex_r = load_yaml(cfg_dir / "texjepa_r.yaml")
    tex_c = load_yaml(cfg_dir / "texjepa_c.yaml")
    check(tex_n["meta"]["read_checkpoint"] == "../ijepa_base_200ep/jepa-latest.pth.tar",
          "TexJEPA-N starts from the 200-epoch base checkpoint")
    check(tex_r["meta"]["read_checkpoint"] == "../texjepa_n/jepa-latest.pth.tar",
          "TexJEPA-R starts from TexJEPA-N")
    check(tex_c["meta"]["read_checkpoint"] == "../texjepa_n/jepa-latest.pth.tar",
          "TexJEPA-C starts from TexJEPA-N")
    check(tex_c["optimization"]["vicreg"]["enabled"] is True,
          "TexJEPA-C enables patch variance/covariance regularization")


def check_register_model() -> None:
    print("[2/4] Checking register-token ViT factory")
    import src.models.vision_transformer_v2 as vit_v2

    model = vit_v2.vit_base_reg(img_size=[224], patch_size=14, num_registers=4)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    check(y.shape == (2, 256, 768),
          f"register encoder strips register tokens, got {tuple(y.shape)}")


def check_context_noise() -> None:
    print("[3/4] Checking context-only texture perturbation")
    cfg = {
        "enabled": True,
        "p_gauss": 1.0,
        "p_poisson": 0.0,
        "p_jpeg": 0.0,
        "sigma_range": [0.01, 0.02],
        "warmup_epochs": 1,
    }
    aug = build_context_noise(cfg, mean=CHEST_XRAY_NORM[0], std=CHEST_XRAY_NORM[1])
    check(aug is not None, "context-noise augmenter builds")
    aug.set_intensity(1.0)
    x = torch.zeros(2, 3, 224, 224)
    y = aug(x)
    check(y.shape == x.shape, "context-noise preserves tensor shape")
    check(torch.isfinite(y).all().item(), "context-noise output is finite")


def check_vicreg_and_masks() -> None:
    print("[4/4] Checking masks and VICReg auxiliary loss")
    collator = MaskCollator(
        input_size=224,
        patch_size=14,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        allow_overlap=False,
        min_keep=10,
    )
    batch = [(torch.randn(3, 224, 224), i) for i in range(4)]
    imgs, enc_masks, pred_masks = collator(batch)
    check(imgs[0].shape == (4, 3, 224, 224), "mask collator batches images")
    check(len(enc_masks) == 1 and len(pred_masks) == 4,
          "mask collator emits 1 context mask and 4 target masks")

    z = torch.randn(4, 96, 128, requires_grad=True)
    losses = vicreg_patch_loss(z)
    check(set(losses) == {"var_loss", "cov_loss", "total"}, "VICReg loss keys")
    check(torch.isfinite(losses["total"]).item(), "VICReg total is finite")
    losses["total"].backward()
    check(z.grad is not None and torch.isfinite(z.grad).all().item(),
          "VICReg gradients are finite")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip tiny model construction for very small CPU nodes")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    print("=" * 70)
    print("  TexJEPA repository sanity check")
    print("=" * 70)
    check_configs()
    if not args.skip_model:
        check_register_model()
    check_context_noise()
    check_vicreg_and_masks()
    print("=" * 70)
    print("  All sanity checks passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
