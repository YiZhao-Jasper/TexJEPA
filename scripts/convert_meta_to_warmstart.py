#!/usr/bin/env python3
"""Convert Meta's official I-JEPA checkpoint to a warm-start checkpoint.

Meta's release (e.g. IN22K-vit.h.14-900e.pth.tar) bundles:
    encoder / predictor / target_encoder weights
    + AdamW optimizer state (from IN22K 900-epoch run)
    + scheduler position (epoch=900)

If we load it as-is via `load_checkpoint`, the scheduler would fast-forward 900
epochs of steps, and the AdamW momentum buffers (from a completely different
dataset/lr/batch-size) would corrupt our MIMIC-CXR finetuning.

This script produces a *clean* warm-start checkpoint that contains only
weights (no opt, no scaler), with epoch=0, so our training loop initializes
a fresh optimizer and scheduler from scratch while inheriting the
pretrained representation.

Usage:
    python scripts/convert_meta_to_warmstart.py \
        --in  checkpoints/IN22K-vit.h.14-900e.pth.tar \
        --out logs/ijepa_300/jepa-latest.pth.tar
"""

import argparse
import os
from collections import OrderedDict

import torch


EXPECTED_KEYS = {"encoder", "predictor", "target_encoder"}


def _strip_module_prefix(sd: OrderedDict) -> OrderedDict:
    """Remove DDP `module.` prefix so downstream `_adapt_state_dict` can
    re-add it if the current encoder happens to be DDP-wrapped."""
    out = OrderedDict()
    for k, v in sd.items():
        out[k.replace("module.", "", 1) if k.startswith("module.") else k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Path to Meta checkpoint (e.g. IN22K-vit.h.14-900e.pth.tar)")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="Where to save the cleaned warm-start checkpoint")
    args = ap.parse_args()

    if not os.path.isfile(args.in_path):
        raise FileNotFoundError(args.in_path)

    print(f"[*] Loading Meta checkpoint from {args.in_path} (this may take a moment for ~10 GB) ...")
    ckpt = torch.load(args.in_path, map_location="cpu", weights_only=False)

    top_keys = set(ckpt.keys())
    print(f"[*] Top-level keys in source checkpoint: {sorted(top_keys)}")

    missing = EXPECTED_KEYS - top_keys
    if missing:
        raise RuntimeError(
            f"Source checkpoint is missing required keys: {missing}. "
            f"This does not look like an I-JEPA checkpoint."
        )

    enc = _strip_module_prefix(ckpt["encoder"])
    pred = _strip_module_prefix(ckpt["predictor"])
    tenc = _strip_module_prefix(ckpt["target_encoder"])
    src_epoch = ckpt.get("epoch", "unknown")

    print(f"[*] Source epoch: {src_epoch}")
    print(f"[*] encoder tensors         : {len(enc)}")
    print(f"[*] predictor tensors       : {len(pred)}")
    print(f"[*] target_encoder tensors  : {len(tenc)}")

    out = {
        "encoder": enc,
        "predictor": pred,
        "target_encoder": tenc,
        "epoch": 0,
        "loss": 0.0,
        "meta": {
            "warm_start_source": os.path.basename(args.in_path),
            "source_epoch": src_epoch,
            "note": ("Warm-start from ImageNet-22K pretrained I-JEPA. "
                     "Opt/scaler intentionally omitted so optimizer + "
                     "scheduler start fresh."),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    tmp = args.out_path + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, args.out_path)

    out_size_gb = os.path.getsize(args.out_path) / (1024 ** 3)
    print(f"[+] Saved warm-start checkpoint to {args.out_path} ({out_size_gb:.2f} GB)")
    print("[+] epoch reset to 0, no opt/scaler — optimizer will initialize fresh.")


if __name__ == "__main__":
    main()
