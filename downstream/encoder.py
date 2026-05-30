"""Load a pre-trained TexJEPA target encoder for downstream tasks."""

import logging
import math
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def _clean_state_dict(state_dict):
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        key = k.replace("module.", "", 1) if k.startswith("module.") else k
        cleaned[key] = v
    return cleaned


def _infer_model_name(state_dict, checkpoint):
    """Infer model factory from checkpoint metadata or tensor shapes."""
    model_name = checkpoint.get("model_name")
    if model_name:
        return model_name

    has_registers = "register_tokens" in state_dict
    embed_dim = state_dict["pos_embed"].shape[-1]
    if embed_dim == 1280:
        return "vit_huge_reg" if has_registers else "vit_huge"
    if embed_dim == 1024:
        return "vit_large_reg" if has_registers else "vit_large"
    if embed_dim == 768:
        return "vit_base_reg" if has_registers else "vit_base"
    raise ValueError(f"Cannot infer ViT size from embed_dim={embed_dim}")


def _infer_img_size(state_dict, patch_size):
    num_patches = state_dict["pos_embed"].shape[1]
    grid = int(math.sqrt(num_patches))
    if grid * grid != num_patches:
        raise ValueError(f"pos_embed has non-square patch count: {num_patches}")
    return grid * patch_size


def build_encoder_from_checkpoint(checkpoint, state_dict, patch_size=14):
    """Build the encoder architecture described by a TexJEPA checkpoint."""
    model_name = _infer_model_name(state_dict, checkpoint)
    img_size = _infer_img_size(state_dict, patch_size)
    num_registers = checkpoint.get("num_registers", 0)
    if "register_tokens" in state_dict:
        num_registers = state_dict["register_tokens"].shape[1]

    if model_name.endswith("_reg"):
        import src.models.vision_transformer_v2 as vit

        encoder = vit.__dict__[model_name](
            img_size=[img_size],
            patch_size=patch_size,
            num_registers=num_registers,
        )
    else:
        import src.models.vision_transformer as vit

        encoder = vit.__dict__[model_name](
            img_size=[img_size],
            patch_size=patch_size,
        )
    return encoder, model_name


def load_target_encoder(checkpoint_path, device="cpu"):
    """Load target-encoder weights from a TexJEPA checkpoint.

    The loader supports ViT-H/14, ViT-L/14, and register-token variants by
    reading checkpoint metadata and falling back to state-dict shape inference.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", "?")
    logger.info(f"Checkpoint: epoch={epoch}, loss={loss}")

    state_dict = _clean_state_dict(ckpt.get("target_encoder", ckpt["encoder"]))
    encoder, model_name = build_encoder_from_checkpoint(ckpt, state_dict)
    logger.info(f"Built downstream encoder: {model_name}")

    msg = encoder.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded target_encoder: {msg}")

    encoder = encoder.to(device)
    encoder.eval()

    param_count = sum(p.numel() for p in encoder.parameters()) / 1e6
    logger.info(f"Target encoder: {param_count:.1f}M parameters")

    del ckpt
    return encoder


class TexJEPAFeatureExtractor(nn.Module):
    """Wraps the target encoder for feature extraction.

    Input: [B, 3, 224, 224]
    Output: [B, D] global average pooled patch features
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim

    @torch.no_grad()
    def forward(self, x):
        patch_tokens = self.encoder(x, masks=None)
        features = patch_tokens.mean(dim=1)
        return features

# Backward-compatible alias for older scripts.
IJEPAFeatureExtractor = TexJEPAFeatureExtractor
