"""
Vision Transformer with optional register tokens on top of the existing
``vision_transformer.py`` architecture.

Why register tokens?
====================
"Vision Transformers Need Registers" (Darcet et al., ICLR 2024) demonstrated
that ViTs at scale develop a small number of "high-norm artefact tokens" in
their patch outputs. These artefacts dominate the global mean-pool, hijack
attention maps, and *destroy* attention interpretability. Adding 4-8
learnable [REG] tokens that participate in self-attention but are discarded
at the output absorbs the artefacts cleanly.

In our setting this matters for two independent reasons:

  1.  The downstream ``model_wrappers.py`` mean-pools every patch token
      before the linear probe. With register tokens, the mean-pool no longer
      averages over noise-spike artefacts, and patch-level attention/
      sensitivity analyses become trustworthy.

  2.  Register tokens give the model a "scratchpad" that improves robustness
      to distribution-shifts (DINOv2 v2.5, Darcet et al. follow-ups), which
      directly addresses the noise-fragility we observed at epoch 97.

Implementation note
===================
The original ViT factories are kept unchanged. Register-token encoders are
opt-in via model names such as ``vit_huge_reg``.

Design constraint
=================
The mask collator and ``apply_masks`` operate on tensors with shape
(B, N_patches, D). To keep them untouched, register tokens are
**concatenated AFTER masking and BEFORE the transformer blocks**, i.e. they
participate in self-attention but never enter the masking arithmetic.
Patch outputs returned by the encoder still have shape (B, N_patches, D).
"""

import math
from functools import partial

import torch
import torch.nn as nn

from src.models.vision_transformer import (
    Block,
    PatchEmbed,
    get_2d_sincos_pos_embed,
)
from src.utils.tensors import trunc_normal_
from src.masks.utils import apply_masks


class VisionTransformerWithRegisters(nn.Module):
    """ViT identical to the original ``VisionTransformer`` but with
    ``num_registers`` learnable tokens that flow through self-attention and
    are stripped from the output."""

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        num_registers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_registers = int(num_registers)

        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Frozen 2D sin-cos positional embedding for the patch tokens
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Learnable register tokens — initialized small, trained from scratch
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.num_registers, embed_dim))
            trunc_normal_(self.register_tokens, std=init_std)
        else:
            self.register_tokens = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    # ------------------------------------------------------------------
    # init helpers (identical to vit.VisionTransformer)
    # ------------------------------------------------------------------

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1]
        N = pos_embed.shape[1]
        if npatch == N:
            return pos_embed
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x, masks=None):
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # 1. Patchify + positional embed
        x = self.patch_embed(x)
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        # 2. Apply masks BEFORE concatenating register tokens so the existing
        #    `apply_masks` (which assumes shape (B, N_patches, D)) is
        #    untouched. After this step x has shape (B*M, N_kept, D).
        if masks is not None:
            x = apply_masks(x, masks)

        # 3. Prepend register tokens (no positional embedding — they have no
        #    spatial meaning; they are a global scratchpad).
        if self.register_tokens is not None and self.num_registers > 0:
            B = x.size(0)
            regs = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([regs, x], dim=1)
            num_reg = self.num_registers
        else:
            num_reg = 0

        # 4. Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        # 5. Strip register tokens before returning so downstream code that
        #    expects (B, N_patches, D) keeps working.
        if num_reg > 0:
            x = x[:, num_reg:]

        return x


# --------------------------------------------------------------------------
# Factory functions matching the names used by ``init_model``
# --------------------------------------------------------------------------

def vit_base_reg(patch_size=16, num_registers=4, **kwargs):
    return VisionTransformerWithRegisters(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_registers=num_registers, **kwargs)


def vit_large_reg(patch_size=16, num_registers=4, **kwargs):
    return VisionTransformerWithRegisters(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_registers=num_registers, **kwargs)


def vit_huge_reg(patch_size=16, num_registers=4, **kwargs):
    return VisionTransformerWithRegisters(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_registers=num_registers, **kwargs)
