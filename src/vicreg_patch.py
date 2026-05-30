"""
VICReg-style variance + covariance regularization on patch tokens.

Background
==========
Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for
Self-Supervised Learning", ICLR 2022.

The TexJEPA-N noise-invariant specialization improves robustness to mild
Gaussian, Poisson, and JPEG-style texture perturbations. A remaining failure
mode is over-smoothing: the encoder can become too globally smooth and lose
patch-level contrast between lesion-scale structure and background anatomy.

VICReg's two terms — applied here on the student encoder's patch tokens —
attack this collapse directly:

  - Variance term: hinge loss max(0, gamma - sqrt(var + eps)) per dim,
    averaged. Forces every feature dim to spread out across the batch of
    patches, so no dim collapses to a constant.
  - Covariance term: penalises off-diagonal entries of the patch-feature
    covariance matrix. Decorrelates dims so they encode independent
    structure rather than redundantly representing the global image gist.

Together these explicitly preserve patch-level diversity, which we expect
to preserve lesion-scale texture sensitivity while keeping the I-JEPA loss
and context-only texture corruption objective unchanged.

DDP correctness — IMPORTANT
===========================
An early implementation used
``torch.distributed.nn.functional.all_gather`` to compute global
variance/covariance across DDP ranks. This *deadlocked* immediately at
ep1 iter 1 because:

  * I-JEPA's MaskCollator sets ``min_keep_enc = min(min_keep_enc, len(mask))``
    PER-BATCH (see src/masks/multiblock.py L132 / L156). Each rank's
    DataLoader produces an independent batch of masks, so each rank's
    ``z_enc`` has a DIFFERENT shape ``(B, N_visible_local, D)``.
  * ``all_gather`` requires equal-sized tensors across ranks; with size
    mismatch NCCL hangs forever (we saw rank0 stuck on backward
    ALLREDUCE while rank1 was still on the next forward's _rebuild_buckets
    BROADCAST — classic collective desync).

The fix below uses **local statistics on each rank, then averages the
SCALAR loss across ranks via AllReduce** — exactly the same pattern as
I-JEPA's main loss. The estimator is biased (each rank only sees
batch_size * N_visible_local ≈ 21k samples for ViT-Huge with batch 128)
but at D=1280 the within-rank sample size is still 16x the feature dim,
which is more than enough to estimate variance and covariance reliably.

We deliberately avoid all_gather for two reasons:
  1. Correctness in the face of variable-shape per-rank tensors.
  2. No additional collective ops beyond what DDP already does, so
     SeqNum stays in lockstep across ranks.

Numerical stability
===================
We always cast patch tokens to float32 before the variance/covariance
computation — under bfloat16 autocast, the variance of well-trained
features (which can be O(1e-2)) underflows the BF16 mantissa, leading to
NaN gradients. Float32 here costs ~few MB and is essential for stability.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Variance + covariance terms (per-rank, no cross-rank gather)
# ---------------------------------------------------------------------------

def variance_term(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Hinge variance loss averaged over feature dims.

    Args:
        z: (M, D) flattened patch tokens (LOCAL rank only).
        gamma: target std per dim.
        eps: numerical floor inside the sqrt.

    Returns:
        Scalar tensor.
    """
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.relu(gamma - std).mean()


def covariance_term(z: torch.Tensor) -> torch.Tensor:
    """Off-diagonal covariance penalty, normalised by feature dim.

    Args:
        z: (M, D) tensor (LOCAL rank only). Will be re-centred internally.

    Returns:
        Scalar tensor = sum(off_diag(cov)^2) / D.
    """
    M, D = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(M - 1, 1)
    # off-diagonal squared sum, normalised by D
    off_diag_sq_sum = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag_sq_sum / D


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def vicreg_patch_loss(
    patch_tokens: torch.Tensor,
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
    gamma: float = 1.0,
    max_patches_for_cov: int = 16384,
) -> dict:
    """Compute VICReg variance + covariance losses on patch tokens (LOCAL).

    The returned ``total`` is a per-rank scalar; the caller is expected
    to fold it into the I-JEPA loss BEFORE the existing ``AllReduce`` on
    the total loss (or to do its own AllReduce). We do NOT call any
    collective op inside this function — see the module-level docstring
    for why this matters.

    Args:
        patch_tokens: (B, N, D) student encoder output. N may differ
            across ranks because I-JEPA's MaskCollator truncates per-batch.
            That is fine here because we never cross ranks.
        var_weight: lambda for the variance hinge term (VICReg paper = 25).
            We default to 1.0 because the I-JEPA smooth-l1 loss is on the
            same scale and we want the auxiliary signal to be a mild
            regulariser, not a dominant force.
        cov_weight: lambda for the covariance term (VICReg paper = 1).
            We default to 0.04 to roughly match the magnitude of the
            variance term at D=1280 (the empirical cov scale at the start
            of TexJEPA-C training is ~15 — multiplied by 0.04 gives ~0.6, the
            same order as the variance hinge ~0.5).
        gamma: target std per dim for the variance hinge.
        max_patches_for_cov: cap M used for the covariance computation,
            which forms an MxD intermediate and an O(D^2) Gram matrix.
            For D=1280 the Gram is ~6.5 MB regardless of M, but the MxD
            intermediate dominates and we sub-sample to keep peak memory
            bounded. Variance is cheap and uses all patches.

    Returns:
        dict with float scalar tensors (all REQUIRE_GRAD on the local
        rank's encoder parameters):
            'var_loss'  — hinge variance term (no weight)
            'cov_loss'  — off-diagonal covariance term (no weight)
            'total'     — var_weight * var_loss + cov_weight * cov_loss
    """
    if patch_tokens.dim() == 3:
        z = patch_tokens.reshape(-1, patch_tokens.size(-1))
    else:
        z = patch_tokens

    # Float32 for numerical stability under BF16 autocast.
    z = z.float()

    v_loss = variance_term(z, gamma=gamma)

    # Covariance is O(M*D + D^2). Sub-sample to keep peak memory bounded.
    if z.size(0) > max_patches_for_cov:
        idx = torch.randperm(z.size(0), device=z.device)[:max_patches_for_cov]
        z_cov = z[idx]
    else:
        z_cov = z
    c_loss = covariance_term(z_cov)

    return {
        'var_loss': v_loss,
        'cov_loss': c_loss,
        'total': var_weight * v_loss + cov_weight * c_loss,
    }
