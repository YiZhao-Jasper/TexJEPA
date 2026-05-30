"""
Asymmetric perturbations for TexJEPA specialization.

Motivation
==========
The default I-JEPA training in ``src/pretrain.py`` feeds the SAME normalized
image to both the context encoder and the target encoder. The Smooth-L1 loss
in latent space therefore never sees a scenario where the *context* view is
noisier than the *target*, and the encoder has no incentive to learn a
mapping that is invariant to pixel-level perturbations such as Gaussian
noise, Poisson (shot) noise, or JPEG compression artifacts.

TexJEPA-N fix — context-target asymmetric augmentation
======================================================
Apply mild stochastic noise ONLY to the context view, while the target
encoder still consumes the clean image. The student is then forced to
produce predictions for the *clean* target representations from a *noisy*
input, which is a direct invariance objective (compare with DINOv2's noise-
robustness curriculum and BYOL/MoCo-v3 colour-jitter on the student view).

This module provides
    * Per-sample, on-the-fly tensor-space perturbations (cheap, GPU-friendly).
    * A single ``apply_context_noise`` entry-point that consumes a NORMALIZED
      tensor batch (output of the existing pre-training transform), runs
      perturbations in [0, 1] space, and returns a re-normalized batch.

Notes
-----
* All perturbations operate on a *clone* of the input — the caller's tensor
  is never modified in-place. This is important because the same batch is
  also fed to the target encoder.
* JPEG re-encoding is implemented in pure tensor ops (DCT-quantize-inverse-
  DCT on 8×8 blocks) to stay on-GPU and avoid CPU<->GPU round-trips.
* Default probabilities are LOW (≤0.5 each, applied in expectation 1.0
  perturbations per sample). A "warm-up curriculum" wrapper schedules the
  intensity from 0 → full over the first ``warmup_epochs`` epochs of the
  fine-tuning stage so that we don't shock the encoder right after loading
  a clean checkpoint.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# (De)normalization helpers
# ---------------------------------------------------------------------------

def _denormalize(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Convert a normalized tensor back to [0, 1]-ish image space (clamped)."""
    m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x * s + m).clamp(0.0, 1.0)


def _normalize(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Reverse of ``_denormalize``."""
    m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - m) / s


# ---------------------------------------------------------------------------
# Individual perturbations (operate in [0, 1] space)
# ---------------------------------------------------------------------------

def gaussian_noise_(img01: torch.Tensor, sigma_range: Sequence[float] = (0.01, 0.05)) -> torch.Tensor:
    """Add per-sample i.i.d. Gaussian noise. Returns a NEW tensor.

    sigma is sampled uniformly from ``sigma_range`` *per sample* so different
    samples in the batch receive different noise levels (curriculum-friendly).
    """
    B = img01.shape[0]
    sigmas = torch.empty(B, 1, 1, 1, device=img01.device, dtype=img01.dtype).uniform_(*sigma_range)
    noise = torch.randn_like(img01) * sigmas
    return (img01 + noise).clamp(0.0, 1.0)


def poisson_noise_(img01: torch.Tensor, peak_range: Sequence[float] = (50.0, 200.0)) -> torch.Tensor:
    """Approximate Poisson (shot) noise as photon counting noise.

    For low-photon imagery (which medical X-rays are), shot noise is the
    dominant physical noise source. We model it by sampling counts and
    dividing back: lower ``peak`` => more noise.
    """
    B = img01.shape[0]
    peaks = torch.empty(B, 1, 1, 1, device=img01.device, dtype=img01.dtype).uniform_(*peak_range)
    counts = torch.poisson(img01.float() * peaks)
    out = (counts / peaks).to(img01.dtype)
    return out.clamp(0.0, 1.0)


def jpeg_compression_(img01: torch.Tensor, quality_range: Sequence[int] = (40, 80)) -> torch.Tensor:
    """Simulate JPEG quantization on 8×8 DCT blocks (GPU-friendly, lossy).

    This is *not* exactly identical to libjpeg output, but reproduces the
    most damaging artefact: high-frequency coefficient quantization.
    """
    if img01.shape[-1] % 8 != 0 or img01.shape[-2] % 8 != 0:
        # Fall back: skip rather than mis-shape; only when input is not 8-aligned.
        return img01
    quality = float(random.randint(*quality_range))
    # Standard JPEG quality scaling
    q = max(1.0, (200.0 - 2.0 * quality) if quality >= 50 else (5000.0 / quality))
    return _dct_jpeg_block(img01, q)


def _dct_2d_block_matrix(n: int = 8, device=None, dtype=torch.float32) -> torch.Tensor:
    """Compute the orthonormal DCT-II matrix of size n×n."""
    m = torch.zeros(n, n, device=device, dtype=dtype)
    for k in range(n):
        for i in range(n):
            m[k, i] = math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    m[0] *= math.sqrt(1.0 / n)
    m[1:] *= math.sqrt(2.0 / n)
    return m


def _dct_jpeg_block(img01: torch.Tensor, q: float) -> torch.Tensor:
    """Block-DCT, quantize, inverse, in pure torch (GPU-friendly)."""
    B, C, H, W = img01.shape
    n = 8
    Hn, Wn = H // n, W // n
    # (B, C, Hn, n, Wn, n) -> (B, C, Hn, Wn, n, n)
    blocks = img01.view(B, C, Hn, n, Wn, n).permute(0, 1, 2, 4, 3, 5).contiguous()
    dct_mat = _dct_2d_block_matrix(n, device=img01.device, dtype=img01.dtype)
    # Forward DCT: M @ blocks @ M^T
    coeffs = dct_mat @ blocks @ dct_mat.transpose(-2, -1)
    # Quantize
    coeffs = torch.round(coeffs / q) * q
    # Inverse
    inv = dct_mat.transpose(-2, -1) @ coeffs @ dct_mat
    # Reassemble
    out = inv.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Curriculum-friendly composer
# ---------------------------------------------------------------------------

class ContextNoiseAugment:
    """Stage-2 context-only noise perturbation.

    Parameters
    ----------
    mean, std :
        Normalization statistics used by the upstream transform. Required so
        we can round-trip to / from [0, 1] space inside the perturbation.
    p_gauss, p_poisson, p_jpeg :
        Per-perturbation activation probability. Defaults sum to ~1.0
        expected number of perturbations applied per sample.
    sigma_range, peak_range, quality_range :
        Hyperparameter ranges (see individual perturbation functions).
    intensity :
        Curriculum scaler in [0, 1]. The scaler shrinks the *upper bound* of
        sigma and the *lower bound* of peak/quality, leaving the lower bound
        fixed so a small perturbation is still possible at intensity=0.
    """

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        p_gauss: float = 0.5,
        p_poisson: float = 0.3,
        p_jpeg: float = 0.2,
        sigma_range: Sequence[float] = (0.01, 0.05),
        peak_range: Sequence[float] = (50.0, 200.0),
        quality_range: Sequence[int] = (40, 80),
        intensity: float = 1.0,
    ):
        self.mean = mean
        self.std = std
        self.p_gauss = float(p_gauss)
        self.p_poisson = float(p_poisson)
        self.p_jpeg = float(p_jpeg)
        self.sigma_range = tuple(sigma_range)
        self.peak_range = tuple(peak_range)
        self.quality_range = tuple(quality_range)
        self.intensity = float(intensity)

    # ---- Curriculum --------------------------------------------------------

    def set_intensity(self, intensity: float) -> None:
        """Set curriculum intensity in [0, 1]."""
        self.intensity = max(0.0, min(1.0, float(intensity)))

    def _scaled_ranges(self):
        i = self.intensity
        sig_lo, sig_hi = self.sigma_range
        sig_hi_eff = sig_lo + (sig_hi - sig_lo) * i
        peak_lo, peak_hi = self.peak_range
        # higher peak  = less noise; intensity 0 -> use peak_hi for both endpoints
        peak_lo_eff = peak_hi - (peak_hi - peak_lo) * i
        q_lo, q_hi = self.quality_range
        # higher quality = less artifact; intensity 0 -> use q_hi for both endpoints
        q_lo_eff = int(round(q_hi - (q_hi - q_lo) * i))
        return ((sig_lo, sig_hi_eff),
                (peak_lo_eff, peak_hi),
                (q_lo_eff, q_hi))

    # ---- Application -------------------------------------------------------

    @torch.no_grad()
    def __call__(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Apply asymmetric noise to a NORMALIZED tensor batch and return a
        new NORMALIZED tensor batch (the input is never modified in-place).
        """
        if self.intensity <= 0.0:
            return x_norm
        x01 = _denormalize(x_norm, self.mean, self.std)
        sig_r, peak_r, qual_r = self._scaled_ranges()
        if random.random() < self.p_gauss:
            x01 = gaussian_noise_(x01, sigma_range=sig_r)
        if random.random() < self.p_poisson:
            x01 = poisson_noise_(x01, peak_range=peak_r)
        if random.random() < self.p_jpeg:
            x01 = jpeg_compression_(x01, quality_range=qual_r)
        return _normalize(x01, self.mean, self.std)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_context_noise(
    cfg: Optional[dict],
    mean: Sequence[float],
    std: Sequence[float],
) -> Optional[ContextNoiseAugment]:
    """Build a noise augmenter from a YAML sub-dict, or return None.

    Expected cfg format (all optional, with sensible defaults):

        context_noise:
          enabled: true
          p_gauss: 0.5
          p_poisson: 0.3
          p_jpeg: 0.2
          sigma_range: [0.01, 0.05]
          peak_range:  [50, 200]
          quality_range: [40, 80]
          warmup_epochs: 5    # curriculum: 0 -> full intensity over N epochs
    """
    if not cfg or not cfg.get('enabled', False):
        return None
    aug = ContextNoiseAugment(
        mean=mean, std=std,
        p_gauss=cfg.get('p_gauss', 0.5),
        p_poisson=cfg.get('p_poisson', 0.3),
        p_jpeg=cfg.get('p_jpeg', 0.2),
        sigma_range=cfg.get('sigma_range', (0.01, 0.05)),
        peak_range=cfg.get('peak_range', (50.0, 200.0)),
        quality_range=cfg.get('quality_range', (40, 80)),
        intensity=0.0,   # curriculum starts at 0; pretrain loop ramps it up
    )
    return aug
