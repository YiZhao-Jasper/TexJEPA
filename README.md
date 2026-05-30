# When Texture Becomes the World: Texture-aware JEPA for Chest X-ray Representation Learning

**TexJEPA** is a texture-aware extension of Image Joint-Embedding Predictive Architecture (I-JEPA) for chest X-ray representation learning. The code in this repository trains a ViT-H/14 I-JEPA backbone on MIMIC-CXR-JPG and then specializes it with context-target asymmetric texture perturbations, register-token routing, and patch-token variance/covariance regularization.

Authors: **Yi Zhao\***, **Ruilang Wang\***, Bowen Liu, Donglong Chen†<br>
\* Equal contribution. † Corresponding author.

## Overview

Chest radiographs are dominated by fine-grained local texture: lung markings, opacities, devices, rib shadows, noise, compression artifacts, and scanner-dependent acquisition patterns. A chest X-ray representation learner can therefore appear strong under clean evaluation while becoming brittle when nuisance texture shifts dominate the input. TexJEPA studies this failure mode and introduces targeted modifications that keep the original I-JEPA predictive objective while improving texture robustness.

The released training sequence is:

| Public name | Initialization | Training length | Main change |
| --- | --- | ---: | --- |
| I-JEPA-300 | none in the default public config | 300 epochs | Standard I-JEPA objective on MIMIC-CXR-JPG |
| TexJEPA-N | I-JEPA-300 checkpoint | 50 epochs | Context-only stochastic texture corruption; clean target branch |
| TexJEPA-R | TexJEPA-N checkpoint | 50 epochs | Register-token encoder plus tighter local masks |
| TexJEPA-C | TexJEPA-N checkpoint | 50 epochs | Patch-token variance/covariance regularization |

The public names are the only names used in this release. Checkpoints and training logs should be reported with these names.

## Repository Layout

```text
configs/pretrain/              # Public pre-training configs
downstream/                    # VinBigData/VinDr-style linear probe and fine-tuning
figures/                       # Paper/release figures
scripts/                       # Data preprocessing, launch, smoke, and sanity checks
src/datasets/                  # MIMIC-CXR-JPG pre-training dataset
src/masks/                     # Multi-block I-JEPA mask collator
src/models/                    # ViT, predictor, and register-token ViT
src/pretrain.py                # I-JEPA-300 baseline training loop
src/pretrain_v2.py             # TexJEPA-N/R/C specialization loop
src/perturbations_pretrain.py  # Context-only texture perturbations
src/vicreg_patch.py            # Patch variance/covariance regularization
```

This repository intentionally contains one README only. Large datasets, checkpoints, and downstream result directories are excluded from Git.

## Environment

The original experiments were run with PyTorch on CUDA GPUs. A minimal environment is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For multi-GPU training, use a recent PyTorch build with NCCL support. The launchers accept a space-separated `DEVICES` list, for example `DEVICES="cuda:0 cuda:1"`.

## Data

Pre-training uses **MIMIC-CXR-JPG v2.0.0**. The repository does not redistribute MIMIC-CXR data. Download access must be obtained through PhysioNet credentialing and the MIMIC-CXR-JPG data-use agreement.

Expected default layout:

```text
data/mimic-cxr-jpg/
├── IMAGE_FILENAMES
├── mimic-cxr-2.0.0-split.csv.gz
├── mimic-cxr-2.0.0-metadata.csv.gz
├── mimic-cxr-2.0.0-chexpert.csv.gz
└── files/
    ├── p10/
    ├── ...
    └── p19/
```

To create the 384-pixel training cache used by the configs:

```bash
python scripts/preprocess_mimic_384.py \
  --src data/mimic-cxr-jpg \
  --dst data/mimic-cxr-384 \
  --size 384 \
  --quality 90 \
  --workers 64
```

The pre-training configs read `data/mimic-cxr-384` by default.

## Pre-training

Run the common I-JEPA-300 baseline:

```bash
DEVICES="cuda:0 cuda:1" bash scripts/pretrain_base.sh
```

Then run TexJEPA-N from the base checkpoint:

```bash
DEVICES="cuda:0 cuda:1" bash scripts/train_texjepa_n.sh
```

Then run the two branches from TexJEPA-N:

```bash
DEVICES="cuda:0 cuda:1" bash scripts/train_texjepa_r.sh
DEVICES="cuda:0 cuda:1" bash scripts/train_texjepa_c.sh
```

The checkpoint paths encoded in the public configs are:

```text
logs/ijepa_300/jepa-latest.pth.tar
logs/texjepa_n/jepa-latest.pth.tar
logs/texjepa_r/jepa-latest.pth.tar
logs/texjepa_c/jepa-latest.pth.tar
```

The specialization configs use relative `read_checkpoint` paths so the lineage is explicit:

```text
TexJEPA-N <- logs/ijepa_300/jepa-latest.pth.tar
TexJEPA-R <- logs/texjepa_n/jepa-latest.pth.tar
TexJEPA-C <- logs/texjepa_n/jepa-latest.pth.tar
```

### Optional External Warm-start

If a researcher uses Meta's official ImageNet-22K I-JEPA ViT-H/14 checkpoint as an initialization, convert it before training and report this initialization in the experimental setup:

```bash
python scripts/convert_meta_to_warmstart.py \
  --in checkpoints/IN22K-vit.h.14-900e.pth.tar \
  --out logs/ijepa_300/jepa-latest.pth.tar
```

The default public base config does not require this external checkpoint.

## Downstream Evaluation

The downstream code supports VinBigData/VinDr-style multi-label chest X-ray classification using either linear probing or end-to-end fine-tuning.

Expected dataset variables:

```bash
export VINBIG_IMAGE_DIR=data/vinbig/images_1024/train
export VINBIG_CSV=data/vinbig/annotations/train.csv
export CHECKPOINT=logs/texjepa_n/jepa-latest.pth.tar
```

Run both evaluations:

```bash
bash downstream/run_downstream.sh all
```

Run one mode only:

```bash
bash downstream/run_downstream.sh linear
bash downstream/run_downstream.sh finetune
```

The downstream loader reads checkpoint metadata and supports ViT-H/14, ViT-L/14, and register-token checkpoints.

## Sanity Checks

Run repository-level checks before launching long jobs:

```bash
python scripts/sanity_check.py
python scripts/smoke_test.py
```

For a short live integration test of the training loop without editing YAML:

```bash
TEXJEPA_SMOKE_MAX_ITERS=30 DEVICES="cuda:0" bash scripts/train_texjepa_n.sh
```

## Method Details

TexJEPA preserves the I-JEPA structure: a context encoder predicts target-encoder patch representations under multi-block masking, and the target encoder is updated by exponential moving average. TexJEPA modifies what the context branch is forced to ignore or route:

TexJEPA-N applies mild stochastic Gaussian, Poisson, and JPEG-like corruption only to the context image. The target branch remains clean, so the model predicts clean latent targets from a corrupted context.

TexJEPA-R adds four learnable register tokens to the encoder and tightens the mask distribution. Register tokens participate in self-attention but are stripped before the patch output, preserving compatibility with the I-JEPA predictor.

TexJEPA-C adds a local patch-token variance/covariance auxiliary loss. The loss is computed per rank and folded into the existing loss path without cross-rank all-gather, avoiding variable-shape DDP deadlocks caused by per-batch mask truncation.

## Reproducibility Notes

Use the public config names and checkpoint directories when reporting results. The repository excludes raw MIMIC-CXR-JPG, derived 384-pixel caches, model checkpoints, and downstream predictions because these artifacts are large or governed by third-party data-use terms.

If releasing trained weights, place them outside the Git repository, for example on GitHub Releases or a model-hosting service, and preserve the four checkpoint names listed above.

## Citation

```bibtex
@misc{zhao2026texjepa,
  title  = {When Texture Becomes the World: Texture-aware JEPA for Chest X-ray Representation Learning},
  author = {Zhao, Yi and Wang, Ruilang and Liu, Bowen and Chen, Donglong},
  year   = {2026},
  note   = {TexJEPA research code}
}
```

## License

This repository is released for non-commercial research use under the license included in `LICENSE`. MIMIC-CXR-JPG and downstream datasets remain governed by their original licenses and data-use agreements.
