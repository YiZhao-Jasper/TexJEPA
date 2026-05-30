#!/bin/bash
# =============================================================
# TexJEPA Downstream Evaluation on VinBigData/VinDr-style labels
# DDP multi-GPU, NCCL protection, and nohup-safe defaults
# =============================================================
# Usage:
#   bash downstream/run_downstream.sh linear      # Linear probe (single GPU)
#   bash downstream/run_downstream.sh finetune     # Fine-tune (DDP, GPU 1,2)
#   bash downstream/run_downstream.sh all          # Both sequentially
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Paths; override with environment variables on each machine. ---
CHECKPOINT="${CHECKPOINT:-logs/texjepa_n/jepa-latest.pth.tar}"
IMAGE_DIR="${VINBIG_IMAGE_DIR:-data/vinbig/images_1024/train}"
CSV_PATH="${VINBIG_CSV:-data/vinbig/annotations/train.csv}"
OUTPUT_BASE="${OUTPUT_BASE:-downstream_results}"

# --- GPU config ---
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"
LINEAR_GPU="${LINEAR_GPU:-0}"

# --- Optional Conda activation. ---
if [ -n "${CONDA_ENV:-}" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

# --- NCCL robustness ---
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

MODE="${1:-all}"

echo "============================================"
echo "TexJEPA Downstream Evaluation"
echo "============================================"
echo "Checkpoint: $CHECKPOINT"
echo "Image dir:  $IMAGE_DIR"
echo "CSV path:   $CSV_PATH"
echo "Mode:       $MODE"
echo "GPUs:       $GPUS (nproc=$NPROC)"
echo "============================================"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"; exit 1
fi
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: CSV not found: $CSV_PATH"; exit 1
fi
if [ ! -d "$IMAGE_DIR" ]; then
    echo "ERROR: Image dir not found: $IMAGE_DIR"; exit 1
fi

run_linear() {
    echo ""
    echo "============================================"
    echo ">>> LINEAR PROBE (GPU $LINEAR_GPU) <<<"
    echo "============================================"
    echo ""
    CUDA_VISIBLE_DEVICES=$LINEAR_GPU python -u downstream/train_linear_probe.py \
        --checkpoint "$CHECKPOINT" \
        --image_dir "$IMAGE_DIR" \
        --csv_path "$CSV_PATH" \
        --output_dir "${OUTPUT_BASE}/linear_probe" \
        --gpu 0 \
        --batch_size 128 \
        --train_batch 256 \
        --num_workers 8 \
        --epochs 100 \
        --lr 0.1 \
        --weight_decay 0.0
}

run_finetune() {
    echo ""
    echo "============================================"
    echo ">>> FINE-TUNING DDP (GPUs: $GPUS) <<<"
    echo "============================================"
    echo ""
    CUDA_VISIBLE_DEVICES=$GPUS torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NPROC \
        downstream/train_finetune.py \
        --checkpoint "$CHECKPOINT" \
        --image_dir "$IMAGE_DIR" \
        --csv_path "$CSV_PATH" \
        --output_dir "${OUTPUT_BASE}/finetune" \
        --batch_size 48 \
        --num_workers 8 \
        --epochs 50 \
        --warmup_epochs 5 \
        --lr 1e-4 \
        --weight_decay 0.05 \
        --layer_decay 0.75 \
        --drop_rate 0.1 \
        --patience 15 \
        --use_amp \
        --resume
}

case "$MODE" in
    linear)
        run_linear
        ;;
    finetune)
        run_finetune
        ;;
    all)
        run_linear
        echo ""
        echo "Linear probe done. Starting fine-tuning..."
        echo ""
        run_finetune
        ;;
    *)
        echo "Usage: $0 {linear|finetune|all}"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "ALL DONE!"
echo "Results in: $OUTPUT_BASE/"
echo "============================================"
