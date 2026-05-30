"""End-to-end fine-tuning with DDP multi-GPU support.

Supports both single-GPU (direct python) and multi-GPU (torchrun) launch.
Industrial features: checkpoint resume, early stopping, AMP, gradient clipping,
layer-wise LR decay, warmup+cosine schedule, NCCL timeout protection.
"""

import os
import sys
import time
import math
import json
import signal
import logging
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.transforms import CHEST_XRAY_NORM
from downstream.encoder import load_target_encoder
from downstream.vinbig_dataset import (
    VinBigDataset, make_train_val_split, NUM_CLASSES, VINBIG_CLASSES,
)
from downstream.metrics import compute_multilabel_metrics

import torchvision.transforms as T

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_STOP_TRAINING = False


def _signal_handler(signum, frame):
    global _STOP_TRAINING
    _STOP_TRAINING = True


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def is_ddp():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_ddp():
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=60),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_rank0():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def log_rank0(msg):
    if is_rank0():
        logger.info(msg)


class FineTuneModel(nn.Module):
    def __init__(self, encoder, num_classes, drop_rate=0.1):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        patch_tokens = self.encoder(x, masks=None)
        features = patch_tokens.mean(dim=1)
        return self.head(features)


def build_train_transform(crop_size=224):
    return T.Compose([
        T.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(*CHEST_XRAY_NORM),
    ])


def build_eval_transform(crop_size=224):
    return T.Compose([
        T.Resize(int(crop_size * 256 / 224)),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(*CHEST_XRAY_NORM),
    ])


def get_layer_wise_param_groups(model, base_lr, layer_decay=0.75, wd=0.05):
    param_groups = []
    num_layers = len(model.encoder.blocks)

    layer_scales = {}
    layer_scales["encoder.patch_embed"] = layer_decay ** (num_layers + 1)
    layer_scales["encoder.pos_embed"] = layer_decay ** (num_layers + 1)
    for i in range(num_layers):
        layer_scales[f"encoder.blocks.{i}"] = layer_decay ** (num_layers - i)
    layer_scales["encoder.norm"] = 1.0
    layer_scales["head"] = 1.0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        scale = 1.0
        for prefix, s in layer_scales.items():
            if name.startswith(prefix):
                scale = s
                break

        is_norm_or_bias = ("norm" in name) or ("bias" in name) or (param.ndim == 1)
        param_groups.append({
            "params": [param],
            "lr": base_lr * scale,
            "weight_decay": 0.0 if is_norm_or_bias else wd,
            "name": name,
        })

    return param_groups


def warmup_cosine_lr(optimizer, step, warmup_steps, total_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        lr_scale = step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        base = pg.get("_base_lr", pg["lr"])
        if "_base_lr" not in pg:
            pg["_base_lr"] = pg["lr"]
        pg["lr"] = max(base * lr_scale, min_lr)


def save_checkpoint(model, optimizer, scaler, epoch, best_auc, best_epoch,
                    patience_counter, output_dir, tag="latest"):
    if not is_rank0():
        return
    state = {
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
    }
    path = os.path.join(output_dir, f"checkpoint_{tag}.pth")
    torch.save(state, path)


def load_checkpoint(model, optimizer, scaler, output_dir, device):
    path = os.path.join(output_dir, "checkpoint_latest.pth")
    if not os.path.exists(path):
        return 0, 0.0, 0, 0

    log_rank0(f"Resuming from {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    log_rank0(f"Resumed from epoch {ckpt['epoch']}, best_auc={ckpt['best_auc']:.4f}")
    return (ckpt["epoch"], ckpt["best_auc"],
            ckpt["best_epoch"], ckpt["patience_counter"])


def train_finetune(args):
    global _STOP_TRAINING

    use_ddp = is_ddp()
    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        log_rank0(f"DDP: rank={dist.get_rank()}, local_rank={local_rank}, "
                  f"world_size={world_size}")
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        world_size = 1
        log_rank0(f"Single GPU: {device}")

    encoder = load_target_encoder(args.checkpoint, device="cpu")
    model = FineTuneModel(encoder, NUM_CLASSES, drop_rate=args.drop_rate).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if use_ddp else model
    total_params = sum(p.numel() for p in raw_model.parameters()) / 1e6
    trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad) / 1e6
    log_rank0(f"Model: {total_params:.1f}M params, {trainable:.1f}M trainable")

    train_ids, val_ids = make_train_val_split(args.csv_path, val_ratio=0.2, seed=42)
    log_rank0(f"Split: {len(train_ids)} train, {len(val_ids)} val")

    train_dataset = VinBigDataset(args.image_dir, args.csv_path,
                                  transform=build_train_transform(), image_ids=train_ids)
    val_dataset = VinBigDataset(args.image_dir, args.csv_path,
                                transform=build_eval_transform(), image_ids=val_ids)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    log_rank0(f"Train: {len(train_dataset)} images, {len(train_loader)} batches/epoch "
              f"(batch_size={args.batch_size} x {world_size} GPUs = "
              f"{args.batch_size * world_size} effective)")
    log_rank0(f"Val: {len(val_dataset)} images")

    param_groups = get_layer_wise_param_groups(
        raw_model, base_lr=args.lr, layer_decay=args.layer_decay, wd=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    all_train_labels = torch.stack([train_dataset.labels[iid] for iid in train_dataset.image_ids])
    pos_counts = all_train_labels.sum(dim=0).clamp(min=1)
    n_total = float(len(all_train_labels))
    pos_weight = ((n_total - pos_counts) / pos_counts).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if is_rank0():
        log_rank0("=== Class distribution (train) ===")
        for i, name in enumerate(VINBIG_CLASSES):
            log_rank0(f"  {name:30s}: {int(pos_counts[i]):5d} pos "
                      f"({100*pos_counts[i]/n_total:.1f}%) | "
                      f"pos_weight={pos_weight[i].item():.1f}")

    scaler = GradScaler("cuda") if args.use_amp else None
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    os.makedirs(args.output_dir, exist_ok=True)

    start_epoch, best_auc, best_epoch, patience_counter = 0, 0.0, 0, 0
    if args.resume:
        start_epoch, best_auc, best_epoch, patience_counter = load_checkpoint(
            model, optimizer, scaler, args.output_dir, device)

    log_rank0("=" * 70)
    log_rank0(f"FINE-TUNING START | epochs={args.epochs} | lr={args.lr} | "
              f"AMP={'ON' if args.use_amp else 'OFF'} | "
              f"layer_decay={args.layer_decay} | patience={args.patience}")
    log_rank0("=" * 70)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if _STOP_TRAINING:
            log_rank0("Received stop signal, saving checkpoint...")
            save_checkpoint(model, optimizer, scaler, epoch - 1, best_auc,
                            best_epoch, patience_counter, args.output_dir)
            break

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            warmup_cosine_lr(optimizer, global_step, warmup_steps, total_steps, args.lr)

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if scaler is not None:
                with autocast("cuda"):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_time = time.time() - t0

        # --- Validation (all ranks run full val set) ---
        model.eval()
        all_probs, all_gt = [], []
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels_d = labels.to(device, non_blocking=True)
                if scaler is not None:
                    with autocast("cuda"):
                        logits = model(imgs)
                        v_loss = criterion(logits, labels_d)
                else:
                    logits = model(imgs)
                    v_loss = criterion(logits, labels_d)
                val_loss += v_loss.item()
                val_batches += 1
                all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
                all_gt.append(labels.numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_gt = np.concatenate(all_gt, axis=0)
        metrics = compute_multilabel_metrics(all_gt, all_probs, VINBIG_CLASSES)

        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        lr_now = optimizer.param_groups[-1]["lr"]

        is_best = metrics["mean_auc"] > best_auc
        if is_best:
            best_auc = metrics["mean_auc"]
            best_epoch = epoch
            patience_counter = 0
            if is_rank0():
                state = {
                    "model_state_dict": raw_model.state_dict(),
                    "epoch": epoch,
                    "mean_auc": best_auc,
                    "mean_ap": metrics["mean_ap"],
                }
                torch.save(state, os.path.join(args.output_dir, "best_finetune.pth"))
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or is_best:
            log_rank0(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train={avg_train_loss:.4f} val={avg_val_loss:.4f} | "
                f"mAUC={metrics['mean_auc']:.4f} mAP={metrics['mean_ap']:.4f} | "
                f"lr={lr_now:.2e} | {train_time:.0f}s"
                f"{' *BEST*' if is_best else ''}"
                f" | patience={patience_counter}/{args.patience}"
            )

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, scaler, epoch, best_auc,
                            best_epoch, patience_counter, args.output_dir)

        if patience_counter >= args.patience:
            log_rank0(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # --- Final evaluation ---
    save_checkpoint(model, optimizer, scaler, epoch, best_auc,
                    best_epoch, patience_counter, args.output_dir, tag="final")

    if is_rank0():
        log_rank0("=" * 70)
        log_rank0("FINE-TUNING COMPLETE")
        log_rank0(f"Best mAUC: {best_auc:.4f} at epoch {best_epoch}")
        log_rank0("=" * 70)

        best_path = os.path.join(args.output_dir, "best_finetune.pth")
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
            raw_model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()

        all_probs, all_gt = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                logits = model(imgs)
                all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
                all_gt.append(labels.numpy())

        all_probs = np.concatenate(all_probs)
        all_gt = np.concatenate(all_gt)
        final_metrics = compute_multilabel_metrics(all_gt, all_probs, VINBIG_CLASSES)

        log_rank0(f"Final mAUC={final_metrics['mean_auc']:.4f}, "
                  f"mAP={final_metrics['mean_ap']:.4f}")
        log_rank0("-" * 70)
        for name, info in final_metrics["per_class"].items():
            if info.get("skipped"):
                log_rank0(f"  {name:30s}: SKIPPED (n_pos={info['n_pos']})")
            else:
                log_rank0(f"  {name:30s}: AUC={info['auc']:.4f}  AP={info['ap']:.4f}  "
                          f"(n_pos={info['n_pos']})")

        results = {
            "best_mAUC": final_metrics["mean_auc"],
            "best_mAP": final_metrics["mean_ap"],
            "best_epoch": best_epoch,
            "total_epochs_trained": epoch,
            "per_class": final_metrics["per_class"],
        }
        results_path = os.path.join(args.output_dir, "finetune_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log_rank0(f"Results saved to {results_path}")

        results_txt = os.path.join(args.output_dir, "finetune_results.txt")
        with open(results_txt, "w") as f:
            f.write(f"Best mAUC: {final_metrics['mean_auc']:.4f}\n")
            f.write(f"Best mAP: {final_metrics['mean_ap']:.4f}\n")
            f.write(f"Best epoch: {best_epoch}\n\n")
            for name, info in final_metrics["per_class"].items():
                if not info.get("skipped"):
                    f.write(f"{name}: AUC={info['auc']:.4f} AP={info['ap']:.4f} "
                            f"n_pos={info['n_pos']}\n")

    if use_ddp:
        cleanup_ddp()


def parse_args():
    parser = argparse.ArgumentParser(description="TexJEPA Fine-tune on VinBigData (DDP)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./downstream_results/finetune")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id for single-GPU mode (ignored in DDP)")
    parser.add_argument("--batch_size", type=int, default=48,
                        help="Per-GPU batch size")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output_dir")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_finetune(args)
