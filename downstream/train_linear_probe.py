"""Linear probing: freeze encoder, train a linear classifier on top.

This is the gold-standard evaluation for self-supervised representations.
If TexJEPA learned good chest X-ray features, a simple linear layer should
achieve strong AUC on VinBigData classification.
"""

import os
import sys
import time
import logging
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.transforms import CHEST_XRAY_NORM
from downstream.encoder import load_target_encoder, TexJEPAFeatureExtractor
from downstream.vinbig_dataset import (
    VinBigDataset, make_train_val_split, NUM_CLASSES, VINBIG_CLASSES,
)
from downstream.metrics import compute_multilabel_metrics

import torchvision.transforms as T

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_eval_transform(crop_size=224):
    return T.Compose([
        T.Resize(int(crop_size * 256 / 224)),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(*CHEST_XRAY_NORM),
    ])


def build_train_transform(crop_size=224):
    return T.Compose([
        T.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(*CHEST_XRAY_NORM),
    ])


@torch.no_grad()
def extract_features(feature_extractor, dataloader, device):
    """Pre-extract all features once to speed up linear probe training."""
    all_feats, all_labels = [], []
    feature_extractor.eval()
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        feats = feature_extractor(imgs)
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


class LinearClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_dim, affine=False)
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.norm(x)
        return self.linear(x)


def train_linear_probe(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Load encoder ---
    encoder = load_target_encoder(args.checkpoint, device=device)
    feature_extractor = TexJEPAFeatureExtractor(encoder).to(device)
    embed_dim = feature_extractor.embed_dim
    logger.info(f"Encoder loaded, embed_dim={embed_dim}")

    # --- Datasets ---
    train_ids, val_ids = make_train_val_split(args.csv_path, val_ratio=0.2, seed=42)
    logger.info(f"Split: {len(train_ids)} train, {len(val_ids)} val")

    train_transform = build_train_transform(crop_size=224)
    val_transform = build_eval_transform(crop_size=224)

    train_dataset = VinBigDataset(args.image_dir, args.csv_path,
                                  transform=train_transform, image_ids=train_ids)
    val_dataset = VinBigDataset(args.image_dir, args.csv_path,
                                transform=val_transform, image_ids=val_ids)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- Check label distribution ---
    all_labels_check = torch.stack([train_dataset.labels[iid] for iid in train_dataset.image_ids])
    pos_per_class = all_labels_check.sum(dim=0)
    for i, name in enumerate(VINBIG_CLASSES):
        logger.info(f"  {name}: {int(pos_per_class[i])} positives "
                     f"({100*pos_per_class[i]/len(train_dataset):.1f}%)")

    extract_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # --- Extract features ---
    logger.info("Extracting train features...")
    t0 = time.time()
    train_feats, train_labels = extract_features(feature_extractor, extract_loader, device)
    logger.info(f"Train features: {train_feats.shape}, took {time.time()-t0:.1f}s")

    logger.info("Extracting val features...")
    t0 = time.time()
    val_feats, val_labels = extract_features(feature_extractor, val_loader, device)
    logger.info(f"Val features: {val_feats.shape}, took {time.time()-t0:.1f}s")

    del feature_extractor, encoder
    torch.cuda.empty_cache()

    # --- Train linear classifier ---
    classifier = LinearClassifier(embed_dim, NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Class-balanced weighting for BCE loss
    n_total = float(len(train_labels))
    pos_counts = train_labels.sum(dim=0).clamp(min=1)
    pos_weight = ((n_total - pos_counts) / pos_counts).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_feats_gpu = train_feats.to(device)
    train_labels_gpu = train_labels.to(device)
    val_feats_gpu = val_feats.to(device)

    best_auc = 0.0
    best_epoch = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        classifier.train()

        n = train_feats_gpu.size(0)
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, args.train_batch):
            idx = perm[i:i + args.train_batch]
            feats = train_feats_gpu[idx]
            labels = train_labels_gpu[idx]

            logits = classifier(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # --- Evaluate ---
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_feats_gpu)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_gt = val_labels.numpy()

        metrics = compute_multilabel_metrics(val_gt, val_probs, VINBIG_CLASSES)
        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now = scheduler.get_last_lr()[0]

        is_best = metrics["mean_auc"] > best_auc
        if is_best:
            best_auc = metrics["mean_auc"]
            best_epoch = epoch
            torch.save(classifier.state_dict(),
                       os.path.join(args.output_dir, "best_linear_probe.pth"))

        if epoch % 10 == 0 or epoch == 1 or is_best:
            logger.info(
                f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | lr={lr_now:.6f} | "
                f"mAUC={metrics['mean_auc']:.4f} | mAP={metrics['mean_ap']:.4f}"
                f"{' *BEST*' if is_best else ''}"
            )

    # --- Final report ---
    logger.info("=" * 60)
    logger.info("LINEAR PROBE COMPLETE")
    logger.info(f"Best mAUC: {best_auc:.4f} at epoch {best_epoch}")
    logger.info("=" * 60)

    classifier.load_state_dict(
        torch.load(os.path.join(args.output_dir, "best_linear_probe.pth"),
                   map_location=device, weights_only=True))
    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(val_feats_gpu)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
    final_metrics = compute_multilabel_metrics(val_labels.numpy(), val_probs, VINBIG_CLASSES)

    logger.info(f"Final mAUC={final_metrics['mean_auc']:.4f}, mAP={final_metrics['mean_ap']:.4f}")
    logger.info("-" * 60)
    for name, info in final_metrics["per_class"].items():
        if info.get("skipped"):
            logger.info(f"  {name:30s}: SKIPPED (n_pos={info['n_pos']})")
        else:
            logger.info(f"  {name:30s}: AUC={info['auc']:.4f}  AP={info['ap']:.4f}  "
                         f"(n_pos={info['n_pos']})")

    results_path = os.path.join(args.output_dir, "linear_probe_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Best mAUC: {final_metrics['mean_auc']:.4f}\n")
        f.write(f"Best mAP: {final_metrics['mean_ap']:.4f}\n")
        f.write(f"Best epoch: {best_epoch}\n\n")
        for name, info in final_metrics["per_class"].items():
            if not info.get("skipped"):
                f.write(f"{name}: AUC={info['auc']:.4f} AP={info['ap']:.4f} n_pos={info['n_pos']}\n")
    logger.info(f"Results saved to {results_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="TexJEPA Linear Probe on VinBigData")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to TexJEPA pretrained checkpoint (.pth.tar)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to VinBigData image directory (train/)")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to VinBigData train.csv")
    parser.add_argument("--output_dir", type=str, default="./downstream_results/linear_probe")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for feature extraction")
    parser.add_argument("--train_batch", type=int, default=256,
                        help="Batch size for linear probe training (on cached features)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_linear_probe(args)
