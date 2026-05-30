"""VinBigData Chest X-ray dataset for multi-label classification downstream task."""

import os
import csv
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image

VINBIG_CLASSES = [
    "Aortic enlargement",       # 0
    "Atelectasis",              # 1
    "Calcification",            # 2
    "Cardiomegaly",             # 3
    "Consolidation",            # 4
    "ILD",                      # 5
    "Infiltration",             # 6
    "Lung Opacity",             # 7
    "Nodule/Mass",              # 8
    "Other lesion",             # 9
    "Pleural effusion",         # 10
    "Pleural thickening",       # 11
    "Pneumothorax",             # 12
    "Pulmonary fibrosis",       # 13
]
NUM_CLASSES = len(VINBIG_CLASSES)  # 14

CLASS_NAME_TO_IDX = {
    "Aortic enlargement": 0,
    "Atelectasis": 1,
    "Calcification": 2,
    "Cardiomegaly": 3,
    "Consolidation": 4,
    "ILD": 5,
    "Infiltration": 6,
    "Lung Opacity": 7,
    "Nodule/Mass": 8,
    "Other lesion": 9,
    "Pleural effusion": 10,
    "Pleural thickening": 11,
    "Pneumothorax": 12,
    "Pulmonary fibrosis": 13,
}


def parse_vinbig_csv(csv_path):
    """Parse VinBigData train.csv into per-image multi-label vectors.

    The CSV has columns: image_id, class_name, class_id, rad_id,
    x_min, y_min, x_max, y_max.  class_id=14 means 'No finding'.
    Multiple rows per image (one per bbox per radiologist).

    Returns:
        dict[str, torch.Tensor]: image_id -> binary label vector of shape [14].
    """
    image_labels = defaultdict(lambda: torch.zeros(NUM_CLASSES))
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            class_name = row["class_name"]
            if class_name == "No finding":
                continue
            if class_name in CLASS_NAME_TO_IDX:
                image_labels[image_id][CLASS_NAME_TO_IDX[class_name]] = 1.0
    all_image_ids = set(image_labels.keys())
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = row["image_id"]
            if iid not in all_image_ids:
                image_labels[iid] = torch.zeros(NUM_CLASSES)
    return dict(image_labels)


class VinBigDataset(Dataset):
    """VinBigData multi-label classification dataset.

    Args:
        image_dir: Path to directory containing PNG images.
        csv_path: Path to train.csv with annotations.
        transform: Torchvision transforms to apply.
        image_ids: Optional list of image IDs to use (for train/val split).
    """

    def __init__(self, image_dir, csv_path, transform=None, image_ids=None):
        self.image_dir = image_dir
        self.transform = transform
        all_labels = parse_vinbig_csv(csv_path)
        if image_ids is not None:
            self.image_ids = [iid for iid in image_ids if iid in all_labels]
        else:
            self.image_ids = sorted(all_labels.keys())
        self.labels = {iid: all_labels[iid] for iid in self.image_ids}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.png")
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[image_id]
        return img, label


def make_train_val_split(csv_path, val_ratio=0.2, seed=42):
    """Deterministic stratified-ish split of VinBigData into train/val.

    Uses hash-based splitting for reproducibility without shuffling.
    """
    all_labels = parse_vinbig_csv(csv_path)
    image_ids = sorted(all_labels.keys())

    import hashlib
    train_ids, val_ids = [], []
    for iid in image_ids:
        h = int(hashlib.md5(iid.encode()).hexdigest(), 16) % 1000
        if h < int(val_ratio * 1000):
            val_ids.append(iid)
        else:
            train_ids.append(iid)

    return train_ids, val_ids
