"""
MIMIC-CXR-JPG v2.0.0 dataset for I-JEPA self-supervised pretraining.

Supports two image-loading modes:
  - ZIP mode (use_zip=True):  reads JPGs directly from p10.zip..p19.zip
  - Extracted mode (use_zip=False): reads from extracted directory tree

Image enumeration strategy:
  - If `splits` is None  → parse IMAGE_FILENAMES (all 377K images)
  - If `splits` is given  → parse mimic-cxr-2.0.0-split.csv.gz with filtering
"""

import os
import gzip
import csv
import zipfile
import io
import random
from logging import getLogger
from multiprocessing import Value

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = getLogger()

MIMIC_ZIP_PREFIXES = [f'p{i}' for i in range(10, 20)]


def _parse_image_filenames(filenames_path):
    """Read IMAGE_FILENAMES and return structured entries.

    Each line has the form:
        files/p10/p10000032/s50414267/02aa804e-...-4e384014.jpg

    Returns list of dicts with keys: rel_path, zip_prefix, path_in_zip.
    """
    entries = []
    with open(filenames_path, 'r') as f:
        for line in f:
            rel_path = line.strip()
            if not rel_path or not rel_path.endswith('.jpg'):
                continue
            parts = rel_path.split('/')
            if len(parts) < 5:
                continue
            zip_prefix = parts[1]
            path_in_zip = '/'.join(parts[1:])
            entries.append({
                'rel_path': rel_path,
                'zip_prefix': zip_prefix,
                'path_in_zip': path_in_zip,
            })
    return entries


def _parse_split_csv(csv_path, splits):
    """Read mimic-cxr-2.0.0-split.csv.gz, filter by split, reconstruct paths.

    CSV columns: dicom_id, study_id, subject_id, split
    Returns list of dicts with keys: rel_path, zip_prefix, path_in_zip.
    """
    entries = []
    with gzip.open(csv_path, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] not in splits:
                continue
            dicom_id = row['dicom_id']
            subject_id = row['subject_id']
            study_id = row['study_id']
            subject_str = f'p{subject_id}'
            zip_prefix = subject_str[:3]
            path_in_zip = f'{zip_prefix}/{subject_str}/s{study_id}/{dicom_id}.jpg'
            rel_path = f'files/{path_in_zip}'
            entries.append({
                'rel_path': rel_path,
                'zip_prefix': zip_prefix,
                'path_in_zip': path_in_zip,
            })
    return entries


class MIMICCXRPretraining(Dataset):
    """MIMIC-CXR-JPG for self-supervised pretraining (images only, no labels).

    Args:
        root_path: Root directory containing IMAGE_FILENAMES, split CSV,
                   and files/ subdirectory with p10.zip..p19.zip.
        transform: torchvision transform pipeline.
        use_zip: If True, load images from ZIP archives. If False, expect
                 extracted directory tree under root_path/files/.
        splits: None to use all images, or a tuple/list of split names
                (e.g. ('train',)) to filter via split CSV.
    """

    def __init__(self, root_path, transform=None, use_zip=True, splits=None):
        super().__init__()
        self.root_path = root_path
        self.transform = transform
        self.use_zip = use_zip
        self._zip_handles = {}
        self._fail_counter = Value('i', 0)

        filenames_path = os.path.join(root_path, 'IMAGE_FILENAMES')
        split_csv_path = os.path.join(root_path, 'mimic-cxr-2.0.0-split.csv.gz')

        if splits is not None:
            if not os.path.exists(split_csv_path):
                raise FileNotFoundError(
                    f'splits={splits} requested but {split_csv_path} not found')
            self.entries = _parse_split_csv(split_csv_path, set(splits))
            logger.info(
                f'Loaded {len(self.entries)} images from split CSV '
                f'(splits={list(splits)})')
        elif os.path.exists(filenames_path):
            self.entries = _parse_image_filenames(filenames_path)
            logger.info(f'Loaded {len(self.entries)} images from IMAGE_FILENAMES')
        elif os.path.exists(split_csv_path):
            self.entries = _parse_split_csv(
                split_csv_path, {'train', 'validate', 'test'})
            logger.info(
                f'Loaded {len(self.entries)} images from split CSV (all splits)')
        else:
            raise FileNotFoundError(
                f'No IMAGE_FILENAMES or split CSV found at {root_path}')

        if use_zip:
            self._check_zip_availability()

        logger.info(f'MIMICCXRPretraining: {len(self.entries)} images ready')

    def _check_zip_availability(self):
        """Filter entries to only include images from available ZIP files."""
        zip_dir = os.path.join(self.root_path, 'files')
        available = set()
        for prefix in MIMIC_ZIP_PREFIXES:
            zpath = os.path.join(zip_dir, f'{prefix}.zip')
            if os.path.exists(zpath):
                fsize = os.path.getsize(zpath)
                if fsize > 1_000_000:
                    available.add(prefix)
                else:
                    logger.warning(
                        f'{prefix}.zip exists but is only {fsize} bytes '
                        f'(possibly incomplete)')
            else:
                logger.warning(f'{prefix}.zip not found at {zip_dir}')

        missing = sorted(set(MIMIC_ZIP_PREFIXES) - available)
        if missing:
            logger.warning(
                f'Missing/incomplete ZIPs: {missing} — '
                f'images from these will be skipped')

        before = len(self.entries)
        self.entries = [e for e in self.entries if e['zip_prefix'] in available]
        skipped = before - len(self.entries)
        if skipped > 0:
            logger.warning(
                f'Filtered out {skipped} images from unavailable ZIPs '
                f'({len(self.entries)} remaining)')

        logger.info(
            f'Available ZIPs: {sorted(available)} '
            f'({len(available)}/{len(MIMIC_ZIP_PREFIXES)})')

    def _open_image(self, entry):
        if self.use_zip:
            zip_prefix = entry['zip_prefix']
            path_in_zip = entry['path_in_zip']
            zpath = os.path.join(
                self.root_path, 'files', f'{zip_prefix}.zip')

            pid = os.getpid()
            key = (pid, zpath)
            if key not in self._zip_handles:
                self._zip_handles[key] = zipfile.ZipFile(zpath, 'r')
            zf = self._zip_handles[key]
            data = zf.read(path_in_zip)
            return Image.open(io.BytesIO(data)).convert('RGB')

        img_path = os.path.join(self.root_path, entry['rel_path'])
        return Image.open(img_path).convert('RGB')

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        for attempt in range(3):
            entry = self.entries[index]
            try:
                img = self._open_image(entry)
                if self.transform is not None:
                    img = self.transform(img)
                return img, index
            except Exception as e:
                with self._fail_counter.get_lock():
                    self._fail_counter.value += 1
                    total_fails = self._fail_counter.value
                if total_fails <= 50 or total_fails % 1000 == 0:
                    logger.warning(
                        f'Load failed (attempt {attempt+1}/3, '
                        f'total_fails={total_fails}): '
                        f'{entry["rel_path"]}: {e}')
                index = random.randint(0, len(self.entries) - 1)

        img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        return img, index


def make_mimic_pretrain(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    drop_last=True,
    use_zip=True,
    splits=None,
    **kwargs,
):
    """Create MIMIC-CXR pre-training dataset, sampler, and dataloader."""
    dataset = MIMICCXRPretraining(
        root_path=root_path,
        transform=transform,
        use_zip=use_zip,
        splits=splits,
    )
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    logger.info('MIMIC-CXR pretraining data loader created')
    return dataset, data_loader, dist_sampler
