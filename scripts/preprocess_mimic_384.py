#!/usr/bin/env python3
"""Preprocess MIMIC-CXR JPEGs to 384x384 for fast training I/O.

Why 384 (not 224 or 256)?
  * Network input is 224x224 (required by Meta's ViT-H/14 pretrained
    pos_embed — 16x16 patches at patch_size=14).
  * At training time we do RandomResizedCrop(scale=(0.3, 1.0), size=224).
  * Storing at 384 preserves a good amount of random-crop scale diversity
    while shrinking 377,110 images from ~560 GB total down to ~8.7 GB,
    which fits 100% into OS page cache after the first epoch — training
    becomes compute-bound on the A6000 GPUs.
  * Grayscale chest X-rays compress well at q=90 (~22 KB/image).

I/O layout:
  Source:  data/mimic-cxr-jpg/files/p{10..19}/p{subj}/s{study}/{dicom}.jpg
  Target:  data/mimic-cxr-384/files/p{10..19}/p{subj}/s{study}/{dicom}.jpg
  Meta  :  Symlinked from the target root so ``MIMICCXRPretraining`` only
           needs ``root_path: data/mimic-cxr-384``.

Run:
  python scripts/preprocess_mimic_384.py \
      --src  data/mimic-cxr-jpg \
      --dst  data/mimic-cxr-384 \
      --size 384 --quality 90 --workers 64
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from PIL import Image


# --------------------------------------------------------------------------
# Worker — pickled and run in a pool. Must be top-level for multiprocessing.
# --------------------------------------------------------------------------

_G = {}  # process-global settings populated via pool initializer


def _init_worker(src_root: str, dst_root: str, size: int, quality: int):
    _G['src_root'] = src_root
    _G['dst_root'] = dst_root
    _G['size'] = size
    _G['quality'] = quality
    # Tell PIL not to spawn sub-threads — we already have 64 processes
    from PIL import features  # noqa
    os.environ.setdefault('OMP_NUM_THREADS', '1')


def _process_one(rel_path: str) -> tuple[str, int, str | None]:
    """Return (rel_path, output_bytes, error_string_or_None)."""
    src_path = os.path.join(_G['src_root'], rel_path)
    dst_path = os.path.join(_G['dst_root'], rel_path)

    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 1024:
        return rel_path, os.path.getsize(dst_path), None  # resume-safe

    try:
        with Image.open(src_path) as im:
            im = im.convert('RGB')
            im = im.resize((_G['size'], _G['size']), Image.LANCZOS)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            tmp = dst_path + '.tmp'
            im.save(tmp, 'JPEG', quality=_G['quality'], optimize=True)
            os.replace(tmp, dst_path)
        return rel_path, os.path.getsize(dst_path), None
    except Exception as e:
        return rel_path, 0, f'{type(e).__name__}: {e}'


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/mimic-cxr-jpg',
                    help='Source root (containing IMAGE_FILENAMES and files/pXX/)')
    ap.add_argument('--dst', default='data/mimic-cxr-384',
                    help='Destination root (will be created)')
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--quality', type=int, default=90)
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--progress-every', type=int, default=5000,
                    help='Print a progress line every N images')
    args = ap.parse_args()

    src_root = os.path.abspath(args.src)
    dst_root = os.path.abspath(args.dst)
    filenames_path = os.path.join(src_root, 'IMAGE_FILENAMES')

    if not os.path.isfile(filenames_path):
        print(f'ERROR: {filenames_path} not found', file=sys.stderr)
        sys.exit(2)

    os.makedirs(dst_root, exist_ok=True)

    # Symlink the metadata files so MIMICCXRPretraining(root_path=dst) just works.
    for meta in ('IMAGE_FILENAMES',
                 'mimic-cxr-2.0.0-split.csv.gz',
                 'mimic-cxr-2.0.0-metadata.csv.gz',
                 'mimic-cxr-2.0.0-chexpert.csv.gz'):
        src_meta = os.path.join(src_root, meta)
        dst_meta = os.path.join(dst_root, meta)
        if os.path.exists(src_meta) and not os.path.exists(dst_meta):
            os.symlink(src_meta, dst_meta)
            print(f'[symlink] {dst_meta} -> {src_meta}')

    # Load the rel_path list
    rel_paths = []
    with open(filenames_path) as f:
        for line in f:
            line = line.strip()
            if line.endswith('.jpg'):
                rel_paths.append(line)

    total = len(rel_paths)
    print(f'[*] Source           : {src_root}')
    print(f'[*] Destination      : {dst_root}')
    print(f'[*] Images to process: {total}')
    print(f'[*] Target size      : {args.size}x{args.size}, JPEG q={args.quality}')
    print(f'[*] Workers          : {args.workers}')
    print('[*] Resume-safe      : YES (skips already-written non-empty files)')

    t0 = time.time()
    done = 0
    fail = 0
    bytes_written = 0
    errors: list[str] = []

    ctx = mp.get_context('spawn')
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(src_root, dst_root, args.size, args.quality),
    ) as pool:
        for rel_path, nbytes, err in pool.imap_unordered(
                _process_one, rel_paths, chunksize=32):
            done += 1
            if err is not None:
                fail += 1
                if len(errors) < 30:
                    errors.append(f'{rel_path} :: {err}')
            else:
                bytes_written += nbytes

            if done % args.progress_every == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta_sec = (total - done) / rate if rate > 0 else 0
                avg_kb = (bytes_written / (done - fail) / 1024) if (done - fail) else 0
                print(f'  [{done:>7}/{total}] '
                      f'{100*done/total:5.1f}% | '
                      f'rate {rate:6.1f} img/s | '
                      f'fail {fail} | '
                      f'avg {avg_kb:4.1f} KB | '
                      f'elapsed {elapsed/60:.1f} min | '
                      f'ETA {eta_sec/60:.1f} min',
                      flush=True)

    total_elapsed = time.time() - t0
    print()
    print(f'[+] DONE in {total_elapsed/60:.1f} minutes')
    print(f'[+] Processed  : {done - fail} / {total} images ({100*(done-fail)/total:.2f}%)')
    print(f'[+] Failures   : {fail}')
    print(f'[+] Total size : {bytes_written/1e9:.2f} GB')
    print(f'[+] Avg size   : {bytes_written/(done-fail)/1024:.1f} KB' if (done-fail) else '')

    if errors:
        print('[!] First errors:')
        for e in errors[:20]:
            print('   ', e)


if __name__ == '__main__':
    main()
