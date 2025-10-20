#!/usr/bin/env python3
"""Preprocess JPG/JPEG images into a mirrored folder under data_preprocessed.

Features:
- Recursively finds .jpg/.jpeg files under input dir
- Resizes to square (default 256) using Lanczos
- Converts to RGB, normalizes to float32 in [0,1]
- Saves as .npy by default (optionally .png or both)
- Mirrors folder structure under output dir
- Optional --max-files to run a small validation run
"""
from pathlib import Path
import argparse
import shutil
import logging
import sys

from PIL import Image
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

IMAGE_EXTS = {".jpg", ".jpeg"}


def process_image(src: Path, dst_dir: Path, img_size: int, save_as: str):
    try:
        im = Image.open(src).convert("RGB")
    except Exception as e:
        logging.warning(f"Unable to open image {src}: {e}")
        return False

    try:
        im = im.resize((img_size, img_size), Image.LANCZOS)
        arr = np.asarray(im).astype(np.float32)
        # normalize to [0,1]
        if arr.max() > 1.0:
            arr = arr / 255.0
    except Exception as e:
        logging.warning(f"Failed to process image {src}: {e}")
        return False

    dst_dir.mkdir(parents=True, exist_ok=True)
    base = dst_dir / src.name
    if save_as in ("png", "both"):
        out_png = base.with_suffix('.png')
        try:
            im.save(out_png)
        except Exception as e:
            logging.warning(f"Failed to save PNG {out_png}: {e}")

    if save_as in ("npy", "both"):
        out_npy = base.with_suffix('.npy')
        try:
            np.save(out_npy, arr, allow_pickle=False)
        except Exception as e:
            logging.warning(f"Failed to save NPY {out_npy}: {e}")

    return True


def find_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def main(argv=None):
    p = argparse.ArgumentParser(description="Preprocess JPG images into data_preprocessed")
    p.add_argument("--input-dir", type=Path, default=Path("data"), help="Input data directory")
    p.add_argument("--output-dir", type=Path, default=Path("data_preprocessed"), help="Output dir")
    p.add_argument("--img-size", type=int, default=256, help="Resize images to this square size")
    p.add_argument("--save-as", choices=("npy", "png", "both"), default="npy", help="Save format")
    p.add_argument("--clean", action="store_true", help="Remove existing output dir before processing")
    p.add_argument("--max-files", type=int, default=0, help="If >0, process at most this many files (for quick test)")
    args = p.parse_args(argv)

    inp = args.input_dir
    out = args.output_dir

    if not inp.exists():
        logging.error(f"Input directory {inp} not found")
        return 2

    if args.clean and out.exists():
        logging.info(f"Removing existing output dir {out}")
        shutil.rmtree(out)

    files = find_images(inp)
    total = len(files)
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    logging.info(f"Found {total} JPG files under {inp}. Processing {len(files)} into {out}")

    succeeded = 0
    for src in tqdm(files, desc="Preprocessing"):
        rel = src.relative_to(inp)
        dst_dir = out / rel.parent
        ok = process_image(src, dst_dir, args.img_size, args.save_as)
        if ok:
            succeeded += 1

    logging.info(f"Done. Succeeded: {succeeded}/{len(files)} (found {total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
