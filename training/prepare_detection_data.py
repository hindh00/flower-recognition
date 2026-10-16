"""Derives YOLO-format single-class ("flower") bounding boxes from Oxford
102's segmentation trimaps, then writes an Ultralytics-ready dataset
directory + dataset.yaml.

    python prepare_detection_data.py --data-root data --out-dir data/detection
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from dataset import download_segmentations, load_all_samples


def mask_to_bbox(mask_path, min_fg_fraction=0.01):
    """Trimap -> normalized (xc, yc, w, h), or None if the mask is empty or
    degenerate.

    Oxford 102's trimaps encode background as blue-dominant and foreground
    as red/white-dominant (Nilsback & Zisserman, 2007). `r > b` is a
    first-pass heuristic -- verify it visually against real masks (see
    notebooks/train_detector_flowers102.ipynb's "Verify mask-to-box
    derivation" cell) before trusting it across all 8,189 images.
    """
    mask = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.int16)
    r, b = mask[..., 0], mask[..., 2]
    fg = r > b
    if fg.mean() < min_fg_fraction:
        return None
    ys, xs = np.nonzero(fg)
    h, w = mask.shape[:2]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h


def build_yolo_dataset(data_root="data", out_dir=None, val_size=0.15, seed=42, min_fg_fraction=0.01):
    out_dir = Path(out_dir or Path(data_root) / "detection")
    seg_dir = download_segmentations(data_root)
    image_paths = [Path(p) for p, _ in load_all_samples(data_root)]  # species label unused here

    pairs = []
    skipped = 0
    for img_path in image_paths:
        idx = img_path.stem.split("_")[1]  # "image_00001" -> "00001"
        mask_path = seg_dir / f"segmim_{idx}.jpg"
        bbox = mask_to_bbox(mask_path, min_fg_fraction=min_fg_fraction)
        if bbox is None:
            skipped += 1
            continue
        pairs.append((img_path, bbox))
    print(f"Derived boxes for {len(pairs)}/{len(image_paths)} images ({skipped} skipped: empty/degenerate mask)")

    train_pairs, val_pairs = train_test_split(pairs, test_size=val_size, random_state=seed)
    for split_name, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, (xc, yc, bw, bh) in split_pairs:
            link_path = img_dir / img_path.name
            if not link_path.exists():
                os.symlink(img_path.resolve(), link_path)  # avoid duplicating ~330MB of jpgs
            (lbl_dir / f"{img_path.stem}.txt").write_text(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: flower\n")
    print(f"Wrote YOLO dataset to {out_dir} ({len(train_pairs)} train / {len(val_pairs)} val)")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()
    build_yolo_dataset(args.data_root, args.out_dir, args.val_size)
