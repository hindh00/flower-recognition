"""Derives YOLO-format single-class ("flower") bounding boxes from Oxford
102's segmentation trimaps, then writes an Ultralytics-ready dataset
directory + dataset.yaml"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split

from dataset import download_segmentations, load_all_samples

BACKGROUND_RGB = np.array([0, 0, 254])
BACKGROUND_TOLERANCE = 60


"""Trimap -> list of normalized (xc, yc, w, h) boxes, one per connected foreground blob
clearing min_fg_fraction of the image area.

Each blob gets its own box instead of one box spanning all foreground
pixels, so images with more than one bloom don't collapse into a single box covering both"""
def mask_to_bboxes(mask_path, min_fg_fraction=0.01):
    mask = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.int16)
    fg = np.linalg.norm(mask - BACKGROUND_RGB, axis=-1) > BACKGROUND_TOLERANCE
    h, w = mask.shape[:2]
    min_pixels = min_fg_fraction * h * w

    labeled, n_components = ndimage.label(fg, structure=np.ones((3, 3)))
    boxes = []
    for component_id in range(1, n_components + 1):
        ys, xs = np.nonzero(labeled == component_id)
        if len(xs) < min_pixels:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        boxes.append(((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h))
    return boxes


def build_yolo_dataset(data_root="data", out_dir=None, val_size=0.15, seed=42, min_fg_fraction=0.01):
    out_dir = Path(out_dir or Path(data_root) / "detection")
    seg_dir = download_segmentations(data_root)
    image_paths = [Path(p) for p, _ in load_all_samples(data_root)]  # species label unused here

    pairs = []
    skipped = 0
    n_boxes = 0
    for img_path in image_paths:
        idx = img_path.stem.split("_")[1]  # "image_00001" -> "00001"
        mask_path = seg_dir / f"segmim_{idx}.jpg"
        boxes = mask_to_bboxes(mask_path, min_fg_fraction=min_fg_fraction)
        if not boxes:
            skipped += 1
            continue
        pairs.append((img_path, boxes))
        n_boxes += len(boxes)
    print(f"Derived {n_boxes} boxes across {len(pairs)}/{len(image_paths)} images ({skipped} skipped: no foreground blob)")

    train_pairs, val_pairs = train_test_split(pairs, test_size=val_size, random_state=seed)
    for split_name, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, boxes in split_pairs:
            link_path = img_dir / img_path.name
            if not link_path.exists():
                link_path.symlink_to(img_path.resolve())  # avoid duplicating ~330MB of jpgs
            lines = [f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n" for xc, yc, bw, bh in boxes]
            (lbl_dir / f"{img_path.stem}.txt").write_text("".join(lines))

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
