"""D-RISE (Detector Randomized Input Sampling for Explanation) saliency maps
for the YOLO flower detector.

D-RISE is model-agnostic and perturbation-based:
It masks random regions of the input hundreds of times and weights each mask by how much
it preserved the detector's ability to re-find the target box, which more directly
reflects what the box+confidence output actually depended on.

Single-class detector, so this drops D-RISE's class-similarity term (always
1 for a one-class model) and keeps only the box-IoU * confidence term."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import zoom
from ultralytics import YOLO

from report_utils import append_results_section

DETECTOR_IMGSZ = 640
PAD_COLOR = 114


def _letterbox(image: Image.Image, size: int = DETECTOR_IMGSZ) -> np.ndarray:
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (PAD_COLOR,) * 3)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return np.asarray(canvas, dtype=np.float32)


"""RISE-style masks: a small random binary grid, upsampled to image
    resolution with bilinear interpolation so mask edges are soft rather
    than hard-edged blocks."""
def _generate_masks(n_masks, grid_size, image_size, keep_prob=0.5, seed=42):
    rng = np.random.default_rng(seed)
    grid = (rng.random((n_masks, grid_size, grid_size)) < keep_prob).astype(np.float32)
    scale = image_size / grid_size
    masks = zoom(grid, (1, scale, scale), order=1)
    return masks[:, :image_size, :image_size]


def _box_iou(box_a, boxes_b):
    x1 = np.maximum(box_a[0], boxes_b[:, 0])
    y1 = np.maximum(box_a[1], boxes_b[:, 1])
    x2 = np.minimum(box_a[2], boxes_b[:, 2])
    y2 = np.minimum(box_a[3], boxes_b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-9, None)


"""Weights each mask by max_i[ IoU(target_box, box_i) * conf_i ] over the
    detections found in that masked image."""
def drise_saliency(yolo, canvas, target_box, n_masks=400, grid_size=8, keep_prob=0.5, conf=0.1, batch_size=16, seed=42):
    size = canvas.shape[0]
    masks = _generate_masks(n_masks, grid_size, size, keep_prob, seed)
    saliency = np.zeros((size, size), dtype=np.float32)
    total_weight = 0.0

    for start in range(0, n_masks, batch_size):
        batch_masks = masks[start:start + batch_size]
        batch_imgs = [(canvas * m[:, :, None]).astype(np.uint8) for m in batch_masks]
        results = yolo.predict(batch_imgs, imgsz=size, conf=conf, verbose=False)

        for mask, result in zip(batch_masks, results):
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            if len(boxes) == 0:
                continue
            weight = float((_box_iou(target_box, boxes) * confs).max())
            saliency += weight * mask
            total_weight += weight

    if total_weight > 0:
        saliency /= total_weight
    return saliency / (saliency.max() + 1e-12)


def run_drise(
    image_paths,
    checkpoint="runs/detect/flower_detector/weights/best.pt",
    out_dir="../docs/images",
    results_path="../docs/results.md",
    n_masks=400,
):
    yolo = YOLO(checkpoint)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        canvas = _letterbox(image)
        detections = yolo.predict(canvas.astype(np.uint8), imgsz=DETECTOR_IMGSZ, verbose=False)[0]
        if len(detections.boxes) == 0:
            print(f"Skipping {path}: detector found nothing on the full image")
            continue
        top_box = detections.boxes.xyxy[detections.boxes.conf.argmax()].cpu().numpy()

        saliency = drise_saliency(yolo, canvas, top_box, n_masks=n_masks)
        visualization = show_cam_on_image(canvas / 255.0, saliency, use_rgb=True)

        fname = f"drise_{i:02d}.png"
        Image.fromarray(visualization).save(out_path / fname)
        rows.append(fname)
        print(f"Saved {out_path / fname}")

    if results_path:
        lines = [f"Random mask used per image ({n_masks})."]
        for fname in rows:
            lines += [f"![D-RISE {fname}](images/{fname})", ""]
        append_results_section(Path(results_path), marker="## Detector explainability (D-RISE)", lines=lines)
        print(f"Updated {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/detect/flower_detector/weights/best.pt")
    parser.add_argument("--out-dir", default="../docs/images")
    parser.add_argument("--results-path", default="../docs/results.md")
    parser.add_argument("--images", nargs="+", required=True, help="Paths to sample images")
    parser.add_argument("--n-masks", type=int, default=400)
    args = parser.parse_args()
    run_drise(args.images, args.checkpoint, args.out_dir, args.results_path, args.n_masks)
