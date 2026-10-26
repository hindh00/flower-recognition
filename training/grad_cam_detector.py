"""EigenCAM visualizations for the YOLO flower detector.
EigenCAM only needs the target layer's activations and never touches the model's output.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO

from report_utils import append_results_section

DETECTOR_IMGSZ = 640
PAD_COLOR = 114


def _letterbox(image: Image.Image, size: int = DETECTOR_IMGSZ) -> Image.Image:
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (PAD_COLOR,) * 3)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def run_eigencam(
    image_paths,
    checkpoint="runs/detect/flower_detector/weights/best.pt",
    out_dir="../docs/images",
    results_path="../docs/results.md",
):
    yolo = YOLO(checkpoint)
    model = yolo.model
    model.eval()

    # The layer right before the Detect head - The last point where features are still a single spatial feature map
    target_layers = [model.model[-2]]
    cam = EigenCAM(model, target_layers)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        canvas = _letterbox(image)
        rgb_img = np.asarray(canvas, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).unsqueeze(0)

        # targets is never read by EigenCAM, it only reads activations
        # but must be non-None, passing None would make BaseCAM try to np.argmax() the detector's raw multi-head output as if it were a
        # single classification logits tensor.
        grayscale_cam = cam(input_tensor=input_tensor, targets=[0])[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        fname = f"eigencam_detector_{i:02d}.png"
        Image.fromarray(visualization).save(out_path / fname)
        rows.append(fname)
        print(f"Saved {out_path / fname}")

    if results_path:
        lines = [
            "EigenCAM heatmaps for the flower detector -- shows which region of the letterboxed "
            "model input (the square, gray-padded view the detector actually sees) most activates "
            "the feature map right before the detection head.",
            "",
        ]
        for fname in rows:
            lines += [f"![EigenCAM {fname}](images/{fname})", ""]
        append_results_section(Path(results_path), marker="## Detector explainability (EigenCAM)", lines=lines)
        print(f"Updated {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/detect/flower_detector/weights/best.pt")
    parser.add_argument("--out-dir", default="../docs/images")
    parser.add_argument("--results-path", default="../docs/results.md")
    parser.add_argument("--images", nargs="+", required=True, help="Paths to sample images")
    args = parser.parse_args()
    run_eigencam(args.images, args.checkpoint, args.out_dir, args.results_path)
