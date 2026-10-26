"""Grad-CAM visualizations for a handful of test-set predictions."""

import argparse
from pathlib import Path

import _sympy_fix
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import IMAGENET_MEAN, IMAGENET_STD, get_transforms, load_class_names
from train import build_model


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def run_gradcam(
    image_paths,
    true_labels,
    checkpoint="checkpoints/best_model.pth",
    backbone="efficientnet_b0",
    cat_to_name_path="cat_to_name.json",
    out_dir="../docs/images",
    results_path="../docs/results.md",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    class_names = load_class_names(cat_to_name_path)
    model = build_model(backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    cam = GradCAM(model=model, target_layers=[model.conv_head])
    _, eval_tf = get_transforms()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (path, true_label) in enumerate(zip(image_paths, true_labels)):
        image = Image.open(path).convert("RGB")
        input_tensor = eval_tf(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_label = model(input_tensor).argmax(1).item()

        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_label)])[0]
        rgb_img = denormalize(input_tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy()
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        fname = f"gradcam_{i:02d}.png"
        Image.fromarray(visualization).save(out_path / fname)
        status = "correct" if pred_label == true_label else "WRONG"
        rows.append((fname, class_names[true_label], class_names[pred_label], status))
        print(f"Saved {out_path / fname}  (true={class_names[true_label]}, pred={class_names[pred_label]}, {status})")

    if results_path:
        _append_gradcam_section(Path(results_path), rows)


"""Appends a Grad-CAM section in
    docs/results.md so all generated results live in one place"""
def _append_gradcam_section(results_path: Path, rows) -> None:
    section_lines = [marker, "", "Heatmaps show which regions of the image most influenced the prediction.", ""]
    for fname, true_name, pred_name, status in rows:
        section_lines.append(f"**{status}** — true: `{true_name}`, predicted: `{pred_name}`")
        section_lines.append("")
        section_lines.append(f"![Grad-CAM {fname}](images/{fname})")
        section_lines.append("")
    new_section = "\n".join(section_lines).rstrip() + "\n"

    existing = results_path.read_text() if results_path.exists() else ""
    if marker in existing:
        before = existing.split(marker)[0].rstrip() + "\n\n"
    else:
        before = existing.rstrip() + "\n\n" if existing else ""
    results_path.write_text(before + new_section)
    print(f"Updated {results_path} with the Grad-CAM section")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--cat-to-name", default="cat_to_name.json")
    parser.add_argument("--out-dir", default="../docs/images")
    parser.add_argument("--results-path", default="../docs/results.md")
    parser.add_argument("--images", nargs="+", required=True, help="Paths to sample images")
    parser.add_argument("--labels", nargs="+", type=int, required=True, help="True label indices (0-indexed)")
    args = parser.parse_args()
    run_gradcam(args.images, args.labels, args.checkpoint, args.backbone, args.cat_to_name, args.out_dir, args.results_path)
