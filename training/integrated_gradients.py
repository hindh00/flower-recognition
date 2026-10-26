"""Integrated Gradients and SmoothGrad pixel-attribution maps (via captum)
for a handful of test-set predictions, finer-grained than Grad-CAM's
coarse conv-layer heatmaps."""

import argparse
from pathlib import Path

import _sympy_fix  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients, NoiseTunnel
from PIL import Image

from dataset import IMAGENET_MEAN, IMAGENET_STD, get_transforms, load_class_names
from report_utils import append_results_section
from train import build_model


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def _attribution_heatmap(attr):
    """(C, H, W) signed attribution -> (H, W) in [0, 1], summed over channels."""
    attr = np.abs(attr).sum(axis=0)
    return attr / (attr.max() + 1e-12)


def plot_attribution(rgb_img, ig_map, smoothgrad_map, out_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb_img)
    axes[0].set_title("input")
    axes[1].imshow(rgb_img)
    axes[1].imshow(ig_map, cmap="inferno", alpha=0.6)
    axes[1].set_title("Integrated Gradients")
    axes[2].imshow(rgb_img)
    axes[2].imshow(smoothgrad_map, cmap="inferno", alpha=0.6)
    axes[2].set_title("SmoothGrad")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_integrated_gradients(
    image_paths,
    true_labels,
    checkpoint="checkpoints/best_model.pth",
    backbone="efficientnet_b0",
    cat_to_name_path="cat_to_name.json",
    out_dir="../docs/images",
    results_path="../docs/results.md",
    ig_steps=50,
    smoothgrad_samples=25,
    smoothgrad_std=0.15,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    class_names = load_class_names(cat_to_name_path)
    model = build_model(backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ig = IntegratedGradients(model)
    smooth_ig = NoiseTunnel(ig)
    _, eval_tf = get_transforms()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (path, true_label) in enumerate(zip(image_paths, true_labels)):
        image = Image.open(path).convert("RGB")
        input_tensor = eval_tf(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_label = model(input_tensor).argmax(1).item()
        baseline = torch.zeros_like(input_tensor)

        ig_attr = ig.attribute(input_tensor, baselines=baseline, target=pred_label, n_steps=ig_steps)
        sg_attr = smooth_ig.attribute(
            input_tensor,
            nt_type="smoothgrad",
            nt_samples=smoothgrad_samples,
            stdevs=smoothgrad_std,
            baselines=baseline,
            target=pred_label,
        )

        rgb_img = denormalize(input_tensor.detach().squeeze(0).cpu()).permute(1, 2, 0).numpy()
        ig_map = _attribution_heatmap(ig_attr.squeeze(0).detach().cpu().numpy())
        sg_map = _attribution_heatmap(sg_attr.squeeze(0).detach().cpu().numpy())

        fname = f"ig_{i:02d}.png"
        status = "correct" if pred_label == true_label else "WRONG"
        title = f"{status} -- true: {class_names[true_label]}, predicted: {class_names[pred_label]}"
        plot_attribution(rgb_img, ig_map, sg_map, out_path / fname, title)
        rows.append((fname, class_names[true_label], class_names[pred_label], status))
        print(f"Saved {out_path / fname}  (true={class_names[true_label]}, pred={class_names[pred_label]}, {status})")

    if results_path:
        lines = ["Integrated Gradients and SmoothGrad pixel-attribution maps."]
        for fname, true_name, pred_name, status in rows:
            lines += [
                f"**{status}** -- true: `{true_name}`, predicted: `{pred_name}`",
                "",
                f"![Integrated Gradients {fname}](images/{fname})",
                "",
            ]
        append_results_section(Path(results_path), marker="## Explainability (Integrated Gradients / SmoothGrad)", lines=lines)
        print(f"Updated {results_path}")


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
    run_integrated_gradients(args.images, args.labels, args.checkpoint, args.backbone, args.cat_to_name, args.out_dir, args.results_path)
