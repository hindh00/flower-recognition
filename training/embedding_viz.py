"""t-SNE visualization of the classifier's penultimate-layer embeddings on
the test set.

A dataset-level view of class separability that complements Grad-CAM's per-image
heatmaps: classes that cluster together or overlap here are exactly the ones
that show up in evaluate.py's "most confused pairs" table, which Grad-CAM
can't demonstrate on its own."""

import argparse
from pathlib import Path

import _sympy_fix  # noqa: F401
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from dataset import get_dataloaders
from report_utils import append_results_section
from train import build_model


"""Collects embeddings from the model's global-pool layer."""
@torch.no_grad()
def collect_embeddings(model, loader, device):
    model.eval()
    embeddings, labels = [], []

    def hook(_module, _input, output):
        embeddings.append(output.flatten(1).cpu())

    handle = model.global_pool.register_forward_hook(hook)
    try:
        for images, batch_labels in loader:
            model(images.to(device))
            labels.append(batch_labels)
    finally:
        handle.remove()
    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()


def plot_tsne(embeddings, labels, out_path, perplexity=30, seed=42):
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca").fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="nipy_spectral", s=8, alpha=0.7)
    ax.set_title("t-SNE of test-set embeddings (penultimate layer), colored by class")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(scatter, ax=ax, label="class index", fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(
    data_root="data",
    checkpoint="checkpoints/best_model.pth",
    backbone="efficientnet_b0",
    out_dir="../docs/images",
    results_path="../docs/results.md",
    perplexity=30,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    loaders, _ = get_dataloaders(data_root)

    model = build_model(backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    embeddings, labels = collect_embeddings(model, loaders["test"], device)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fname = "tsne_embeddings.png"
    plot_tsne(embeddings, labels, out_path / fname, perplexity=perplexity)
    print(f"Saved {out_path / fname}")

    if results_path:
        append_results_section(
            Path(results_path),
            marker="## Explainability (t-SNE embeddings)",
            lines=[
                "t-SNE projection of test-set embeddings, colored by true class.",
                "",
                f"![t-SNE embeddings](images/{fname})",
            ],
        )
        print(f"Updated {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--out-dir", default="../docs/images")
    parser.add_argument("--results-path", default="../docs/results.md")
    parser.add_argument("--perplexity", type=float, default=30)
    args = parser.parse_args()
    run(args.data_root, args.checkpoint, args.backbone, args.out_dir, args.results_path, args.perplexity)
