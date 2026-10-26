"""Confidence calibration: reliability diagram and Expected Calibration
Error (ECE) on the test set. 
It answers "should I trust a 0.97 confidence score?"
"""

import argparse
from pathlib import Path

import _sympy_fix  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import get_dataloaders
from evaluate import collect_predictions
from report_utils import append_results_section
from train import build_model


def expected_calibration_error(confidences, correct, n_bins=15):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidences, bin_edges[1:-1]), 0, n_bins - 1)

    ece = 0.0
    bin_accs, bin_confs = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        count = mask.sum()
        bin_center = (bin_edges[b] + bin_edges[b + 1]) / 2
        if count == 0:
            bin_accs.append(np.nan)
            bin_confs.append(bin_center)
            continue
        acc, conf = correct[mask].mean(), confidences[mask].mean()
        bin_accs.append(acc)
        bin_confs.append(conf)
        ece += (count / len(confidences)) * abs(acc - conf)
    return ece, np.array(bin_confs), np.array(bin_accs)


def plot_reliability_diagram(bin_confs, bin_accs, ece, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    valid = ~np.isnan(bin_accs)
    width = (bin_confs[1] - bin_confs[0]) * 0.9 if len(bin_confs) > 1 else 0.05
    ax.bar(bin_confs[valid], bin_accs[valid], width=width, alpha=0.7, edgecolor="black", label="observed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (predicted probability)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram (ECE = {ece:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(
    data_root="data",
    checkpoint="checkpoints/best_model.pth",
    backbone="efficientnet_b0",
    out_dir="../docs/images",
    results_path="../docs/results.md",
    n_bins=15,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    loaders, _ = get_dataloaders(data_root)

    model = build_model(backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    logits, labels = collect_predictions(model, loaders["test"], device)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = preds == labels.numpy()

    ece, bin_confs, bin_accs = expected_calibration_error(confidences, correct, n_bins=n_bins)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fname = "reliability_diagram.png"
    plot_reliability_diagram(bin_confs, bin_accs, ece, out_path / fname)
    print(f"ECE: {ece:.4f}")
    print(f"Saved {out_path / fname}")

    if results_path:
        append_results_section(
            Path(results_path),
            marker="## Explainability (Confidence calibration)",
            lines=[
                f"Expected Calibration Error (ECE): **{ece:.4f}** over {n_bins} bins -- lower is "
                "better; 0 means predicted confidence always matches observed accuracy. Bars below "
                "the diagonal mean the model is overconfident in that confidence range, which is the "
                "failure mode that matters most for trusting a single prediction (see `LOW_CONFIDENCE_THRESHOLD` "
                "in `api/predict.py`).",
                "",
                f"![Reliability diagram](images/{fname})",
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
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()
    run(args.data_root, args.checkpoint, args.backbone, args.out_dir, args.results_path, args.n_bins)
