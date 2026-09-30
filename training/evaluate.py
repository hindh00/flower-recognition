"""Test-set evaluation: top-1/top-5 accuracy, a full per-class report, a
confusion-matrix heatmap, and a "most confused pairs" table (the full
102x102 matrix isn't readable at a glance). Writes docs/results.md and
docs/images/confusion_matrix.png for the README.

    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score

from dataset import NUM_CLASSES, get_dataloaders, load_class_names
from train import build_model


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_logits = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        all_logits.append(outputs.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def most_confused_pairs(cm, class_names, top_n=15):
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    return pairs[:top_n]


def plot_confusion_matrix(cm, out_path, normalize=True):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (102 classes, row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate(
    data_root="data",
    checkpoint="checkpoints/best_model.pth",
    backbone="efficientnet_b0",
    cat_to_name_path="cat_to_name.json",
    out_dir="../docs",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, _ = get_dataloaders(data_root)
    class_names = load_class_names(cat_to_name_path)

    model = build_model(backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    logits, labels = collect_predictions(model, loaders["test"], device)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    labels_np = labels.numpy()

    top1 = float((preds == labels_np).mean())
    top5 = float(top_k_accuracy_score(labels_np, probs, k=5, labels=list(range(NUM_CLASSES))))
    report = classification_report(labels_np, preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(labels_np, preds, labels=list(range(NUM_CLASSES)))
    confused = most_confused_pairs(cm, class_names)

    out_path = Path(out_dir)
    (out_path / "images").mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, out_path / "images" / "confusion_matrix.png")

    with open(out_path / "results.md", "w") as f:
        f.write("# Evaluation results\n\n")
        f.write(f"- **Top-1 test accuracy**: {top1:.4f}\n")
        f.write(f"- **Top-5 test accuracy**: {top5:.4f}\n\n")
        f.write("## Most confused class pairs\n\n")
        f.write("| Count | True class | Predicted as |\n|---|---|---|\n")
        for count, true_name, pred_name in confused:
            f.write(f"| {count} | {true_name} | {pred_name} |\n")
        f.write("\n## Full classification report\n\n```\n")
        f.write(report)
        f.write("```\n\n")
        f.write("![Confusion matrix](images/confusion_matrix.png)\n")

    print(f"Top-1: {top1:.4f}  Top-5: {top5:.4f}")
    print(f"Wrote {out_path / 'results.md'} and confusion_matrix.png")
    return top1, top5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--cat-to-name", default="cat_to_name.json")
    parser.add_argument("--out-dir", default="../docs")
    args = parser.parse_args()
    evaluate(args.data_root, args.checkpoint, args.backbone, args.cat_to_name, args.out_dir)
