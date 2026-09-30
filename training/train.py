"""Two-stage transfer-learning fine-tune of an ImageNet-pretrained backbone
(via timm) on Oxford 102 Flowers.

Stage 1 warms up a fresh classifier head with the backbone frozen.
Stage 2 unfreezes the whole network and fine-tunes end-to-end at a lower,
cosine-annealed learning rate. Mixed precision is used automatically on GPU.

Run from Colab (T4 GPU, free tier) or locally with a CUDA GPU:
    python train.py --backbone efficientnet_b0 --finetune-epochs 15
"""

import argparse
import time
from pathlib import Path

import _sympy_fix  # noqa: F401  (must run before any torch import below)
import timm
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import NUM_CLASSES, get_dataloaders


def build_model(backbone="efficientnet_b0", num_classes=NUM_CLASSES, pretrained=True):
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)


def set_backbone_trainable(model, trainable: bool):
    """Freezes/unfreezes everything except the final classifier layer, which
    stays trainable in both stages."""
    classifier_param_names = {name for name, _ in model.named_parameters() if "classifier" in name or "fc" in name or "head" in name}
    for name, param in model.named_parameters():
        param.requires_grad = trainable or name in classifier_param_names


def run_epoch(model, loader, criterion, optimizer, device, train, scaler=None):
    model.train() if train else model.eval()
    use_amp = scaler is not None and device.type == "cuda"
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()

            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train(
    data_root="data",
    backbone="efficientnet_b0",
    warmup_epochs=3,
    finetune_epochs=15,
    batch_size=32,
    warmup_lr=1e-3,
    finetune_lr=1e-4,
    out_dir="checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders, _ = get_dataloaders(data_root, batch_size=batch_size)
    model = build_model(backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    def maybe_save(val_acc):
        nonlocal best_val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_path / "best_model.pth")

    print("\n=== Stage 1: warm-up (frozen backbone) ===")
    set_backbone_trainable(model, trainable=False)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=warmup_lr)
    for epoch in range(warmup_epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True, scaler=scaler)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        maybe_save(val_acc)
        print(
            f"[warmup {epoch + 1}/{warmup_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({time.time() - t0:.1f}s)"
        )

    print("\n=== Stage 2: full fine-tune ===")
    set_backbone_trainable(model, trainable=True)
    optimizer = AdamW(model.parameters(), lr=finetune_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs)
    for epoch in range(finetune_epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True, scaler=scaler)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        scheduler.step()
        maybe_save(val_acc)
        print(
            f"[finetune {epoch + 1}/{finetune_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={scheduler.get_last_lr()[0]:.2e} ({time.time() - t0:.1f}s)"
        )

    print(f"\nBest val accuracy: {best_val_acc:.4f}  ->  {out_path / 'best_model.pth'}")
    return out_path / "best_model.pth"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", default="checkpoints")
    args = parser.parse_args()
    train(
        data_root=args.data_root,
        backbone=args.backbone,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        warmup_lr=args.warmup_lr,
        finetune_lr=args.finetune_lr,
        out_dir=args.out_dir,
    )
