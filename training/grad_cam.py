"""Grad-CAM visualizations for a handful of test-set predictions (a mix of
correct and incorrect ones makes the most interesting README material).

Optional stretch step — run after evaluate.py:
    python grad_cam.py --checkpoint checkpoints/best_model.pth \\
        --images data/flowers-102/jpg/image_00001.jpg data/flowers-102/jpg/image_00002.jpg \\
        --labels 0 1
"""

import argparse
from pathlib import Path

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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names(cat_to_name_path)
    model = build_model(backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # `conv_head` is the final 1x1 conv before global pooling in timm's
    # EfficientNet implementation — the standard Grad-CAM target layer for
    # this architecture family.
    cam = GradCAM(model=model, target_layers=[model.conv_head])
    _, eval_tf = get_transforms()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, (path, true_label) in enumerate(zip(image_paths, true_labels)):
        image = Image.open(path).convert("RGB")
        input_tensor = eval_tf(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_label = model(input_tensor).argmax(1).item()

        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_label)])[0]
        rgb_img = denormalize(input_tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy()
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        status = "correct" if pred_label == true_label else "wrong"
        fname = f"gradcam_{i:02d}_{status}_true-{class_names[true_label]}_pred-{class_names[pred_label]}.png"
        fname = fname.replace(" ", "-")
        Image.fromarray(visualization).save(out_path / fname)
        print(f"Saved {out_path / fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--cat-to-name", default="cat_to_name.json")
    parser.add_argument("--out-dir", default="../docs/images")
    parser.add_argument("--images", nargs="+", required=True, help="Paths to sample images")
    parser.add_argument("--labels", nargs="+", type=int, required=True, help="True label indices (0-indexed)")
    args = parser.parse_args()
    run_gradcam(args.images, args.labels, args.checkpoint, args.backbone, args.cat_to_name, args.out_dir)
