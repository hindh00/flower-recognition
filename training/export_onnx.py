"""Export the fine-tuned PyTorch checkpoint to ONNX for the deployed API,
then verify the exported model's outputs match PyTorch's before trusting it
in production.

    python export_onnx.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
from pathlib import Path

import _sympy_fix  # noqa: F401  (must run before any torch import below)
import numpy as np
import onnx
import onnxruntime as ort
import torch

from dataset import load_class_names
from train import build_model


def export(
    checkpoint="checkpoints/best_model.pth",
    backbone="efficientnet_b0",
    cat_to_name_path="cat_to_name.json",
    out_path="../api/model/flower_model.onnx",
    opset=17,
    image_size=224,
):
    model = build_model(backbone, pretrained=False).to("cpu")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    onnx.checker.check_model(str(out_path))
    print(f"Exported ONNX model to {out_path}")

    # Parity check: PyTorch vs ONNX Runtime on several random inputs. If
    # this assertion ever fails, do not deploy the model — something in the
    # export (op support, opset mismatch) silently changed the math.
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(1, 3, image_size, image_size)
            torch_out = model(x).numpy()
            onnx_out = session.run(None, {input_name: x.numpy()})[0]
            max_diff = float(np.abs(torch_out - onnx_out).max())
            assert max_diff < 1e-3, f"PyTorch/ONNX outputs diverge: max diff {max_diff}"
    print("Parity check passed: PyTorch and ONNX Runtime outputs match within tolerance.")

    class_names = load_class_names(cat_to_name_path)
    class_names_path = out_path.parent / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Wrote {class_names_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--cat-to-name", default="cat_to_name.json")
    parser.add_argument("--out-path", default="../api/model/flower_model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.checkpoint, args.backbone, args.cat_to_name, args.out_path, args.opset)
