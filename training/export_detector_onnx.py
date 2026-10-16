"""Exports the fine-tuned YOLO detector to ONNX with NMS baked in
(ultralytics>=8.3.67 `nms=True`), then verifies the exported model's boxes
match the PyTorch model's boxes on real sample images before trusting it.

    python export_detector_onnx.py --checkpoint runs/detect/flower_detector/weights/best.pt
"""

import argparse
import glob
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def _box_iou(boxes_a, boxes_b):
    """Pairwise IoU between two (N, 4) xyxy box arrays."""
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def export(checkpoint="runs/detect/flower_detector/weights/best.pt",
           out_path="../api/model/flower_detector.onnx",
           imgsz=640, conf=0.25, iou=0.45,
           sample_glob="data/detection/images/val/*.jpg", num_samples=10):
    pt_model = YOLO(checkpoint)
    exported = Path(pt_model.export(format="onnx", nms=True, imgsz=imgsz, opset=17, conf=conf, iou=iou))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exported.replace(out_path)

    # Parity check against real validation images (not random tensors --
    # needs to exercise real detection+NMS, not just matrix-multiply
    # equivalence). Mirrors export_onnx.py's "don't ship a broken export"
    # philosophy for the classifier.
    onnx_model = YOLO(str(out_path))
    sample_paths = sorted(glob.glob(sample_glob))[:num_samples]
    assert sample_paths, f"No sample images found at {sample_glob}"
    for img_path in sample_paths:
        pt_boxes = pt_model.predict(img_path, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0].boxes
        onnx_boxes = onnx_model.predict(img_path, imgsz=imgsz, verbose=False)[0].boxes
        assert len(pt_boxes) == len(onnx_boxes), (
            f"{img_path}: PyTorch found {len(pt_boxes)} boxes, ONNX found {len(onnx_boxes)}"
        )
        if len(pt_boxes):
            iou_matrix = _box_iou(pt_boxes.xyxy.cpu().numpy(), onnx_boxes.xyxy.cpu().numpy())
            assert iou_matrix.diagonal().min() > 0.9, f"{img_path}: PyTorch/ONNX boxes diverge (IoU < 0.9)"
    print(f"Parity check passed on {len(sample_paths)} sample images: PyTorch and ONNX Runtime boxes match.")
    print(f"Exported ONNX detector to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # The serving code's expected output shape is (1, max_det, 6) =
    # [x1, y1, x2, y2, conf, cls], zero-padded for unused detection slots.
    # Print it here so a mismatch is caught immediately, not discovered
    # later as a silent wrong-shape bug in api/predict.py.
    import onnxruntime as ort

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    print(f"Detector ONNX output spec: {session.get_outputs()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/detect/flower_detector/weights/best.pt")
    parser.add_argument("--out-path", default="../api/model/flower_detector.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    export(args.checkpoint, args.out_path, args.imgsz)
