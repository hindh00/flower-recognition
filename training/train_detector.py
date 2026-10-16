"""Fine-tunes a YOLO detector (single class: "flower") on the Oxford 102
segmentation-derived boxes produced by prepare_detection_data.py.

    python train_detector.py --data data/detection/dataset.yaml --epochs 50
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train_detector(data="data/detection/dataset.yaml", model="yolo11n.pt", epochs=50, imgsz=640,
                    out_dir="runs/detect", seed=42):
    yolo = YOLO(model)
    results = yolo.train(data=data, epochs=epochs, imgsz=imgsz, project=out_dir, name="flower_detector", seed=seed)
    return Path(results.save_dir) / "weights" / "best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/detection/dataset.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-dir", default="runs/detect")
    args = parser.parse_args()
    print(train_detector(args.data, args.model, args.epochs, args.imgsz, args.out_dir))
