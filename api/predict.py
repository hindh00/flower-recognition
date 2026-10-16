"""Vercel Python Function: flower classification inference.

Accepts a raw image (POST body, produced by the frontend's client-side
resize step) and returns the top-5 predicted classes. Uses onnxruntime
rather than full PyTorch so the deployed bundle stays small and fast.
"""

import io
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_DIR = Path(__file__).parent / "model"
IMAGE_SIZE = 224
RESIZE_SHORT_SIDE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_BODY_BYTES = 4_500_000  # Vercel Functions cap request/response bodies at 4.5 MB
LOW_CONFIDENCE_THRESHOLD = 0.5  # below this, the top prediction is shown as "not sure"

# Loaded once at module scope so warm Fluid Compute instances reuse the
# session across invocations instead of reloading the model every request.
_session = ort.InferenceSession(
    str(MODEL_DIR / "flower_model.onnx"), providers=["CPUExecutionProvider"]
)
_input_name = _session.get_inputs()[0].name
with open(MODEL_DIR / "class_names.json") as f:
    _class_names = json.load(f)


def _to_tensor(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)  # HWC -> CHW


def _five_crop(image: Image.Image, crop_size: int) -> list[Image.Image]:
    """Center crop + 4 corner crops. Used as test-time augmentation so a
    photo with multiple flowers or a cluttered background isn't judged on a
    single, possibly off-center, crop."""
    w, h = image.size
    left_max, top_max = w - crop_size, h - crop_size
    boxes = [
        (0, 0),
        (left_max, 0),
        (0, top_max),
        (left_max, top_max),
        (left_max // 2, top_max // 2),  # center
    ]
    return [image.crop((l, t, l + crop_size, t + crop_size)) for l, t in boxes]


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Resize-256 + 5-crop-224 (center + 4 corners) with ImageNet
    normalization. The center crop alone matches the eval-time transform
    used during training (see training/dataset.py); the corner crops are
    test-time augmentation only, averaged over in predict_topk."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    scale = RESIZE_SHORT_SIDE / min(w, h)
    image = image.resize((round(w * scale), round(h * scale)), Image.BILINEAR)

    crops = _five_crop(image, IMAGE_SIZE)
    batch = np.stack([_to_tensor(c) for c in crops])  # (5, 3, 224, 224)
    return batch.astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def predict_topk(image_bytes: bytes, k: int = 5):
    input_batch = preprocess(image_bytes)
    logits = _session.run(None, {_input_name: input_batch})[0]
    probs = softmax(logits).mean(axis=0)  # average over the 5 crops
    top_idx = np.argsort(probs)[::-1][:k]
    predictions = [{"label": _class_names[i], "confidence": float(probs[i])} for i in top_idx]
    low_confidence = bool(probs[top_idx[0]] < LOW_CONFIDENCE_THRESHOLD)
    return predictions, low_confidence


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self._send_json({"error": "Empty request body"}, status=400)
                return
            if content_length > MAX_BODY_BYTES:
                self._send_json({"error": "Image too large"}, status=413)
                return

            image_bytes = self.rfile.read(content_length)
            predictions, low_confidence = predict_topk(image_bytes)
            self._send_json({"predictions": predictions, "low_confidence": low_confidence})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
