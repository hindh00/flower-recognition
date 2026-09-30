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
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_BODY_BYTES = 4_500_000  # Vercel Functions cap request/response bodies at 4.5 MB

# Loaded once at module scope so warm Fluid Compute instances reuse the
# session across invocations instead of reloading the model every request.
_session = ort.InferenceSession(
    str(MODEL_DIR / "flower_model.onnx"), providers=["CPUExecutionProvider"]
)
_input_name = _session.get_inputs()[0].name
with open(MODEL_DIR / "class_names.json") as f:
    _class_names = json.load(f)


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Resize-256 / center-crop-224 + ImageNet normalization — matches the
    eval-time transform used during training (see training/dataset.py)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    scale = 256 / min(w, h)
    image = image.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = image.size
    left, top = (w - IMAGE_SIZE) // 2, (h - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # HWC -> NCHW
    return arr.astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def predict_topk(image_bytes: bytes, k: int = 5):
    input_tensor = preprocess(image_bytes)
    logits = _session.run(None, {_input_name: input_tensor})[0]
    probs = softmax(logits)[0]
    top_idx = np.argsort(probs)[::-1][:k]
    return [{"label": _class_names[i], "confidence": float(probs[i])} for i in top_idx]


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
            predictions = predict_topk(image_bytes)
            self._send_json({"predictions": predictions})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
