"""Vercel Python Function: flower classification inference.

Accepts a raw image (POST body, produced by the frontend's client-side
resize step) and returns the top-5 predicted classes. Uses onnxruntime
rather than full PyTorch so the deployed bundle stays small and fast.

Two-stage pipeline: a YOLO detector first locates flower region(s) in the
photo (so a cluttered background or multiple blooms don't drown out the
classifier), and each detected region is classified separately and
aggregated. If the detector finds nothing -- or its model file isn't
present at all -- this falls back to classifying the whole image via
5-crop test-time augmentation, so the app still works without a trained
detector.
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

DETECTOR_IMGSZ = 640
DETECTOR_PAD_COLOR = 114  # Ultralytics' standard letterbox fill value
BOX_PAD_FRAC = 0.15  # context padding around each detected box before classifying
MAX_BOXES_TO_CLASSIFY = 5  # bounds latency/payload on busy, multi-flower photos

# Loaded once at module scope so warm Fluid Compute instances reuse the
# session across invocations instead of reloading the model every request.
_session = ort.InferenceSession(
    str(MODEL_DIR / "flower_model.onnx"), providers=["CPUExecutionProvider"]
)
_input_name = _session.get_inputs()[0].name
with open(MODEL_DIR / "class_names.json") as f:
    _class_names = json.load(f)

# The detector is optional: if its ONNX file isn't present (e.g. not yet
# trained/committed), detect_boxes() below returns [] and predict_topk()
# falls back to whole-image classification -- the app still works.
try:
    _detector_session = ort.InferenceSession(
        str(MODEL_DIR / "flower_detector.onnx"), providers=["CPUExecutionProvider"]
    )
    _detector_input_name = _detector_session.get_inputs()[0].name
except Exception as exc:
    print(f"Detector model unavailable ({exc!r}); serving whole-image classification only.")
    _detector_session = None
    _detector_input_name = None


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


def _resize_short_side(image: Image.Image, size: int) -> Image.Image:
    w, h = image.size
    scale = size / min(w, h)
    return image.resize((round(w * scale), round(h * scale)), Image.BILINEAR)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _letterbox(image: Image.Image, size: int):
    """Resize preserving aspect ratio, pad to a size x size square -- the
    standard YOLO input transform. Returns the canvas plus the scale/offset
    needed to map detector-space boxes back to the original image."""
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (DETECTOR_PAD_COLOR,) * 3)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def detect_boxes(image: Image.Image) -> list[dict]:
    """Returns [{"box": (x1, y1, x2, y2), "confidence": c}, ...] in the
    ORIGINAL image's pixel coordinates, sorted by confidence desc, capped at
    MAX_BOXES_TO_CLASSIFY. Empty list means "no detections" -- the signal
    that triggers the whole-image fallback in predict_topk()."""
    if _detector_session is None:
        return []

    canvas, scale, pad_x, pad_y = _letterbox(image, DETECTOR_IMGSZ)
    arr = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
    output = _detector_session.run(None, {_detector_input_name: arr.astype(np.float32)})[0]
    rows = output[0]  # (max_det, 6): [x1, y1, x2, y2, conf, cls] -- NMS already applied at export

    boxes = []
    for row in rows:
        x1, y1, x2, y2, conf = row[0], row[1], row[2], row[3], row[4]
        if conf <= 0:  # NMS-baked export zero-pads unused detection slots
            continue
        boxes.append({
            "box": (
                max((x1 - pad_x) / scale, 0),
                max((y1 - pad_y) / scale, 0),
                min((x2 - pad_x) / scale, image.width),
                min((y2 - pad_y) / scale, image.height),
            ),
            "confidence": float(conf),
        })
    boxes.sort(key=lambda b: -b["confidence"])
    return boxes[:MAX_BOXES_TO_CLASSIFY]


def _pad_and_letterbox_crop(image: Image.Image, box) -> np.ndarray:
    """Crops a detected box (padded for context) and letterboxes it to
    IMAGE_SIZE x IMAGE_SIZE -- preserves aspect ratio instead of stretching,
    since the box is already a tight, detector-isolated flower region."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = w * BOX_PAD_FRAC, h * BOX_PAD_FRAC
    x1, y1 = max(x1 - pad_w, 0), max(y1 - pad_h, 0)
    x2, y2 = min(x2 + pad_w, image.width), min(y2 + pad_h, image.height)
    crop = image.crop((round(x1), round(y1), round(x2), round(y2)))

    cw, ch = crop.size
    scale = IMAGE_SIZE / max(cw, ch)
    nw, nh = round(cw * scale), round(ch * scale)
    resized = crop.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    canvas.paste(resized, ((IMAGE_SIZE - nw) // 2, (IMAGE_SIZE - nh) // 2))
    return _to_tensor(canvas)


def classify_from_boxes(image: Image.Image, boxes: list[dict], k: int):
    batch = np.stack([_pad_and_letterbox_crop(image, b["box"]) for b in boxes]).astype(np.float32)
    logits = _session.run(None, {_input_name: batch})[0]
    probs = softmax(logits)  # (num_boxes, 102)

    weights = np.array([b["confidence"] for b in boxes], dtype=np.float32)
    weights /= weights.sum()
    avg_probs = (probs * weights[:, np.newaxis]).sum(axis=0)

    top_idx = np.argsort(avg_probs)[::-1][:k]
    predictions = [{"label": _class_names[i], "confidence": float(avg_probs[i])} for i in top_idx]
    low_confidence = bool(avg_probs[top_idx[0]] < LOW_CONFIDENCE_THRESHOLD)
    return predictions, low_confidence


def classify_whole_image_tta(image: Image.Image, k: int):
    """Fallback when the detector finds nothing (or isn't available):
    resize-256 + 5-crop-224 (center + 4 corners), averaged -- the original
    whole-image test-time augmentation."""
    crops = _five_crop(_resize_short_side(image, RESIZE_SHORT_SIDE), IMAGE_SIZE)
    batch = np.stack([_to_tensor(c) for c in crops]).astype(np.float32)
    logits = _session.run(None, {_input_name: batch})[0]
    probs = softmax(logits).mean(axis=0)
    top_idx = np.argsort(probs)[::-1][:k]
    predictions = [{"label": _class_names[i], "confidence": float(probs[i])} for i in top_idx]
    low_confidence = bool(probs[top_idx[0]] < LOW_CONFIDENCE_THRESHOLD)
    return predictions, low_confidence


def predict_topk(image_bytes: bytes, k: int = 5):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    boxes = detect_boxes(image)

    if boxes:
        predictions, low_confidence = classify_from_boxes(image, boxes, k)
    else:
        predictions, low_confidence = classify_whole_image_tta(image, k)

    detections = [
        {
            "box": [
                float(b["box"][0] / image.width),
                float(b["box"][1] / image.height),
                float(b["box"][2] / image.width),
                float(b["box"][3] / image.height),
            ],
            "confidence": b["confidence"],
        }
        for b in boxes
    ]
    return predictions, low_confidence, detections


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
            predictions, low_confidence, detections = predict_topk(image_bytes)
            self._send_json({"predictions": predictions, "low_confidence": low_confidence, "detections": detections})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
