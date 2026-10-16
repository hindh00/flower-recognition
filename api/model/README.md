# Model artifacts

This folder is populated by `training/export_onnx.py` (run at the end of the
training notebook) and is **not** committed empty — it needs two files before
`api/predict.py` will work or the app can be deployed:

- `flower_model.onnx` — the exported, fine-tuned classifier
- `class_names.json` — a JSON array of 102 flower names, index-matched to the
  model's output classes (index 0 = model output 0, etc.)

Run the training notebook (`notebooks/train_flowers102.ipynb`) end to end,
then copy (or `git add`) the two generated files here before deploying.

## Optional: `flower_detector.onnx`

A third, **optional** artifact: a single-class ("flower") YOLO detector,
exported with NMS baked in by `training/export_detector_onnx.py` (run at the
end of `notebooks/train_detector_flowers102.ipynb`). When present,
`api/predict.py` detects flower region(s) first and classifies each one
separately, which helps with cluttered backgrounds and photos with multiple
blooms. When **absent**, `api/predict.py` loads the classifier only and
falls back to classifying the whole image (5-crop test-time augmentation) —
the app works correctly either way, this file just improves accuracy on
harder real-world photos.
