# Model artifacts

This folder is populated by `training/export_onnx.py` (run at the end of the
training notebook) and is **not** committed empty — it needs two files before
`api/predict.py` will work or the app can be deployed:

- `flower_model.onnx` — the exported, fine-tuned classifier
- `class_names.json` — a JSON array of 102 flower names, index-matched to the
  model's output classes (index 0 = model output 0, etc.)

Run the training notebook (`notebooks/train_flowers102.ipynb`) end to end,
then copy (or `git add`) the two generated files here before deploying.
