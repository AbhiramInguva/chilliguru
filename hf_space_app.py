"""
Gradio Space entrypoint — upload this file as app.py on inguvaaa/chilliguru-detector.

Expects in the Space repo root:
  - chilli_pest_v2.pt
  - model_info.json

JSON shape matches ChilliGuru Flask client (telugu, type, confidence on top_detection).
"""
import json
from pathlib import Path

import gradio as gr
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "chilli_pest_v2.pt"
INFO_PATH = Path(__file__).resolve().parent / "model_info.json"

model = YOLO(str(MODEL_PATH))
with open(INFO_PATH, encoding="utf-8") as f:
    MODEL_INFO = json.load(f)

TELUGU = MODEL_INFO.get("telugu_names", {})
CLASS_TYPES = MODEL_INFO.get("class_types", {})
HIGH_CONF = float(MODEL_INFO.get("high_confidence_min", 55))
MARGIN_THR = float(MODEL_INFO.get("margin_threshold", 0.15))
CONF_PRED = float(MODEL_INFO.get("confidence_threshold", 0.25))
IMGSZ = int(MODEL_INFO.get("imgsz", 640))


def _name_for_id(class_id: int) -> str:
    names = model.names
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    return names[class_id] if class_id < len(names) else str(class_id)


def predict(image: Image.Image | None):
    if image is None:
        return {"success": False, "error": "No image provided"}

    results = model.predict(image, conf=CONF_PRED, verbose=False, imgsz=IMGSZ)
    r0 = results[0]
    boxes = r0.boxes

    if boxes is None or len(boxes) == 0:
        return {
            "success": True,
            "low_confidence": True,
            "top_detection": None,
            "all_detections": [],
            "message": "No pest or disease detected above threshold. Try a clearer close-up.",
        }

    best_idx = int(boxes.conf.argmax())
    confidence_pct = float(boxes.conf[best_idx]) * 100.0
    class_id = int(boxes.cls[best_idx])
    label = _name_for_id(class_id)

    if len(boxes) > 1:
        sorted_confs = boxes.conf.sort(descending=True).values
        margin = float(sorted_confs[0] - sorted_confs[1])
    else:
        margin = 1.0

    is_low = confidence_pct < HIGH_CONF or (
        confidence_pct < 70.0 and margin < MARGIN_THR
    )

    all_detections = []
    for i in range(len(boxes)):
        det_id = int(boxes.cls[i])
        det_label = _name_for_id(det_id)
        conf_pct = round(float(boxes.conf[i]) * 100.0, 1)
        all_detections.append(
            {
                "label": det_label,
                "telugu": TELUGU.get(det_label, ""),
                "confidence": conf_pct,
                "type": CLASS_TYPES.get(det_label, "pest"),
            }
        )

    top = {
        "label": label,
        "telugu": TELUGU.get(label, ""),
        "confidence": round(confidence_pct, 1),
        "type": CLASS_TYPES.get(label, "pest"),
        "raw_label": label,
    }

    return {
        "success": True,
        "low_confidence": is_low,
        "top_detection": top,
        "all_detections": all_detections,
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload chilli leaf / pest photo"),
    outputs=gr.JSON(label="Detection result"),
    title="ChilliGuru pest detector V2",
    description=(
        "YOLOv8m — 15 merged classes. "
        "Upload a close-up of an affected leaf or visible pest."
    ),
    api_name="predict",
)

if __name__ == "__main__":
    demo.launch()
