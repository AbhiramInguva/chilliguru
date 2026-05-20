import json
from pathlib import Path

import gradio as gr
from PIL import Image

import detector

MODEL_PATH = Path(__file__).resolve().parent / "chilli_pest_v2.pt"
INFO_PATH = Path(__file__).resolve().parent / "model_info.json"

# Original Roboflow 18-class index → merged 15-class index (same as training notebook)
CLASS_MAP_18_TO_15: dict[int, int] = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 11,
    17: 14,
}

def _get_model():
    """Lazy-load primary chilli model from singleton cache in detector.py."""
    model = detector.get_model(1)
    if model is None:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
    return model

model = _get_model()

with open(INFO_PATH, encoding="utf-8") as f:
    MODEL_INFO = json.load(f)

V2_CLASSES: list[str] = list(MODEL_INFO["classes"])
TELUGU: dict[str, str] = MODEL_INFO.get("telugu_names", {})
CLASS_TYPES: dict[str, str] = MODEL_INFO.get("class_types", {})
HIGH_CONF = float(MODEL_INFO.get("high_confidence_min", 55))
MARGIN_THR = float(MODEL_INFO.get("margin_threshold", 0.15))
CONF_PRED = float(MODEL_INFO.get("confidence_threshold", 0.25))
IMGSZ = int(MODEL_INFO.get("imgsz", 640))

# Snapshot names baked into the checkpoint (for raw_label + remap decisions)
_baked = model.names
BAKED_NAMES: dict[int, str] = (
    dict(_baked) if isinstance(_baked, dict) else {i: str(n) for i, n in enumerate(_baked)}
)
NUM_BAKED = len(BAKED_NAMES)

if NUM_BAKED == 15:
    # Force display strings to match model_info (Ultralytics ≥8.1: model.names is read-only)
    name_map = {i: V2_CLASSES[i] for i in range(len(V2_CLASSES))}
    model.model.names = name_map


def resolve_detection_class(cls_id: int) -> tuple[int, str, str]:
    """
    Returns (canonical_15_id, display_label, raw_baked_name).
    """
    raw = BAKED_NAMES.get(cls_id, str(cls_id))

    if NUM_BAKED == 18:
        if cls_id not in CLASS_MAP_18_TO_15:
            return 0, V2_CLASSES[0], raw
        cid = CLASS_MAP_18_TO_15[cls_id]
        return cid, V2_CLASSES[cid], raw

    if NUM_BAKED == 15:
        if 0 <= cls_id < len(V2_CLASSES):
            return cls_id, V2_CLASSES[cls_id], raw
        return cls_id, raw, raw

    # Unknown layout — pass through baked name
    return cls_id, raw, raw


def predict(image: Image.Image | None):
    if image is None:
        return {"success": False, "error": "No image provided"}

    # ── Preprocessing / Validation: Out-of-Domain Guardrail ──────────────────
    try:
        yolo = detector.get_model(3)
        if yolo:
            results = yolo.predict(image, verbose=False, conf=0.10)
            boxes = results[0].boxes
            names_map = results[0].names

            AGRICULTURAL_CLASSES = {58, 50, 51, 46, 47, 49}
            AGRICULTURAL_NAMES = {"potted plant", "broccoli", "carrot", "banana", "apple", "orange"}

            # If the model returns 0 detections (solid canvas, blank wall/background)
            if boxes is None or len(boxes) == 0:
                return {
                    "success": False,
                    "error": "Please upload chili plant images",
                    "phase": 3,
                    "low_confidence": True
                }

            # Check if the detected object classes are purely non-agricultural
            purely_non_agricultural = True
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = str(names_map.get(cls_id, cls_id)).lower()
                is_ag = (cls_id in AGRICULTURAL_CLASSES) or (class_name in AGRICULTURAL_NAMES)
                if is_ag:
                    purely_non_agricultural = False
                    break

            if purely_non_agricultural:
                return {
                    "success": False,
                    "error": "Please upload chili plant images",
                    "phase": 3,
                    "low_confidence": True
                }
    except Exception as e:
        print(f"   [OOD Preprocessing] YOLOv8n check error: {e}", flush=True)

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
    _, label, raw_label = resolve_detection_class(class_id)

    if len(boxes) > 1:
        sorted_confs = boxes.conf.sort(descending=True).values
        margin = float(sorted_confs[0] - sorted_confs[1])
    else:
        margin = 1.0

    is_low = confidence_pct < HIGH_CONF or (confidence_pct < 70.0 and margin < MARGIN_THR)

    all_detections = []
    for i in range(len(boxes)):
        det_id = int(boxes.cls[i])
        _, det_label, det_raw = resolve_detection_class(det_id)
        conf_pct = round(float(boxes.conf[i]) * 100.0, 1)
        all_detections.append(
            {
                "label": det_label,
                "telugu": TELUGU.get(det_label, ""),
                "confidence": conf_pct,
                "type": CLASS_TYPES.get(det_label, "pest"),
                "raw_label": det_raw,
            }
        )

    top = {
        "label": label,
        "telugu": TELUGU.get(label, ""),
        "confidence": round(confidence_pct, 1),
        "type": CLASS_TYPES.get(label, "pest"),
        "raw_label": raw_label,
    }

    return {
        "success": True,
        "low_confidence": is_low,
        "top_detection": top,
        "all_detections": all_detections,
        "model_classes_baked": NUM_BAKED,
        "labels_source": "model_info.json + remap" if NUM_BAKED == 18 else "model_info.json",
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload chilli leaf / pest photo"),
    outputs=gr.JSON(label="Detection result"),
    title="ChilliGuru pest detector V2",
    description=(
        "YOLO — 15 merged classes. "
        "Upload a close-up of an affected leaf or visible pest."
    ),
    api_name="predict",
)

if __name__ == "__main__":
    demo.launch(show_error=True)

# Sync: 2026-05-20T23:42:29
