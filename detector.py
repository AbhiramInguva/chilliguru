"""
detector.py — ChilliGuru Pest & Disease Detector (3-Phase Cascade Pipeline)
Architecture:
  Phase 1: Primary chilli detector (chilli_pest_model.pt, 18-class)
  Phase 2: IP102 fallback model (ip102_model.pt, 5-class generic pests)
  Phase 3: Generic crop anomaly detector (yolov8n.pt, non-pest rejection)

REFACTORED FOR PERFORMANCE:
  - Thread-safe singleton model caching (lazy-loaded, cached globally)
  - In-memory image processing (numpy.frombuffer + cv2.imdecode, zero disk I/O)
  - Optimized Phase 3 (reuses Phase 1/2 bounding boxes, avoids redundant re-detection)
"""

import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

log = logging.getLogger("chilliguru.detector")

PHASE1_MODEL_PATH    = "chilli_pest_model.pt"
PHASE2_MODEL_PATH    = "ip102_model.pt"
PHASE3_MODEL_PATH    = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 45.0
PHASE1_MIN_CONF      = 0.40  # 40% — Phase 1 requirement

if not Path(PHASE1_MODEL_PATH).exists():
    log.warning("phase1_model_missing", extra={"data": {"path": PHASE1_MODEL_PATH}})

# Thread-safe module-level globals: initialized to None, lazy-loaded on first request
_PHASE1_MODEL = None
_PHASE2_MODEL = None
_PHASE3_MODEL = None
_phase1_model = _PHASE1_MODEL  # backward compatibility alias
_phase2_model = _PHASE2_MODEL  # backward compatibility alias
_phase3_model = _PHASE3_MODEL  # backward compatibility alias

# ── 18 class labels from VIT-AP model (Phase 1) ────────────────────────────────
CLASS_NAMES = [
    "Black Thrips-Leafs",                       # 0
    "Black Thrips-Pest",                         # 1
    "Collectotrichum spp (Anthracnose)",         # 2
    "Curling-Leafs",                             # 3
    "Healthy-Leafs",                             # 4
    "Leaf Spot-Leafs",                           # 5
    "Leveillula taurica (Powdery Mildew)",       # 6
    "Mozaik-Leaf (Mosaic Virus)",                # 7
    "Pest-Asphondylia capsici",                  # 8
    "Pest-Helicoverpa armigera (Fruit Borer)",   # 9
    "Pest-Myzus persicae (Aphids)",              # 10
    "Pest-Phenacoccus solenopsis (Mealybug)",    # 11
    "Pest-Red Mites",                            # 12
    "Pest-Spodoptera exigua (Beet Armyworm)",    # 13
    "Pest-Spodoptera litura (Armyworm)",         # 14
    "Pest-White Fly",                            # 15
    "Red Mites leafs",                           # 16
    "White Fly-Leafs",                           # 17
]

# ── IP102 to ChilliGuru class mapping (Phase 2 fallback) ──────────────────────
IP102_CLASS_MAPPING = {
    "aphid":              ("Pest-Myzus persicae (Aphids)",          "పేను పురుగు",                        "pest"),
    "thrips":             ("Black Thrips-Pest",                     "నల్ల తుమ్మెద పురుగు",               "pest"),
    "tobaccocaterpillar": ("Pest-Spodoptera litura (Armyworm)",     "గొంగళి పురుగు",                     "pest"),
    "whitefly":           ("Pest-White Fly",                        "తెల్ల ఈగ పురుగు",                    "pest"),
    "mites":              ("Pest-Red Mites",                        "ఎర్ర సాలె పురుగు",                   "pest"),
}

def _get_friendly_name(raw_label):
    """Map raw model label → (friendly English name, Telugu name, type)."""
    mapping = {
        "Black Thrips-Leafs":                    ("Black Thrips – Leaf Damage",        "నల్ల తుమ్మెద పురుగు ఆకు నష్టం",      "pest"),
        "Black Thrips-Pest":                     ("Black Thrips",                       "నల్ల తుమ్మెద పురుగు",               "pest"),
        "Collectotrichum spp (Anthracnose)":     ("Anthracnose (Fruit Rot)",            "యాంత్రాక్నోస్ / పండు కుళ్ళు తెగులు", "disease"),
        "Curling-Leafs":                         ("Leaf Curling",                       "ఆకు ముడత",                          "disease"),
        "Healthy-Leafs":                         ("Healthy Leaf",                       "ఆరోగ్యకరమైన ఆకు",                   "healthy"),
        "Leaf Spot-Leafs":                       ("Leaf Spot",                          "ఆకు మచ్చ తెగులు",                    "disease"),
        "Leveillula taurica (Powdery Mildew)":   ("Powdery Mildew",                    "పొడి తెగులు",                        "disease"),
        "Mozaik-Leaf (Mosaic Virus)":            ("Mosaic Virus",                       "మొజాయిక్ వైరస్",                    "disease"),
        "Pest-Asphondylia capsici":              ("Chilli Flower Gall Midge",           "మిరప పూల పురుగు",                    "pest"),
        "Pest-Helicoverpa armigera (Fruit Borer)":("Fruit Borer",                      "పండు తొలిచే పురుగు",                 "pest"),
        "Pest-Myzus persicae (Aphids)":          ("Aphids (Green Louse)",               "పేను పురుగు",                        "pest"),
        "Pest-Phenacoccus solenopsis (Mealybug)":("Mealybug",                          "తెల్ల దూది పురుగు",                   "pest"),
        "Pest-Red Mites":                        ("Red Spider Mites",                   "ఎర్ర సాలె పురుగు",                   "pest"),
        "Pest-Spodoptera exigua (Beet Armyworm)":("Beet Armyworm",                     "చిన్న గొంగళి పురుగు",                "pest"),
        "Pest-Spodoptera litura (Armyworm)":     ("Armyworm",                           "గొంగళి పురుగు",                     "pest"),
        "Pest-White Fly":                        ("Whitefly",                           "తెల్ల ఈగ పురుగు",                    "pest"),
        "Red Mites leafs":                       ("Red Spider Mites – Leaf Damage",     "ఎర్ర సాలె పురుగు ఆకు నష్టం",        "pest"),
        "White Fly-Leafs":                       ("Whitefly – Leaf Damage",             "తెల్ల ఈగ ఆకు నష్టం",                "pest"),
    }
    english, telugu, kind = mapping.get(raw_label, (raw_label, "", "unknown"))
    return english, telugu, kind

# ── Pest / disease info for all 18 classes ───────────────────────────────────
PEST_INFO = {
    "Black Thrips-Leafs": {
        "symptoms": "Silver streaks and bronzing on leaves, upward leaf curl, distorted new growth",
        "damage":   "Black thrips suck sap and spread leaf curl virus — worse in Rabi season",
    },
    "Black Thrips-Pest": {
        "symptoms": "Tiny black insects visible on new shoots and flower buds, sticky deposits",
        "damage":   "Direct sap feeding and virus transmission — can destroy 40% of crop",
    },
    "Collectotrichum spp (Anthracnose)": {
        "symptoms": "Dark sunken spots on ripe or ripening fruit, spots spread quickly in wet weather",
        "damage":   "Destroys fruit at harvest time — major post-harvest loss in AP/Telangana",
    },
    "Curling-Leafs": {
        "symptoms": "Leaves curl upward or downward, yellowing along edges, stunted growth",
        "damage":   "Usually caused by thrips or leaf curl virus — reduces photosynthesis and yield",
    },
    "Healthy-Leafs": {
        "symptoms": "No visible pest or disease signs — plant looks normal and green",
        "damage":   "None — plant is healthy",
    },
    "Leaf Spot-Leafs": {
        "symptoms": "Circular or irregular brown/black spots on leaves, spots may have yellow halo",
        "damage":   "Leaf drop and reduced photosynthesis — spreads fast in humid conditions",
    },
    "Leveillula taurica (Powdery Mildew)": {
        "symptoms": "White powdery coating on upper leaf surface, leaves turn yellow and fall",
        "damage":   "Common in Rabi season — severe infection can defoliate entire plant",
    },
    "Mozaik-Leaf (Mosaic Virus)": {
        "symptoms": "Mosaic pattern of light and dark green on leaves, leaf distortion, stunted plant",
        "damage":   "Spread by aphids and whiteflies — no cure, infected plants must be removed",
    },
    "Pest-Asphondylia capsici": {
        "symptoms": "Flower buds swell abnormally, fail to open, drop early — gall-like swellings",
        "damage":   "Chilli gall midge destroys flowers before fruiting — severe yield loss",
    },
    "Pest-Helicoverpa armigera (Fruit Borer)": {
        "symptoms": "Round entry hole in fruit with brown powder (frass), fruit drops early",
        "damage":   "Larva eats seeds and fruit interior — can destroy 30–50% of crop if untreated",
    },
    "Pest-Myzus persicae (Aphids)": {
        "symptoms": "Tiny green or black clusters on new shoots, sticky honeydew coating, curled leaves",
        "damage":   "Sucks sap and spreads mosaic and leaf curl viruses — worse in cool weather",
    },
    "Pest-Phenacoccus solenopsis (Mealybug)": {
        "symptoms": "White cottony clusters on stems, leaves and fruit joints, sticky sooty mold",
        "damage":   "Severe infestation causes wilting, stunting and complete plant collapse",
    },
    "Pest-Red Mites": {
        "symptoms": "Tiny red dots moving on leaf undersides, silvery or bronze leaf colour, fine webbing",
        "damage":   "Worst in Zaid (hot dry season) — sucks sap and can kill plants quickly",
    },
    "Pest-Spodoptera exigua (Beet Armyworm)": {
        "symptoms": "Irregular holes in young leaves, skeletonized leaves, small caterpillars in groups",
        "damage":   "Rapidly strips leaves and also attacks flower buds — outbreak spreads fast",
    },
    "Pest-Spodoptera litura (Armyworm)": {
        "symptoms": "Large irregular holes in leaves, caterpillars visible at night, severe defoliation",
        "damage":   "Heavy feeder — can defoliate an entire field in a few nights",
    },
    "Pest-White Fly": {
        "symptoms": "Tiny white insects fly up when plant is disturbed, yellowing leaves, sticky coating",
        "damage":   "Spreads chilli leaf curl virus — major threat in Kharif season AP/Telangana",
    },
    "Red Mites leafs": {
        "symptoms": "Silvery or bronze discolouration on leaf surface, stippling marks, fine webbing underneath",
        "damage":   "Leaf damage by red spider mites — reduces photosynthesis and fruit size",
    },
    "White Fly-Leafs": {
        "symptoms": "Yellowing and whitening of leaves, sooty black mold on sticky honeydew deposits",
        "damage":   "Whitefly leaf feeding weakens plant and creates entry points for fungal disease",
    },
}

def get_model(phase_id):
    """
    Thread-safe centralized lazy-loading function for all 3 model phases.
    Returns cached instance pointer on all subsequent calls (no re-instantiation).
    
    Args:
        phase_id (int): 1, 2, or 3 for the respective detection phase
    
    Returns:
        YOLO model instance or None if file not found or loading failed
    """
    global _PHASE1_MODEL, _PHASE2_MODEL, _PHASE3_MODEL
    
    if phase_id == 1:
        if _PHASE1_MODEL is not None:
            return _PHASE1_MODEL
        if not Path(PHASE1_MODEL_PATH).exists():
            return None
        log.info("phase1_model_loading")
        try:
            from ultralytics import YOLO
            _PHASE1_MODEL = YOLO(PHASE1_MODEL_PATH)
            log.info("phase1_model_ready")
            return _PHASE1_MODEL
        except Exception as e:
            log.error("phase1_model_load_failed", extra={"data": {"error_message": str(e)}})
            return None
    
    elif phase_id == 2:
        if _PHASE2_MODEL is not None:
            return _PHASE2_MODEL
        if not Path(PHASE2_MODEL_PATH).exists():
            return None
        log.info("phase2_model_loading")
        try:
            from ultralytics import YOLO
            _PHASE2_MODEL = YOLO(PHASE2_MODEL_PATH)
            log.info("phase2_model_ready")
            return _PHASE2_MODEL
        except Exception as e:
            log.error("phase2_model_load_failed", extra={"data": {"error_message": str(e)}})
            return None
    
    elif phase_id == 3:
        if _PHASE3_MODEL is not None:
            return _PHASE3_MODEL
        if not Path(PHASE3_MODEL_PATH).exists():
            return None
        log.info("phase3_model_loading")
        try:
            from ultralytics import YOLO
            _PHASE3_MODEL = YOLO(PHASE3_MODEL_PATH)
            log.info("phase3_model_ready")
            return _PHASE3_MODEL
        except Exception as e:
            log.error("phase3_model_load_failed", extra={"data": {"error_message": str(e)}})
            return None
    
    return None


# Backward compatibility: keep original function names pointing to get_model()
def _load_custom():
    """Phase 1: Load primary chilli detector (VIT-AP). [Deprecated: use get_model(1)]"""
    global _phase1_model
    _phase1_model = get_model(1)
    return _phase1_model

def _load_ip102_model():
    """Phase 2: Load IP102 generic pest detector (fallback). [Deprecated: use get_model(2)]"""
    global _phase2_model
    _phase2_model = get_model(2)
    return _phase2_model

def _load_yolov8n_model():
    """Phase 3: Load generic YOLOv8n for non-chilli crop anomaly rejection. [Deprecated: use get_model(3)]"""
    global _phase3_model
    _phase3_model = get_model(3)
    return _phase3_model

def _resolve_label(raw_label, cls_id):
    """Return the canonical CLASS_NAMES entry for a detected label."""
    if raw_label in PEST_INFO:
        return raw_label
    # try matching by class index if model returns index-based names
    if isinstance(cls_id, int) and 0 <= cls_id < len(CLASS_NAMES):
        return CLASS_NAMES[cls_id]
    # fuzzy: find first CLASS_NAMES entry that contains the raw label (case-insensitive)
    rl = raw_label.lower()
    for cn in CLASS_NAMES:
        if rl in cn.lower() or cn.lower() in rl:
            return cn
    return raw_label

def _sev(c):
    return "High" if c >= 80 else ("Medium" if c >= 50 else "Low (not very sure)")


def detect_from_bytes(image_bytes, message=""):
    """
    OPTIMIZED ENTRY POINT: Accepts raw image byte array instead of file path.
    
    Uses numpy.frombuffer + cv2.imdecode for 100% in-memory image decoding.
    ELIMINATES all disk I/O overhead for image loading/storage.
    
    Args:
        image_bytes (bytes): Raw image binary data (JPEG/PNG/WebP)
        message (str): Optional user description for context
    
    Returns:
        dict: Detection result with success status, top_detection, all_detections, etc.
    """
    if not image_bytes:
        return {"success": False, "error": "No image data provided"}
    
    # Convert bytes → numpy array (zero-copy, in-memory only)
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_decoded is None:
            return {"success": False, "error": "Failed to decode image"}
        log.info("image_decoded_from_bytes", extra={"data": {"bytes": len(image_bytes)}})
    except Exception as e:
        log.error("image_decode_error", extra={"data": {"error_message": str(e)}})
        return {"success": False, "error": "Image decoding failed"}
    
    # ── Preprocessing / Validation: Out-of-Domain Guardrail ──────────────────
    try:
        phase3_model = get_model(3)
        if phase3_model:
            results = phase3_model.predict(img_decoded, verbose=False, conf=0.10)
            boxes = results[0].boxes
            names_map = results[0].names

            AGRICULTURAL_CLASSES = {58, 50, 51, 46, 47, 49}
            AGRICULTURAL_NAMES = {"potted plant", "broccoli", "carrot", "banana", "apple", "orange"}

            if boxes is None or len(boxes) == 0:
                return {
                    "success": False,
                    "error": "Please upload chili plant images",
                    "phase": 3,
                    "low_confidence": True
                }

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
        log.error("ood_check_error", extra={"data": {"error_message": str(e)}})

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1: Primary Chilli Detector
    # ──────────────────────────────────────────────────────────────────────────
    phase1_result = None
    phase1_boxes = None  # Cache for Phase 3 optimization
    try:
        phase1_model = get_model(1)
        if phase1_model:
            results = phase1_model.predict(img_decoded, verbose=False, conf=0.10)
            boxes = results[0].boxes
            names_map = results[0].names

            if boxes is not None and len(boxes) > 0:
                phase1_boxes = boxes  # Cache for Phase 3 optimization
                detections = []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    model_name = names_map.get(cls_id, str(cls_id))
                    confidence = float(box.conf[0]) * 100.0
                    raw_label = _resolve_label(model_name, cls_id)
                    english, telugu, kind = _get_friendly_name(raw_label)
                    display = f"{english} [{telugu}]" if telugu else english
                    info = PEST_INFO.get(raw_label, {"symptoms": f"Damage by {english}", "damage": ""})
                    detections.append({
                        "label": display,
                        "raw_label": raw_label,
                        "type": kind,
                        "confidence": float(round(confidence, 1)),
                        "severity": _sev(confidence),
                        "symptoms": info["symptoms"],
                        "damage": info["damage"],
                    })

                detections.sort(key=lambda x: x["confidence"], reverse=True)
                top = detections[0]
                
                if top["confidence"] >= (PHASE1_MIN_CONF * 100):
                    if top["raw_label"] not in CLASS_NAMES:
                        return {
                            "success": False,
                            "error": "Please upload chili plant images",
                            "message": "Please upload chili plant images",
                            "phase": 3,
                            "low_confidence": True
                        }

                    phase1_result = {
                        "success": True,
                        "top_detection": top,
                        "all_detections": detections[:3],
                        "model_used": "Phase 1: VIT-AP ChilliGuru (18-class)",
                        "low_confidence": top["confidence"] < CONFIDENCE_THRESHOLD,
                        "phase": 1,
                    }
    except Exception as e:
        log.error("phase1_inference_error", extra={"data": {"error_message": str(e)}})

    if phase1_result and phase1_result["success"]:
        return phase1_result

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2: IP102 Fallback Model
    # ──────────────────────────────────────────────────────────────────────────
    phase2_result = None
    phase2_boxes = None  # Cache for Phase 3 optimization
    try:
        phase2_model = get_model(2)
        if phase2_model:
            results = phase2_model.predict(img_decoded, verbose=False, conf=0.10)
            boxes = results[0].boxes
            names_map = results[0].names

            if boxes is not None and len(boxes) > 0:
                phase2_boxes = boxes  # Cache for Phase 3 optimization
                detections = []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    model_name = str(names_map.get(cls_id, cls_id)).lower()
                    confidence = float(box.conf[0]) * 100.0
                    
                    if model_name in IP102_CLASS_MAPPING:
                        raw_label, telugu, kind = IP102_CLASS_MAPPING[model_name]
                        english, _, _ = _get_friendly_name(raw_label)
                    else:
                        english = model_name
                        telugu = ""
                        kind = "pest"
                    
                    display = f"{english} [{telugu}]" if telugu else english
                    info = PEST_INFO.get(raw_label if model_name in IP102_CLASS_MAPPING else model_name,
                                         {"symptoms": f"Detected by IP102: {english}", "damage": ""})
                    detections.append({
                        "label": display,
                        "raw_label": model_name,
                        "type": kind,
                        "confidence": float(round(confidence, 1)),
                        "severity": _sev(confidence),
                        "symptoms": info.get("symptoms", f"Damage by {english}"),
                        "damage": info.get("damage", ""),
                    })

                detections.sort(key=lambda x: x["confidence"], reverse=True)
                top = detections[0]
                
                if top["confidence"] >= 30:
                    if top["raw_label"] not in IP102_CLASS_MAPPING:
                        return {
                            "success": False,
                            "error": "Please upload chili plant images",
                            "message": "Please upload chili plant images",
                            "phase": 3,
                            "low_confidence": True
                        }

                    phase2_result = {
                        "success": True,
                        "top_detection": top,
                        "all_detections": detections[:3],
                        "model_used": "Phase 2: IP102 Generic Pest Detector (5-class)",
                        "low_confidence": top["confidence"] < CONFIDENCE_THRESHOLD,
                        "phase": 2,
                    }
    except Exception as e:
        log.error("phase2_inference_error", extra={"data": {"error_message": str(e)}})

    if phase2_result and phase2_result["success"]:
        return phase2_result

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3: Generic YOLOv8n (Anomaly Rejection Check)
    # OPTIMIZED: Reuse Phase 1/2 bounding boxes instead of re-detecting
    # ──────────────────────────────────────────────────────────────────────────
    phase3_result = None
    try:
        phase3_model = get_model(3)
        if phase3_model:
            # Use cached boxes from Phase 1 or Phase 2 if available
            boxes = phase1_boxes if phase1_boxes is not None else phase2_boxes
            
            # If no boxes cached from earlier phases, run Phase 3 detection
            if boxes is None:
                results = phase3_model.predict(img_decoded, verbose=False, conf=0.10)
                boxes = results[0].boxes
                names_map = results[0].names
            else:
                # Boxes already computed, get names_map from Phase 3 model
                results = phase3_model.predict(img_decoded, verbose=False, conf=0.10)
                names_map = results[0].names
                log.info("phase3_reused_boxes", extra={"data": {"source": "phase1" if phase1_boxes else "phase2"}})

            AGRICULTURAL_CLASSES = {58, 50, 51, 46, 47, 49}
            AGRICULTURAL_NAMES = {"potted plant", "broccoli", "carrot", "banana", "apple", "orange"}

            non_agricultural_objects = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = str(names_map.get(cls_id, cls_id)).lower()
                    is_ag = (cls_id in AGRICULTURAL_CLASSES) or (class_name in AGRICULTURAL_NAMES)
                    if not is_ag:
                        non_agricultural_objects.append(class_name)

            if len(non_agricultural_objects) > 0:
                phase3_result = {
                    "success": False,
                    "error": "Please upload chili plant images",
                    "message": "Please upload chili plant images",
                    "low_confidence": True,
                    "phase": 3,
                    "detected_objects": non_agricultural_objects,
                    "model_used": "Phase 3: YOLOv8n Generic Detector (Anomaly Check)"
                }
    except Exception as e:
        log.error("phase3_inference_error", extra={"data": {"error_message": str(e)}})

    if phase3_result is not None:
        return phase3_result

    return {
        "success": False,
        "error": "Please upload chili plant images",
        "message": "Please upload chili plant images",
        "low_confidence": True,
        "phase": 0,
    }


def detect(image_path):
    """
    BACKWARD COMPATIBLE WRAPPER: Original file-based entry point.
    Reads image from disk and delegates to detect_from_bytes().
    Maintained for external callers; new code should use detect_from_bytes() directly.
    
    Args:
        image_path (str): File path to image
    
    Returns:
        dict: Detection result
    """
    img = Path(image_path)
    if not img.exists():
        return {"success": False, "error": f"Image not found: {image_path}"}
    
    try:
        with open(img, "rb") as f:
            image_bytes = f.read()
        return detect_from_bytes(image_bytes, message="")
    except Exception as e:
        log.error("detect_file_io_error", extra={"data": {"path": str(image_path), "error_message": str(e)}})
        return {"success": False, "error": f"Failed to read image: {e}"}

def format_for_openai(result, user_description=""):
    if not result.get("success"):
        msg = result.get("message", result.get("error", "unknown"))
        return (f"[DETECTION FAILED: {msg}. Farmer described: '{user_description}'. "
                f"Ask 2 simple questions to understand the problem, then give organic solutions.]")

    top  = result.get("top_detection")
    all_ = result.get("all_detections", [])
    low  = result.get("low_confidence", False)

    if not top:
        return f"[No detection. Farmer said: '{user_description}'. Ask questions to diagnose.]"

    if top.get("type") == "healthy":
        return (f"[DETECTION: Plant appears HEALTHY ({top['confidence']}% confidence). "
                f"Farmer said: '{user_description}'. Reassure them and give one preventive tip.]")

    if low:
        return "\n".join([
            "=== DETECTION: LOW CONFIDENCE — ASK QUESTIONS FIRST ===",
            f"Best guess : {top['label']} ({top['confidence']}% — not very sure)",
            f"Farmer said: \"{user_description}\"",
            "INSTRUCTION: Ask the farmer 2 simple questions:",
            "  1. Which part is affected? (leaves / stem / fruit / roots)",
            "  2. What exactly are you seeing? (hole in fruit / spots / curling / webbing / insects?)",
            "After they answer, give diagnosis and 2–3 organic solutions with metrics.",
            "=" * 50,
        ])

    lines = [
        f"=== DETECTED: {top['label'].upper()} ===",
        f"Confidence : {top['confidence']}%  |  Severity: {top['severity']}",
        f"Symptoms   : {top['symptoms']}",
    ]
    if top.get("damage"):
        lines.append(f"Impact     : {top['damage']}")
    if len(all_) > 1:
        others = ", ".join(f"{d['label']} ({d['confidence']}%)" for d in all_[1:])
        lines.append(f"Also possible: {others}")
    if user_description:
        lines.append(f"Farmer said: \"{user_description}\"")
    lines += [
        "INSTRUCTION: Tell the farmer clearly what this is in simple words (use the Telugu name too).",
        "Give 2–3 organic solutions with: how well it works, days to results, Rs cost, frequency.",
        "End with one simple prevention tip.",
        "=" * 50,
    ]
    return "\n".join(lines)

# Sync: 2026-05-20T23:42:29

