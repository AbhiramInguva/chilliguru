"""
detector.py — ChilliGuru Pest & Disease Detector
Model: VIT-AP (YOLO + ViT backbone, 18-class chilli pest & disease detector)
Falls back to Groq questions when confidence is low.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

CUSTOM_MODEL_PATH    = "chilli_pest_model.pt"
CONFIDENCE_THRESHOLD = 45.0

if not Path(CUSTOM_MODEL_PATH).exists():
    print(f"WARNING: {CUSTOM_MODEL_PATH} not found — pest detector disabled.", flush=True)

_custom_model = None

# ── 18 class labels from VIT-AP model ────────────────────────────────────────
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

def _load_custom():
    global _custom_model
    if _custom_model is not None:
        return _custom_model
    if not Path(CUSTOM_MODEL_PATH).exists():
        return None
    print("   Loading VIT-AP chilli model...", end="", flush=True)
    try:
        from ultralytics import YOLO
        _custom_model = YOLO(CUSTOM_MODEL_PATH)
        print(" Ready!")
        return _custom_model
    except Exception as e:
        print(f" Failed ({e})")
        return None

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

def detect(image_path):
    img = Path(image_path)
    if not img.exists():
        return {"success": False, "error": f"Image not found: {image_path}"}

    custom = _load_custom()
    if not custom:
        return {"success": False, "error": "VIT-AP model could not be loaded"}

    try:
        results    = custom.predict(str(img), verbose=False, conf=0.10)
        boxes      = results[0].boxes
        names_map  = results[0].names  # {int: str} from model

        if boxes is None or len(boxes) == 0:
            return {"success": False, "error": "No detections", "low_confidence": True}

        detections = []
        for box in boxes:
            cls_id     = int(box.cls[0])
            model_name = names_map.get(cls_id, str(cls_id))
            confidence = round(float(box.conf[0]) * 100, 1)
            raw_label  = _resolve_label(model_name, cls_id)
            english, telugu, kind = _get_friendly_name(raw_label)
            display    = f"{english} [{telugu}]" if telugu else english
            info       = PEST_INFO.get(raw_label, {"symptoms": f"Damage by {english}", "damage": ""})
            detections.append({
                "label":      display,
                "raw_label":  raw_label,
                "type":       kind,
                "confidence": confidence,
                "severity":   _sev(confidence),
                "symptoms":   info["symptoms"],
                "damage":     info["damage"],
            })

        detections.sort(key=lambda x: x["confidence"], reverse=True)
        top = detections[0]
        return {
            "success":        True,
            "top_detection":  top,
            "all_detections": detections[:3],
            "model_used":     "VIT-AP ChilliGuru (18-class)",
            "low_confidence": top["confidence"] < CONFIDENCE_THRESHOLD,
        }
    except Exception as e:
        return {"success": False, "error": "exception", "message": str(e)}

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
