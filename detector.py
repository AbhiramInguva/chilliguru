"""
detector.py — ChilliGuru Pest & Disease Detector
Priority:
  1. chilli_pest_model.pt  (custom trained YOLOv8 — 11 chilli pest classes)
  2. PlantVillage CNN      (disease fallback)
  3. Low confidence        (Groq asks farmer questions)
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

CUSTOM_MODEL_PATH    = "chilli_pest_model.pt"
CONFIDENCE_THRESHOLD = 45.0

_custom_model   = None
_vit_model      = None
_vit_labels     = None
_vit_transforms = None

# ── Chilli relevance ──────────────────────────────────────────────────────────
NON_CHILLI = {"corn","maize","apple","cherry","grape","peach","orange",
              "squash","strawberry","raspberry","blueberry","soybean",
              "wheat","rice","cotton","banana"}

def is_chilli_related(label):
    l = label.lower()
    for c in NON_CHILLI:
        if c in l: return False
    return True

# ── Pest info for all 11 trained classes ─────────────────────────────────────
PEST_INFO = {
    "Fruit Borer": {
        "symptoms": "Round entry hole in fruit, brown powder (frass) around hole, fruit drops early",
        "damage":   "Larva eats seeds and inside of fruit — destroys 30-50% of crop if untreated",
    },
    "Armyworm": {
        "symptoms": "Irregular holes in leaves, skeletonized leaves, caterpillars visible at night",
        "damage":   "Rapidly defoliates plants, also bores into fruit",
    },
    "Aphids": {
        "symptoms": "Tiny green or black clusters on new shoots, sticky honeydew coating, curled leaves",
        "damage":   "Sucks sap and spreads leaf curl virus",
    },
    "Whitefly": {
        "symptoms": "Tiny white insects fly up when plant is disturbed, yellowing leaves, sticky coating",
        "damage":   "Spreads chilli leaf curl virus — major threat in AP/Telangana",
    },
    "Thrips": {
        "symptoms": "Silver streaks on leaves, upward leaf curling, stunted new growth",
        "damage":   "Sucks plant sap and spreads viruses",
    },
    "Chilli Thrips": {
        "symptoms": "Upward leaf curling, silvery streaks, distorted new growth, flower drop",
        "damage":   "Spreads chilli leaf curl disease — very common in Rabi season AP/Telangana",
    },
    "Spider Mites": {
        "symptoms": "Silvery or bronze coloured leaves, fine webbing on leaf undersides, tiny moving dots",
        "damage":   "Severe in Zaid (hot/dry) season — can kill plant if untreated",
    },
    "Mealybug": {
        "symptoms": "White cottony clusters on stems, leaves and fruit",
        "damage":   "Sucks sap, causes wilting and stunted growth",
    },
    "Leaf Miner": {
        "symptoms": "Winding white tunnels or trails visible inside leaves",
        "damage":   "Reduces photosynthesis, weakens plant",
    },
    "Broad Mite": {
        "symptoms": "Distorted new leaves, downward curling, bronze or brown colour on young growth",
        "damage":   "Very tiny, hard to see — common in nursery stage",
    },
    "Cutworm": {
        "symptoms": "Seedlings cut at ground level overnight, wilted plants, caterpillar in soil nearby",
        "damage":   "Damages nursery and young transplants at night",
    },
}

def _load_custom():
    global _custom_model
    if _custom_model is not None: return _custom_model
    if not Path(CUSTOM_MODEL_PATH).exists(): return None
    print("   Loading custom chilli pest model...", end="", flush=True)
    try:
        from ultralytics import YOLO
        _custom_model = YOLO(CUSTOM_MODEL_PATH)
        print(" Ready!")
        return _custom_model
    except Exception as e:
        print(f" Failed ({e})")
        return None

def _load_vit():
    global _vit_model, _vit_labels, _vit_transforms
    if _vit_model is not None: return True
    print("   Loading disease classifier (fallback)...", end="", flush=True)
    try:
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageClassification
        MODEL_ID        = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
        _vit_model      = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        _vit_model.eval()
        _vit_labels     = _vit_model.config.id2label
        _vit_transforms = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        print(" Ready!")
        return True
    except Exception as e:
        print(f" Failed ({e})")
        return False

DISEASE_MAP = {
    "Pepper__bell___Bacterial_spot":          {"name":"Bacterial Leaf Spot",   "type":"disease","symptoms":"Dark water-soaked spots with yellow halo"},
    "Pepper__bell___healthy":                 {"name":"Healthy Plant",         "type":"healthy","symptoms":"No visible issue"},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"name":"Spider Mites",  "type":"pest",   "symptoms":"Silvery leaves, fine webbing on undersides"},
    "Tomato___Late_blight":                   {"name":"Phytophthora Blight",   "type":"disease","symptoms":"Dark water-soaked lesions, sudden wilting"},
    "Tomato___Leaf_Mold":                     {"name":"Cercospora Leaf Spot",  "type":"disease","symptoms":"Yellow patches on top, olive mold below"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"name":"Chilli Leaf Curl Virus","type":"disease","symptoms":"Upward curling, yellowing, stunted growth"},
    "Tomato___Tomato_mosaic_virus":           {"name":"Chilli Mosaic Virus",   "type":"disease","symptoms":"Mosaic pattern, distorted leaves"},
    "Tomato___Target_Spot":                   {"name":"Anthracnose/Fruit Rot", "type":"disease","symptoms":"Dark sunken spots on ripe fruit"},
    "Tomato___Early_blight":                  {"name":"Alternaria Blight",     "type":"disease","symptoms":"Target-board spots on older leaves"},
    "Tomato___Bacterial_spot":                {"name":"Bacterial Spot",        "type":"disease","symptoms":"Raised scabby spots on leaves and fruit"},
    "Tomato___Powdery_mildew":                {"name":"Powdery Mildew",        "type":"disease","symptoms":"White powdery patches on upper leaf"},
    "Tomato___healthy":                       {"name":"Healthy Plant",         "type":"healthy","symptoms":"No visible issue"},
    "Tomato___Septoria_leaf_spot":            {"name":"Septoria Leaf Spot",    "type":"disease","symptoms":"Circular spots with dark border"},
}

def _sev(c):
    return "High" if c>=80 else ("Medium" if c>=50 else "Low (not very sure)")

def detect(image_path):
    img = Path(image_path)
    if not img.exists():
        return {"success":False,"error":f"Image not found: {image_path}"}

    # ── Custom model ──────────────────────────────────────────────────────────
    custom = _load_custom()
    if custom:
        try:
            results = custom.predict(str(img), verbose=False, conf=0.10)
            boxes   = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                detections = []
                for box in boxes:
                    cls_id     = int(box.cls[0])
                    label      = results[0].names.get(cls_id, str(cls_id))
                    confidence = round(float(box.conf[0])*100, 1)
                    if not is_chilli_related(label):
                        return {"success":False,"not_chilli":True,"detected_crop":label}
                    info = PEST_INFO.get(label, {"symptoms":f"Damage by {label}","damage":""})
                    detections.append({
                        "label":      label,
                        "type":       "pest",
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
                    "model_used":     "Custom ChilliGuru YOLOv8",
                    "low_confidence": top["confidence"] < CONFIDENCE_THRESHOLD,
                }
        except Exception:
            pass

    # ── PlantVillage fallback ─────────────────────────────────────────────────
    if _load_vit():
        try:
            import torch
            from PIL import Image as PILImage
            pil_img = PILImage.open(img).convert("RGB")
            tensor  = _vit_transforms(pil_img).unsqueeze(0)
            with torch.no_grad():
                logits = _vit_model(tensor).logits
            probs = torch.softmax(logits,dim=-1)[0]
            top3  = torch.topk(probs,k=3)
            detections = []
            for score,idx in zip(top3.values.tolist(),top3.indices.tolist()):
                label      = _vit_labels[idx]
                confidence = round(score*100,1)
                if not is_chilli_related(label):
                    return {"success":False,"not_chilli":True,"detected_crop":label}
                m = DISEASE_MAP.get(label,{"name":label.replace("_"," "),"type":"unknown","symptoms":"Visual anomaly"})
                detections.append({"label":m["name"],"type":m.get("type","unknown"),
                    "confidence":confidence,"severity":_sev(confidence),
                    "symptoms":m.get("symptoms",""),"damage":""})
            if not detections:
                return {"success":False,"not_chilli":True,"detected_crop":"unknown plant"}
            top = detections[0]
            return {"success":True,"top_detection":top,"all_detections":detections,
                    "model_used":"PlantVillage CNN (fallback)","low_confidence":top["confidence"]<CONFIDENCE_THRESHOLD}
        except Exception as e:
            return {"success":False,"error":"exception","message":str(e)}

    return {"success":False,"error":"No models available"}

def format_for_openai(result, user_description=""):
    if result.get("not_chilli"): return "NOT_CHILLI"
    if not result.get("success"):
        msg = result.get("message", result.get("error","unknown"))
        return (f"[DETECTION FAILED: {msg}. Farmer described: '{user_description}'. "
                f"Ask 2 simple questions to understand the problem, then give organic solutions.]")
    top  = result.get("top_detection")
    all_ = result.get("all_detections",[])
    low  = result.get("low_confidence",False)
    if not top:
        return f"[No detection. Farmer said: '{user_description}'. Ask questions to diagnose.]"
    if low:
        return "\n".join([
            "=== DETECTION: LOW CONFIDENCE - ASK QUESTIONS FIRST ===",
            f"Best guess : {top['label']} ({top['confidence']}% - not confident)",
            f"Farmer said: \"{user_description}\"",
            "INSTRUCTION: Ask the farmer 2 simple questions:",
            "  1. Which part is affected? (leaves / stem / fruit / roots)",
            "  2. What exactly are you seeing? (hole in fruit / spots / curling / webbing / insects?)",
            "After they answer, give diagnosis and 2-3 organic solutions with metrics.",
            "="*50])
    lines = [
        f"=== PEST DETECTED: {top['label'].upper()} ===",
        f"Confidence : {top['confidence']}%  |  Severity: {top['severity']}",
        f"Symptoms   : {top['symptoms']}",
    ]
    if top.get("damage"): lines.append(f"Impact     : {top['damage']}")
    if len(all_)>1: lines.append("Also possible: " + ", ".join(f"{d['label']} ({d['confidence']}%)" for d in all_[1:]))
    if user_description: lines.append(f"Farmer said: \"{user_description}\"")
    lines += [
        "INSTRUCTION: Tell the farmer clearly what pest this is in simple words.",
        "Give 2-3 organic solutions with: how well it works, days to results, Rs cost, frequency.",
        "End with one simple prevention tip.",
        "="*44]
    return "\n".join(lines)
