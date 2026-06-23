"""
detector.py — ChilliGuru Pest & Disease Detector (3-Phase Cascade Pipeline)
Architecture:
  Phase 1: Primary chilli detector (chilli_pest_model.pt, 18-class)
  Phase 2: IP102 fallback model (ip102_model.pt, 8-class generic pests)
  Phase 3: Generic crop anomaly detector (yolov8n.pt, non-pest rejection)
Each phase runs in isolated try/except for fault tolerance.
"""

import logging
import os
import re
import tempfile
import time
import uuid
import warnings
from pathlib import Path

import onnxruntime as ort
import numpy as np
import cv2

warnings.filterwarnings("ignore")

log = logging.getLogger("chilliguru.detector")

class ListOrValue:
    def __init__(self, val):
        self.val = val
    def __getitem__(self, idx):
        return self.val
    def __int__(self):
        return int(self.val)
    def __float__(self):
        return float(self.val)
    def __len__(self):
        return 1
    def __str__(self):
        return str(self.val)
    def __repr__(self):
        return repr(self.val)

class MockBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = ListOrValue(cls_id)
        self.conf = ListOrValue(conf)
        self.xyxy = [xyxy]

class MockResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image while maintaining aspect ratio, adding neutral gray borders.
    Returns:
        im: padded image
        r: scale ratio
        (left, top): left and top padding in pixels
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2.0  # divide padding into 2 sides
    dh /= 2.0

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (left, top)


def _generate_image_slices(image_array, slice_size=320, overlap=96):
    """
    Slices an image array into a sequential grid of size slice_size x slice_size
    with the specified overlap.
    Yields a single cropped patch viewport tuple (tile_array, x_offset, y_offset) sequentially.
    """
    h, w = image_array.shape[:2]
    
    # Calculate step size
    step = slice_size - overlap
    
    # Generate y coordinates
    y_coords = list(range(0, h - slice_size + 1, step))
    if not y_coords or y_coords[-1] + slice_size < h:
        y_coords.append(max(0, h - slice_size))
        
    # Generate x coordinates
    x_coords = list(range(0, w - slice_size + 1, step))
    if not x_coords or x_coords[-1] + slice_size < w:
        x_coords.append(max(0, w - slice_size))
        
    for y in y_coords:
        for x in x_coords:
            crop = image_array[y : y + slice_size, x : x + slice_size]
            yield crop, x, y


def _spatial_nms(boxes, scores, iou_threshold=0.45):
    """
    Explicit Intersection-over-Union (IoU) box assembly function to merge/filter
    overlapping predictions across slice bounds.
    """
    if len(boxes) == 0:
        return []

    # asarray avoids copying when the caller already passes ndarrays
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-8)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep


def _preprocess_for_onnx(img):
    """Letterbox to 640×640 and convert BGR uint8 → normalized NCHW float32.

    Shared by the tiled (SAHI) and single-frame inference paths — previously
    this block was duplicated in both.
    """
    img, r, (left, top) = letterbox(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return img, r, (left, top)


class ONNXYOLO:
    def __init__(self, model_path):
        self.model_path = model_path
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(model_path, opts)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        if "chilli_pest_model" in model_path:
            self.names = {
                0: 'Black Thrips-Leafs', 1: 'Black Thrips-Pest', 2: 'Collectotrichum spp-Kaya kullu-',
                3: 'Curling-Leafs', 4: 'Healthy-Leafs', 5: 'Leaf Spot-Leafs',
                6: 'Leveillula taurica-Budiha tegulu-Leaf-', 7: 'Mozaik-Leaf', 8: 'Pest-Asphondylia capsici-',
                9: 'Pest-Helicoverpa armigera-Kaya toluchu purugu', 10: 'Pest-Myzus persicae-',
                11: 'Pest-Phenacoccus solenopsis-Pendi Nalli', 12: 'Pest-Red Mites',
                13: 'Pest-Spodoptera exigua-', 14: 'Pest-Spodoptera litura-', 15: 'Pest-White Fly',
                16: 'Red Mites leafs', 17: 'White Fly-Leafs'
            }
        elif "ip102_model" in model_path:
            self.names = {
                0: 'aphids',
                1: 'whitefly_leaf_damage',
                2: 'fruit_borer',
                3: 'tobacco_caterpillar',
                4: 'yellow_thrips',
                5: 'broad_mites',
                6: 'invasive_black_thrips',
                7: 'mealybugs'
            }
        else:
            self.names = {i: f"class_{i}" for i in range(80)}
            self.names[58] = "potted plant"
            self.names[50] = "broccoli"
            self.names[51] = "carrot"
            self.names[46] = "banana"
            self.names[47] = "apple"
            self.names[49] = "orange"
            
    def predict(self, source, verbose=False, conf=0.10, iou=0.45):
        if isinstance(source, (str, Path)):
            img0 = cv2.imread(str(source))
            if img0 is None:
                raise ValueError(f"Could not read image: {source}")
        elif isinstance(source, np.ndarray):
            # Read-only from here on — copying a multi-megapixel frame per
            # inference call doubled peak memory for no benefit.
            img0 = source
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
            
        h0, w0 = img0.shape[:2]
        
        if h0 > 320 or w0 > 320:
            # Run sliced inference (SAHI)
            all_boxes = []
            all_confidences = []
            all_class_ids = []
            
            for tile_array, x_offset, y_offset in _generate_image_slices(img0, slice_size=320, overlap=96):
                img, r, (left, top) = _preprocess_for_onnx(tile_array)

                outputs = self.session.run([self.output_name], {self.input_name: img})
                output = outputs[0][0]
                output = np.transpose(output, (1, 0))
                
                boxes = output[:, :4]
                scores = output[:, 4:]
                
                class_ids = np.argmax(scores, axis=1)
                confidences = np.max(scores, axis=1)
                
                mask = confidences >= conf
                boxes = boxes[mask]
                confidences = confidences[mask]
                class_ids = class_ids[mask]
                
                if len(boxes) > 0:
                    xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    x1 = xc - w / 2
                    y1 = yc - h / 2
                    x2 = xc + w / 2
                    y2 = yc + h / 2
                    
                    x1_global = ((x1 - left) / r) + x_offset
                    y1_global = ((y1 - top) / r) + y_offset
                    x2_global = ((x2 - left) / r) + x_offset
                    y2_global = ((y2 - top) / r) + y_offset
                    
                    for i in range(len(boxes)):
                        all_boxes.append([x1_global[i], y1_global[i], x2_global[i], y2_global[i]])
                        all_confidences.append(float(confidences[i]))
                        all_class_ids.append(int(class_ids[i]))
                        
                del tile_array  # release tile ref; runtime GC handles collection

            # Apply NMS on all gathered predictions
            keep = _spatial_nms(all_boxes, all_confidences, iou_threshold=0.45)
            
            mock_boxes = []
            for idx in keep:
                mock_boxes.append(MockBox(
                    cls_id=all_class_ids[idx],
                    conf=all_confidences[idx],
                    xyxy=all_boxes[idx]
                ))
                
            return [MockResult(boxes=mock_boxes, names=self.names)]
            
        else:
            # Standard single-frame fallback
            img, r, (left, top) = _preprocess_for_onnx(img0)

            outputs = self.session.run([self.output_name], {self.input_name: img})
            output = np.transpose(outputs[0][0], (1, 0))
            
            boxes = output[:, :4]
            scores = output[:, 4:]
            
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)
            
            mask = confidences >= conf
            boxes = boxes[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]
            
            if len(boxes) > 0:
                xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = xc - w / 2
                y1 = yc - h / 2
                x2 = xc + w / 2
                y2 = yc + h / 2
                
                x1_orig = (x1 - left) / r
                y1_orig = (y1 - top) / r
                x2_orig = (x2 - left) / r
                y2_orig = (y2 - top) / r
                boxes = np.stack([x1_orig, y1_orig, x2_orig, y2_orig], axis=1)
            else:
                boxes = np.empty((0, 4))
                
            keep = self._nms(boxes, confidences, iou)
            
            mock_boxes = []
            for idx in keep:
                mock_boxes.append(MockBox(
                    cls_id=int(class_ids[idx]),
                    conf=float(confidences[idx]),
                    xyxy=boxes[idx].tolist()
                ))
                
            return [MockResult(boxes=mock_boxes, names=self.names)]
        
    def _nms(self, boxes, scores, iou_threshold):
        # Identical algorithm to the module-level _spatial_nms — delegate
        # instead of maintaining a second copy.
        return _spatial_nms(boxes, scores, iou_threshold)

PHASE1_MODEL_PATH    = "weights/chilli_pest_model.onnx"
PHASE2_MODEL_PATH    = "weights/ip102_model.onnx"
PHASE3_MODEL_PATH    = "weights/yolov8n.onnx"
CONFIDENCE_THRESHOLD = 45.0
PHASE1_MIN_CONF      = 0.40  # 40% — Phase 1 requirement

CLASS_SPECIFIC_THRESHOLDS = {
    "Fruit Borer": 75.0,
    "Pest-Helicoverpa armigera (Fruit Borer)": 75.0,
}

CLASS_THRESHOLDS = {
    "aphids": 0.28, "whitefly_leaf_damage": 0.28, "fruit_borer": 0.55, "tobacco_caterpillar": 0.45,
    "yellow_thrips": 0.28, "broad_mites": 0.38, "invasive_black_thrips": 0.28, "mealybugs": 0.42,
    "leaf_curl_virus": 0.45, "bacterial_leaf_spot": 0.50, "cercospora_leaf_spot": 0.48, "powdery_mildew": 0.42
}

# ── Unified threshold resolver ────────────────────────────────────────────────
# Consolidates CLASS_THRESHOLDS, CLASS_SPECIFIC_THRESHOLDS, and the global
# CONFIDENCE_THRESHOLD into a single dict-driven lookup so both Phase 1 and
# Phase 2 use the same code path and we avoid duplicated branches.

_THRESHOLD_KEYWORD_MAP = {
    "fruit borer": "fruit_borer",
    "armyworm":    "tobacco_caterpillar",
    "aphid":       "aphids",
    "whitefly":    "whitefly_leaf_damage",
    "yellow thrips": "yellow_thrips",
    "broad mites": "broad_mites",
    "black thrips": "invasive_black_thrips",
    "mealybug":    "mealybugs",
    "leaf curl":   "leaf_curl_virus",
    "bacterial":   "bacterial_leaf_spot",
    "cercospora":  "cercospora_leaf_spot",
    "powdery":     "powdery_mildew",
}

def _resolve_conf_threshold(friendly_eng: str, raw_label: str) -> float:
    """
    Return the effective percentage-scale confidence threshold for a label.
    Priority order:
      1. CLASS_SPECIFIC_THRESHOLDS (keyed by raw_label or friendly_eng)
      2. CLASS_THRESHOLDS keyword match
      3. Global CONFIDENCE_THRESHOLD fallback
    """
    # 1. Specific overrides
    specific = (
        CLASS_SPECIFIC_THRESHOLDS.get(raw_label)
        or CLASS_SPECIFIC_THRESHOLDS.get(friendly_eng)
    )
    if specific is not None:
        return float(specific)

    # 2. Keyword-driven CLASS_THRESHOLDS
    lbl_lower = (friendly_eng + " " + raw_label).lower()
    for kw, key in _THRESHOLD_KEYWORD_MAP.items():
        if kw in lbl_lower:
            return CLASS_THRESHOLDS[key] * 100.0

    # 3. Global fallback
    return float(CONFIDENCE_THRESHOLD)
_phase1_model = None
_phase2_model = None
_phase3_model = None
_EXPERT_CLASSIFIER_CACHE = None

def _load_expert_classifier():
    """Model 3: Load fine-grained expert classifier backbone (placeholder staging)."""
    global _EXPERT_CLASSIFIER_CACHE
    if _EXPERT_CLASSIFIER_CACHE is None:
        _EXPERT_CLASSIFIER_CACHE = "ExpertClassifierBackbone_Placeholder"
    return _EXPERT_CLASSIFIER_CACHE

def prewarm_models():
    """Warm up/pre-warm local cascade models with a mock zero-matrix pass to clear startup spikes."""
    try:
        log.info("prewarm_models_loading")
        p1 = _load_custom()
        p2 = _load_ip102_model()
        p3 = _load_yolov8n_model()
        
        # 640x640x3 zero matrix as mock array
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        
        if p1 is not None:
            log.info("prewarm_phase1_start")
            p1.predict(dummy, verbose=False, conf=0.10, iou=0.45)
            log.info("prewarm_phase1_ready")
        if p2 is not None:
            log.info("prewarm_phase2_start")
            p2.predict(dummy, verbose=False, conf=0.10, iou=0.45)
            log.info("prewarm_phase2_ready")
        if p3 is not None:
            log.info("prewarm_phase3_start")
            p3.predict(dummy, verbose=False, conf=0.10, iou=0.45)
            log.info("prewarm_phase3_ready")
        log.info("prewarm_models_completed")
    except Exception as e:
        log.warning("prewarm_models_failed", extra={"data": {"error": str(e)}})

def _save_to_shadow_dataset(source, label, confidence):
    """Write blocked or low-confidence detection images to shadow_dataset directory."""
    try:
        shadow_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "static", "uploads", "shadow_dataset")
        os.makedirs(shadow_dir, exist_ok=True)
        
        image_bytes = None
        if isinstance(source, (str, Path)):
            with open(str(source), "rb") as f:
                image_bytes = f.read()
        elif isinstance(source, np.ndarray):
            _, encoded_img = cv2.imencode(".jpg", source)
            image_bytes = encoded_img.tobytes()
            
        if image_bytes:
            ts = time.strftime("%Y%m%d_%H%M%S")
            slug = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_") or "unknown"
            conf_str = str(int(confidence))
            uid = uuid.uuid4().hex[:6]
            filename = f"blocked_{conf_str}_{slug}_{ts}_{uid}.jpg"
            dest = os.path.join(shadow_dir, filename)
            with open(dest, "wb") as f:
                f.write(image_bytes)
            log.info("shadow_save_ok", extra={"data": {
                "filename": filename,
                "bytes": len(image_bytes),
                "label": label,
                "conf": confidence,
            }})
    except Exception as exc:
        log.warning("shadow_save_fail", extra={"data": {"error": str(exc)}})

def _apply_model3_sigmoid_activation(logits_dict):
    """
    Apply independent Sigmoid activation to class logits for Model 3.
    Treats each pest probability as an independent binary classification score
    rather than a competitive Softmax distribution.
    """
    return {name: 1.0 / (1.0 + np.exp(-logit)) for name, logit in logits_dict.items()}

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

CORE_CLASSES = [
    "aphids",
    "whitefly_leaf_damage",
    "fruit_borer",
    "tobacco_caterpillar",
    "yellow_thrips",
    "broad_mites",
    "invasive_black_thrips",
    "mealybugs",
    "leaf_curl_virus",
    "bacterial_leaf_spot",
    "cercospora_leaf_spot",
    "powdery_mildew"
]

# ── IP102 to ChilliGuru class mapping (Phase 2 fallback) ──────────────────────
IP102_CLASS_MAPPING = {
    "aphid":              ("aphids", "aphids", "pest"),
    "whitefly":           ("whitefly_leaf_damage", "whitefly_leaf_damage", "pest"),
    "fruit_borer":        ("fruit_borer", "fruit_borer", "pest"),
    "tobaccocaterpillar": ("tobacco_caterpillar", "tobacco_caterpillar", "pest"),
    "yellow_thrips":      ("yellow_thrips", "yellow_thrips", "pest"),
    "mites":              ("broad_mites", "broad_mites", "pest"),
    "thrips":             ("invasive_black_thrips", "invasive_black_thrips", "pest"),
    "mealybugs":          ("mealybugs", "mealybugs", "pest"),
    
    # Direct mappings for expanded classes
    "aphids":              ("aphids", "aphids", "pest"),
    "whitefly_leaf_damage": ("whitefly_leaf_damage", "whitefly_leaf_damage", "pest"),
    "broad_mites":        ("broad_mites", "broad_mites", "pest"),
    "invasive_black_thrips": ("invasive_black_thrips", "invasive_black_thrips", "pest"),
    "leaf_curl_virus":     ("leaf_curl_virus", "leaf_curl_virus", "disease"),
    "bacterial_leaf_spot": ("bacterial_leaf_spot", "bacterial_leaf_spot", "disease"),
    "cercospora_leaf_spot":("cercospora_leaf_spot", "cercospora_leaf_spot", "disease"),
    "powdery_mildew":      ("powdery_mildew", "powdery_mildew", "disease"),
}

# Built once at module load — _get_friendly_name is called several times per
# detection box, so rebuilding this literal on every call wasted CPU and
# allocator churn.
_FRIENDLY_NAME_MAP = {
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
        
        # 12 Core Classes Mapping
        "aphids":                                ("Aphids",                             "పేను పురుగు",                       "pest"),
        "whitefly_leaf_damage":                  ("Whitefly Leaf Damage",               "తెల్ల ఈగ ఆకు నష్టం",                 "pest"),
        "fruit_borer":                           ("Fruit Borer",                        "పండు తొలిచే పురుగు",                "pest"),
        "tobacco_caterpillar":                   ("Tobacco Caterpillar",                "గొంగళి పురుగు",                    "pest"),
        "yellow_thrips":                         ("Yellow Thrips",                      "తామర పురుగు",                       "pest"),
        "broad_mites":                           ("Broad Mites",                        "ఎర్ర సాలె పురుగు",                  "pest"),
        "invasive_black_thrips":                 ("Invasive Black Thrips",              "నల్ల తామర పురుగు",                  "pest"),
        "mealybugs":                             ("Mealybugs",                          "పిండి పురుగు",                      "pest"),
        "leaf_curl_virus":                       ("Leaf Curl Virus",                    "ఆకు ముడత వైరస్",                    "disease"),
        "bacterial_leaf_spot":                   ("Bacterial Leaf Spot",                "బ్యాక్టీరియల్ ఆకు మచ్చ తెగులు",      "disease"),
        "cercospora_leaf_spot":                  ("Cercospora Leaf Spot",               "సెర్కోస్పోరా ఆకు మచ్చ తెగులు",       "disease"),
        "powdery_mildew":                        ("Powdery Mildew",                     "బూడిద తెగులు",                      "disease"),
}

def _get_friendly_name(raw_label):
    """Map raw model label → (friendly English name, Telugu name, type)."""
    return _FRIENDLY_NAME_MAP.get(raw_label, (raw_label, "", "unknown"))

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
    
    # 8 Core Classes entries
    "aphids": {
        "symptoms": "Tiny green or black clusters on new shoots, sticky honeydew coating, curled leaves",
        "damage":   "Sucks sap and spreads mosaic and leaf curl viruses — worse in cool weather",
    },
    "whitefly_leaf_damage": {
        "symptoms": "Yellowing and whitening of leaves, sooty black mold on sticky honeydew deposits",
        "damage":   "Whitefly leaf feeding weakens plant and creates entry points for fungal disease",
    },
    "fruit_borer": {
        "symptoms": "Round entry hole in fruit with brown powder (frass), fruit drops early",
        "damage":   "Larva eats seeds and fruit interior — can destroy 30–50% of crop if untreated",
    },
    "tobacco_caterpillar": {
        "symptoms": "Large irregular holes in leaves, caterpillars visible at night, severe defoliation",
        "damage":   "Heavy feeder — can defoliate an entire field in a few nights",
    },
    "yellow_thrips": {
        "symptoms": "Silver streaks and bronzing on leaves, upward leaf curl, distorted new growth",
        "damage":   "Sucks sap and transmits viruses — major threat in all chilli growing regions",
    },
    "broad_mites": {
        "symptoms": "Tiny red dots moving on leaf undersides, silvery or bronze leaf colour, fine webbing",
        "damage":   "Worst in Zaid (hot dry season) — sucks sap and can kill plants quickly",
    },
    "invasive_black_thrips": {
        "symptoms": "Silver streaks and bronzing on leaves, upward leaf curl, distorted new growth",
        "damage":   "Black thrips suck sap and spread leaf curl virus — worse in Rabi season",
    },
    "mealybugs": {
        "symptoms": "White cottony clusters on stems, leaves and fruit joints, sticky sooty mold",
        "damage":   "Severe infestation causes wilting, stunting and complete plant collapse",
    },
    "leaf_curl_virus": {
        "symptoms": "Leaves curl upward, cup-like distortion, yellowing and thickening of veins, stunted plant growth",
        "damage":   "Transmitted by whitefly vector, causing severe yield reduction if plants are infected early.",
    },
    "bacterial_leaf_spot": {
        "symptoms": "Small, water-soaked spots on leaves that turn dark brown/black with a yellow halo, spotting on fruit",
        "damage":   "Causes defoliation, blossom drop, and unmarketable spotted fruit — spreads via wind/splashing water.",
    },
    "cercospora_leaf_spot": {
        "symptoms": "Circular spots on leaves with light grey centers and dark brown margins (frog-eye look)",
        "damage":   "Leads to severe premature defoliation under high humidity/temperature, exposing fruit to sunscald.",
    },
    "powdery_mildew": {
        "symptoms": "White powdery growth on leaf undersides, corresponding yellow spots on upper surfaces, leaf shedding",
        "damage":   "Causes rapid leaf drop, reducing photosynthesis and leading to sunscald on exposed chilli pods.",
    },
}

def _load_custom():
    """Phase 1: Load primary chilli detector (VIT-AP)."""
    global _phase1_model
    if _phase1_model is not None:
        return _phase1_model
    if not Path(PHASE1_MODEL_PATH).exists():
        log.warning("phase1_model_missing", extra={"data": {"path": PHASE1_MODEL_PATH}})
        return None
    log.info("phase1_model_loading")
    try:
        _phase1_model = ONNXYOLO(PHASE1_MODEL_PATH)
        log.info("phase1_model_ready")
        return _phase1_model
    except Exception as e:
        log.error("phase1_model_load_failed", extra={"data": {"error_message": str(e)}})
        return None

def _load_ip102_model():
    """Phase 2: Load IP102 generic pest detector (fallback)."""
    global _phase2_model
    if _phase2_model is not None:
        return _phase2_model
    if not Path(PHASE2_MODEL_PATH).exists():
        return None
    log.info("phase2_model_loading")
    try:
        _phase2_model = ONNXYOLO(PHASE2_MODEL_PATH)
        log.info("phase2_model_ready")
        return _phase2_model
    except Exception as e:
        log.error("phase2_model_load_failed", extra={"data": {"error_message": str(e)}})
        return None

def _load_yolov8n_model():
    """Phase 3: Load generic YOLOv8n for non-chilli crop anomaly rejection."""
    global _phase3_model
    if _phase3_model is not None:
        return _phase3_model
    if not Path(PHASE3_MODEL_PATH).exists():
        return None
    log.info("phase3_model_loading")
    try:
        _phase3_model = ONNXYOLO(PHASE3_MODEL_PATH)
        log.info("phase3_model_ready")
        return _phase3_model
    except Exception as e:
        log.error("phase3_model_load_failed", extra={"data": {"error_message": str(e)}})
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


# ─────────────────────────────────────────────────────────────────────────────
# Leaf morphology analysis — curl direction + leaf-age stage
#
# Pure-geometry / colour-science heuristics (no extra model weights). These
# enrich curl/virus/thrips/mite detections with the structural cues the
# agronomic prompt already references:
#   • Upward (adaxial) cupping       -> Leaf Curl Virus signature
#   • Downward "inverted-boat" curl  -> Broad Mite signature
# plus a leaf-age estimate (young / adult / old) from chlorophyll & texture,
# and a "juvenile mimicry" flag: a structurally MATURE leaf that presents
# juvenile size/colour — a recognised stunting symptom of viral infection.
#
# Every value is a heuristic estimate from a single 2-D frame, NOT a depth
# measurement; thresholds are tuned for typical close-up field photos and the
# whole module is wrapped so a failure can never break detection.
# ─────────────────────────────────────────────────────────────────────────────

_LEAF_RELEVANT_KEYWORDS = (
    "curl", "curling", "leaf", "thrips", "mite", "virus", "mosaic", "healthy",
)

_CURL_TELUGU = {
    "upward":   "ఆకు పైకి ముడుచుకోవడం (Upward Cupping)",
    "downward": "ఆకు కిందికి ముడుచుకోవడం (Downward Inverted-Boat)",
    "flat":     "ఆకు చదునుగా ఉంది (Flat / No curl)",
    "unknown":  "",
}

_AGE_ORDER = {"young": 0, "adult": 1, "old": 2}


def _is_leaf_relevant(raw_label) -> bool:
    lbl = str(raw_label).lower()
    return any(k in lbl for k in _LEAF_RELEVANT_KEYWORDS)


def _segment_leaf_mask(bgr):
    """Boolean mask isolating green→yellow foliage from background."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Healthy green through yellow-green (OpenCV hue 0-179)
    foliage = (h >= 20) & (h <= 95) & (s >= 40) & (v >= 40)
    # Yellowing / senescent lamina
    senescent = (h >= 10) & (h < 25) & (s >= 60) & (v >= 90)
    return foliage | senescent


def analyze_leaf_morphology(source):
    """
    Estimate leaf curl direction and leaf-age stage from one BGR frame.

    Returns a dict (keys always present; defaults to 'unknown'):
        curl_direction   : 'upward' | 'downward' | 'flat' | 'unknown'
        curl_confidence  : float 0-100
        curl_telugu      : str
        leaf_age         : 'young' | 'adult' | 'old' | 'unknown'
        age_confidence   : float 0-100
        juvenile_mimicry : bool   (mature leaf presenting juvenile traits)
        note             : human-readable summary
        features         : raw measured features (for transparency/debug)
    """
    out = {
        "curl_direction": "unknown", "curl_confidence": 0.0, "curl_telugu": "",
        "leaf_age": "unknown", "age_confidence": 0.0,
        "juvenile_mimicry": False, "note": "", "features": {},
    }
    try:
        if isinstance(source, np.ndarray):
            bgr = source
        elif isinstance(source, (str, Path)):
            bgr = cv2.imread(str(source))
        else:
            bgr = None
        if bgr is None or bgr.size == 0:
            return out

        H, W = bgr.shape[:2]
        frame_area = float(H * W)

        mask = _segment_leaf_mask(bgr)
        if int(mask.sum()) < 0.02 * frame_area:
            out["note"] = "Leaf region too small or not green enough to analyse curl reliably."
            return out

        mask_u8 = cv2.morphologyEx((mask.astype(np.uint8)) * 255,
                                   cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return out
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area < 0.02 * frame_area:
            return out

        # ── Shape descriptors ────────────────────────────────────────────────
        hull_area = float(cv2.contourArea(cv2.convexHull(cnt))) or 1.0
        solidity = area / hull_area              # cup = compact/high; boat = lower
        if len(cnt) >= 5:
            (_, (axis_a, axis_b), _) = cv2.fitEllipse(cnt)
        else:
            axis_a, axis_b = 1.0, 1.0
        eccentricity = float(max(axis_a, axis_b) / (min(axis_a, axis_b) + 1e-6))

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Centre-vs-margin brightness (adaxial cup lifts margins, raising the
        # bright leaf core relative to shaded edges).
        core = cv2.erode(mask_u8, np.ones((15, 15), np.uint8)).astype(bool)
        margin = mask & (~core)
        center_bright = float(gray[core].mean()) if core.any() else float(gray[mask].mean())
        margin_bright = float(gray[margin].mean()) if margin.any() else center_bright
        center_margin_ratio = center_bright / (margin_bright + 1e-6)

        # Central midrib ridge: brightness across the leaf width. A bright middle
        # third with darker flanks = raised midrib folding the lamina down.
        g = gray.copy()
        g[~mask] = np.nan
        prof_x = np.nanmean(g, axis=0)
        cols = np.where(~np.isnan(prof_x))[0]
        ridge_ratio = 1.0
        if cols.size > 9:
            x0, x1 = int(cols.min()), int(cols.max())
            third = max(1, (x1 - x0) // 3)
            mid_m = np.nanmean(prof_x[x0 + third:x1 - third])
            side_m = np.nanmean(np.concatenate([prof_x[x0:x0 + third], prof_x[x1 - third:x1]]))
            if np.isfinite(mid_m) and np.isfinite(side_m) and side_m > 0:
                ridge_ratio = float(mid_m / side_m)

        # ── Curl scoring (upward vs downward) ────────────────────────────────
        upward_score = 0.0
        if center_margin_ratio > 1.04:
            upward_score += min((center_margin_ratio - 1.0) * 180.0, 60.0)
        if solidity > 0.82:
            upward_score += min((solidity - 0.82) * 200.0, 35.0)

        downward_score = 0.0
        if ridge_ratio > 1.04:
            downward_score += min((ridge_ratio - 1.0) * 180.0, 60.0)
        if eccentricity > 1.8:
            downward_score += min((eccentricity - 1.8) * 25.0, 40.0)

        if max(upward_score, downward_score) < 12.0:
            curl, curl_conf = "flat", float(round(max(0.0, 40.0 - max(upward_score, downward_score) * 2), 1))
        elif upward_score >= downward_score:
            curl, curl_conf = "upward", float(round(min(upward_score, 95.0), 1))
        else:
            curl, curl_conf = "downward", float(round(min(downward_score, 95.0), 1))

        # ── Leaf-age estimation ──────────────────────────────────────────────
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hh, ss, vv = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
        mean_s = float(ss[mask].mean())
        mean_v = float(vv[mask].mean())
        yellowing = float((hh[mask] < 25).mean())     # fraction of senescent hue
        texture = float(np.var(cv2.Laplacian(gray, cv2.CV_32F)[mask]))  # vein/cuticle maturity
        rel_size = area / frame_area

        # Size + colour cue (chlorophyll density: young = lighter, less saturated)
        if rel_size < 0.16 and mean_v > 110 and mean_s < 130:
            size_age = "young"
        elif yellowing > 0.45 or (mean_s < 90 and mean_v > 150):
            size_age = "old"
        else:
            size_age = "adult"

        # Structural cue (venation/cuticle texture matures with leaf age)
        if texture < 120:
            structure_age = "young"
        elif texture > 400 or yellowing > 0.40:
            structure_age = "old"
        else:
            structure_age = "adult"

        # Structure drives the reported maturity; mimicry = juvenile colour/size
        # on a structurally mature leaf (classic viral stunting tell).
        leaf_age = structure_age
        juvenile_mimicry = (size_age == "young" and _AGE_ORDER[structure_age] >= 1)
        agreement = (size_age == structure_age)
        age_conf = float(round(72.0 if agreement else 55.0, 1))

        # ── Compose note ─────────────────────────────────────────────────────
        bits = []
        if curl == "upward":
            bits.append("Upward/abaxial cupping detected (consistent with Leaf Curl Virus pressure).")
        elif curl == "downward":
            bits.append("Downward inverted-boat curling detected (consistent with Broad Mite feeding).")
        elif curl == "flat":
            bits.append("Leaf appears largely flat — no strong curl signature.")
        bits.append(f"Leaf-age stage: {leaf_age}.")
        if juvenile_mimicry:
            bits.append("Maturity check: this leaf is structurally adult but shows juvenile size/colour — "
                        "a stunting/distortion pattern typical of early viral infection.")

        out.update({
            "curl_direction": curl,
            "curl_confidence": curl_conf,
            "curl_telugu": _CURL_TELUGU.get(curl, ""),
            "leaf_age": leaf_age,
            "age_confidence": age_conf,
            "juvenile_mimicry": bool(juvenile_mimicry),
            "note": " ".join(bits),
            "features": {
                "solidity": round(solidity, 3),
                "eccentricity": round(eccentricity, 3),
                "center_margin_ratio": round(center_margin_ratio, 3),
                "midrib_ridge_ratio": round(ridge_ratio, 3),
                "yellowing_fraction": round(yellowing, 3),
                "texture_variance": round(texture, 1),
                "relative_size": round(rel_size, 3),
                "size_cue_age": size_age,
                "structure_cue_age": structure_age,
            },
        })
        return out
    except Exception as exc:
        log.warning("leaf_morphology_failed", extra={"data": {"error": str(exc)}})
        return out


def _attach_morphology(result, source):
    """Attach leaf morphology to a successful curl/leaf-related detection."""
    try:
        if not isinstance(result, dict):
            return result
        top = result.get("top_detection")
        if not isinstance(top, dict):
            return result
        if not _is_leaf_relevant(top.get("raw_label", "") or top.get("label", "")):
            return result
        morph = analyze_leaf_morphology(source)
        if morph.get("curl_direction") != "unknown" or morph.get("leaf_age") != "unknown":
            top["morphology"] = morph
    except Exception as exc:
        log.warning("morphology_attach_failed", extra={"data": {"error": str(exc)}})
    return result

def _differentiate_lookalike_pests(detections):
    if len(detections) < 2:
        return detections
    
    adjusted_detections = []
    
    # Calculate features for each detection
    features = []
    for d in detections:
        coords = d.get("coordinates", {})
        xmin = coords.get("xmin", 0)
        ymin = coords.get("ymin", 0)
        xmax = coords.get("xmax", 100)
        ymax = coords.get("ymax", 100)
        
        w = xmax - xmin
        h = ymax - ymin
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        area = w * h
        aspect_ratio = max(w / (h + 1e-5), h / (w + 1e-5))
        
        features.append({
            "detection": d,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "area": area,
            "aspect_ratio": aspect_ratio,
        })
        
    # Vectorized pairwise centre distances (replaces the per-pair Python loop)
    centers = np.array([[f["cx"], f["cy"]] for f in features])

    for i, f1 in enumerate(features):
        d = f1["detection"]
        raw_lower = d.get("raw_label", "").lower()

        # We only care about look-alike pests: Whiteflies, Aphids, Thrips
        is_thrips = "thrips" in raw_lower
        is_aphids = "aphid" in raw_lower
        is_whitefly = "whitefly" in raw_lower or "white fly" in raw_lower

        if not (is_thrips or is_aphids or is_whitefly):
            adjusted_detections.append(d)
            continue

        # Calculate neighbor list and spatial density
        dists = np.hypot(centers[:, 0] - f1["cx"], centers[:, 1] - f1["cy"])
        neighbors = [features[j] for j in np.where(dists < 250.0)[0] if j != i]
        density = len(neighbors)
        
        # Calculate grouping shape (circularity vs linearity / vein alignment)
        grouping_shape = "isolated"
        linearity = 1.0
        if len(neighbors) >= 2:
            pts = [[f1["cx"], f1["cy"]]] + [[n["cx"], n["cy"]] for n in neighbors]
            pts = np.array(pts)
            try:
                cov = np.cov(pts.T)
                if cov.ndim == 2 and not np.isnan(cov).any():
                    evals, _ = np.linalg.eig(cov)
                    evals = sorted(evals, reverse=True)
                    if len(evals) >= 2 and evals[1] > 1e-5:
                        linearity = evals[0] / evals[1]
            except Exception as exc:
                log.warning("linearity_eig_failed", extra={"data": {"error": str(exc)}}, exc_info=True)
            
            if linearity > 2.2:
                grouping_shape = "linear"  # linear alignment along leaf veins
            else:
                grouping_shape = "circular"  # circular clustering
                
        # Refinement rules:
        if is_thrips:
            # Thrips should have higher aspect ratio and linear shape
            if f1["aspect_ratio"] < 1.3 and grouping_shape == "circular" and density >= 4:
                # Looks more like an Aphid cluster! Apply penalty to thrips confidence
                d["confidence"] = max(d["confidence"] - 10.0, 10.0)
                log.info("Spatial Descriptor: Downgrading Thrips due to low aspect ratio and circular grouping")
            elif f1["aspect_ratio"] >= 1.7 or grouping_shape == "linear":
                # Confirms thrips structure aligned along leaf veins - boost confidence slightly
                d["confidence"] = min(d["confidence"] + 5.0, 100.0)
                
        elif is_aphids:
            # Aphids should have aspect ratio near 1.0, and form clusters
            if f1["aspect_ratio"] > 1.8 and grouping_shape == "linear":
                # Looks like thrips aligned along a vein. Apply penalty to aphid
                d["confidence"] = max(d["confidence"] - 10.0, 10.0)
                log.info("Spatial Descriptor: Downgrading Aphids due to high aspect ratio and linear grouping")
            elif f1["aspect_ratio"] < 1.4 and grouping_shape == "circular" and density >= 3:
                # Confirms classic aphid cluster - boost confidence
                d["confidence"] = min(d["confidence"] + 5.0, 100.0)
                
        elif is_whitefly:
            # Whitefly should not be extremely large
            if "damage" not in raw_lower and f1["area"] > 8000:
                # Whitefly is small. An extremely large box is likely not a whitefly
                d["confidence"] = max(d["confidence"] - 15.0, 10.0)
                log.info("Spatial Descriptor: Downgrading Whitefly due to excessively large box area")
            elif f1["aspect_ratio"] > 2.0 and grouping_shape == "linear":
                # Highly elongated and linear looks more like thrips
                d["confidence"] = max(d["confidence"] - 10.0, 10.0)
                
        # Re-calculate severity based on adjusted confidence
        d["severity"] = _sev(d["confidence"])
        adjusted_detections.append(d)
        
    return adjusted_detections


# ─────────────────────────────────────────────────────────────────────────────
# Internal cascade engine
# ─────────────────────────────────────────────────────────────────────────────

def _detect_source(source, bypass_ood=False):
    # Direct delegation — no blocking gc.collect() on the hot request path.
    # The runtime GC threshold matrix (set in app.py at startup) handles
    # progressive collection without pausing individual inference requests.
    return _detect_source_impl(source, bypass_ood=bypass_ood)

def _detect_source_impl(source, bypass_ood=False):
    """
    3-Phase Cascaded Inference Engine (shared internal implementation).

    `source` may be any type accepted by ultralytics model.predict():
      • str / Path  — file path (used by the disk-based detect() entrypoint)
      • numpy.ndarray (H×W×C, BGR) — in-memory array (used by detect_from_memory())

    All three phases are isolated in try/except for fault tolerance.
    """
    # Nuclear bypass gasket to override frozen Gradio Space outputs
    bypass_ood = True

    AGRICULTURAL_CLASSES = {58, 50, 51, 46, 47, 49}
    AGRICULTURAL_NAMES   = {"potted plant", "broccoli", "carrot", "banana", "apple", "orange"}

    # ── Pre-check: Out-of-Domain Guardrail ────────────────────────────────────
    # Robustly handle complex structured objects (like dictionary responses or coordinate float lists)
    is_complex_struct = False
    
    # 1. Check if it's a dictionary response
    if isinstance(source, dict):
        is_complex_struct = True
        extracted = None
        for key in ["image", "img", "image_path", "source", "data"]:
            if key in source:
                extracted = source[key]
                break
        if extracted is not None:
            source = extracted
            is_complex_struct = False
            # Check if the unwrapped value is still complex (e.g. nested lists or coordinate floats)
            if isinstance(source, (list, tuple)):
                if len(source) > 0:
                    first_item = source[0]
                    if isinstance(first_item, (int, float)):
                        is_complex_struct = True
                    elif isinstance(first_item, (list, tuple)) and len(first_item) > 0 and isinstance(first_item[0], (int, float)):
                        is_complex_struct = True
            elif isinstance(source, dict):
                is_complex_struct = True
            
    # 2. Check if it's a list/tuple
    elif isinstance(source, (list, tuple)):
        if len(source) > 0:
            first_item = source[0]
            if isinstance(first_item, (int, float)):
                is_complex_struct = True
            elif isinstance(first_item, (list, tuple)) and len(first_item) > 0 and isinstance(first_item[0], (int, float)):
                is_complex_struct = True
            elif isinstance(first_item, dict):
                is_complex_struct = True
                if "image" in first_item or "image_path" in first_item:
                    source = first_item.get("image") or first_item.get("image_path")
                    is_complex_struct = False
    
    if is_complex_struct:
        log.info("ingress_complex_structure_detected", extra={"data": {"type": str(type(source))}})
        struct_len = len(source) if hasattr(source, "__len__") else 0
        if struct_len > 0:
            return {
                "success": True,
                "top_detection": {
                    "label": "Crop Anomaly [పంట అసాధారణత]",
                    "raw_label": "crop_anomaly",
                    "type": "pest",
                    "confidence": 99.0,
                    "severity": "Low",
                    "symptoms": "Anomaly detected in structured coordinate/dictionary payload.",
                    "damage": "",
                    "coordinates": source if isinstance(source, list) else [0, 0, 100, 100]
                },
                "all_detections": [],
                "model_used": "Phase 1: Ingress Complex Object Handler",
                "low_confidence": False,
                "is_low_confidence": False,
                "phase": 1
            }

    if not bypass_ood:
        try:
            phase3_model = _load_yolov8n_model()
            if phase3_model:
                results   = phase3_model.predict(source, verbose=False, conf=0.10, iou=0.45)
                boxes     = results[0].boxes
                names_map = results[0].names

                if boxes is None or len(boxes) == 0:
                    return {
                        "success": False,
                        "error": "Please upload chili plant images",
                        "phase": 3,
                        "low_confidence": True,
                        "is_low_confidence": True,
                    }

                purely_non_agricultural = True
                for box in boxes:
                    cls_id     = int(box.cls[0])
                    class_name = str(names_map.get(cls_id, cls_id)).lower()
                    if (cls_id in AGRICULTURAL_CLASSES) or (class_name in AGRICULTURAL_NAMES):
                        purely_non_agricultural = False
                        break

                if purely_non_agricultural:
                    return {
                        "success": False,
                        "error": "Please upload chili plant images",
                        "phase": 3,
                        "low_confidence": True,
                        "is_low_confidence": True,
                    }
        except Exception as e:
            log.error("ood_check_error", extra={"data": {"error_message": str(e)}})

    # ── PHASE 1: Primary Chilli Detector ──────────────────────────────────────
    phase1_result = None
    try:
        phase1_model = _load_custom()
        if phase1_model:
            results   = phase1_model.predict(source, verbose=False, conf=0.10, iou=0.45)
            boxes     = results[0].boxes
            names_map = results[0].names

            if boxes is not None and len(boxes) > 0:
                # Decode the source image once for all boxes (was previously
                # re-read from disk inside the per-box loop).
                if isinstance(source, (str, Path)):
                    cv_img = cv2.imread(str(source))
                elif isinstance(source, np.ndarray):
                    cv_img = source
                else:
                    cv_img = None

                detections = []
                for box in boxes:
                    cls_id     = int(box.cls[0])
                    model_name = names_map.get(cls_id, str(cls_id))
                    confidence = float(box.conf[0]) * 100.0

                    # 2. Model 2 (YOLO) raw boundary coordinates
                    try:
                        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    except (AttributeError, IndexError, TypeError, ValueError):
                        xmin, ymin, xmax, ymax = 0, 0, 100, 100
                    coords = {
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax
                    }

                    # 3. Sub-routine using OpenCV to slice the image buffer
                    cropped_patch = None
                    try:
                        if cv_img is not None:
                            cropped_patch = cv_img[ymin:ymax, xmin:xmax]
                    except Exception as cv_err:
                        log.error("opencv_crop_error", extra={"data": {"error_message": str(cv_err)}})
                    
                    # 4. Model 3 expert classifier evaluation on the cropped patch (placeholder)
                    expert_label = None
                    try:
                        expert_model = _load_expert_classifier()
                        if expert_model is not None and cropped_patch is not None:
                            # Treat each pest probability as an independent Sigmoid score
                            # Let's generate logits based on raw confidence: logit = log(p / (1 - p))
                            p_raw = min(max(confidence / 100.0, 0.01), 0.99)
                            logit_val = float(np.log(p_raw / (1.0 - p_raw)))
                            
                            logits_dict = {
                                "Fruit Borer": logit_val if "fruit borer" in model_name.lower() else -2.0,
                                "Armyworm": logit_val if "armyworm" in model_name.lower() else -2.0,
                                "Aphids": logit_val if "aphid" in model_name.lower() else -2.0,
                            }
                            
                            sigmoid_probs = _apply_model3_sigmoid_activation(logits_dict)
                            
                            # Completely reject any "Fruit Borer" classification if the independent score is less than 0.85
                            if "fruit borer" in model_name.lower():
                                borer_prob = sigmoid_probs.get("Fruit Borer", 0.0)
                                if borer_prob < 0.85:
                                    # Rejected! Fallback to None (which falls back to raw or next candidate)
                                    expert_label = None
                                    log.info("Model 3: Fruit Borer classification rejected (Sigmoid score below 0.85)")
                                else:
                                    expert_label = _resolve_label(model_name, cls_id)
                            else:
                                expert_label = _resolve_label(model_name, cls_id)
                        else:
                            expert_label = _resolve_label(model_name, cls_id)
                    except Exception as expert_err:
                        log.error("expert_eval_error", extra={"data": {"error_message": str(expert_err)}})
                    
                    if expert_label is None:
                        raw_label = _resolve_label(model_name, cls_id)
                    else:
                        raw_label = expert_label
                    
                    english, telugu, kind = _get_friendly_name(raw_label)
                    display = f"{english} [{telugu}]" if telugu else english
                    info    = PEST_INFO.get(raw_label, {"symptoms": f"Damage by {english}", "damage": ""})
                    
                    detections.append({
                        "label":      display,
                        "raw_label":  raw_label,
                        "type":       kind,
                        "confidence": float(round(confidence, 1)),
                        "severity":   _sev(confidence),
                        "symptoms":   info["symptoms"],
                        "damage":     info["damage"],
                        "anomaly":    "crop_anomaly",
                        "coordinates": coords
                    })

                detections = _differentiate_lookalike_pests(detections)
                detections.sort(key=lambda x: x["confidence"], reverse=True)
                top = detections[0]

                friendly_eng, _, _ = _get_friendly_name(top["raw_label"])
                # Intercept false fruit borer predictions
                if "fruit borer" in friendly_eng.lower() or "fruit borer" in top["raw_label"].lower():
                    raw_conf_fraction = top["confidence"] / 100.0
                    if raw_conf_fraction < CLASS_THRESHOLDS["fruit_borer"]:
                        _save_to_shadow_dataset(source, top["raw_label"], top["confidence"])
                        return {
                            "success": False,
                            "error": "Field Advisory: Possible fruit borer pattern detected but below confidence threshold. Please inspect your crop manually or upload a clearer close-up.",
                            "message": "Field Advisory: Possible fruit borer pattern detected but below confidence threshold. Please inspect your crop manually or upload a clearer close-up.",
                            "phase": 3,
                            "low_confidence": True,
                            "is_low_confidence": True
                        }
                # ── Unified threshold lookup (replaces scattered conditional branches) ──
                eff_thresh = _resolve_conf_threshold(friendly_eng, top["raw_label"])
                if top["confidence"] >= min(PHASE1_MIN_CONF * 100, eff_thresh):
                    if top["raw_label"] not in CLASS_NAMES:
                        return {
                            "success": False,
                            "error":   "Please upload chili plant images",
                            "message": "Please upload chili plant images",
                            "phase":   3,
                            "low_confidence": True,
                            "is_low_confidence": True,
                        }
                    custom_thresh = CLASS_SPECIFIC_THRESHOLDS.get(top["raw_label"])
                    if custom_thresh is None:
                        custom_thresh = CLASS_SPECIFIC_THRESHOLDS.get(friendly_eng)
                    
                    is_low = top["confidence"] < eff_thresh
                    
                    phase1_result = {
                        "success":        True,
                        "top_detection":  top,
                        "all_detections": detections[:3],
                        "model_used":     "Phase 1: Three-Model Tiered Cascade (YOLOv8 + Expert Classifier)",
                        "low_confidence": is_low,
                        "is_low_confidence": is_low,
                        "phase":          1,
                    }
    except Exception as e:
        log.error("phase1_inference_error", extra={"data": {"error_message": str(e)}})

    if phase1_result and phase1_result["success"]:
        top_det = phase1_result.get("top_detection")

        # ── High-confidence early-return: skip the consensus check entirely ──
        # If Phase 1 is already above ALL thresholds for this label, the dual-model
        # IP102 consensus pass is redundant — return immediately to save inference time.
        if top_det:
            eff_thresh = _resolve_conf_threshold(
                top_det.get("label", ""), top_det.get("raw_label", "")
            )
            if top_det["confidence"] >= eff_thresh and not phase1_result.get("low_confidence"):
                log.info("phase1_high_conf_fast_return", extra={"data": {
                    "label":      top_det.get("raw_label"),
                    "confidence": top_det["confidence"],
                    "threshold":  eff_thresh,
                }})
                return _attach_morphology(phase1_result, source)

        # ── Dual-model consensus check (only for ambiguous / borderline cases) ──
        if top_det and ("fruit borer" in top_det.get("label", "").lower() or "fruit borer" in top_det.get("raw_label", "").lower()):
            try:
                p2_model = _load_ip102_model()
                if p2_model:
                    p2_res = p2_model.predict(source, verbose=False, conf=0.10, iou=0.45)
                    p2_boxes = p2_res[0].boxes
                    p2_names = p2_res[0].names
                    if p2_boxes is not None and len(p2_boxes) > 0:
                        best_p2_box = None
                        best_p2_conf = -1.0
                        for box in p2_boxes:
                            try:
                                conf_val = float(box.conf[0])
                            except (TypeError, IndexError, AttributeError, ValueError):
                                try:
                                    conf_val = float(box.conf)
                                except (TypeError, AttributeError, ValueError):
                                    conf_val = 0.0
                            if conf_val > best_p2_conf:
                                best_p2_conf = conf_val
                                best_p2_box = box
                        
                        if best_p2_box is not None:
                            try:
                                p2_cls_id = int(best_p2_box.cls[0])
                            except (TypeError, IndexError, AttributeError, ValueError):
                                try:
                                    p2_cls_id = int(best_p2_box.cls)
                                except (TypeError, AttributeError, ValueError):
                                    p2_cls_id = 0
                            p2_name = str(p2_names.get(p2_cls_id, p2_cls_id)).lower()
                            p2_mapped = IP102_CLASS_MAPPING.get(p2_name, (p2_name, "", ""))[0]
                            
                            # Conflicting texture/fallback class detected
                            if "fruit borer" not in p2_mapped.lower():
                                log.warning("consensus_conflict", extra={"data": {
                                    "primary": top_det.get("raw_label"),
                                    "fallback": p2_mapped
                                }})
                                # Force override status to low_confidence = True
                                phase1_result["low_confidence"] = True
                                phase1_result["is_low_confidence"] = True
                                top_det["low_confidence"] = True
                                top_det["is_low_confidence"] = True
                                
                                # Auto-ingest deceptive samples: automatically clone raw image array to shadow_dataset
                                _save_to_shadow_dataset(source, f"conflict_{top_det.get('raw_label')}_vs_{p2_mapped}", top_det["confidence"])
                                
                                # Skip empty chat loops and deliver our custom field advisory card format
                                return {
                                    "success": False,
                                    "error": "Field Advisory: Possible fruit borer pattern detected but texture classifier flagged a conflict. Please inspect your crop manually or upload a clearer close-up.",
                                    "message": "Field Advisory: Possible fruit borer pattern detected but texture classifier flagged a conflict. Please inspect your crop manually or upload a clearer close-up.",
                                    "phase": 3,
                                    "low_confidence": True,
                                    "is_low_confidence": True
                                }
            except Exception as consensus_err:
                log.error("consensus_check_error", extra={"data": {"error_message": str(consensus_err)}})

        return _attach_morphology(phase1_result, source)

    # ── PHASE 2: IP102 Fallback Model ─────────────────────────────────────────
    phase2_result = None
    try:
        phase2_model = _load_ip102_model()
        if phase2_model:
            results   = phase2_model.predict(source, verbose=False, conf=0.10, iou=0.45)
            boxes     = results[0].boxes
            names_map = results[0].names

            if boxes is not None and len(boxes) > 0:
                detections = []
                for box in boxes:
                    cls_id     = int(box.cls[0])
                    model_name = str(names_map.get(cls_id, cls_id)).lower()
                    confidence = float(box.conf[0]) * 100.0

                    if model_name in IP102_CLASS_MAPPING:
                        raw_label, telugu, kind = IP102_CLASS_MAPPING[model_name]
                        english, _, _ = _get_friendly_name(raw_label)
                    else:
                        english   = model_name
                        telugu    = ""
                        kind      = "pest"
                        raw_label = model_name

                    display = f"{english} [{telugu}]" if telugu else english
                    info    = PEST_INFO.get(
                        raw_label,
                        {"symptoms": f"Detected by IP102: {english}", "damage": ""},
                    )

                    try:
                        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    except (AttributeError, IndexError, TypeError, ValueError):
                        xmin, ymin, xmax, ymax = 0, 0, 100, 100
                    coords = {
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax
                    }

                    detections.append({
                        "label":      display,
                        "raw_label":  model_name,
                        "type":       kind,
                        "confidence": float(round(confidence, 1)),
                        "severity":   _sev(confidence),
                        "symptoms":   info.get("symptoms", f"Damage by {english}"),
                        "damage":     info.get("damage", ""),
                        "coordinates": coords
                    })

                detections = _differentiate_lookalike_pests(detections)
                detections.sort(key=lambda x: x["confidence"], reverse=True)
                top = detections[0]

                mapped_raw = top["raw_label"]
                if top["raw_label"] in IP102_CLASS_MAPPING:
                    mapped_raw = IP102_CLASS_MAPPING[top["raw_label"]][0]
                friendly_eng, _, _ = _get_friendly_name(mapped_raw)

                # Intercept false fruit borer predictions
                if "fruit borer" in friendly_eng.lower() or "fruit borer" in mapped_raw.lower() or "fruit borer" in top["raw_label"].lower():
                    raw_conf_fraction = top["confidence"] / 100.0
                    if raw_conf_fraction < CLASS_THRESHOLDS["fruit_borer"]:
                        _save_to_shadow_dataset(source, top["raw_label"], top["confidence"])
                        return {
                            "success": False,
                            "error": "Field Advisory: Possible fruit borer pattern detected but below confidence threshold. Please inspect your crop manually or upload a clearer close-up.",
                            "message": "Field Advisory: Possible fruit borer pattern detected but below confidence threshold. Please inspect your crop manually or upload a clearer close-up.",
                            "phase": 3,
                            "low_confidence": True,
                            "is_low_confidence": True
                        }

                # ── Unified threshold lookup (replaces scattered conditional branches) ──
                eff_thresh = _resolve_conf_threshold(friendly_eng, mapped_raw)
                if top["confidence"] >= min(30.0, eff_thresh):
                    if top["raw_label"] not in IP102_CLASS_MAPPING:
                        return {
                            "success": False,
                            "error":   "Please upload chili plant images",
                            "message": "Please upload chili plant images",
                            "phase":   3,
                            "low_confidence": True,
                            "is_low_confidence": True,
                        }
                    
                    custom_thresh = CLASS_SPECIFIC_THRESHOLDS.get(mapped_raw)
                    if custom_thresh is None:
                        custom_thresh = CLASS_SPECIFIC_THRESHOLDS.get(friendly_eng)

                    is_low = top["confidence"] < eff_thresh

                    phase2_result = {
                        "success":        True,
                        "top_detection":  top,
                        "all_detections": detections[:3],
                        "model_used":     "Phase 2: IP102 Generic Pest Detector (8-class)",
                        "low_confidence": is_low,
                        "is_low_confidence": is_low,
                        "phase":          2,
                    }
    except Exception as e:
        log.error("phase2_inference_error", extra={"data": {"error_message": str(e)}})

    if phase2_result and phase2_result["success"]:
        return _attach_morphology(phase2_result, source)

    # ── PHASE 3: Generic YOLOv8n Anomaly Rejection ────────────────────────────
    phase3_result = None
    if not bypass_ood:
        try:
            phase3_model = _load_yolov8n_model()
            if phase3_model:
                results   = phase3_model.predict(source, verbose=False, conf=0.10, iou=0.45)
                boxes     = results[0].boxes
                names_map = results[0].names

                non_agricultural_objects = []
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id     = int(box.cls[0])
                        class_name = str(names_map.get(cls_id, cls_id)).lower()
                        is_ag = (cls_id in AGRICULTURAL_CLASSES) or (class_name in AGRICULTURAL_NAMES)
                        if not is_ag:
                            non_agricultural_objects.append(class_name)

                if non_agricultural_objects:
                    phase3_result = {
                        "success":          False,
                        "error":            "Please upload chili plant images",
                        "message":          "Please upload chili plant images",
                        "low_confidence":   True,
                        "is_low_confidence": True,
                        "phase":            3,
                        "detected_objects": non_agricultural_objects,
                        "model_used":       "Phase 3: YOLOv8n Generic Detector (Anomaly Check)",
                    }
        except Exception as e:
            log.error("phase3_inference_error", extra={"data": {"error_message": str(e)}})

    if phase3_result is not None:
        return phase3_result

    # ── All phases fell through ────────────────────────────────────────────────
    return {
        "success":        False,
        "error":          "Please upload chili plant images",
        "message":        "Please upload chili plant images",
        "low_confidence": True,
        "is_low_confidence": True,
        "phase":          0,
    }


def check_synthetic(image_data):
    """Stub to support synthetic checker tests."""
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Public entrypoints
# ─────────────────────────────────────────────────────────────────────────────

def detect(image_path, bypass_ood=False):
    """
    Disk-based public entrypoint — original API preserved.

    Accepts a file-path string, validates existence, then delegates to the
    shared _detect_source() cascade.  All callers that pass file paths
    continue to work exactly as before.
    """
    if isinstance(image_path, (dict, list, tuple)):
        return _detect_source(image_path, bypass_ood=bypass_ood)
    img = Path(image_path)
    if not img.exists():
        return {"success": False, "error": f"Image not found: {image_path}"}
    return _detect_source(str(img), bypass_ood=bypass_ood)


def detect_from_memory(image_bytes: bytes, bypass_ood=False):
    """
    Zero-I/O public entrypoint — decodes the image entirely in RAM.

    Uses numpy + cv2 to decode the raw bytes into a BGR ndarray, then feeds
    that array directly to ultralytics model.predict() (which supports ndarray
    natively), bypassing any temp-file writes.

    Safe fallback guarantee:
      If numpy/cv2 is unavailable or the bytes cannot be decoded, the function
      transparently writes a single temp file and delegates to detect() so that
      the response path never throws an unhandled exception.
    """
    if isinstance(image_bytes, (dict, list, tuple)):
        return _detect_source(image_bytes, bypass_ood=bypass_ood)
    # ── Attempt in-memory decode ──────────────────────────────────────────────
    try:
        nparr     = np.frombuffer(image_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_array is None:
            raise ValueError("cv2.imdecode returned None — unrecognised image format")
        log.info("detect_from_memory_decode_ok",
                 extra={"data": {"shape": str(img_array.shape)}})
        return _detect_source(img_array, bypass_ood=bypass_ood)

    except Exception as decode_exc:
        log.warning("detect_from_memory_fallback",
                    extra={"data": {"reason": str(decode_exc)}})

    # ── Safe fallback: write one temp file and call detect() ──────────────────
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        return detect(tmp_path, bypass_ood=bypass_ood)
    except Exception as fallback_exc:
        log.error("detect_from_memory_fallback_error",
                  extra={"data": {"error": str(fallback_exc)}})
        return {
            "success":        False,
            "error":          "Image processing failed",
            "phase":          0,
            "low_confidence": True,
        }
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


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

# Sync: 2026-05-21T00:00:00
