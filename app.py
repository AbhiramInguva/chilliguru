import functools
import gc
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
import requests

# ── Calibrate GC thresholds at process launch ─────────────────────────────────
# Raises gen-0 threshold so the collector runs less aggressively per request,
# preventing blocking pauses inside the hot /chat and /detect handlers.
gc.set_threshold(1000, 10, 10)
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from gradio_client import Client, handle_file
from groq import Groq
import detector

# ── Global background worker pool (replaces per-request daemon Thread spawns) ──
# max_workers=2 keeps total thread headroom well within the 150 MB container.
_shadow_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="shadow-worker")

# ── Persistent HTTP session (connection-pool reuse for Open-Meteo calls) ──
_http_session = requests.Session()

# ── Structured JSON logger ────────────────────────────────────────────────────
class _JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts":    self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        if hasattr(record, "data") and isinstance(record.data, dict):
            payload.update(record.data)
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_JsonFormatter())
logging.root.setLevel(logging.INFO)
logging.root.handlers = [_handler]
log = logging.getLogger("chilliguru")
REGIONAL_TRANSLATION_MAP = {
    "aphids": {
        "en": "Aphids",
        "hi": "माहू (एफिड्स)",
        "te": "పేను పురుగు",
        "kn": "ಸೇಬು ಜೇನು ನೊಣ (ಅಫಿಡ್ಸ್)",
        "ta": "அசுவினி"
    },
    "whitefly_leaf_damage": {
        "en": "Whitefly Leaf Damage",
        "hi": "सफेद मक्खी का नुकसान",
        "te": "తెల్ల ఈగ ఆకు నష్టం",
        "kn": "ಬಿಳಿ ನೊಣದ ಎಲೆ ಹಾನಿ",
        "ta": "வெள்ளை ஈ இலை சேதம்"
    },
    "fruit_borer": {
        "en": "Fruit Borer",
        "hi": "फल छेदक",
        "te": "పండు తొలిచే పురుగు",
        "kn": "ಕಾಯಿ ಕೊರಕ",
        "ta": "காய்ப்புழு"
    },
    "tobacco_caterpillar": {
        "en": "Tobacco Caterpillar",
        "hi": "तंबाकू की इल्ली",
        "te": "పొగాకు లద్దె పురుగు",
        "kn": "ತಂಬಾಕು ಪತಂಗ",
        "ta": "புகையிலை வெட்டுப்புழு"
    },
    "yellow_thrips": {
        "en": "Yellow Thrips",
        "hi": "पीला थ्रिप्स",
        "te": "పసుపు తామర పురుగు",
        "kn": "ಹಳದಿ ನುಸಿ",
        "ta": "மஞ்சள் இலைப்பேன்"
    },
    "broad_mites": {
        "en": "Broad Mites",
        "hi": "चौड़ी मक्खियाँ (माइट्स)",
        "te": "ఎర్ర నల్లి / తామర పురుగు",
        "kn": "ಅಗಲವಾದ ನುಸಿ",
        "ta": "பரந்த சிலந்திப் பேன்"
    },
    "invasive_black_thrips": {
        "en": "Invasive Black Thrips",
        "hi": "आक्रामक काला थ्रिप्स",
        "te": "నల్ల తామర పురుగు",
        "kn": "ಆಕ್ರಮಣಕಾರಿ ಕಪ್ಪು ನುಸಿ",
        "ta": "ஊடுருவும் கருப்பு இலைப்பேன்"
    },
    "mealybugs": {
        "en": "Mealybugs",
        "hi": "मीलीबग",
        "te": "పిండి పురుగు",
        "kn": "ಹಿಟ್ಟು ತಿಗಣೆ",
        "ta": "மாவுப்பூச்சி"
    },
    "leaf_curl_virus": {
        "en": "Leaf Curl Virus",
        "hi": "पत्ती मरोड़ वायरस",
        "te": "ఆకు ముడత వైరస్",
        "kn": "ಎಲೆ ಮುರುಟು ವೈರಸ್",
        "ta": "இலை சுருள் வைரஸ்"
    },
    "bacterial_leaf_spot": {
        "en": "Bacterial Leaf Spot",
        "hi": "जीवाणु पत्ती धब्बा",
        "te": "బ్యాక్టీరిयల్ ఆకు మచ్చ తెగులు",
        "kn": "ಬ್ಯಾಕ್ಟೀರಿಯಾದ ಎಲೆ ಚುಕ್ಕೆ",
        "ta": "பாக்டீரியா இலைப்புள்ளி"
    },
    "cercospora_leaf_spot": {
        "en": "Cercospora Leaf Spot",
        "hi": "सर्कोस्पोरा पत्ता धब्बा",
        "te": "సెర్కోస్పోరా ఆకు మచ్చ తెగులు",
        "kn": "ಸೆರ್ಕೋಸ್ಪೊರಾ ಎಲೆ ಚುಕ್ಕೆ",
        "ta": "செர்கோஸ்போரா இலைப்புள்ளி"
    },
    "powdery_mildew": {
        "en": "Powdery Mildew",
        "hi": "पाउडर जैसी फफूंदी (चूर्णी फफूंद)",
        "te": "బూడిద తెగులు",
        "kn": "ಬೂದಿ ರೋಗ",
        "ta": "சாம்பல் நோய்"
    }
}

def to_core_class(label_or_name):
    if not label_or_name:
        return None
    lbl = str(label_or_name).lower()
    if "black thrips" in lbl or "invasive" in lbl:
        return "invasive_black_thrips"
    if "yellow thrips" in lbl:
        return "yellow_thrips"
    if "thrips" in lbl:
        return "invasive_black_thrips"
    if "aphid" in lbl:
        return "aphids"
    if "white fly" in lbl or "whitefly" in lbl:
        return "whitefly_leaf_damage"
    if "fruit borer" in lbl or "helicoverpa" in lbl or "borer" in lbl:
        return "fruit_borer"
    if "armyworm" in lbl or "tobaccocaterpillar" in lbl or "tobacco_caterpillar" in lbl or "spodoptera" in lbl:
        return "tobacco_caterpillar"
    if "red mites" in lbl or "broad mites" in lbl or "broad_mites" in lbl or "mites" in lbl:
        return "broad_mites"
    if "mealybug" in lbl:
        return "mealybugs"
    if "leaf curl" in lbl or "curling" in lbl or "mosaic" in lbl or "mozaik" in lbl:
        return "leaf_curl_virus"
    if "cercospora" in lbl:
        return "cercospora_leaf_spot"
    if "bacterial" in lbl:
        return "bacterial_leaf_spot"
    if "spot" in lbl:
        return "bacterial_leaf_spot"
    if "powdery" in lbl or "mildew" in lbl or "leveillula" in lbl:
        return "powdery_mildew"
    return None

# ── Pre-compiled script regex map (built once at module load) ─────────────────
# Eliminates repeated re.compile() calls inside strip_cross_contamination at
# runtime — the compiled Pattern objects are reused across every request.
_SCRIPT_RE: dict = {
    "hi": re.compile(r"[\u0900-\u097f]"),
    "te": re.compile(r"[\u0c00-\u0c7f]"),
    "kn": re.compile(r"[\u0c80-\u0cff]"),
    "ta": re.compile(r"[\u0b80-\u0bff]"),
}
_CLEANUP_RE = {
    "brackets_with_content": re.compile(r"\s*\[[^\]]*\]"),
    "empty_brackets":        re.compile(r"\s*\[\s*\]"),
    "empty_parens":          re.compile(r"\s*\(\s*\)"),
    "multi_space":           re.compile(r"\s+"),
}

# lru_cache memoises identical (text, target_lang) pairs across a request
# burst — repeated SYSTEM_PROMPT cleaning calls become O(1) dict lookups.
@functools.lru_cache(maxsize=128)
def strip_cross_contamination(text, target_lang):
    if not isinstance(text, str):
        return text

    if target_lang == "en":
        text = _CLEANUP_RE["brackets_with_content"].sub("", text)
        for script_re in _SCRIPT_RE.values():
            text = script_re.sub("", text)
    else:
        for lang_code, script_re in _SCRIPT_RE.items():
            if lang_code != target_lang:
                text = script_re.sub("", text)
        text = _CLEANUP_RE["empty_brackets"].sub("", text)

    text = _CLEANUP_RE["empty_parens"].sub("", text)
    text = _CLEANUP_RE["empty_brackets"].sub("", text)
    text = _CLEANUP_RE["multi_space"].sub(" ", text).strip()
    return text

MODEL          = "llama-3.3-70b-versatile"
MAX_TOKENS     = 1200
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB hard limit

# Recognised image magic-byte signatures (read first 12 bytes)
_IMAGE_SIGNATURES = [
    b'\xff\xd8\xff',       # JPEG
    b'\x89PNG\r\n\x1a\n', # PNG
    b'GIF87a',             # GIF87a
    b'GIF89a',             # GIF89a
    b'RIFF',               # WebP  (bytes 0-3; bytes 8-11 must also be b'WEBP')
]

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Shadow dataset directory (created once at startup) ────────────────────────
SHADOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "static", "uploads", "shadow_dataset")
os.makedirs(SHADOW_DIR, exist_ok=True)

limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
    default_limits=[],
)

@app.errorhandler(429)
def rate_limit_exceeded(e):
    log.warning("rate_limit_exceeded", extra={"data": {"remote_addr": request.remote_addr}})
    return jsonify({
        "success": False,
        "error":   "Too many uploads. Please wait a minute before trying again.",
    }), 429

_hf_client = None
_hf_connect_error = None
_hf_initialized = False

def get_hf_client():
    global _hf_client, _hf_connect_error, _hf_initialized
    if not _hf_initialized:
        log.info("hf_connect_start")
        try:
            hf_token = os.environ.get("HF_TOKEN")
            try:
                _hf_client = Client("inguvaaa/comprehensive", token=hf_token, verbose=False,
                                   httpx_kwargs={"timeout": 30.0})
            except TypeError:
                # Older gradio_client versions don't accept httpx_kwargs — fall back gracefully
                _hf_client = Client("inguvaaa/comprehensive", token=hf_token, verbose=False)
            log.info("hf_connect_ok")
        except Exception as exc:
            _hf_connect_error = str(exc)
            log.error("hf_connect_fail", extra={"data": {"error_message": _hf_connect_error}})
        _hf_initialized = True
    return _hf_client

def get_hf_connect_error():
    global _hf_connect_error
    return _hf_connect_error

# ── Circuit Breaker state ─────────────────────────────────────────────────────
HF_CIRCUIT_OPEN      = False
HF_FAILURE_COUNT     = 0
HF_RECOVERY_TIME     = None   # datetime.time when the circuit may close again
CB_FAILURE_THRESHOLD = 3      # consecutive failures before opening
CB_COOLDOWN_SECS     = 60     # seconds to wait before retrying


def _cb_is_open():
    """Return True when the circuit is open and the cooldown has not yet elapsed."""
    global HF_CIRCUIT_OPEN, HF_FAILURE_COUNT, HF_RECOVERY_TIME
    if not HF_CIRCUIT_OPEN:
        return False
    if time.monotonic() >= HF_RECOVERY_TIME:
        log.info("circuit_breaker_reset")
        HF_CIRCUIT_OPEN  = False
        HF_FAILURE_COUNT = 0
        HF_RECOVERY_TIME = None
        return False
    return True


def _cb_record_success():
    global HF_FAILURE_COUNT
    HF_FAILURE_COUNT = 0


def _cb_record_failure():
    global HF_CIRCUIT_OPEN, HF_FAILURE_COUNT, HF_RECOVERY_TIME
    HF_FAILURE_COUNT += 1
    if HF_FAILURE_COUNT >= CB_FAILURE_THRESHOLD:
        HF_CIRCUIT_OPEN  = True
        HF_RECOVERY_TIME = time.monotonic() + CB_COOLDOWN_SECS
        log.warning("circuit_breaker_open", extra={"data": {
            "failure_count": HF_FAILURE_COUNT,
            "cooldown_secs": CB_COOLDOWN_SECS,
        }})


def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY", ""))


def call_hf_detector(image_bytes):
    client = get_hf_client()
    if client is None:
        return {"error": f"HF client unavailable: {get_hf_connect_error() or 'startup connection failed'}"}

    tmp_path = None
    _t_hf = time.time()
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        log.info("hf_call_start")
        result = client.predict(handle_file(tmp_path), api_name="/predict")
        hf_ms = round((time.time() - _t_hf) * 1000)
        _cb_record_success()
        log.info("hf_call_ok", extra={"data": {"duration_ms": hf_ms, "phase": "hf"}})
        
        # Robustly decode and unwrap result
        parsed_result = result
        if isinstance(result, str):
            try:
                import json
                parsed_result = json.loads(result)
            except Exception:
                pass

        if isinstance(parsed_result, list) and len(parsed_result) > 0:
            parsed_result = parsed_result[0]

        if isinstance(parsed_result, dict):
            # Extract nested data if present
            if "data" in parsed_result:
                inner_data = parsed_result["data"]
                if isinstance(inner_data, (dict, list)):
                    parsed_result = inner_data
                    if isinstance(parsed_result, list) and len(parsed_result) > 0:
                        parsed_result = parsed_result[0]

        # Extract core detection labels or target values directly if wrapped
        if isinstance(parsed_result, dict) and "top_detection" not in parsed_result:
            if "label" in parsed_result:
                parsed_result = {
                    "success": parsed_result.get("success", True),
                    "low_confidence": parsed_result.get("low_confidence", False),
                    "is_low_confidence": parsed_result.get("is_low_confidence", False),
                    "phase": parsed_result.get("phase", 1),
                    "top_detection": parsed_result,
                    "all_detections": [parsed_result]
                }

        if isinstance(parsed_result, dict):
            return parsed_result
        return {"result": str(parsed_result)}
    except Exception as exc:
        log.error("hf_call_error", extra={"data": {
            "error_message": str(exc),
            "duration_ms":   round((time.time() - _t_hf) * 1000),
            "phase":         "hf",
        }}, exc_info=True)
        _cb_record_failure()
        return {"error": str(exc)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _groq_stream_generator(messages, detection, is_low):
    """
    SSE generator for the /detect streaming response.

    Event frames (each separated by a blank line, prefixed with 'data: '):
      {"type": "meta",  "detection": {...}|null, "low_confidence": bool}
          — First frame. Carries the detection-card data so the frontend can
            render the card before any text arrives.
      {"type": "text",  "text": "<chunk>"}
          — One frame per Groq delta token.
      {"type": "done"}
          — Clean end-of-stream signal.
      {"type": "error", "error": "<message>"}
          — Mid-stream Groq error; client renders it inside the active bubble.
    """
    # ── Frame 0: metadata ─────────────────────────────────────────────────────
    meta = {"type": "meta", "detection": detection, "low_confidence": is_low}
    yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

    # ── Frames 1-N: Groq token stream ─────────────────────────────────────────
    _t = time.time()
    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                payload = {"type": "text", "text": delta}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        log.info("groq_stream_done", extra={"data": {
            "duration_ms": round((time.time() - _t) * 1000),
        }})
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as exc:
        log.error("groq_stream_error", extra={"data": {"error_message": str(exc)}},
                  exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"


# ── Data Flywheel — shadow dataset helpers ────────────────────────────────────

def _shadow_save(image_bytes: bytes, label: str, confidence, trigger: str) -> None:
    """
    Write *image_bytes* into SHADOW_DIR for offline active-learning review.

    Filename schema:
        {trigger}_{confidence}_{label_slug}_{YYYYMMDD_HHMMSS}_{uuid6}.jpg

    Examples:
        low_conf_38_whitefly_20260521_143025_a1b2c3.jpg
        phase3_0_non_chilli_20260521_094512_d4e5f6.jpg

    All filesystem errors are caught and logged as warnings so they can
    never interrupt the HTTP response already in flight.
    """
    try:
        ts         = time.strftime("%Y%m%d_%H%M%S")
        slug       = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_") or "unknown"
        conf_str   = str(int(confidence)) if confidence else "0"
        uid        = uuid.uuid4().hex[:6]
        filename   = f"{trigger}_{conf_str}_{slug}_{ts}_{uid}.jpg"
        dest       = os.path.join(SHADOW_DIR, filename)

        with open(dest, "wb") as fh:
            fh.write(image_bytes)

        log.info("shadow_save_ok", extra={"data": {
            "filename":  filename,
            "bytes":     len(image_bytes),
            "trigger":   trigger,
            "label":     label,
            "conf":      confidence,
        }})
    except Exception as exc:
        # Intentionally silent — a disk error must never crash the request
        log.warning("shadow_save_fail", extra={"data": {
            "trigger": trigger,
            "error":   str(exc),
        }})


def _trigger_shadow_save(image_bytes: bytes, label: str,
                         confidence, trigger: str) -> None:
    """Submit _shadow_save to the global thread-pool instead of spawning a new daemon thread."""
    _shadow_executor.submit(_shadow_save, image_bytes, label, confidence, trigger)


SYSTEM_PROMPT = """IMPORTANT: Always respond in English unless the farmer writes in Telugu, Hindi, or Tamil first. Default language is English.

You are ChilliGuru, a friendly farming assistant for chilli farmers in Andhra Pradesh and Telangana. Talk like a trusted friend — simple, warm, easy to understand.

VARIETIES: Teja, Guntur Sannam, LCA 334, Wonder Hot, Pusa Jwala, Byadgi.
SEASONS: Kharif (Jun-Oct), Rabi (Nov-Feb), Zaid (Mar-May).
PESTS: Thrips, Spider Mites, Aphids, Whiteflies, Fruit Borer, Mealybugs, Leaf Miners, Armyworm.
DISEASES: Leaf Curl Virus, Powdery Mildew, Anthracnose, Damping Off, Phytophthora, Bacterial Wilt, Cercospora Leaf Spot, Bacterial Leaf Spot.

SOLUTION FORMAT:
  Solution name (Home-made OR Shop):
  How to make/use it: [simple steps]
  How well it works: X out of 10
  Days to see results: X-X days
  Cost: Rs X to Rs X
  How often: every X days for X weeks
  Where to get: [AP/Telangana]

When performing a plant diagnosis, structure your response strictly with these three headers:
### Climate-Pest Correlation Analysis
[Provide a brief analysis of the weather/climate factors correlated with this pest/disease pressure.
- You must explicitly analyze leaf curling directions and insect grouping patterns to confirm your findings.
- For leaf_curl_virus, you must explicitly identify "Upward/Abaxial Cupping and Vein Thickening" (ఆకు పైకి ముడుచుకోవడం).
- For broad_mites, you must explicitly identify "Downward Inverted-Boat Curling" (ఆకు కిందికి ముడుచుకోవడం).]

### Biological & Organic Interventions
[Provide 2-3 biological or organic solutions using the SOLUTION FORMAT above.
- You must use explicit biological sub-class targeting: recommend specific Bacterial vectors (such as Bacillus thuringiensis), Viral vectors (such as NPV blocks), or Fungal pathogens (such as Beauveria bassiana or others) tailored strictly to the diagnosed pest/disease lifecycle. Every recommendation must include specific biological sub-class vectors (Bt or NPV blocks).
- For viral profiles (such as Leaf Curl Virus), you must mandate organic recommendations targeting the whitefly vector using biological fungal spores (specifically Beauveria bassiana).
- For fungal/bacterial profile detections (Cercospora Leaf Spot, Bacterial Leaf Spot, Powdery Mildew), you must recommend organic treatments like Copper Hydroxide, Pseudomonas fluorescens, or systemic bio-agents.]

### Targeted Chemical Interventions
[Provide a brief overview of targeted chemical/inorganic alternatives. Provide chemical details (e.g., active ingredients) but advise biological/organic alternatives first since you are ChilliGuru.
- For viral profiles (such as Leaf Curl Virus), you must mandate recommendations targeting the whitefly vector using systemic chemical neonicotinoids (specifically Acetamiprid).
- For fungal/bacterial spot/mildew profile detections (Cercospora Leaf Spot, Bacterial Leaf Spot, Powdery Mildew), you must output a clear markdown table contrasting organic choices (specifically Copper Hydroxide or Pseudomonas fluorescens) with targeted chemical choices.]

For every recommended solution/intervention (both biological/organic and chemical), you must feature an explicit "Cost-Effectiveness & Speed Evaluation Table" in markdown format. The table must detail:
- Intervention (name of the solution)
- Estimated Cost per Acre in INR (₹)
- Efficacy Speed (e.g., 'Immediate 24hr knockdown' vs. '5-day systemic spread')
- Environmental Residual Protection (residual window, e.g., '7 days' or '14 days')

End with one prevention tip.
ORGANIC ONLY (except when listing chemical details in Targeted Chemical Interventions). LANGUAGE: reply in the same language the user writes in."""

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "hf_connected": get_hf_client() is not None,
        "groq_ready": bool(os.getenv("GROQ_API_KEY", "")),
    })

@app.route("/api/regional-risk")
def regional_risk():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        lat = 16.5
        lon = 79.5

    temp = 28.0
    humidity = 60.0
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        resp = _http_session.get(url, timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            current = data.get("current", {})
            if "temperature_2m" in current:
                temp = float(current["temperature_2m"])
            if "relative_humidity_2m" in current:
                humidity = float(current["relative_humidity_2m"])
    except Exception as e:
        log.warning("open_meteo_api_error", extra={"data": {"error": str(e)}})

    def calculate_risks(t, h):
        risks = []
        # 1. Invasive Black Thrips
        if 20.0 <= t <= 33.0 and h < 55.0:
            risks.append({
                "pest": "invasive_black_thrips",
                "label": "Invasive Black Thrips",
                "telugu": "నల్ల తామర పురుగు",
                "level": "Critical",
                "description": "Warm and dry conditions are highly optimal for Invasive Black Thrips expansion."
            })
        elif 18.0 <= t <= 35.0 and h < 65.0:
            risks.append({
                "pest": "invasive_black_thrips",
                "label": "Invasive Black Thrips",
                "telugu": "నల్ల తామర పురుగు",
                "level": "High",
                "description": "Favorable conditions for thrips activity. Monitor leaf undersides."
            })
        elif 15.0 <= t <= 38.0:
            risks.append({
                "pest": "invasive_black_thrips",
                "label": "Invasive Black Thrips",
                "telugu": "నల్ల తామర పురుగు",
                "level": "Moderate",
                "description": "Moderate thrips activity. Keep field borders clean."
            })

        # 2. Aphids
        if t < 26.0 and h > 65.0:
            risks.append({
                "pest": "aphids",
                "label": "Aphids",
                "telugu": "పేను పురుగు",
                "level": "High",
                "description": "Cooler temperatures and high humidity promote rapid aphid colonization."
            })
        elif t < 30.0 and h > 50.0:
            risks.append({
                "pest": "aphids",
                "label": "Aphids",
                "telugu": "పేను పురుగు",
                "level": "Moderate",
                "description": "Moderate risk of aphids. Look for ants or honey-dew deposits."
            })

        # 3. Whitefly
        if 26.0 <= t <= 38.0 and h > 75.0:
            risks.append({
                "pest": "whitefly_leaf_damage",
                "label": "Whitefly",
                "telugu": "తెల్ల ఈగ",
                "level": "Critical",
                "description": "Hot and humid microclimate triggers massive whitefly outbreak."
            })
        elif 24.0 <= t <= 40.0 and h > 60.0:
            risks.append({
                "pest": "whitefly_leaf_damage",
                "label": "Whitefly",
                "telugu": "తెల్ల ఈగ",
                "level": "High",
                "description": "High risk of whitefly migration. Yellow sticky traps recommended."
            })
        elif 20.0 <= t <= 42.0:
            risks.append({
                "pest": "whitefly_leaf_damage",
                "label": "Whitefly",
                "telugu": "తెల్ల ఈగ",
                "level": "Moderate",
                "description": "Moderate whitefly presence. Inspect shoots regularly."
            })

        # 4. Broad Mites / Red Mites
        if t > 33.0 and h < 45.0:
            risks.append({
                "pest": "broad_mites",
                "label": "Broad Mites",
                "telugu": "ఎర్ర సాలె పురుగు",
                "level": "Critical",
                "description": "Very hot and dry weather causes rapid broad mite infestation cycles."
            })
        elif t > 30.0 and h < 55.0:
            risks.append({
                "pest": "broad_mites",
                "label": "Broad Mites",
                "telugu": "ఎర్ర సాలె పురుగు",
                "level": "High",
                "description": "High temperature and dry wind favor mite propagation."
            })
        elif t > 25.0:
            risks.append({
                "pest": "broad_mites",
                "label": "Broad Mites",
                "telugu": "ఎర్ర సాలె పురుగు",
                "level": "Moderate",
                "description": "Moderate risk. Overhead irrigation can suppress mite build-up."
            })

        # 5. Fruit Borer
        if 24.0 <= t <= 35.0 and h > 65.0:
            risks.append({
                "pest": "fruit_borer",
                "label": "Fruit Borer",
                "telugu": "పండు తొలిచే పురుగు",
                "level": "High",
                "description": "Warm, humid conditions speed up egg-hatching and fruit borer damage."
            })
        elif 20.0 <= t <= 38.0:
            risks.append({
                "pest": "fruit_borer",
                "label": "Fruit Borer",
                "telugu": "పండు తొలిచే పురుగు",
                "level": "Moderate",
                "description": "Moderate risk of fruit borer. Check for bored entry holes in fruits."
            })

        # 6. Tobacco Caterpillar
        if 25.0 <= t <= 36.0 and h > 70.0:
            risks.append({
                "pest": "tobacco_caterpillar",
                "label": "Tobacco Caterpillar",
                "telugu": "గొంగళి పురుగు",
                "level": "High",
                "description": "High humidity and temperature increase risk of Spodoptera caterpillar activity."
            })
        elif 22.0 <= t <= 38.0:
            risks.append({
                "pest": "tobacco_caterpillar",
                "label": "Tobacco Caterpillar",
                "telugu": "గొంగళి పురుగు",
                "level": "Moderate",
                "description": "Moderate threat. Watch for skeletonized leaf patches."
            })

        # 7. Yellow Thrips
        if 20.0 <= t <= 32.0 and h < 60.0:
            risks.append({
                "pest": "yellow_thrips",
                "label": "Yellow Thrips",
                "telugu": "తామర పురుగు",
                "level": "High",
                "description": "Favorable dry temperature range for yellow thrips feeding on new leaves."
            })
        elif 18.0 <= t <= 36.0:
            risks.append({
                "pest": "yellow_thrips",
                "label": "Yellow Thrips",
                "telugu": "తామర పురుగు",
                "level": "Moderate",
                "description": "Moderate risk. Upward leaf curling might begin."
            })

        # 8. Mealybugs
        if t > 25.0 and h > 60.0:
            risks.append({
                "pest": "mealybugs",
                "label": "Mealybugs",
                "telugu": "పిండి పురుగు",
                "level": "High",
                "description": "Warmth and humidity favor white cottony mealybug cluster formation."
            })
        elif t > 20.0:
            risks.append({
                "pest": "mealybugs",
                "label": "Mealybugs",
                "telugu": "పిండి పురుగు",
                "level": "Moderate",
                "description": "Moderate risk. Prune heavily infested shoots."
            })

        level_order = {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3}
        risks.sort(key=lambda r: level_order.get(r["level"], 4))
        return risks

    locations = []
    locations.append({
        "latitude": lat,
        "longitude": lon,
        "name": "Requested Field Centroid",
        "temperature": temp,
        "humidity": humidity,
        "risks": calculate_risks(temp, humidity)
    })
    locations.append({
        "latitude": lat,
        "longitude": lon + 0.05,
        "name": "East Regional Watchpoint",
        "temperature": round(temp + 2.0, 1),
        "humidity": round(max(5.0, humidity - 15.0), 1),
        "risks": calculate_risks(temp + 2.0, max(5.0, humidity - 15.0))
    })
    locations.append({
        "latitude": lat + 0.05,
        "longitude": lon,
        "name": "North Regional Watchpoint",
        "temperature": round(temp - 2.0, 1),
        "humidity": round(min(99.0, humidity + 15.0), 1),
        "risks": calculate_risks(temp - 2.0, min(99.0, humidity + 15.0))
    })
    locations.append({
        "latitude": lat - 0.04,
        "longitude": lon - 0.04,
        "name": "South-West Regional Station",
        "temperature": round(temp + 1.0, 1),
        "humidity": round(min(99.0, humidity + 5.0), 1),
        "risks": calculate_risks(temp + 1.0, min(99.0, humidity + 5.0))
    })
    return jsonify(locations)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()
        history = data.get("history", [])
        if not message:
            return jsonify({"error": "No message"}), 400

        # Sub-module 1: resolve language (explicit tag fast-path or script scan)
        lang = _resolve_request_language(message)
        # JSON payload 'lang' field is highest-priority override
        payload_lang = data.get("lang", "").strip().lower()
        if payload_lang:
            tag = re.split(r'[-,;]', payload_lang)[0].strip()
            if tag in _SUPPORTED_LANGS:
                lang = tag  # early-return fast-path — no further script scanning needed

        system_content = strip_cross_contamination(SYSTEM_PROMPT, lang)
        system_content += _LANG_INSTRUCTION_MAP.get(lang, "\nIMPORTANT: You must respond in English.")
        system_content = strip_cross_contamination(system_content, lang)
        message_clean  = strip_cross_contamination(message, lang)

        messages = (
            [{"role": "system", "content": system_content}]
            + history
            + [{"role": "user", "content": message_clean}]
        )
        response = get_client().chat.completions.create(
            model=MODEL, messages=messages, max_tokens=MAX_TOKENS, temperature=0.7
        )
        return jsonify({"reply": response.choices[0].message.content.strip()})
    except Exception as e:
        log.error("chat_error", extra={"data": {"error_message": str(e)}}, exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/detect", methods=["POST"])
@limiter.limit("5 per minute")
def detect():
    _t0 = time.time()
    try:
        resp = _detect_inner()
        status_code = resp[1] if isinstance(resp, tuple) else resp.status_code
        log.info("detect_complete", extra={"data": {
            "duration_ms": round((time.time() - _t0) * 1000),
            "status_code": status_code,
        }})
        return resp
    except Exception as e:
        log.error("detect_unhandled_error", extra={"data": {
            "error_message": str(e),
            "duration_ms":   round((time.time() - _t0) * 1000),
        }}, exc_info=True)
        return jsonify({"success": False, "error": "Internal Processing Error"}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Decoupled sub-functions extracted from _detect_inner()
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORTED_LANGS = frozenset(["en", "hi", "te", "kn", "ta"])
_LANG_INSTRUCTION_MAP = {
    "en": "\nIMPORTANT: You must respond in English.",
    "hi": "\nIMPORTANT: You must respond in Hindi (हिंदी).",
    "te": "\nIMPORTANT: You must respond in Telugu (తెలుగు).",
    "kn": "\nIMPORTANT: You must respond in Kannada (ಕನ್ನಡ).",
    "ta": "\nIMPORTANT: You must respond in Tamil (தமிழ்).",
}


def _resolve_request_language(user_msg: str) -> str:
    """
    Sub-module 1: Resolve the target language for this request.

    Priority:
      1. Explicit 'lang' header / form param / query param.
      2. Accept-Language header (first tag only).
      3. Unicode script auto-detection on user_msg.
      4. Default → 'en'.

    Early-return fast-path: if an explicit, valid language tag is supplied
    we skip all downstream script-scanning gates immediately.
    """
    # Priority 1 & 2 — explicit tag sources
    raw = (
        request.headers.get("lang", "").strip().lower()
        or request.headers.get("Accept-Language", "").strip().lower()
        or request.form.get("lang", "").strip().lower()
        or request.args.get("lang", "").strip().lower()
    )
    if raw:
        tag = re.split(r'[-,;]', raw)[0].strip()
        if tag in _SUPPORTED_LANGS:
            # Fast-path: definitive explicit code — no script scanning needed
            return tag

    # Priority 3 — Unicode script heuristic (uses pre-compiled patterns)
    if _SCRIPT_RE["te"].search(user_msg):
        return "te"
    if _SCRIPT_RE["hi"].search(user_msg):
        return "hi"
    if _SCRIPT_RE["kn"].search(user_msg):
        return "kn"
    if _SCRIPT_RE["ta"].search(user_msg):
        return "ta"

    # Priority 4 — default
    return "en"


def _execute_guardrail_check(image_bytes: bytes, result: dict) -> dict | None:
    """
    Sub-module 2: Apply the HF-space guardrail result filters.

    Returns the cleaned result dict if it should proceed to Groq,
    or None to signal that the local cascade must be invoked instead.
    Raises a Flask Response directly for hard 4xx rejections.
    """
    from flask import jsonify

    response_data = None
    if isinstance(result, dict):
        if "error" in result:
            response_data = result["error"]
        elif "label" in result:
            response_data = result["label"]
        elif "top_detection" in result and result["top_detection"]:
            td = result["top_detection"]
            response_data = td.get("label") if isinstance(td, dict) else td
    elif isinstance(result, (list, tuple)) and result:
        response_data = result[0]
    else:
        response_data = result

    label = str(response_data).strip().lower() if response_data is not None else ""

    # Confidence extraction
    conf_val = None
    if isinstance(result, dict):
        conf_val = result.get("confidence")
        if conf_val is None and isinstance(result.get("top_detection"), dict):
            conf_val = result["top_detection"].get("confidence")
    try:
        conf = float(conf_val) if conf_val is not None else 0.0
    except Exception:
        conf = None

    # Hotfix gasket
    if (label == "non_chilli" and conf == 0) or conf is None:
        return None  # force local cascade with bypass_ood=True

    # Hard out-of-domain rejection
    if label == "non_chilli":
        log.warning("out_of_domain_crop")
        raise _GuardrailReject(
            jsonify({
                "error": "Cannot identify crop. Please upload a clear photo of a chilli plant.",
                "request_id": str(uuid.uuid4())
            }), 422
        )

    # Guardrail / phase-3 rejection → push to shadow dataset, fall through to local
    is_guardrail = (
        label == "0"
        or (isinstance(result, dict) and (result.get("success") is False or result.get("phase") == 3))
    )
    if is_guardrail:
        _trigger_shadow_save(image_bytes, label="non_chilli", confidence=0, trigger="phase3")
        log.info("guardrail_rejection_bypass_to_local_cascade")
        return None

    if isinstance(result, dict) and "error" in result:
        log.warning("hf_result_error_fallback", extra={"data": {"error_message": result["error"]}})
        return None

    return result


class _GuardrailReject(Exception):
    """Internal sentinel: carry a Flask response tuple out of _execute_guardrail_check."""
    def __init__(self, response, status):
        self.response = response
        self.status   = status


def _compile_groq_payload(
    result: dict,
    top: dict | None,
    is_low: bool,
    lang: str,
    user_msg: str,
    image_bytes: bytes | None,
) -> str:
    """
    Sub-module 3: Compile the Groq context string from the detection result.

    Handles both the success path (top detection found) and the fallback path
    (no confident detection — triggers questioning mode).
    """
    lang_names = {"en": "English", "hi": "Hindi", "te": "Telugu",
                  "kn": "Kannada", "ta": "Tamil"}
    low_note_lang_hints = {
        "en": "not fully certain",
        "hi": "पूरी तरह से आश्वस्त नहीं",
        "te": "పూర్తిగా ఖచ్చితంగా తెలియదు",
        "kn": "ಖಚಿತವಾಗಿಲ್ಲ",
        "ta": "முழுமையாக உறுதியாக தெரியவில்லை",
    }

    if top and not result.get("error"):
        raw_label_name = top.get("raw_label") or top.get("label", "")
        core_cls = to_core_class(raw_label_name)

        if core_cls:
            english, telugu_name, kind = detector._get_friendly_name(core_cls)
            top["raw_label"] = english
            top["telugu"]    = telugu_name
            top["label"]     = f"{english} [{telugu_name}]" if telugu_name else english
            top["type"]      = kind

        label      = top.get("label", "unknown pest")
        telugu     = top.get("telugu", "")
        confidence = top.get("confidence", 0)
        kind       = top.get("type", "pest")

        # Regional translation lookup
        translated_label = label
        if core_cls and core_cls in REGIONAL_TRANSLATION_MAP:
            translated_label = REGIONAL_TRANSLATION_MAP[core_cls].get(
                lang, REGIONAL_TRANSLATION_MAP[core_cls]["en"]
            )

        translated_label_clean = strip_cross_contamination(translated_label, lang)
        label_clean            = strip_cross_contamination(label, lang)
        telugu_clean           = strip_cross_contamination(telugu, lang)
        user_msg_clean         = strip_cross_contamination(user_msg, lang)

        target_lang_name = lang_names.get(lang, "English")
        low_note_hint    = low_note_lang_hints.get(lang, "not fully certain")

        if is_low and image_bytes:
            _trigger_shadow_save(image_bytes, label=label, confidence=confidence, trigger="low_conf")

        low_note = (
            f"\nNOTE: This is a low-confidence detection. "
            f"Mention to the farmer that you are {low_note_hint} and ask one short clarifying question."
        ) if is_low else ""

        return (
            f"=== CNN DETECTION RESULT ===\n"
            f"Detected: {translated_label_clean}\n"
            f"Type: {kind} | Confidence: {confidence}%\n"
            f"Farmer described: '{user_msg_clean}'\n"
            f"INSTRUCTION: Tell the farmer clearly what this {kind} is in simple words in {target_lang_name} "
            f"(mention the name '{translated_label_clean}' if helpful). "
            f"Provide a dual-structured treatment approach with 'Biological & Organic Interventions' and 'Targeted Chemical Interventions'. Enforce explicit biological sub-class targeting inside Biological sections (recommend specific bacterial vectors like Bacillus thuringiensis, viral vectors like NPV, or fungal pathogens tailored strictly to the diagnosed lifecycle). "
            f"If the diagnosis is a viral profile (Leaf Curl Virus), you must mandate recommendations targeting the whitefly vector using biological fungal spores (specifically Beauveria bassiana) or systemic chemical neonicotinoids (specifically Acetamiprid). "
            f"If the diagnosis is a fungal/bacterial profile (Cercospora Leaf Spot, Bacterial Leaf Spot, Powdery Mildew), you must output clear tables contrasting organic treatments (such as Copper Hydroxide, Pseudomonas fluorescens, or systemic bio-agents) with targeted chemical choices. "
            f"For every suggested solution, you must explicitly render a 'Cost-Effectiveness & Speed Evaluation Table' in markdown showing: Estimated Cost per Acre in INR (₹), Efficacy Speed (e.g., Immediate 24hr knockdown vs 5-day systemic spread), and Environmental Residual Protection windows. End with one prevention tip."
            + low_note
        )
    else:
        # No confident detection — questioning mode
        err = result.get("error", "") if isinstance(result, dict) else ""
        if err:
            log.warning("detector_error", extra={"data": {"error_message": err}})
        user_msg_clean = strip_cross_contamination(user_msg, lang)
        return (
            f"A farmer uploaded a photo of their chilli plant. "
            f"They described: '{user_msg_clean}'. "
            f"The AI detector could not identify the problem with confidence. "
            f"Ask them 2 specific questions about what they can see "
            f"(colour of affected area, location on plant — leaves/stem/fruit/roots, "
            f"any insects visible, holes in fruit, webbing, powder, spots etc) "
            f"then give a diagnosis followed by a dual-structured treatment plan under 'Biological & Organic Interventions' and 'Targeted Chemical Interventions'. Enforce explicit biological sub-class targeting (specific bacterial, viral, or fungal pathogens tailored to the lifecycle). "
            f"If the diagnosis resolves to a viral profile (Leaf Curl Virus), you must mandate recommendations targeting the whitefly vector using biological fungal spores (Beauveria bassiana) or systemic chemical neonicotinoids (Acetamiprid). "
            f"If the diagnosis resolves to a fungal/bacterial profile (Cercospora Leaf Spot, Bacterial Leaf Spot, Powdery Mildew), contrast organic treatments (Copper Hydroxide, Pseudomonas fluorescens, or systemic bio-agents) with targeted chemical choices in clear tables. "
            f"For every suggestion, include a 'Cost-Effectiveness & Speed Evaluation Table' in markdown showing: Estimated Cost per Acre in INR (₹), Efficacy Speed, and Environmental Residual Protection windows. End with one prevention tip."
        )


def _detect_inner():
    user_msg    = request.form.get("message", "").strip()
    history_raw = request.form.get("history", "[]")
    try:
        history = json.loads(history_raw)
    except Exception:
        history = []

    if not user_msg:
        user_msg = "I uploaded a photo of my chilli plant but I am not sure what the problem is."

    # ── Try HF Space detector first ───────────────────────────────────────────
    image_file = request.files.get("image")
    detection    = None
    is_low       = False   # default; overwritten below if image is present
    groq_context = None

    if image_file:
        # ── Size check (seek without loading the full stream) ─────────────────
        image_file.seek(0, 2)
        file_size = image_file.tell()
        image_file.seek(0)
        if file_size > MAX_IMAGE_BYTES:
            log.warning("payload_too_large", extra={"data": {"size_bytes": file_size}})
            return jsonify({
                "error": "Payload too large",
                "request_id": str(uuid.uuid4())
            }), 413

        # ── Magic-byte type verification (first 12 bytes only) ────────────────
        header = image_file.read(12)
        image_file.seek(0)
        is_webp = header[:4] == b'RIFF' and header[8:12] == b'WEBP'
        is_known = is_webp or any(header.startswith(sig) for sig in _IMAGE_SIGNATURES if sig != b'RIFF')
        if not is_known:
            log.warning("invalid_image_magic", extra={"data": {"header_hex": header.hex()}})
            return jsonify({
                "success": False,
                "error":   "Invalid or corrupted image format submitted",
                "phase":   0,
            }), 400

        # ── Full read (validation passed) ─────────────────────────────────────
        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({"error": "Empty file"}), 400

        result = None
        force_bypass_ood = False
        # Try HF space call first (skip if circuit is open)
        if _cb_is_open():
            log.warning("circuit_breaker_skip_hf")
        else:
            try:
                if get_hf_client() is not None:
                    result = call_hf_detector(image_bytes)
                    
                    # ── Guardrail check (delegated to sub-module 2) ──────────
                    try:
                        result = _execute_guardrail_check(image_bytes, result)
                        if result is None:
                            force_bypass_ood = True
                    except _GuardrailReject as gr:
                        return gr.response, gr.status
                else:
                    log.warning("hf_client_unavailable_fallback")
            except _GuardrailReject:
                raise
            except Exception as exc:
                log.error("hf_unreachable_fallback", extra={"data": {"error_message": str(exc)}},
                          exc_info=True)
                result = None

        if result is None:
            log.info("local_cascade_start")
            _t_local = time.time()
            try:
                result = detector.detect_from_memory(image_bytes, bypass_ood=force_bypass_ood)
                log.info("local_cascade_ok", extra={"data": {
                    "duration_ms": round((time.time() - _t_local) * 1000),
                    "phase":       result.get("phase") if isinstance(result, dict) else None,
                }})
                if isinstance(result, dict) and (result.get("success") is False or result.get("phase") == 3):
                    return jsonify(result)
            except Exception as local_exc:
                log.error("local_cascade_error", extra={"data": {
                    "error_message": str(local_exc),
                    "duration_ms":   round((time.time() - _t_local) * 1000),
                }}, exc_info=True)
                result = {"error": str(local_exc)}

        top_label = (result.get("top_detection", {}) or {}).get("label") if isinstance(result, dict) else None
        log.info("detector_result", extra={"data": {
            "phase":       result.get("phase") if isinstance(result, dict) else None,
            "top_label":   top_label,
            "success":     result.get("success") if isinstance(result, dict) else None,
            "low_confidence": result.get("low_confidence") if isinstance(result, dict) else None,
        }})

        top    = result.get("top_detection") if isinstance(result, dict) else None
        is_low = result.get("low_confidence", False) if isinstance(result, dict) else True

        # ── Sub-module 1: Resolve language (with early-return fast-path) ───────
        lang = _resolve_request_language(user_msg)

        # ── Sub-module 3: Compile Groq payload ──────────────────────────────
        if top is not None:
            detection = top
        groq_context = _compile_groq_payload(
            result=result,
            top=top,
            is_low=is_low,
            lang=lang,
            user_msg=user_msg,
            image_bytes=image_bytes if image_file else None,
        )
        if top:
            is_low = result.get("low_confidence", False)
    else:
        # No image — text-only chat-fallback path
        lang = _resolve_request_language(user_msg)
        user_msg_clean = strip_cross_contamination(user_msg, lang)
        groq_context = (
            f"A farmer uploaded a photo of their chilli plant. "
            f"They described: '{user_msg_clean}'. "
            f"Ask them 2 specific questions about what they can see, "
            f"then give a diagnosis and 2-3 organic solutions with metrics."
        )

    # ── Stream Groq response via SSE ──────────────────────────────────────────
    system_content = strip_cross_contamination(SYSTEM_PROMPT, lang)
    system_content += _LANG_INSTRUCTION_MAP.get(lang, "\nIMPORTANT: You must respond in English.")
    system_content = strip_cross_contamination(system_content, lang)

    messages = [{"role": "system", "content": system_content}] + history + [{"role": "user", "content": groq_context}]
    return Response(
        stream_with_context(_groq_stream_generator(messages, detection, is_low)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx/Render proxy buffering
        },
    )

# ── Pre-warm models ───────────────────────────────────────────────────────────
log.info("prewarm_start")
try:
    detector.prewarm_models()
except Exception as e:
    log.warning("prewarm_failed", extra={"data": {"error": str(e)}})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
