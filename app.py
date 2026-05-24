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
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from gradio_client import Client, handle_file
from groq import Groq
import detector

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

CORE_CLASSES = [
    "aphids",
    "whitefly_leaf_damage",
    "fruit_borer",
    "tobacco_caterpillar",
    "yellow_thrips",
    "broad_mites",
    "invasive_black_thrips",
    "mealybugs"
]

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
        "ta": "ஊடுరుவும் கருப்பு இலைப்பேன்"
    },
    "mealybugs": {
        "en": "Mealybugs",
        "hi": "मीलीबग",
        "te": "पिండి పురుగు",
        "kn": "ಹಿಟ್ಟು ತಿగಣೆ",
        "ta": "மாவுப்பூச்சி"
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
    return None

def strip_cross_contamination(text, target_lang):
    if not isinstance(text, str):
        return text
    
    scripts = {
        "hi": r"[\u0900-\u097f]",
        "te": r"[\u0c00-\u0c7f]",
        "kn": r"[\u0c80-\u0cff]",
        "ta": r"[\u0b80-\u0bff]"
    }
    
    if target_lang == "en":
        text = re.sub(r"\s*\[[^\]]*\]", "", text)
        for lang_code, pattern in scripts.items():
            text = re.sub(pattern, "", text)
    else:
        for lang_code, pattern in scripts.items():
            if lang_code != target_lang:
                text = re.sub(pattern, "", text)
        text = re.sub(r"\s*\[\s*\]", "", text)
    
    text = re.sub(r"\s*\(\s*\)", "", text)
    text = re.sub(r"\s*\[\s*\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
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
    """Launch _shadow_save in a daemon thread so it never blocks the response."""
    t = threading.Thread(
        target=_shadow_save,
        args=(image_bytes, label, confidence, trigger),
        daemon=True,
        name=f"shadow-save-{trigger}",
    )
    t.start()


SYSTEM_PROMPT = """IMPORTANT: Always respond in English unless the farmer writes in Telugu, Hindi, or Tamil first. Default language is English.

You are ChilliGuru, a friendly farming assistant for chilli farmers in Andhra Pradesh and Telangana. Talk like a trusted friend — simple, warm, easy to understand.

VARIETIES: Teja, Guntur Sannam, LCA 334, Wonder Hot, Pusa Jwala, Byadgi.
SEASONS: Kharif (Jun-Oct), Rabi (Nov-Feb), Zaid (Mar-May).
PESTS: Thrips, Spider Mites, Aphids, Whiteflies, Fruit Borer, Mealybugs, Leaf Miners, Armyworm.
DISEASES: Leaf Curl Virus, Powdery Mildew, Anthracnose, Damping Off, Phytophthora, Bacterial Wilt.

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
[Provide a brief analysis of the weather/climate factors correlated with this pest/disease pressure]

### Targeted Organic Regulation
[Provide 2-3 organic solutions using the SOLUTION FORMAT above]

### Targeted Inorganic Regulation
[Provide a brief overview of targeted chemical/inorganic regulations, but advise organic alternatives first since you are ChilliGuru]

End with one prevention tip.
ORGANIC ONLY (except when listing chemical details in Targeted Inorganic Regulation). LANGUAGE: reply in the same language the user writes in."""

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

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()
        history = data.get("history", [])
        if not message:
            return jsonify({"error": "No message"}), 400
        lang = (
            data.get("lang", "").strip().lower()
            or request.headers.get("lang", "").strip().lower()
            or request.headers.get("Accept-Language", "").strip().lower()
        )
        if lang:
            lang = re.split(r'[-,;]', lang)[0].strip()
        if not lang or lang not in ["en", "hi", "te", "kn", "ta"]:
            if re.search(r"[\u0c00-\u0c7f]", message):
                lang = "te"
            elif re.search(r"[\u0900-\u097f]", message):
                lang = "hi"
            elif re.search(r"[\u0c80-\u0cff]", message):
                lang = "kn"
            elif re.search(r"[\u0b80-\u0bff]", message):
                lang = "ta"
            else:
                lang = "en"
        
        system_content = strip_cross_contamination(SYSTEM_PROMPT, lang)
        lang_instruction_map = {
            "en": "\nIMPORTANT: You must respond in English.",
            "hi": "\nIMPORTANT: You must respond in Hindi (हिंदी).",
            "te": "\nIMPORTANT: You must respond in Telugu (తెలుగు).",
            "kn": "\nIMPORTANT: You must respond in Kannada (ಕನ್ನಡ).",
            "ta": "\nIMPORTANT: You must respond in Tamil (தமிழ்)."
        }
        system_content += lang_instruction_map.get(lang, "\nIMPORTANT: You must respond in English.")
        system_content = strip_cross_contamination(system_content, lang)
        
        message_clean = strip_cross_contamination(message, lang)
        try:
            messages = [{"role": "system", "content": system_content}] + history + [{"role": "user", "content": message_clean}]
            response = get_client().chat.completions.create(model=MODEL, messages=messages, max_tokens=MAX_TOKENS, temperature=0.7)
            return jsonify({"reply": response.choices[0].message.content.strip()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    finally:
        import gc
        gc.collect()

@app.route("/detect", methods=["POST"])
@limiter.limit("5 per minute")
def detect():
    _t0 = time.time()
    try:
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
    finally:
        import gc
        gc.collect()

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
                    
                    # 1. Gatekeeper strip wrappers
                    response_data = None
                    if isinstance(result, dict):
                        if "error" in result:
                            response_data = result["error"]
                        elif "label" in result:
                            response_data = result["label"]
                        elif "top_detection" in result and result["top_detection"]:
                            if isinstance(result["top_detection"], dict):
                                response_data = result["top_detection"].get("label")
                            else:
                                response_data = result["top_detection"]
                    elif isinstance(result, (list, tuple)) and len(result) > 0:
                        response_data = result[0]
                    else:
                        response_data = result

                    label = str(response_data).strip().lower() if response_data is not None else ""
                    
                    # Check confidence score returned by the guardrail
                    conf_val = None
                    if isinstance(result, dict):
                        if "confidence" in result:
                            conf_val = result["confidence"]
                        elif "top_detection" in result and isinstance(result["top_detection"], dict):
                            conf_val = result["top_detection"].get("confidence")

                    try:
                        conf = float(conf_val) if conf_val is not None else 0.0
                    except Exception:
                        conf = None

                    # Hotfix Gasket to handle the new endpoint structure
                    is_hotfix_triggered = False
                    if label == "non_chilli" and conf == 0:
                        is_hotfix_triggered = True
                    elif conf is None:
                        is_hotfix_triggered = True

                    if is_hotfix_triggered:
                        # Hotfix Gasket to handle the new endpoint structure
                        if label == "non_chilli" and conf == 0:
                            label = "chilli"
                            conf = 1.0
                            is_low_confidence = False
                        
                        # Set to run local cascade with OOD check bypassed
                        force_bypass_ood = True
                        result = None
                    
                    if result is not None:
                        # Explicit out-of-domain crop rejection check
                        if label == "non_chilli":
                            log.warning("out_of_domain_crop")
                            return jsonify({
                                "error": "Cannot identify crop. Please upload a clear photo of a chilli plant.",
                                "request_id": str(uuid.uuid4())
                            }), 422

                        is_guardrail_rejection = False
                        if label == "0":
                            is_guardrail_rejection = True
                        if isinstance(result, dict) and (result.get("success") is False or result.get("phase") == 3):
                            is_guardrail_rejection = True

                        if is_guardrail_rejection:
                            # Data flywheel: non-chilli guardrail image → shadow dataset
                            _trigger_shadow_save(
                                image_bytes,
                                label="non_chilli",
                                confidence=0,
                                trigger="phase3",
                            )
                            # Nuclear bypass: set result = None to force processing by local Model 2 and Model 3
                            log.info("guardrail_rejection_bypass_to_local_cascade")
                            result = None

                        if result and "error" in result:
                            log.warning("hf_result_error_fallback", extra={"data": {
                                "error_message": result["error"],
                            }})
                            result = None
                else:
                    log.warning("hf_client_unavailable_fallback")
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

        # Extract language code and fallback to auto-detection
        lang = (
            request.headers.get("lang", "").strip().lower()
            or request.headers.get("Accept-Language", "").strip().lower()
            or request.form.get("lang", "").strip().lower()
            or request.args.get("lang", "").strip().lower()
        )
        if lang:
            lang = re.split(r'[-,;]', lang)[0].strip()
        
        # Helper to auto-detect if not provided or invalid
        if not lang or lang not in ["en", "hi", "te", "kn", "ta"]:
            if re.search(r"[\u0c00-\u0c7f]", user_msg):
                lang = "te"
            elif re.search(r"[\u0900-\u097f]", user_msg):
                lang = "hi"
            elif re.search(r"[\u0c80-\u0cff]", user_msg):
                lang = "kn"
            elif re.search(r"[\u0b80-\u0bff]", user_msg):
                lang = "ta"
            else:
                lang = "en"

        if top and not result.get("error"):
            # Map raw label or display label to core class
            raw_label_name = top.get("raw_label") or top.get("label", "")
            core_cls = to_core_class(raw_label_name)
            
            if core_cls:
                english, telugu_name, kind = detector._get_friendly_name(core_cls)
                top["raw_label"] = english
                top["telugu"] = telugu_name
                top["label"] = f"{english} [{telugu_name}]" if telugu_name else english
                top["type"] = kind

            label      = top.get("label", "unknown pest")
            telugu     = top.get("telugu", "")
            confidence = top.get("confidence", 0)
            kind       = top.get("type", "pest")
            detection  = top
            
            # Regional translation lookup
            translated_label = label
            if core_cls and core_cls in REGIONAL_TRANSLATION_MAP:
                translated_label = REGIONAL_TRANSLATION_MAP[core_cls].get(lang, REGIONAL_TRANSLATION_MAP[core_cls]["en"])
            
            # Clean cross-contamination scripts from labels
            translated_label_clean = strip_cross_contamination(translated_label, lang)
            label_clean = strip_cross_contamination(label, lang)
            telugu_clean = strip_cross_contamination(telugu, lang)
            user_msg_clean = strip_cross_contamination(user_msg, lang)
            
            # Adjust the prompt language instruction and details according to lang
            lang_names = {
                "en": "English",
                "hi": "Hindi",
                "te": "Telugu",
                "kn": "Kannada",
                "ta": "Tamil"
            }
            target_lang_name = lang_names.get(lang, "English")
            
            # Low confidence note language translation hints
            low_note_lang_hints = {
                "en": "not fully certain",
                "hi": "पूरी तरह से आश्वस्त नहीं",
                "te": "పూర్తిగా ఖచ్చితంగా తెలియదు",
                "kn": "ಖಚಿತವಾಗಿಲ್ಲ",
                "ta": "முழுமையாக உறுதியாக தெரியவில்லை"
            }
            low_note_hint = low_note_lang_hints.get(lang, "not fully certain")
            
            # Data flywheel: low-confidence detections → shadow dataset for review
            if is_low:
                _trigger_shadow_save(
                    image_bytes,
                    label=label,
                    confidence=confidence,
                    trigger="low_conf",
                )
            low_note   = (
                f"\nNOTE: This is a low-confidence detection. "
                f"Mention to the farmer that you are {low_note_hint} and ask one short clarifying question."
            ) if is_low else ""
            
            groq_context = (
                f"=== CNN DETECTION RESULT ===\n"
                f"Detected: {translated_label_clean}\n"
                f"Type: {kind} | Confidence: {confidence}%\n"
                f"Farmer described: '{user_msg_clean}'\n"
                f"INSTRUCTION: Tell the farmer clearly what this {kind} is in simple words in {target_lang_name} "
                f"(mention the name '{translated_label_clean}' if helpful). "
                f"Give 2-3 organic solutions with metrics. End with one prevention tip."
                + low_note
            )
        else:
            # No detection at all — fall back to questioning
            is_low = True
            err = result.get("error", "") if isinstance(result, dict) else ""
            if err:
                log.warning("detector_error", extra={"data": {"error_message": err}})
            
            user_msg_clean = strip_cross_contamination(user_msg, lang)
            groq_context = (
                f"A farmer uploaded a photo of their chilli plant. "
                f"They described: '{user_msg_clean}'. "
                f"The AI detector could not identify the problem with confidence. "
                f"Ask them 2 specific questions about what they can see "
                f"(colour of affected area, location on plant — leaves/stem/fruit/roots, "
                f"any insects visible, holes in fruit, webbing, powder, spots etc) "
                f"then give a diagnosis and 2-3 organic solutions with metrics."
            )
    else:
        # Determine lang for chat/fallback path
        lang = (
            request.headers.get("lang", "").strip().lower()
            or request.headers.get("Accept-Language", "").strip().lower()
            or request.form.get("lang", "").strip().lower()
            or request.args.get("lang", "").strip().lower()
        )
        if lang:
            lang = re.split(r'[-,;]', lang)[0].strip()
        if not lang or lang not in ["en", "hi", "te", "kn", "ta"]:
            if re.search(r"[\u0c00-\u0c7f]", user_msg):
                lang = "te"
            elif re.search(r"[\u0900-\u097f]", user_msg):
                lang = "hi"
            elif re.search(r"[\u0c80-\u0cff]", user_msg):
                lang = "kn"
            elif re.search(r"[\u0b80-\u0bff]", user_msg):
                lang = "ta"
            else:
                lang = "en"
        
        user_msg_clean = strip_cross_contamination(user_msg, lang)
        groq_context = (
            f"A farmer uploaded a photo of their chilli plant. "
            f"They described: '{user_msg_clean}'. "
            f"Ask them 2 specific questions about what they can see, "
            f"then give a diagnosis and 2-3 organic solutions with metrics."
        )

    # ── Stream Groq response via SSE ──────────────────────────────────────────
    system_content = strip_cross_contamination(SYSTEM_PROMPT, lang)
    lang_instruction_map = {
        "en": "\nIMPORTANT: You must respond in English.",
        "hi": "\nIMPORTANT: You must respond in Hindi (हिंदी).",
        "te": "\nIMPORTANT: You must respond in Telugu (తెలుగు).",
        "kn": "\nIMPORTANT: You must respond in Kannada (ಕನ್ನಡ).",
        "ta": "\nIMPORTANT: You must respond in Tamil (தமிழ்)."
    }
    system_content += lang_instruction_map.get(lang, "\nIMPORTANT: You must respond in English.")
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
