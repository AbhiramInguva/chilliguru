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
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, g, has_request_context, make_response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from gradio_client import Client, handle_file
from groq import Groq
import detector

class SchemaValidationError(Exception):
    pass

TRANSLATION_DICTIONARY = {
    4: "Mealybugs [పిండి పురుగు]",
    "4": "Mealybugs [పిండి పురుగు]",
    "Mealybugs": "Mealybugs [పిండి పురుగు]",
    "Mealybug": "Mealybugs [పిండి పురుగు]"
}

# ── Structured JSON logger ────────────────────────────────────────────────────
class StructuredJsonFormatter(logging.Formatter):
    def format(self, record):
        # Format timestamp matching standard asctime format
        if not hasattr(record, 'asctime'):
            record.asctime = self.formatTime(record, self.datefmt)
        timestamp = record.asctime

        # Get request_id
        request_id = "system"
        if has_request_context():
            request_id = g.get("request_id", "system")
        elif hasattr(record, "request_id"):
            request_id = record.request_id

        # Determine status
        status = "error" if record.levelno >= logging.ERROR else "success"

        # Determine event name
        event_name = record.getMessage()

        # Build metrics dict
        metrics = {}
        if has_request_context() and hasattr(g, "performance") and isinstance(g.performance, dict):
            metrics.update(g.performance)

        if hasattr(record, "data") and isinstance(record.data, dict):
            for k, v in record.data.items():
                if k == "metrics" and isinstance(v, dict):
                    metrics.update(v)
                elif k != "event":
                    metrics[k] = v

        payload = {
            "timestamp": timestamp,
            "request_id": request_id,
            "event": event_name,
            "metrics": metrics,
            "status": status
        }

        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)

# Configure logging on the root logger and remove all existing handlers
for _h in logging.root.handlers[:]:
    logging.root.removeHandler(_h)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(StructuredJsonFormatter())
logging.root.addHandler(_handler)
logging.root.setLevel(logging.INFO)

# Force propagation and clean handlers from common loggers
for logger_name in ["werkzeug", "flask", "gunicorn", "gunicorn.error", "gunicorn.access", "chilliguru"]:
    _l = logging.getLogger(logger_name)
    _l.propagate = True
    for _h in _l.handlers[:]:
        _l.removeHandler(_h)

log = logging.getLogger("chilliguru")

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
    t_start = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        log.info("hf_call_start")
        result = client.predict(handle_file(tmp_path), api_name="/predict")
        duration_us = int((time.perf_counter() - t_start) * 1_000_000)
        if has_request_context() and hasattr(g, "performance"):
            g.performance["hf_api_call"] = duration_us
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
        duration_us = int((time.perf_counter() - t_start) * 1_000_000)
        if has_request_context() and hasattr(g, "performance"):
            g.performance["hf_api_call"] = duration_us
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
    t_start = time.perf_counter()
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

        duration_us = int((time.perf_counter() - t_start) * 1_000_000)
        if has_request_context() and hasattr(g, "performance"):
            g.performance["llm_generation_time"] = duration_us

        log.info("groq_stream_done", extra={"data": {
            "duration_ms": round((time.time() - _t) * 1000),
        }})
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as exc:
        duration_us = int((time.perf_counter() - t_start) * 1_000_000)
        if has_request_context() and hasattr(g, "performance"):
            g.performance["llm_generation_time"] = duration_us
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

Always give 2-3 solutions. End with one prevention tip.
ORGANIC ONLY. LANGUAGE: reply in the same language the user writes in."""

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
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": message}]
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
    import uuid
    g.request_id = f"req-{uuid.uuid4().hex[:8]}"
    g.performance = {
        "hf_api_call": 0,
        "local_inference_processing": 0,
        "llm_generation_time": 0
    }
    _t0 = time.time()
    try:
        try:
            resp = _detect_inner()
            log.info("detect_complete", extra={"data": {
                "duration_ms": round((time.time() - _t0) * 1000),
                "status_code": resp.status_code,
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
            request_id = g.get("request_id", "")
            response_obj = make_response(jsonify({"error": "Payload too large", "request_id": request_id}))
            response_obj.status_code = 413
            return response_obj

        # ── Magic-byte type verification (first 12 bytes only) ────────────────
        header = image_file.read(12)
        image_file.seek(0)
        is_webp = header[:4] == b'RIFF' and header[8:12] == b'WEBP'
        is_known = is_webp or any(header.startswith(sig) for sig in _IMAGE_SIGNATURES if sig != b'RIFF')
        if not is_known:
            log.warning("invalid_image_magic", extra={"data": {"header_hex": header.hex()}})
            request_id = g.get("request_id", "")
            response_obj = make_response(jsonify({
                "success": False,
                "error":   "Invalid or corrupted image format submitted",
                "phase":   0,
                "request_id": request_id
            }))
            response_obj.status_code = 415
            return response_obj

        # ── Full read (validation passed) ─────────────────────────────────────
        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({"error": "Empty file"}), 400

        # AI Synthetic Payload Firewall Check (Model 0)
        try:
            synthetic_score = detector.check_synthetic(image_bytes)
            if synthetic_score > 0.85:
                request_id = g.get("request_id", "")
                
                # Standard Structured Warning Log
                log.warning("ai_generation_detected", extra={"data": {"status": "rejected"}})
                
                # Intentional Raw Structured Event Log matching prompt requirements exactly
                import sys
                sys.stdout.write(json.dumps({
                    "event": "ai_generation_detected",
                    "request_id": request_id,
                    "status": "rejected"
                }) + "\n")
                sys.stdout.flush()

                response_obj = make_response(jsonify({
                    "error": "AI generated image matrices are blocked from agronomic diagnostic queues.",
                    "request_id": request_id
                }))
                response_obj.status_code = 400
                return response_obj
        except Exception as filter_exc:
            log.warning("ai_filter_failed", extra={"data": {"error_message": str(filter_exc)}})

        result = None
        force_bypass_ood = False
        # Try HF space call first (skip if circuit is open)
        if _cb_is_open():
            log.warning("circuit_breaker_skip_hf")
        else:
            try:
                if get_hf_client() is not None:
                    result = call_hf_detector(image_bytes)
                    
                    try:
                        if not isinstance(result, dict):
                            raise SchemaValidationError("Response is not a dictionary")
                        
                        if "success" not in result:
                            raise SchemaValidationError("Missing 'success' key in response")
                        if not isinstance(result["success"], bool):
                            raise SchemaValidationError("Invalid 'success' data type, expected boolean")
                        
                        if result["success"]:
                            # Expected layout contract for successful prediction
                            if "top_detection" not in result:
                                raise SchemaValidationError("Missing 'top_detection' key")
                            if "all_detections" not in result:
                                raise SchemaValidationError("Missing 'all_detections' key")
                            
                            top_det = result["top_detection"]
                            if top_det is not None:
                                if not isinstance(top_det, dict):
                                    raise SchemaValidationError("'top_detection' must be a dictionary or None")
                                
                                for key in ["label", "confidence", "type", "raw_label"]:
                                    if key not in top_det:
                                        raise SchemaValidationError(f"Missing '{key}' in 'top_detection'")
                                
                                if not isinstance(top_det["label"], str):
                                    raise SchemaValidationError("'label' in 'top_detection' must be a string")
                                if not isinstance(top_det["type"], str):
                                    raise SchemaValidationError("'type' in 'top_detection' must be a string")
                                if not isinstance(top_det["raw_label"], str):
                                    raise SchemaValidationError("'raw_label' in 'top_detection' must be a string")
                                
                                try:
                                    conf_val = float(top_det["confidence"])
                                except (ValueError, TypeError):
                                    raise SchemaValidationError("Invalid 'confidence' format in 'top_detection'")
                            
                            if not isinstance(result["all_detections"], list):
                                raise SchemaValidationError("'all_detections' must be a list")
                            
                            for idx, det in enumerate(result["all_detections"]):
                                if not isinstance(det, dict):
                                    raise SchemaValidationError(f"Detection at index {idx} in 'all_detections' must be a dictionary")
                                for key in ["label", "confidence", "type", "raw_label"]:
                                    if key not in det:
                                        raise SchemaValidationError(f"Missing '{key}' in detection at index {idx}")
                                if not isinstance(det["label"], str):
                                    raise SchemaValidationError(f"'label' in detection at index {idx} must be a string")
                                if not isinstance(det["type"], str):
                                    raise SchemaValidationError(f"'type' in detection at index {idx} must be a string")
                                if not isinstance(det["raw_label"], str):
                                    raise SchemaValidationError(f"'raw_label' in detection at index {idx} must be a string")
                                try:
                                    float(det["confidence"])
                                except (ValueError, TypeError):
                                    raise SchemaValidationError(f"Invalid 'confidence' in detection at index {idx}")
                            
                            for k in ["low_confidence", "is_low_confidence"]:
                                if k in result and not isinstance(result[k], bool):
                                    raise SchemaValidationError(f"Invalid '{k}' data type, expected boolean")
                                    
                            if top_det is not None:
                                label = top_det["label"].strip().lower()
                                conf = float(top_det["confidence"])
                            else:
                                label = ""
                                conf = 0.0
                        else:
                            # Response returned success=False (an error or guardrail rejection)
                            if "error" not in result or not isinstance(result["error"], str):
                                raise SchemaValidationError("Missing or invalid 'error' string in failed response")
                            label = ""
                            conf = 0.0
                            
                    except SchemaValidationError as sve:
                        log.warning("schema_validation_failed_fallback", extra={"data": {
                            "error_message": str(sve),
                            "raw_response": str(result)
                        }})
                        result = None
                    
                    if result is not None:
                        # Explicit out-of-domain crop rejection check (The Elephant Fix)
                        if label == "non_chilli":
                            log.warning("out_of_domain_crop")
                            request_id = g.get("request_id", "")
                            response_obj = make_response(jsonify({
                                "error": "Cannot identify crop. Please upload a clear photo of a chilli plant.",
                                "request_id": request_id
                            }))
                            response_obj.status_code = 422
                            return response_obj

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
            t_local_start = time.perf_counter()
            try:
                result = detector.detect_from_memory(image_bytes, bypass_ood=force_bypass_ood)
                duration_us = int((time.perf_counter() - t_local_start) * 1_000_000)
                if has_request_context() and hasattr(g, "performance"):
                    g.performance["local_inference_processing"] = duration_us
                
                log.info("local_cascade_ok", extra={"data": {
                    "duration_ms": round(duration_us / 1000.0),
                    "phase":       result.get("phase") if isinstance(result, dict) else None,
                }})
                if isinstance(result, dict) and (result.get("success") is False or result.get("phase") == 3):
                    return jsonify(result)
            except Exception as local_exc:
                duration_us = int((time.perf_counter() - t_local_start) * 1_000_000)
                if has_request_context() and hasattr(g, "performance"):
                    g.performance["local_inference_processing"] = duration_us
                log.error("local_cascade_error", extra={"data": {
                    "error_message": str(local_exc),
                    "duration_ms":   round(duration_us / 1000.0),
                }}, exc_info=True)
                result = {"error": str(local_exc)}

        # Downstream translation mapping for Class 4 / Mealybugs
        if isinstance(result, dict):
            for det in [result.get("top_detection")] + result.get("all_detections", []):
                if isinstance(det, dict):
                    lbl = det.get("label", "")
                    raw_lbl = det.get("raw_label", "")
                    cls_val = det.get("class_id")
                    if lbl in (4, "4", "Mealybugs", "Mealybug") or raw_lbl in (4, "4", "Mealybugs", "Mealybug") or cls_val in (4, "4"):
                        det["label"] = "Mealybugs [పిండి పురుగు]"
                        det["telugu"] = "పిండి పురుగు"
                        det["raw_label"] = "Mealybugs"
                        det["type"] = "pest"

        top_label = (result.get("top_detection", {}) or {}).get("label") if isinstance(result, dict) else None
        log.info("detector_result", extra={"data": {
            "phase":       result.get("phase") if isinstance(result, dict) else None,
            "top_label":   top_label,
            "success":     result.get("success") if isinstance(result, dict) else None,
            "low_confidence": result.get("low_confidence") if isinstance(result, dict) else None,
        }})

        top    = result.get("top_detection") if isinstance(result, dict) else None
        is_low = result.get("low_confidence", False) if isinstance(result, dict) else True

        if top and not result.get("error"):
            # Detection present (confident or low-confidence) — give Groq full context
            label      = top.get("label", "unknown pest")
            telugu     = top.get("telugu", "")
            confidence = top.get("confidence", 0)

            # Enforce an absolute confidence threshold floor (0.80 / 80%)
            try:
                conf_val = float(confidence)
            except (ValueError, TypeError):
                conf_val = 0.0
            
            # Normalize scale to 0.0-1.0 if greater than 1.0 (since confidence could be out of 100.0)
            norm_conf = conf_val / 100.0 if conf_val > 1.0 else conf_val

            if norm_conf < 0.80:
                request_id = g.get("request_id", "")
                top_label = label
                conf = conf_val
                
                # Standard Structured Warning Log
                log.warning("low_confidence_prediction_blocked", extra={"data": {
                    "top_class": top_label,
                    "confidence": conf
                }})
                
                # Intentional Raw Structured Event Log matching prompt requirements exactly
                import sys
                sys.stdout.write(json.dumps({
                    "event": "low_confidence_prediction_blocked",
                    "request_id": request_id,
                    "top_class": top_label,
                    "confidence": conf
                }) + "\n")
                sys.stdout.flush()

                # Return standardized cannot identify card
                response_obj = make_response(jsonify({
                    "status": "error",
                    "message": "Cannot reliably identify image features. Please ensure you are uploading a clear, well-lit photo of a chilli plant or pest anomaly.",
                    "request_id": request_id
                }))
                response_obj.status_code = 422
                return response_obj

            kind       = top.get("type", "pest")
            detection  = top
            # Data flywheel: low-confidence detections → shadow dataset for review
            if is_low:
                _trigger_shadow_save(
                    image_bytes,
                    label=label,
                    confidence=confidence,
                    trigger="low_conf",
                )
            low_note   = (
                "\nNOTE: This is a low-confidence detection. "
                "Mention to the farmer that you are not fully certain and ask one short clarifying question."
            ) if is_low else ""
            groq_context = (
                f"=== CNN DETECTION RESULT ===\n"
                f"Detected: {label}" + (f" [{telugu}]" if telugu else "") + f"\n"
                f"Type: {kind} | Confidence: {confidence}%\n"
                f"Farmer described: '{user_msg}'\n"
                f"INSTRUCTION: Tell the farmer clearly what this {kind} is in simple words "
                f"(mention the Telugu name {telugu} if helpful). "
                f"Give 2-3 organic solutions with metrics. End with one prevention tip."
                + low_note
            )
        else:
            # No detection at all — fall back to questioning
            is_low = True
            err = result.get("error", "") if isinstance(result, dict) else ""
            if err:
                log.warning("detector_error", extra={"data": {"error_message": err}})
            groq_context = (
                f"A farmer uploaded a photo of their chilli plant. "
                f"They described: '{user_msg}'. "
                f"The AI detector could not identify the problem with confidence. "
                f"Ask them 2 specific questions about what they can see "
                f"(colour of affected area, location on plant — leaves/stem/fruit/roots, "
                f"any insects visible, holes in fruit, webbing, powder, spots etc) "
                f"then give a diagnosis and 2-3 organic solutions with metrics."
            )
    else:
        groq_context = (
            f"A farmer uploaded a photo of their chilli plant. "
            f"They described: '{user_msg}'. "
            f"Ask them 2 specific questions about what they can see, "
            f"then give a diagnosis and 2-3 organic solutions with metrics."
        )

    # ── Stream Groq response via SSE ──────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": groq_context}]
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
