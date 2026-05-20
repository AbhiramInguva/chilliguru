import json
import os
import tempfile
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gradio_client import Client, handle_file
from groq import Groq

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1200

app = Flask(__name__, static_folder="static")
CORS(app)

print("Connecting to HF Space...", flush=True)
hf_client = None
hf_connect_error = None
try:
    hf_client = Client("inguvaaa/chilliguru-detector", verbose=False)
    print("HF Space connected.", flush=True)
except Exception as exc:
    hf_connect_error = str(exc)
    print(f"HF Space connection failed at startup: {hf_connect_error}", flush=True)


def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY", ""))


def call_hf_detector(image_bytes):
    if hf_client is None:
        return {"error": f"HF client unavailable: {hf_connect_error or 'startup connection failed'}"}

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        print("Calling HF Space...", flush=True)
        result = hf_client.predict(image=handle_file(tmp_path), api_name="/predict")
        print(f"HF result: {result}", flush=True)
        return result if isinstance(result, dict) else {"result": str(result)}
    except Exception as exc:
        print(f"HF detector exception: {exc}", flush=True)
        traceback.print_exc()
        return {"error": str(exc)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

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
        "hf_connected": hf_client is not None,
        "groq_ready": bool(os.getenv("GROQ_API_KEY", "")),
    })

@app.route("/chat", methods=["POST"])
def chat():
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

@app.route("/detect", methods=["POST"])
def detect():
    try:
        return _detect_inner()
    except Exception as e:
        print(f"Unhandled /detect error: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def _detect_inner():
    user_msg    = request.form.get("message", "").strip()
    history_raw = request.form.get("history", "[]")
    try:
        history = json.loads(history_raw)
    except Exception:
        history = []

    if not user_msg:
        user_msg = "I uploaded a photo of my chilli plant but I am not sure what the problem is."

    # ── Try HF Space detector ─────────────────────────────────────────────────
    image_file = request.files.get("image")
    detection  = None
    groq_context = None

    if image_file:
        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({"error": "Empty file"}), 400

        result      = call_hf_detector(image_bytes)
        print(f"HF detector result: {result}", flush=True)

        top = result.get("top_detection") if isinstance(result, dict) else None

        if top and not result.get("low_confidence") and not result.get("error"):
            # Confident detection — give Groq full context
            label      = top.get("label", "unknown pest")
            telugu     = top.get("telugu", "")
            confidence = top.get("confidence", 0)
            kind       = top.get("type", "pest")
            detection  = top
            groq_context = (
                f"=== CNN DETECTION RESULT ===\n"
                f"Detected: {label}" + (f" [{telugu}]" if telugu else "") + f"\n"
                f"Type: {kind} | Confidence: {confidence}%\n"
                f"Farmer described: '{user_msg}'\n"
                f"INSTRUCTION: Tell the farmer clearly what this {kind} is in simple words "
                f"(mention the Telugu name {telugu} if helpful). "
                f"Give 2-3 organic solutions with metrics. End with one prevention tip."
            )
        else:
            # No confident detection — fall back to questioning
            err = result.get("error", "") if isinstance(result, dict) else ""
            if err:
                print(f"HF detector error: {err}", flush=True)
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

    # ── Ask Groq ──────────────────────────────────────────────────────────────
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": groq_context}]
        response = get_client().chat.completions.create(model=MODEL, messages=messages, max_tokens=MAX_TOKENS, temperature=0.7)
        return jsonify({"reply": response.choices[0].message.content.strip(), "detection": detection})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
