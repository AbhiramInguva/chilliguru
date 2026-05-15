import os
import json
import tempfile
import urllib.request
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq

MODEL      = "llama-3.3-70b-versatile"
MAX_TOKENS = 1200
MODEL_PATH = "chilli_pest_model.pt"

# ── Model download ────────────────────────────────────────────────────────────
def download_model():
    if os.path.exists(MODEL_PATH):
        return True
    print("Downloading chilli_pest_model.pt from Google Drive...", flush=True)
    try:
        url = "https://drive.usercontent.google.com/download?id=10vNM_1Gd-oOA6NBsKgZPxAe2Xr9k3Ttm&export=download&confirm=t"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Model downloaded successfully.", flush=True)
        return True
    except Exception as e:
        print(f"Model download failed: {e} — running without pest detector.", flush=True)
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False

MODEL_READY = download_model()

# ── Conditionally load detector ───────────────────────────────────────────────
detector = None
if MODEL_READY:
    try:
        import detector as _detector
        detector = _detector
    except Exception as e:
        print(f"detector import failed: {e}", flush=True)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
CORS(app)

def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY", ""))

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

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/health")
def health():
    return jsonify({
        "status":      "ok",
        "model_ready": MODEL_READY and detector is not None,
        "groq_ready":  bool(os.getenv("GROQ_API_KEY", "")),
    })

@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json()
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
    user_msg    = request.form.get("message", "").strip()
    history_raw = request.form.get("history", "[]")
    try:
        history = json.loads(history_raw)
    except Exception:
        history = []

    # Default English message when farmer sends photo with no description
    if not user_msg:
        user_msg = "Please analyse this chilli plant photo and tell me what pest or disease you see, and give organic solutions."

    # ── Try CNN detector first ────────────────────────────────────────────────
    image = request.files.get("image")
    detection_context = None

    if detector and image:
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                result = detector.detect(tmp.name)
            os.unlink(tmp.name)
            if result.get("success"):
                detection_context = detector.format_for_openai(result, user_msg)
            else:
                detection_context = None
        except Exception as e:
            print(f"Detection error: {e}", flush=True)
            detection_context = None

    # ── Build Groq message ────────────────────────────────────────────────────
    english_instruction = "IMPORTANT: Respond in English unless the farmer's message is in Telugu/Hindi/Tamil."
    if detection_context:
        full_msg = f"{english_instruction}\n{detection_context}\n\nFarmer says: {user_msg}"
    else:
        full_msg = f"{english_instruction}\n[Farmer uploaded a chilli plant photo. They described: '{user_msg}']\nAsk 2 simple questions to diagnose the problem, then give 2-3 organic solutions with metrics."

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": full_msg}]
        response = get_client().chat.completions.create(model=MODEL, messages=messages, max_tokens=MAX_TOKENS, temperature=0.7)
        return jsonify({
            "reply":     response.choices[0].message.content.strip(),
            "detection": result.get("top_detection") if detection_context else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
