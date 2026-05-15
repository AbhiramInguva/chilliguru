import os
import tempfile
import urllib.request
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq

MODEL      = "llama-3.3-70b-versatile"
MAX_TOKENS = 1200

def download_model():
    model_path = 'chilli_pest_model.pt'
    if not os.path.exists(model_path):
        print('Downloading model...')
        url = 'https://drive.usercontent.google.com/download?id=10vNM_1Gd-oOA6NBsKgZPxAe2Xr9k3Ttm&export=download&confirm=t'
        urllib.request.urlretrieve(url, model_path)
        print('Model downloaded!')

download_model()

app = Flask(__name__, static_folder="static")
CORS(app)

def get_client():
    return Groq(api_key=os.getenv('GROQ_API_KEY', ''))

SYSTEM_PROMPT = """You are ChilliGuru, a friendly farming assistant for chilli farmers in Andhra Pradesh and Telangana. Talk like a trusted friend — simple, warm, easy to understand.

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
    return jsonify({"status": "ok", "model_ready": False, "groq_ready": bool(os.getenv('GROQ_API_KEY', ''))})

@app.route("/chat", methods=["POST"])
def chat():
    print(f'DEBUG KEY: {os.getenv("GROQ_API_KEY", "NOT FOUND")[:10]}')
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
        import json
        history = json.loads(history_raw)
    except:
        history = []
    context = f"[Farmer uploaded a photo of their chilli plant. They described: '{user_msg}'. Ask 2 simple questions to diagnose the problem, then give 2-3 organic solutions with metrics.]"
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": context}]
        response = get_client().chat.completions.create(model=MODEL, messages=messages, max_tokens=MAX_TOKENS, temperature=0.7)
        return jsonify({"reply": response.choices[0].message.content.strip(), "detection": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
