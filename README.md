# ChilliGuru 🌶️

An organic-first farming assistant for chilli (pepper) growers in Andhra Pradesh and
Telangana. Upload a photo of a plant and ChilliGuru identifies the pest or disease,
then chats back with home-made and shop-bought organic remedies (with a chemical
fallback section) in English, Telugu, Hindi, Kannada, or Tamil.

## What it does

- **Photo diagnosis** — upload a leaf/fruit photo, get a pest/disease detection with
  a confidence score, then a streamed, farmer-friendly explanation and treatment plan.
- **Text chat** — ask farming questions directly (varieties, seasons, fertilizer,
  planting schedules) without a photo.
- **Regional pest-risk map** — a Leaflet map showing live temperature/humidity-driven
  risk levels for the 8 core pests around the farmer's location.
- **Multi-language** — auto-detects the farmer's language from script or browser
  locale; replies in English, Telugu (తెలుగు), Hindi (हिंदी), Kannada (ಕನ್ನಡ), or
  Tamil (தமிழ்).
- **Organic-first** — every response leads with home-made/organic solutions; chemical
  options are only ever shown in a clearly separated "Targeted Chemical
  Interventions" section.

## Architecture

```
Browser (static/index.html + static/js/app.js)
        │
        ▼
Flask app (app.py, served by gunicorn)
        │
        ├── POST /detect  (image upload)
        │     │
        │     ├── 1. PRIMARY: Hugging Face Space "inguvaaa/comprehensive"
        │     │      (chilli_pest_v2.pt, YOLOv8m, 15 classes — see model_info.json)
        │     │      called via gradio_client, guarded by a circuit breaker
        │     │      (opens after 3 consecutive failures, 60s cooldown)
        │     │
        │     └── 2. FALLBACK: local 3-phase ONNX cascade (detector.py)
        │            used when the HF Space is unreachable/circuit-open, OR
        │            when the HF result fails the out-of-domain guardrail.
        │            Phase 1 — weights/chilli_pest_model.onnx (18→8 core classes)
        │            Phase 2 — weights/ip102_model.onnx (IP102, 8 generic classes)
        │            Phase 3 — weights/yolov8n.onnx (COCO, non-crop rejection)
        │            This is a DIFFERENT model family from the HF Space model —
        │            see "Model weights" below.
        │
        ├── POST /chat  (text-only)
        │     └── Groq LLM (llama-3.3-70b-versatile), system prompt grounded in
        │         the curated agronomy reference at knowledge/chilli_kb.json
        │         (see "Agronomy knowledge base" below)
        │
        └── GET /api/regional-risk
              └── Open-Meteo current weather + a rule table mapping
                  temperature/humidity bands to pest risk levels
```

Both `/chat` and `/detect` build their Groq system prompt/context server-side in
`app.py` (`SYSTEM_PROMPT`, `_compile_groq_payload`) — no prompt, API key, or
knowledge-base content is ever sent to or stored in the browser. The CLI
(`chilliguru.py`) is a separate, terminal-only client that talks to the same Groq
account and the same local `detector.py` cascade directly (no Flask/HF Space).

### Agronomy knowledge base

`knowledge/chilli_kb.json` is a curated reference (nutrient deficiencies, pests,
diseases) transcribed from an expert agronomy document. `knowledge/kb_class_map.json`
links the 15 vision-model classes in `model_info.json` to their KB entry. When a
detection maps to a KB entry, `app.py` injects that entry's distinguishing symptoms
and organic-first management into the Groq context so advice is grounded in the
reference rather than free-generated. Nutrient deficiencies are advice-layer only —
the vision model cannot detect them — so `/chat` also carries a lightweight
symptom→deficiency summary (iron vs. magnesium vs. manganese vs. zinc, etc.) for
text-only questions. Run `python3 scripts/validate_kb.py` after editing the KB.

### Model weights — what's actually local vs. remote

| | Primary (remote) | Fallback (local) |
|---|---|---|
| Where it runs | Hugging Face Space `inguvaaa/comprehensive` | This repo, in-process |
| Code | `hf_space_app.py` (deployed to the Space) | `detector.py` |
| Weights | `chilli_pest_v2.pt` (YOLOv8m, **15** classes) | `weights/*.onnx` (3 separate ONNX models) |
| Metadata | `model_info.json` — read by `hf_space_app.py` only | class maps/thresholds defined inline in `detector.py` |

`model_info.json`'s `model_file: chilli_pest_v2.pt` is **not** a local path the Flask
app reads — there is no `.pt` file in this repo (model weights are uploaded to the HF
Space via `upload_to_hf.py`/`deploy.sh`, never committed — see `.gitignore`). The
local fallback cascade in `detector.py` loads the three ONNX files under `weights/`
directly via `onnxruntime`, with its own, different class taxonomy (8 core pest
classes consolidated from an 18-class Phase-1 model, plus IP102 Phase-2 and a
YOLOv8n Phase-3 out-of-domain check). Both detectors' raw labels are normalized to
the same 8 core pest classes (plus 4 disease classes) in `app.py`'s `to_core_class()`
before being shown to the farmer or matched against the knowledge base.

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serves `static/index.html` |
| `/health` | GET | `{status, hf_connected, groq_ready}` — liveness/readiness check |
| `/api/regional-risk?lat=&lon=` | GET | Weather-driven pest risk levels for 4 nearby points |
| `/chat` | POST | `{message, history, lang}` → `{reply}` (JSON; rate-limited 15/min) |
| `/detect` | POST (multipart) | `image`, `message`, `history`, `lang`, optional MCQ fields (`plant_age`, `observed`, `affected_part`, `curl_dir`) → Server-Sent Events stream (rate-limited 5/min) |

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key used server-side by `/chat`, `/detect`, and the CLI |
| `HF_TOKEN` | Recommended | Auth token for the Hugging Face Space client (primary detector) |
| `CORS_ALLOWED_ORIGINS` | No | Comma-separated allowed origins; defaults to `localhost:8080` for local dev |
| `DATASET_PATH` | No | Overrides the training dataset directory (defaults to `./dataset`); used by `train.py`/`self_heal.py` only, not by the served app |
| `PORT` | No | Port for `app.run()` when running without gunicorn (defaults to `8080`) |
| `ROBOFLOW_API_KEY` | Training only | Used by `train.py` to pull the IP102/PlantVillage dataset |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Deploy only | Used by `deploy.sh`/`upload_to_hf.py` to push weights to the HF Space |
| `KAGGLE_KERNEL` | Deploy only, optional | `owner/slug` to auto-download trained weights via the `kaggle` CLI in `deploy.sh` |

**On Render, these are set in the dashboard's Environment tab — never commit a
`.env` file.** Locally, `chilliguru.py` and `app.py` (via `python-dotenv`/your shell)
read a `.env` file that is gitignored (`.gitignore` excludes `.env`, `.env.*`).

## Local run

```bash
# Quick start — installs deps, runs the HF connectivity smoke test, starts gunicorn
./run_local.sh
# → http://localhost:5000

# Or via Makefile:
make test        # python3 test_local.py — local unit/structure tests
make test-live    # python3 test_live.py — live endpoint smoke test
make status       # git status + check for stray *.pt files
```

`run_local.sh` activates `venv/` or `.venv/` if present, installs
`requirements.txt`, runs `test_hf_connection.py`, then starts
`gunicorn app:app --timeout 120 --workers 1 --bind 0.0.0.0:5000`.

## Deployment

- **Render** auto-deploys the Flask app from the `Procfile`:
  `web: gunicorn app:app --timeout 120 --workers 1 --threads 1`. Environment
  variables (`GROQ_API_KEY`, `HF_TOKEN`, `CORS_ALLOWED_ORIGINS`) are configured in
  the Render dashboard, not in the repo. The live instance is
  https://chilliguru.onrender.com.
- **Hugging Face Space** (the primary detector model) is deployed separately via
  `deploy.sh`, which runs local tests, uploads `chilli_pest_v2.pt` + `model_info.json`
  + `hf_space_app.py` + `hf_space_requirements.txt` to the Space with
  `upload_to_hf.py`, waits for the Space to rebuild, then runs `test_live.py`
  against the live endpoint. Equivalent to `make upload` without the
  test/wait/smoke-test wrapper.
- The local ONNX fallback weights in `weights/` are committed directly to this repo
  (they're small) and need no separate deploy step — Render picks them up on the
  normal git-based deploy.

## Repo layout (selected)

```
app.py                  Flask backend (routes, Groq orchestration, circuit breaker)
detector.py              Local 3-phase ONNX fallback cascade
chilliguru.py            Standalone terminal CLI (Groq + detector.py, no Flask)
knowledge/chilli_kb.json        Curated agronomy reference (deficiencies/pests/diseases)
knowledge/kb_class_map.json     Maps the 15 vision-model classes to KB entries
scripts/validate_kb.py          Validates the KB and class map are complete/consistent
static/index.html, static/js/app.js, static/css/style.css   Served frontend
index.html               Standalone frontend variant (also fixed to call /chat,
                          not Groq directly — see Security note below)
weights/*.onnx           Local fallback model weights (committed)
model_info.json          Metadata for the REMOTE HF Space model only (see above)
pan_india_pests.yaml     YOLO dataset config for training (train.py/self_heal.py)
train.py, self_heal.py   Offline training / self-calibration scripts
deploy.sh, upload_to_hf.py, Makefile, Procfile, run_local.sh   Ops scripts
```

## Security note

All LLM calls (web frontend and CLI) go through the backend — `GROQ_API_KEY` is
read server-side via `os.getenv` and never reaches the browser. The web frontend
calls the Flask `/chat`/`/detect` routes with a relative path so it works
identically on Render and locally without hardcoding a domain.
