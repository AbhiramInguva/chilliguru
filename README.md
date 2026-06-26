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
| `/health` | GET | Cheap, side-effect-free: `{status, hf_initialized, hf_connected, hf_circuit_open, groq_key_configured}` |
| `/health/deep` | GET | Active probe: actually calls the HF Space + Groq, returns `{status, hf:{...}, groq:{...}}` |
| `/api/regional-risk?lat=&lon=` | GET | Weather-driven pest risk levels for 4 nearby points |
| `/chat` | POST | `{message, history, lang}` → `{reply}` (JSON; rate-limited 15/min) |
| `/detect` | POST (multipart) | `image`, `message`, `history`, `lang`, optional MCQ fields (`plant_age`, `observed`, `affected_part`, `curl_dir`) → SSE stream, or `{triage:{...}}` JSON if the adaptive triage flow needs to ask a follow-up question first (rate-limited 5/min) |
| `/detect/triage-answer` | POST (JSON) | One turn of the adaptive triage Q&A loop (`state`, `question_id`, `answer_key`, `history`) → next `{triage:{...}}` question or the resolved SSE stream (rate-limited 20/min) |
| `/case/<case_id>` | GET | Serves the standalone outcome-tracking follow-up page (`static/case.html`) |
| `/api/case/<case_id>` | GET | Returns the saved case record (diagnosis + treatment summary) for the follow-up page |
| `/api/case/<case_id>/outcome` | POST (multipart) | `outcome` (`better`/`same`/`worse`/`not_sure`), optional `image` → records the outcome, tags any follow-up photo into the shadow dataset, and returns an escalation note if `worse`/`same` (rate-limited 10/min) |

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
| `SENTRY_DSN` | No, optional | If set, enables Sentry error tracking (`sentry-sdk[flask]`) for unhandled exceptions; if unset, Sentry is never initialized — no-op |
| `RETENTION_DAYS` | No | Days to keep farmer photos in `static/uploads/shadow_dataset/`, `static/uploads/telemetry_logs/`, and outcome-tracking cases in `static/uploads/cases/` before `scripts/purge_old_telemetry.py` deletes them; defaults to `90`. See [PRIVACY.md](PRIVACY.md) |
| `CASE_STORE_PATH` | No, **production-critical** | Directory the outcome-tracking case store (`case_store.py`) writes to; defaults to `static/uploads/cases/`. **On Render's free tier this default is on ephemeral disk and cases will NOT survive a restart/redeploy.** Point this at a mounted Render persistent disk (or replace `case_store.build_case_store()` with a DB-backed `CaseStore`) before relying on outcome tracking in production. See "Outcome tracking" below |

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
  `web: gunicorn app:app --worker-class gthread --workers 1 --threads 4 --timeout 120`.
  This uses a single process with 4 worker *threads* (not the default sync
  worker) so a slow Groq/HF call doesn't block every other concurrent request —
  important on the free tier's single worker. `render.yaml` pins the identical
  `startCommand` for Blueprint-based deploys. Environment variables
  (`GROQ_API_KEY`, `HF_TOKEN`, `CORS_ALLOWED_ORIGINS`, optional `SENTRY_DSN`)
  are configured in the Render dashboard, not in the repo. The live instance is
  https://chilliguru.onrender.com.

  **Concurrency / threaded worker — dashboard override warning:** if the
  Render service's **Settings → Start Command** field has ever been set
  manually in the dashboard, it **overrides the `Procfile` entirely** — Render
  will keep running whatever command is in that field (e.g. the old
  `--workers 1 --threads 1` sync invocation) regardless of what `Procfile` or
  `render.yaml` say. This is a dashboard setting Claude Code cannot change.
  **A human must open the service in the Render dashboard, check Settings →
  Start Command, and either clear it (so the `Procfile` takes effect) or
  manually update it to match** `gunicorn app:app --worker-class gthread
  --workers 1 --threads 4 --timeout 120`, then trigger a redeploy.
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
triage.py                Vision-guided adaptive triage engine (question selection/re-rank)
case_store.py            Outcome-tracking case storage interface (see "Outcome tracking" above)
chilliguru.py            Standalone terminal CLI (Groq + detector.py, no Flask)
knowledge/chilli_kb.json        Curated agronomy reference (deficiencies/pests/diseases)
knowledge/kb_class_map.json     Maps the 15 vision-model classes to KB entries
knowledge/triage_rules.json     Adaptive-triage question rulebook (sourced from chilli_kb.json)
scripts/validate_kb.py          Validates the KB and class map are complete/consistent
static/index.html, static/js/app.js, static/css/style.css   Served frontend (main chat SPA)
static/case.html, static/js/case.js   Standalone outcome-tracking follow-up page
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

## Self-heal model promotion gate

`self_heal.py` retrains the local detector on curated farmer telemetry
(`static/uploads/telemetry_logs/`) and exports new weights to
`weights/chilli_pest_model.onnx`. Before those weights are ever committed and
deployed, `scripts/eval/run_validation_gate.py` must approve them:

1. **Held-out set** — `scripts/eval/validation_manifest.json` lists
   manually-verified images + labels that are never part of training. It
   ships **empty**; you must populate it with real images before the gate can
   promote anything — until then it fails closed (refuses to promote) rather
   than approving blindly.
2. **Seed a baseline** — once the manifest has real images, run
   `python3 scripts/eval/run_validation_gate.py --seed-baseline` once to
   record the current production model's macro-F1 in `models/registry.json`.
3. **Normal runs** — every subsequent gate run compares a freshly retrained
   candidate's macro-F1 against the registry's active baseline (minus a small
   tolerance). Pass → promoted, backed up to `models/versions/<id>/`, and
   recorded in `models/registry.json`. Fail → the existing model is kept and
   the workflow's deploy step is skipped (see `.github/workflows/self_heal_deploy.yml`).
4. **Rollback** — `python3 scripts/rollback_model.py` (or `make rollback-model`)
   restores the previous registered version. Local-only; commit/push/redeploy
   yourself afterwards if you're satisfied.
5. **Manual approval (optional, human-only step)** — the workflow's deploy job
   uses `environment: production-deploy`. To require a human to click Approve
   before any auto-retrained model goes live, create that environment in this
   repo's **Settings → Environments** and add required reviewers — this
   cannot be configured from the workflow file itself.

## Privacy & disclaimers

The app serves chemical pesticide dosages and stores farmer photos/coarse
location — see [PRIVACY.md](PRIVACY.md) for what's collected, retention
(`RETENTION_DAYS`, default 90 days, purged by `scripts/purge_old_telemetry.py`),
and how to request deletion. The UI shows a persistent notice covering both
the advice disclaimer (confirm chemical dosages with a local KVK/agriculture
officer before use) and the data notice; the same disclaimer is appended
server-side to any `/chat` or `/detect` reply that mentions chemical
treatment.

## Outcome tracking (farmer-initiated, first version)

After a diagnosis (a pest/disease card, or a triage-resolved deficiency —
see "Vision-guided adaptive triage" — high-confidence detections included),
the app creates a small "case" record (`app.py`'s `_create_case`) and shows
the farmer a short code + recovery link (`/case/<case_id>`) they can save.
There are no accounts, reminders, or SMS in this first version — the farmer
has to come back and open the link themselves.

- **Recovery screen** (`static/case.html` + `static/js/case.js`, served at
  `/case/<case_id>`): "How is your crop now?" with four tappable
  icon+text options (Better/Same/Worse/Not sure) and an optional follow-up
  photo upload. Standalone page, no login, works from a saved link days
  later on any device.
- **The flywheel**: when an outcome (and optionally a follow-up photo)
  comes in, the photo is saved into the *same* `static/uploads/shadow_dataset/`
  directory the existing active-learning capture paths already use, but with
  a `.json` sidecar (`app.py`'s `_outcome_shadow_save`) carrying the original
  diagnosis, the treatment given, days elapsed, and the farmer-reported
  outcome — i.e. real labelled diagnosis→treatment→outcome data, not just a
  raw image. This does **not** trigger retraining by itself; a human still
  curates a batch of these into `static/uploads/telemetry_logs/` before
  `self_heal.py` runs, exactly as today.
- **Escalation**: if the farmer reports "Worse" or "Same", the follow-up
  screen shows a short escalation note — the existing
  confirm-with-KVK/agriculture-officer chemical framing for pests/diseases,
  or a soil-test recommendation for deficiencies (reusing language already
  established elsewhere in the app, not new advice).
- **Storage interface** (`case_store.py`): a `CaseStore` abstract interface
  with a `JSONFileCaseStore` default (one small JSON file per case under
  `CASE_STORE_PATH`, default `static/uploads/cases/`) for local dev. **This
  is NOT durable on Render's free tier** — see the `CASE_STORE_PATH` row
  above and the module docstring in `case_store.py`. Case creation/lookup is
  best-effort everywhere it's called: if the store is unavailable for any
  reason, `/detect` and `/detect/triage-answer` still return the full
  diagnosis exactly as before, just without a case_id/save-this-case card —
  the core diagnosis flow never blocks on or fails because of this feature.
