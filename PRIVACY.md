# ChilliGuru — Data Privacy & Retention

ChilliGuru processes farmer-submitted plant photos and approximate location to
diagnose pests/diseases and give regional risk guidance. This document
explains what is collected, why, where it is stored, how long it is kept, and
how to request deletion — required context under India's Digital Personal
Data Protection (DPDP) Act, and linked directly from the app's UI notice.

## What is collected

| Data | When | Why |
|---|---|---|
| Uploaded plant photo | When you use the camera-scan / photo-upload feature | Sent to the detector (remote HF Space or local fallback) to identify the pest/disease |
| Approximate device location (browser geolocation) | When you grant location permission, for the Regional Risk Map or alongside a photo upload | Looks up local weather (temperature/humidity) to estimate regional pest risk |
| Chat messages | When you type a question | Sent to Groq's LLM API to generate farming advice; conversation history is kept only in your browser tab, not stored server-side |
| Low-confidence / rejected detection photos | Automatically, when the detector is unsure or rejects an image as not-a-chilli-plant | Saved for human review and potential future model retraining ("shadow dataset") |
| Outcome-tracking case record (diagnosis label, treatment summary, language, a short case code) | Automatically, after every completed diagnosis | Lets you return later via a saved link (`/case/<case_id>`) to report whether the treatment worked |
| Follow-up outcome (Better/Same/Worse/Not sure) + optional follow-up photo | Only if you open your saved case link and choose to report back | Saved alongside the original diagnosis as labelled training data (see "Outcome tracking" below) and used to offer further help if the treatment isn't working |

**No account, name, phone number, or other direct identity is collected.**
The case code (e.g. `7K2QXJ9`) is an opaque, randomly generated identifier —
it identifies the *case*, not you. It is shown to you once and only usable
by whoever has the saved link; it does not appear in any URL query string,
analytics tool, or third-party service.
Photos and coordinates are not linked to any login or identity — the app has
no user accounts.

## Where it's stored

- Plant photos saved for active learning go to `static/uploads/shadow_dataset/`
  on the server (Render). A human may later curate a subset of these into
  `static/uploads/telemetry_logs/` for retraining (see "Model retraining"
  below) — this is a manual review step, not automatic.
- Chat conversation history lives only in your browser tab's memory
  (JavaScript variable) and is sent with each new message so the assistant
  has context — it is **not** written to a database or file on the server.
- Approximate location (latitude/longitude) is used to query a weather API
  (Open-Meteo) and immediately discarded after computing the risk response —
  it is **never written to disk or to the shadow dataset alongside a photo**.
  See "Geolocation and photos are not linked" below.
- Outcome-tracking case records go to `static/uploads/cases/` (one small
  JSON file per case, written by `case_store.py`'s default
  `JSONFileCaseStore`). **Important:** on Render's free tier this disk is
  ephemeral — case records do not survive a redeploy/restart unless a
  persistent disk is mounted and `CASE_STORE_PATH` points at it. See
  [README.md](README.md#outcome-tracking-farmer-initiated-first-version)
  and `case_store.py`'s module docstring. Outcome follow-up photos are
  saved into the same `static/uploads/shadow_dataset/` directory as other
  active-learning captures, with a `.json` sidecar tagging the original
  diagnosis, treatment, days elapsed, and reported outcome.

## How long it's kept — retention

Stored photos (`static/uploads/shadow_dataset/` and
`static/uploads/telemetry_logs/`) and outcome-tracking case records
(`static/uploads/cases/`) are subject to a retention window controlled by
the `RETENTION_DAYS` environment variable (default: **90 days**, see
[README.md](README.md#environment-variables)). Files older than this
window are deleted by [scripts/purge_old_telemetry.py](scripts/purge_old_telemetry.py),
which can be run:

- Manually: `python3 scripts/purge_old_telemetry.py --execute` (or `make purge-telemetry` for a dry-run preview)
- On a schedule: [.github/workflows/purge_telemetry.yml](.github/workflows/purge_telemetry.yml) runs weekly via GitHub Actions cron, or can be triggered manually from the Actions tab

The script defaults to a **dry-run** (lists what it would delete without
deleting) unless `--execute` is passed, so a misconfigured schedule can never
silently delete data without first having been run in dry-run mode during
development.

## Geolocation and photos are not linked

The detector's shadow-dataset save path
([app.py](app.py)'s `_shadow_save` / [detector.py](detector.py)'s
`_save_to_shadow_dataset`) writes only the image bytes, the predicted label,
a confidence score, and a timestamp into the filename — **it never receives
or writes latitude/longitude**. The `/detect` endpoint accepts optional
`latitude`/`longitude` form fields from the browser for potential future
regional features, but the current server code does not read or persist
them anywhere; they are dropped. This is intentionally verified to remain
true — if a future change ever threads geolocation into the shadow-dataset
save path, it must not be stored in a way that links a precise coordinate to
a specific photo.

The separate Regional Risk Map feature (`/api/regional-risk`) uses coarse
coordinates (rounded browser geolocation, or a fixed Guntur-region default)
purely to query a weather API for a temperature/humidity estimate — the
response is computed and returned immediately; nothing is written to disk.
This endpoint accepts coordinates via a POST JSON body (not a URL query
string) specifically so they don't appear in server access logs or browser
history. A legacy GET `?lat=&lon=` form is still accepted for backward
compatibility with existing automated tests, but the bundled frontend never
uses it.

Our own application logs (`app.py`'s structured JSON logger) never include
latitude/longitude in any log line. The one place coordinates appear in a
URL is the *outbound* call to the third-party Open-Meteo weather API, which
requires them as query parameters per its own API contract — this is a
standard weather-API pattern and uses the same coarse, non-photo-linked
coordinates described above.

## Model retraining on telemetry

`static/uploads/telemetry_logs/` is the curated input for `self_heal.py`'s
automated retraining. Promoting a retrained model to production now requires
passing a held-out validation gate
([scripts/eval/run_validation_gate.py](scripts/eval/run_validation_gate.py))
— see [README.md](README.md#self-heal-model-promotion-gate) for details. This
is a model-quality safeguard, not a privacy control, but it's worth noting:
retraining only ever touches model *weights*; it does not extract or expose
any individual farmer's photo or location to anyone outside the retraining
job itself.

Outcome-tracking follow-up photos (saved to `static/uploads/shadow_dataset/`
with a `.json` metadata sidecar — diagnosis, treatment, days elapsed,
reported outcome) follow the exact same path as any other shadow-dataset
capture: a human must manually curate them into `telemetry_logs/` before
they're ever used in a retraining run. Capturing and tagging an outcome
does **not** trigger retraining by itself.

## How to request deletion

Since the app has no accounts, there's no per-user "my data" page. To request
deletion of specific photos:

1. Open a GitHub issue on this repository (or contact the maintainer
   directly) describing the upload (approximate date/time and what was in
   the photo) so it can be located in `static/uploads/shadow_dataset/` or
   `static/uploads/telemetry_logs/` and removed manually.
2. Alternatively, wait for the retention window (`RETENTION_DAYS`, default 90
   days) — the photo will be automatically purged without any request needed.

If you believe a photo was retained beyond the documented window or used in
a way inconsistent with this document, please raise it the same way.
