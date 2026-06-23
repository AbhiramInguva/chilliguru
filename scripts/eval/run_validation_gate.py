"""
run_validation_gate.py — self-heal model-promotion gate.

Run after self_heal.py has retrained and exported weights/chilli_pest_model.onnx
(the candidate) but BEFORE that candidate is committed/pushed/deployed. Decides
whether the candidate is allowed to replace the current production model.

FAILS CLOSED on every ambiguous case — it only ever promotes when it has
positive evidence the candidate is at least as good as production:
  - scripts/eval/validation_manifest.json missing, unreadable, or "images": []
    -> refuse (exit 1). This is the most important guarantee in this script:
       a missing/empty held-out set must never be treated as "no problem found".
  - models/registry.json missing or has no active baseline metric yet
    -> refuse (exit 1) and explain how to seed one (--seed-baseline).
  - candidate macro-F1 < baseline macro-F1 - PROMOTION_TOLERANCE
    -> refuse (exit 1), candidate weights are left in place on disk but NOT
       added to the registry, so the calling workflow's subsequent commit
       step (gated on this script's exit code) never runs.
  - candidate macro-F1 >= baseline - PROMOTION_TOLERANCE
    -> promote: back up the candidate into models/versions/<version_id>/,
       append a new active registry entry, flip the previous entry inactive.

Usage:
    python3 scripts/eval/run_validation_gate.py
    python3 scripts/eval/run_validation_gate.py --seed-baseline   # one-time bootstrap, see below

--seed-baseline is for the human operator only (NOT used by the automated
self-heal workflow): run it once, after scripts/eval/validation_manifest.json
has been populated with real held-out images, to record the CURRENT
production weights/chilli_pest_model.onnx's metric as the v0 baseline in
models/registry.json. Until that has been done, the gate has nothing to
compare a candidate against and will keep refusing to promote.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH  = ROOT / "scripts" / "eval" / "validation_manifest.json"
REGISTRY_PATH  = ROOT / "models" / "registry.json"
VERSIONS_DIR   = ROOT / "models" / "versions"
CANDIDATE_PATH = ROOT / "weights" / "chilli_pest_model.onnx"
TELEMETRY_DIR  = ROOT / "static" / "uploads" / "telemetry_logs"

PROMOTION_TOLERANCE = 0.02  # candidate may score up to this much below baseline and still promote


def fail(message: str) -> "NoReturn":
    print(f"\n[GATE] PROMOTION BLOCKED: {message}\n", file=sys.stderr)
    sys.exit(1)


def load_manifest():
    if not MANIFEST_PATH.exists():
        fail(f"validation manifest not found at {MANIFEST_PATH} -- refusing to promote on a missing held-out set.")
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"validation manifest at {MANIFEST_PATH} is not valid JSON ({exc}) -- refusing to promote.")
        return  # unreachable, keeps linters happy
    images = manifest.get("images") or []
    if not images:
        fail(
            f"{MANIFEST_PATH} has no images (the held-out validation set is empty). "
            f"Populate it with real, manually-verified examples before this gate can promote anything."
        )
    valid_labels = set(manifest.get("valid_labels") or [])
    resolved = []
    for entry in images:
        img_path = ROOT / entry["path"]
        if not img_path.exists():
            fail(f"validation image listed in manifest does not exist on disk: {img_path}")
        if valid_labels and entry["label"] not in valid_labels:
            fail(f"validation entry {entry['path']!r} has label {entry['label']!r} not in valid_labels")
        resolved.append({"path": img_path, "label": entry["label"]})
    return resolved


def load_registry():
    if not REGISTRY_PATH.exists():
        return {"versions": []}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"models/registry.json is not valid JSON ({exc}). Fix or remove it before re-running.")


def save_registry(registry):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def active_entry(registry):
    for entry in registry.get("versions", []):
        if entry.get("active"):
            return entry
    return None


def _normalize_label(raw: str) -> str:
    return (raw or "").strip().lower().replace(" ", "_").replace("-", "_")


def predict_label(image_path: Path):
    """
    Runs the candidate model (weights/chilli_pest_model.onnx, already on disk
    after self_heal.py's export step) on a single held-out image via
    detector.py's real inference path -- the gate must score exactly what
    production would run, not a reimplementation of it.
    """
    import detector  # imported lazily so --help works without onnxruntime installed
    result = detector.detect(str(image_path), bypass_ood=True)
    if not isinstance(result, dict):
        return "non_chilli"
    top = result.get("top_detection") or {}
    raw = top.get("raw_label") or top.get("label") or result.get("label")
    if not raw and result.get("success") is False:
        return "non_chilli"
    return _normalize_label(raw)


def compute_macro_f1(y_true, y_pred):
    """Pure-Python macro-F1 -- no sklearn dependency for a handful of held-out images."""
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def evaluate_candidate(images):
    y_true = [_normalize_label(e["label"]) for e in images]
    y_pred = [predict_label(e["path"]) for e in images]
    macro_f1 = compute_macro_f1(y_true, y_pred)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    return {"macro_f1": round(macro_f1, 4), "accuracy": round(accuracy, 4), "n_images": len(y_true)}


def source_data_summary():
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    n = len([f for f in TELEMETRY_DIR.glob("**/*") if f.suffix.lower() in valid_exts]) if TELEMETRY_DIR.exists() else 0
    return f"{n} telemetry image(s) from {TELEMETRY_DIR.relative_to(ROOT)} at promotion time"


def promote(registry, metrics):
    version_id = f"v_{time.strftime('%Y%m%d_%H%M%S')}"
    dest_dir = VERSIONS_DIR / version_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(CANDIDATE_PATH, dest_dir / CANDIDATE_PATH.name)

    for entry in registry.get("versions", []):
        entry["active"] = False

    registry.setdefault("versions", []).append({
        "version_id": version_id,
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_data_summary": source_data_summary(),
        "metrics": metrics,
        "weights_backup_path": str((dest_dir / CANDIDATE_PATH.name).relative_to(ROOT)),
        "active": True,
    })
    save_registry(registry)
    print(f"\n[GATE] PROMOTION APPROVED -- registered as {version_id}. metrics={metrics}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-baseline", action="store_true",
                         help="One-time human-run bootstrap: record the CURRENT weights/chilli_pest_model.onnx's "
                              "held-out metric as the v0 baseline. Does not require an existing baseline to compare "
                              "against (that's the whole point of seeding one).")
    args = parser.parse_args()

    if not CANDIDATE_PATH.exists():
        fail(f"no model found at {CANDIDATE_PATH} -- nothing to evaluate.")

    images = load_manifest()  # exits the process if missing/empty -- fail closed
    registry = load_registry()

    print(f"[GATE] Evaluating {CANDIDATE_PATH} against {len(images)} held-out image(s)...")
    metrics = evaluate_candidate(images)
    print(f"[GATE] Candidate metrics: {metrics}")

    if args.seed_baseline:
        if active_entry(registry):
            fail("an active baseline already exists in models/registry.json -- "
                 "--seed-baseline is only for the very first run. Use the normal "
                 "(no-flag) invocation, or edit models/registry.json by hand if you "
                 "really need to reseed.")
        promote(registry, metrics)
        return

    baseline = active_entry(registry)
    if not baseline or not isinstance(baseline.get("metrics"), dict) or baseline["metrics"].get("macro_f1") is None:
        fail(
            "models/registry.json has no active baseline with a recorded macro_f1 yet. "
            "Populate scripts/eval/validation_manifest.json with real held-out images, then run "
            "'python3 scripts/eval/run_validation_gate.py --seed-baseline' once to record the current "
            "production model's metric before this gate can evaluate future candidates."
        )

    baseline_f1 = baseline["metrics"]["macro_f1"]
    if metrics["macro_f1"] < baseline_f1 - PROMOTION_TOLERANCE:
        fail(
            f"candidate macro-F1 {metrics['macro_f1']} is below baseline {baseline_f1} "
            f"minus tolerance {PROMOTION_TOLERANCE}. Keeping existing production model "
            f"({baseline['version_id']})."
        )

    promote(registry, metrics)


if __name__ == "__main__":
    main()
