"""
rollback_model.py — one-command rollback of the production detection model.

Restores weights/chilli_pest_model.onnx from the PREVIOUS entry in
models/registry.json (the one promoted immediately before the current active
entry) and flips the registry's active flag back to it. This is a LOCAL
filesystem + registry change only -- it does not commit, push, or redeploy
anything; do that yourself afterwards if you're satisfied with the rollback.

Usage:
    python3 scripts/rollback_model.py            # roll back one version
    python3 scripts/rollback_model.py --dry-run  # show what would happen, change nothing
    python3 scripts/rollback_model.py --to v_20260601_120000   # roll back to a specific version_id
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH  = ROOT / "models" / "registry.json"
CANDIDATE_PATH = ROOT / "weights" / "chilli_pest_model.onnx"


def fail(message: str):
    print(f"\n[ROLLBACK] FAILED: {message}\n", file=sys.stderr)
    sys.exit(1)


def load_registry():
    if not REGISTRY_PATH.exists():
        fail(f"{REGISTRY_PATH} does not exist -- nothing has ever been promoted, there is nothing to roll back to.")
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"{REGISTRY_PATH} is not valid JSON ({exc}).")


def save_registry(registry):
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without changing anything.")
    parser.add_argument("--to", metavar="VERSION_ID", default=None,
                         help="Roll back to a specific version_id instead of just the immediately preceding one.")
    args = parser.parse_args()

    registry = load_registry()
    versions = registry.get("versions", [])
    if len(versions) < 2 and not args.to:
        fail("models/registry.json has fewer than 2 versions -- there is no previous version to roll back to.")

    active_idx = next((i for i, v in enumerate(versions) if v.get("active")), None)
    if active_idx is None:
        fail("no active entry found in models/registry.json -- cannot determine what to roll back from.")

    if args.to:
        target_idx = next((i for i, v in enumerate(versions) if v.get("version_id") == args.to), None)
        if target_idx is None:
            fail(f"no version with version_id={args.to!r} found in models/registry.json.")
    else:
        target_idx = active_idx - 1
        if target_idx < 0:
            fail("the active entry is already the oldest version on record -- nothing to roll back to.")

    current  = versions[active_idx]
    target   = versions[target_idx]
    backup   = ROOT / target["weights_backup_path"]

    if not backup.exists():
        fail(f"backup weights for {target['version_id']} not found at {backup} -- cannot restore.")

    print(f"[ROLLBACK] Current active version: {current['version_id']} (metrics={current.get('metrics')})")
    print(f"[ROLLBACK] Rolling back to:         {target['version_id']} (metrics={target.get('metrics')})")
    print(f"[ROLLBACK] Restoring {backup} -> {CANDIDATE_PATH}")

    if args.dry_run:
        print("\n[ROLLBACK] --dry-run: no files were changed.\n")
        return

    shutil.copy(backup, CANDIDATE_PATH)
    for i, v in enumerate(versions):
        v["active"] = (i == target_idx)
    versions.append({
        "version_id": f"rollback_{time.strftime('%Y%m%d_%H%M%S')}_to_{target['version_id']}",
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_data_summary": f"Manual rollback from {current['version_id']} to {target['version_id']}.",
        "metrics": target.get("metrics"),
        "weights_backup_path": target["weights_backup_path"],
        "active": False,  # the restored target entry above is the active one; this is just an audit record
    })
    save_registry(registry)

    print(f"\n[ROLLBACK] Done. weights/chilli_pest_model.onnx now matches {target['version_id']}.")
    print("[ROLLBACK] This only changed local files -- commit and push/deploy yourself if you want this live.\n")


if __name__ == "__main__":
    main()
