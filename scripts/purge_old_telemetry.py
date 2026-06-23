"""
purge_old_telemetry.py — deletes farmer-uploaded photos older than RETENTION_DAYS.

Targets the two directories the app writes farmer photos into:
  - static/uploads/shadow_dataset/   (app.py / detector.py active-learning captures)
  - static/uploads/telemetry_logs/   (curated set self_heal.py retrains on)

DRY-RUN BY DEFAULT — prints what it would delete without deleting anything.
Pass --execute to actually delete. This is a deliberate safety default: a
cron/Action misconfiguration should never silently delete real telemetry.

Usage:
    python3 scripts/purge_old_telemetry.py                  # dry-run, default RETENTION_DAYS
    python3 scripts/purge_old_telemetry.py --execute         # actually delete
    RETENTION_DAYS=30 python3 scripts/purge_old_telemetry.py --execute
    python3 scripts/purge_old_telemetry.py --retention-days 14 --execute

Scheduling: run this on a schedule via cron or a GitHub Actions scheduled
workflow (see .github/workflows/purge_telemetry.yml) so old farmer photos
don't accumulate indefinitely. See PRIVACY.md for the documented retention
policy and RETENTION_DAYS' default.
"""
import argparse
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RETENTION_DAYS = 90

TARGET_DIRS = [
    ROOT / "static" / "uploads" / "shadow_dataset",
    ROOT / "static" / "uploads" / "telemetry_logs",
]

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".txt"}  # .txt = YOLO label sidecars


def find_old_files(directory: Path, cutoff_epoch: float):
    if not directory.exists():
        return []
    old = []
    for f in directory.glob("**/*"):
        if not f.is_file() or f.name == ".gitkeep":
            continue
        if f.suffix.lower() not in VALID_EXTS:
            continue
        if f.stat().st_mtime < cutoff_epoch:
            old.append(f)
    return old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-days", type=int, default=None,
                         help=f"Override RETENTION_DAYS env var / default ({DEFAULT_RETENTION_DAYS}).")
    parser.add_argument("--execute", action="store_true",
                         help="Actually delete files. Without this flag, only lists what would be deleted.")
    args = parser.parse_args()

    retention_days = args.retention_days or int(os.environ.get("RETENTION_DAYS", DEFAULT_RETENTION_DAYS))
    cutoff_epoch = time.time() - (retention_days * 86400)
    cutoff_str = time.strftime("%Y-%m-%d", time.localtime(cutoff_epoch))

    print(f"[PURGE] RETENTION_DAYS={retention_days} -> deleting files last modified before {cutoff_str}")
    print(f"[PURGE] Mode: {'EXECUTE (will delete)' if args.execute else 'DRY-RUN (no files will be touched)'}")

    total_found = 0
    total_bytes = 0
    for directory in TARGET_DIRS:
        old_files = find_old_files(directory, cutoff_epoch)
        print(f"\n[PURGE] {directory.relative_to(ROOT)}: {len(old_files)} file(s) older than {retention_days} days")
        for f in old_files:
            size = f.stat().st_size
            total_found += 1
            total_bytes += size
            age_days = (time.time() - f.stat().st_mtime) / 86400
            action = "DELETING" if args.execute else "would delete"
            print(f"  [{action}] {f.relative_to(ROOT)} (age={age_days:.1f}d, {size} bytes)")
            if args.execute:
                f.unlink()

    print(f"\n[PURGE] Total: {total_found} file(s), {total_bytes / 1024:.1f} KB"
          f" {'deleted' if args.execute else 'identified (dry-run — pass --execute to delete)'}.")


if __name__ == "__main__":
    main()
