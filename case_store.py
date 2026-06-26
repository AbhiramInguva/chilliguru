"""
case_store.py — outcome-tracking case persistence behind a small interface.

WHY THIS EXISTS: after a diagnosis, a farmer can come back later, report
whether the treatment worked, and optionally upload a follow-up photo. That
needs *some* record to survive between the first request and the follow-up
request, potentially days apart.

PRODUCTION WARNING — RENDER FREE TIER HAS EPHEMERAL DISK:
JSONFileCaseStore (the default below) writes one small JSON file per case to
local disk. On Render's free tier (and most PaaS free tiers), that disk does
NOT survive a redeploy or dyno restart — every case created since the last
restart is silently lost. This is FINE for local dev/testing and even for a
single long-lived production process between deploys, but it is NOT a durable
production guarantee. To make outcome-tracking durable in production, either:

  (a) Mount a Render *persistent* disk (Render dashboard → Disks) and point
      CASE_STORE_PATH at a path on it, or
  (b) Swap build_case_store() below for a subclass backed by an external
      store (Postgres/Redis/etc. — not implemented here; this module only
      ships the interface + the local-disk default).

If neither is configured, the app does NOT crash or refuse to serve
diagnoses — case creation/lookup is best-effort everywhere it's called (see
app.py's _create_case) and simply returns None/False on any failure, which
the caller treats as "outcome-tracking unavailable this request" while the
core diagnosis flow proceeds exactly as before. A startup log warning is
emitted (see build_case_store) so this degradation is visible in production
logs rather than silent.
"""
import json
import logging
import os
import re
import secrets
import string
import threading

log = logging.getLogger("chilliguru")

_CASE_ID_ALPHABET = string.ascii_uppercase + string.digits  # no lowercase -- easier to read/say aloud
_CASE_ID_LEN = 7


def generate_case_id() -> str:
    """
    Short, human-friendly, URL-safe case id (e.g. '7K2QXJ9'). This is NOT a
    security boundary -- case records hold no personal identity (see
    PRIVACY.md), just a diagnosis label/treatment summary/outcome, so it only
    needs to be impractical to guess by casual enumeration, not
    adversary-proof. 36^7 (~78 billion) combinations is more than enough for
    that bar without needing a cryptographic token a farmer would struggle to
    write down or read back over a phone call.
    """
    return "".join(secrets.choice(_CASE_ID_ALPHABET) for _ in range(_CASE_ID_LEN))


class CaseStore:
    """
    Storage interface for outcome-tracking cases. Every method MUST fail
    soft: catch its own errors, log a warning, and return None/False rather
    than raising. Callers (app.py) treat any falsy return as "tracking
    unavailable this request" and continue the core flow regardless.
    """

    def create(self, case: dict) -> bool:
        raise NotImplementedError

    def get(self, case_id: str) -> dict | None:
        raise NotImplementedError

    def update(self, case_id: str, **fields) -> bool:
        raise NotImplementedError


class JSONFileCaseStore(CaseStore):
    """
    One JSON file per case under `base_dir` (default: static/uploads/cases/).
    A process-wide lock prevents two threads in this app's own gunicorn
    thread pool (see Procfile: --workers 1 --threads 4) from corrupting a
    file mid-write. It does NOT coordinate across multiple processes/dynos,
    which is fine for this app's current single-worker deployment but would
    need a real DB if that ever changes.

    NOT DURABLE ON EPHEMERAL DISK — see module docstring.
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._lock = threading.Lock()
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, case_id: str) -> str:
        # Defensive: case_id should already be from generate_case_id(), but
        # never let an externally-supplied id (URL path param) escape base_dir.
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "", str(case_id))[:32]
        return os.path.join(self.base_dir, f"{safe_id}.json")

    def create(self, case: dict) -> bool:
        try:
            with self._lock:
                path = self._path(case["case_id"])
                if os.path.exists(path):
                    return False  # collision -- caller retries with a new id
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(case, fh, ensure_ascii=False, indent=2)
            return True
        except Exception as exc:
            log.warning("case_store_create_fail", extra={"data": {"error": str(exc)}})
            return False

    def get(self, case_id: str) -> dict | None:
        try:
            path = self._path(case_id)
            if not os.path.exists(path):
                return None
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            log.warning("case_store_get_fail", extra={"data": {"error": str(exc)}})
            return None

    def update(self, case_id: str, **fields) -> bool:
        try:
            with self._lock:
                path = self._path(case_id)
                if not os.path.exists(path):
                    return False
                with open(path, encoding="utf-8") as fh:
                    case = json.load(fh)
                case.update(fields)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(case, fh, ensure_ascii=False, indent=2)
            return True
        except Exception as exc:
            log.warning("case_store_update_fail", extra={"data": {"error": str(exc)}})
            return False


class NullCaseStore(CaseStore):
    """
    No-op store used only if even constructing JSONFileCaseStore fails (e.g.
    completely read-only filesystem). Guarantees outcome-tracking degrades to
    "silently does nothing" rather than ever taking down the core app.
    """

    def create(self, case: dict) -> bool:
        return False

    def get(self, case_id: str):
        return None

    def update(self, case_id: str, **fields) -> bool:
        return False


def build_case_store() -> CaseStore:
    """
    Picks the case store implementation from CASE_STORE_PATH (default:
    static/uploads/cases/ next to this file). Logs a one-time warning when
    running on Render without an explicit override, since Render's default
    disk is ephemeral and this is the most common way this feature would
    silently lose data in production.
    """
    base_dir = os.environ.get(
        "CASE_STORE_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads", "cases"),
    )
    if "CASE_STORE_PATH" not in os.environ and os.environ.get("RENDER"):
        log.warning("case_store_ephemeral_disk_warning", extra={"data": {
            "message": (
                "CASE_STORE_PATH is not set and this looks like a Render deployment. "
                "Outcome-tracking cases are being written to the default ephemeral "
                "disk path and will NOT survive a redeploy/restart. Mount a Render "
                "persistent disk and set CASE_STORE_PATH, or swap in a durable "
                "CaseStore implementation, before relying on this feature in "
                "production. See case_store.py module docstring."
            ),
            "path": base_dir,
        }})
    try:
        return JSONFileCaseStore(base_dir)
    except Exception as exc:
        log.warning("case_store_init_fail_using_null_store", extra={"data": {"error": str(exc)}})
        return NullCaseStore()
