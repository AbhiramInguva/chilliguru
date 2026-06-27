import functools
import gc
import json
import logging
import os
import random
import re
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import requests

# ── Calibrate GC thresholds at process launch ─────────────────────────────────
# Raises gen-0 threshold so the collector runs less aggressively per request,
# preventing blocking pauses inside the hot /chat and /detect handlers.
gc.set_threshold(1000, 10, 10)
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, g, has_request_context
from werkzeug.exceptions import HTTPException
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from gradio_client import Client, handle_file
from groq import Groq
import detector
import triage
import case_store

# ── Optional error tracking (Sentry) ──────────────────────────────────────────
# Entirely no-op unless the SENTRY_DSN env var is set — local dev and any
# deploy that doesn't configure it behaves exactly as before.
_SENTRY_DSN = os.environ.get("SENTRY_DSN")
if _SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.flask import FlaskIntegration
    sentry_sdk.init(dsn=_SENTRY_DSN, integrations=[FlaskIntegration()],
                     traces_sample_rate=0.0)

# ── Global background worker pool (replaces per-request daemon Thread spawns) ──
# max_workers=2 keeps total thread headroom well within the 150 MB container.
_shadow_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="shadow-worker")

# ── Persistent HTTP session (connection-pool reuse for Open-Meteo calls) ──
_http_session = requests.Session()

# ── Structured JSON logger ────────────────────────────────────────────────────
class _JsonFormatter(logging.Formatter):
    """
    Custom logging formatter that outputs log records as single-line JSON structures.
    This ensures compatibility with structured logging systems (e.g., GCP, AWS, ELK stacks).
    """
    def format(self, record):
        """
        Formats a standard LogRecord object into a structured JSON string.
        
        Args:
            record (logging.LogRecord): The log record to format.
            
        Returns:
            str: A JSON serialized string containing timestamp, log level, message,
                 and any optional extra context or exception traceback.
        """
        payload = {
            "ts":    self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        # Thread the per-request correlation ID (set in _assign_request_id's
        # before_request hook) into every log line emitted during that
        # request, so a farmer's request_id in an error response can be
        # grepped straight to the matching server logs.
        if has_request_context():
            rid = getattr(g, "request_id", None)
            if rid:
                payload["request_id"] = rid
        if hasattr(record, "data") and isinstance(record.data, dict):
            payload.update(record.data)
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_JsonFormatter())
logging.root.setLevel(logging.INFO)
logging.root.handlers = [_handler]
log = logging.getLogger("chilliguru")
REGIONAL_TRANSLATION_MAP = {
    "aphids": {
        "en": "Aphids",
        "hi": "माहू (एफिड्स)",
        "te": "పేను పురుగు",
        "kn": "ಸೇಬು ಜೇನು ನೊಣ (ಅಫಿಡ್ಸ್)",
        "ta": "அசுவினி"
    },
    "whitefly_leaf_damage": {
        "en": "Whitefly Leaf Damage",
        "hi": "सफेद मक्खी का नुकसान",
        "te": "తెల్ల ఈగ ఆకు నష్టం",
        "kn": "ಬಿಳಿ ನೊಣದ ಎಲೆ ಹಾನಿ",
        "ta": "வெள்ளை ஈ இலை சேதம்"
    },
    "fruit_borer": {
        "en": "Fruit Borer",
        "hi": "फल छेदक",
        "te": "పండు తొలిచే పురుగు",
        "kn": "ಕಾಯಿ ಕೊರಕ",
        "ta": "காய்ப்புழு"
    },
    "tobacco_caterpillar": {
        "en": "Tobacco Caterpillar",
        "hi": "तंबाकू की इल्ली",
        "te": "పొగాకు లద్దె పురుగు",
        "kn": "ತಂಬಾಕು ಪತಂಗ",
        "ta": "புகையிலை வெட்டுப்புழு"
    },
    "yellow_thrips": {
        "en": "Yellow Thrips",
        "hi": "पीला थ्रिप्स",
        "te": "పసుపు తామర పురుగు",
        "kn": "ಹಳದಿ ನುಸಿ",
        "ta": "மஞ்சள் இலைப்பேன்"
    },
    "broad_mites": {
        "en": "Broad Mites",
        "hi": "चौड़ी मक्खियाँ (माइट्स)",
        "te": "ఎర్ర నల్లి / తామర పురుగు",
        "kn": "ಅಗಲವಾದ ನುಸಿ",
        "ta": "பரந்த சிலந்திப் பேன்"
    },
    "invasive_black_thrips": {
        "en": "Invasive Black Thrips",
        "hi": "आक्रामक काला थ्रिप्स",
        "te": "నల్ల తామర పురుగు",
        "kn": "ಆಕ್ರಮಣಕಾರಿ ಕಪ್ಪು ನುಸಿ",
        "ta": "ஊடுருவும் கருப்பு இலைப்பேன்"
    },
    "mealybugs": {
        "en": "Mealybugs",
        "hi": "मीलीबग",
        "te": "పిండి పురుగు",
        "kn": "ಹಿಟ್ಟು ತಿಗಣೆ",
        "ta": "மாவுப்பூச்சி"
    },
    "leaf_curl_virus": {
        "en": "Leaf Curl Virus",
        "hi": "पत्ती मरोड़ वायरस",
        "te": "ఆకు ముడత వైరస్",
        "kn": "ಎಲೆ ಮುರುಟು ವೈರಸ್",
        "ta": "இலை சுருள் வைரஸ்"
    },
    "bacterial_leaf_spot": {
        "en": "Bacterial Leaf Spot",
        "hi": "जीवाणु पत्ती धब्बा",
        "te": "బ్యాక్టీరిयల్ ఆకు మచ్చ తెగులు",
        "kn": "ಬ್ಯಾಕ್ಟೀರಿಯಾದ ಎಲೆ ಚುಕ್ಕೆ",
        "ta": "பாக்டீரியா இலைப்புள்ளி"
    },
    "cercospora_leaf_spot": {
        "en": "Cercospora Leaf Spot",
        "hi": "सर्कोस्पोरा पत्ता धब्बा",
        "te": "సెర్కోస్పోరా ఆకు మచ్చ తెగులు",
        "kn": "ಸೆರ್ಕೋಸ್ಪೊರಾ ಎಲೆ ಚುಕ್ಕೆ",
        "ta": "செர்கோஸ்போரா இலைப்புள்ளி"
    },
    "powdery_mildew": {
        "en": "Powdery Mildew",
        "hi": "पाउडर जैसी फफूंदी (चूर्णी फफूंद)",
        "te": "బూడిద తెగులు",
        "kn": "ಬೂದಿ ರೋಗ",
        "ta": "சாம்பல் நோய்"
    }
}

def to_core_class(label_or_name):
    """
    Normalizes raw disease/pest labels (from vision models or user strings) 
    into standard internal core class identifiers.
    
    This function bridges the gap between different vision model version outputs 
    and the keys used in REGIONAL_TRANSLATION_MAP and _CORE_CLASS_TO_KB.
    
    Args:
        label_or_name (str): Raw string label or classification name to normalize.
        
    Returns:
        str | None: The normalized core class string (e.g., 'invasive_black_thrips'), 
                    or None if no match is found.
    """
    if not label_or_name:
        return None
    # Normalize underscores so Phase-2 labels like "yellow_thrips" match the
    # space-separated keywords below (previously they fell through to the
    # generic "thrips" rule and were misreported as invasive black thrips).
    lbl = str(label_or_name).lower().replace("_", " ")
    if "black thrips" in lbl or "invasive" in lbl:
        return "invasive_black_thrips"
    if "yellow thrips" in lbl:
        return "yellow_thrips"
    if "thrips" in lbl:
        return "invasive_black_thrips"
    if "aphid" in lbl:
        return "aphids"
    if "white fly" in lbl or "whitefly" in lbl:
        return "whitefly_leaf_damage"
    if "fruit borer" in lbl or "helicoverpa" in lbl or "borer" in lbl:
        return "fruit_borer"
    if "armyworm" in lbl or "tobaccocaterpillar" in lbl or "tobacco caterpillar" in lbl or "spodoptera" in lbl:
        return "tobacco_caterpillar"
    if "red mites" in lbl or "broad mites" in lbl or "broad_mites" in lbl or "mites" in lbl:
        return "broad_mites"
    if "mealybug" in lbl:
        return "mealybugs"
    if "leaf curl" in lbl or "curling" in lbl or "mosaic" in lbl or "mozaik" in lbl:
        return "leaf_curl_virus"
    if "cercospora" in lbl:
        return "cercospora_leaf_spot"
    if "bacterial" in lbl:
        return "bacterial_leaf_spot"
    if "spot" in lbl:
        return "bacterial_leaf_spot"
    if "powdery" in lbl or "mildew" in lbl or "leveillula" in lbl:
        return "powdery_mildew"
    return None

# ── Agronomy knowledge base (server-side only — never sent raw to the client) ──
_KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")

def _load_kb_json(filename):
    """
    Loads and parses a JSON file from the local knowledge base directory.
    
    Args:
        filename (str): Name of the JSON file (e.g. 'chilli_kb.json') located in knowledge/.
        
    Returns:
        dict: Parsed JSON data, or an empty dictionary if loading/parsing fails.
    """
    path = os.path.join(_KB_DIR, filename)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("kb_load_failed", extra={"data": {"path": path, "error": str(exc)}})
        return {}

_CHILLI_KB = _load_kb_json("chilli_kb.json")

# Maps this module's internal core-class ids (the keys used by to_core_class()
# and REGIONAL_TRANSLATION_MAP above) to a "section.id" entry in chilli_kb.json.
# This is a different keyspace from knowledge/kb_class_map.json, which maps the
# 15 *raw vision-model* class names from model_info.json — both ultimately
# resolve into the same chilli_kb.json. Core classes with no reference entry
# (cercospora_leaf_spot) are intentionally omitted, not invented.
_CORE_CLASS_TO_KB = {
    "aphids":                "pests.aphids",
    "whitefly_leaf_damage":  "pests.whitefly",
    "fruit_borer":           "pests.fruit_borer",
    "tobacco_caterpillar":   "pests.tobacco_caterpillar",
    "yellow_thrips":         "pests.thrips",
    "broad_mites":           "pests.mites",
    "invasive_black_thrips": "pests.thrips",
    "mealybugs":             "pests.mealybugs",
    "leaf_curl_virus":       "diseases.leaf_curl_virus",
    "bacterial_leaf_spot":   "diseases.bacterial_leaf_spot",
    "powdery_mildew":        "diseases.powdery_mildew",
}


def _get_kb_entry(core_cls):
    """
    Resolves a standardized core class identifier to its matching entry 
    in the agronomy knowledge base (chilli_kb.json).
    
    Args:
        core_cls (str): Standardized core class string (e.g., 'aphids').
        
    Returns:
        dict | None: The raw dictionary containing agronomy data (symptoms, agents, 
                    management practices) for the resolved entry, or None if the class
                    is not mapped or is missing from the database.
    """
    ref = _CORE_CLASS_TO_KB.get(core_cls or "")
    if not ref:
        return None
    section, _, entry_id = ref.partition(".")
    return _CHILLI_KB.get(section, {}).get(entry_id)


def _format_kb_context(core_cls):
    """
    Renders the curated agronomy reference details for a given core class into a structured
    text block to be injected into the LLM prompt context.
    
    This grounds the LLM in highly specific, verified agricultural rules (organic/chemical)
    to prevent hallucinations and ensure local regulatory compliance.
    
    Args:
        core_cls (str): Standardized core class identifier.
        
    Returns:
        str: A formatted text block with symptoms, causal agents, and management methods,
             or an empty string if no agronomy reference exists.
    """
    entry = _get_kb_entry(core_cls)
    if not entry:
        return ""
    lines = [
        "\n=== CURATED AGRONOMY REFERENCE (ground your diagnosis in this) ===",
        f"Reference name: {entry.get('display_name')}",
    ]
    if entry.get("causal_agent"):
        lines.append(f"Causal agent: {entry['causal_agent']}")
    if entry.get("key_symptoms"):
        lines.append(f"Distinguishing symptoms: {entry['key_symptoms']}")
    mgmt = entry.get("management") or {}
    if mgmt.get("organic"):
        lines.append(f"Organic/biological management (reference): {mgmt['organic']}")
    if mgmt.get("chemical"):
        lines.append(
            "Chemical management (reference — use ONLY inside 'Targeted Chemical "
            f"Interventions', after organic options): {mgmt['chemical']}"
        )
    lines.append(
        "Use this reference to ground the diagnosis and treatment plan; do not "
        "contradict it, but still phrase things simply for the farmer.\n"
    )
    return "\n".join(lines)


def _build_deficiency_summary():
    """
    Builds a lightweight symptom-to-nutrient-deficiency text summary mapping 
    from the deficiencies section of the knowledge base.
    
    This summary is appended to the system prompt of the text-only chat path (/chat)
    to help the LLM diagnose issues like nitrogen/magnesium deficiencies based 
    solely on the farmer's textual descriptions.
    
    Returns:
        str: A multi-line string containing quick symptoms and primary diagnostic 
             questions for common nutrient deficiencies.
    """
    deficiencies = _CHILLI_KB.get("deficiencies", {})
    if not deficiencies:
        return ""
    lines = [
        "\n=== NUTRIENT DEFICIENCY QUICK-REFERENCE (text-only diagnosis aid) ===",
        "The vision model cannot detect nutrient deficiencies — use these "
        "symptom patterns to distinguish them when a farmer describes leaf "
        "colour/curling without a photo, or when a photo is inconclusive:",
    ]
    for entry in deficiencies.values():
        name = entry.get("display_name", "")
        symptoms = entry.get("key_symptoms", "")
        first_sentence = symptoms.split(". ")[0].strip().rstrip(".")
        if name and first_sentence:
            lines.append(f"- {name}: {first_sentence}.")
    lines.append(
        "Ask the farmer whether the discolouration is on the YOUNGEST top leaves "
        "or the OLDER lower leaves first — this single question separates most of "
        "the above (iron/zinc/manganese/boron/calcium/copper show on young leaves "
        "first; nitrogen/magnesium/molybdenum/phosphorous/potassium show on older "
        "leaves first).\n"
    )
    return "\n".join(lines)

_DEFICIENCY_SUMMARY = _build_deficiency_summary()


# ── Field-context priors (farmer's MCQ answers) ───────────────────────────────
# Each MCQ answer maps to the core pest/disease classes it makes more (or less)
# likely. Used to re-rank the vision model's detections. Boost/penalty are
# bounded so an answer can nudge the ranking but never single-handedly override
# the image evidence.
_CTX_BOOST    = 10.0
_CTX_PENALTY  = 12.0
_CTX_MAX_UP   = 30.0
_CTX_MAX_DOWN = 25.0

_OBSERVED_BOOST = {
    "tiny_insects":  {"aphids", "whitefly_leaf_damage", "invasive_black_thrips", "yellow_thrips"},
    "holes":         {"fruit_borer", "tobacco_caterpillar"},
    "white_cottony": {"mealybugs"},
    "webbing_dots":  {"broad_mites"},
    "curl_yellow":   {"leaf_curl_virus", "invasive_black_thrips", "yellow_thrips"},
    "white_powder":  {"powdery_mildew"},
    "spots":         {"bacterial_leaf_spot", "cercospora_leaf_spot"},
}
_PLANT_AGE_BOOST = {
    "seedling":   {"aphids", "invasive_black_thrips", "yellow_thrips", "leaf_curl_virus"},
    "vegetative": {"invasive_black_thrips", "yellow_thrips", "aphids", "broad_mites",
                   "leaf_curl_virus", "whitefly_leaf_damage"},
    "flowering":  {"invasive_black_thrips", "broad_mites", "mealybugs",
                   "whitefly_leaf_damage", "fruit_borer"},
    "fruiting":   {"fruit_borer", "tobacco_caterpillar", "mealybugs",
                   "bacterial_leaf_spot", "cercospora_leaf_spot", "powdery_mildew"},
}
# Pests that physically cannot be the main issue at a given stage (no fruit yet)
_PLANT_AGE_IMPOSSIBLE = {
    "seedling":   {"fruit_borer"},
    "vegetative": {"fruit_borer"},
}
_AFFECTED_BOOST = {
    "new_leaves": {"invasive_black_thrips", "yellow_thrips", "aphids",
                   "leaf_curl_virus", "broad_mites"},
    "all_leaves": {"leaf_curl_virus", "broad_mites", "bacterial_leaf_spot",
                   "cercospora_leaf_spot", "powdery_mildew", "whitefly_leaf_damage"},
    "stem":       {"mealybugs"},
    "fruit":      {"fruit_borer", "mealybugs"},
    "flowers":    {"invasive_black_thrips", "yellow_thrips", "fruit_borer"},
}
# Leaf-curl direction ties directly into the morphology feature.
_CURL_BOOST = {
    "upward":   {"leaf_curl_virus", "invasive_black_thrips", "yellow_thrips"},
    "downward": {"broad_mites"},
}
_CURL_PENALTY = {
    "upward":   {"broad_mites"},
    "downward": {"leaf_curl_virus"},
}


def _sev_label(c):
    return "High" if c >= 80 else ("Medium" if c >= 50 else "Low (not very sure)")


def _apply_field_context(result, ctx):
    """
    Re-ranks and adjusts detection confidence scores using a Bayesian-like prior system 
    grounded in the farmer's multiple-choice field responses (e.g. observed symptoms, 
    plant age, affected parts, and leaf curling direction).
    
    This function boosts confidence for logical class alignments (e.g. upward curling and 
    Leaf Curl Virus) and applies penalties or hard limits for impossible scenarios (e.g. 
    fruit borer on a seedling). Score updates are clamped to stay within a reasonable range 
    and avoid overriding actual vision model evidence.
    
    Args:
        result (dict): The detection results structure returned by the YOLO/HF cascade.
        ctx (dict): The farmer's field context multiple-choice selections.
        
    Returns:
        dict: The updated result structure with adjusted confidence scores, severity labels, 
              and re-sorted detection lists.
    """
    if not isinstance(result, dict) or not ctx or not any(ctx.values()):
        return result
    dets = result.get("all_detections")
    top  = result.get("top_detection")
    work = dets if isinstance(dets, list) and dets else ([top] if isinstance(top, dict) else [])
    if not work:
        return result

    age  = ctx.get("plant_age", "")
    obs  = ctx.get("observed", "")
    part = ctx.get("affected_part", "")
    curl = ctx.get("curl_dir", "")

    for d in work:
        if not isinstance(d, dict):
            continue
        core = to_core_class(d.get("raw_label") or d.get("label"))
        if not core:
            continue
        delta = 0.0
        if core in _OBSERVED_BOOST.get(obs, ()):        delta += _CTX_BOOST
        if core in _PLANT_AGE_BOOST.get(age, ()):       delta += _CTX_BOOST
        if core in _AFFECTED_BOOST.get(part, ()):       delta += _CTX_BOOST
        if core in _CURL_BOOST.get(curl, ()):           delta += _CTX_BOOST
        if core in _CURL_PENALTY.get(curl, ()):         delta -= _CTX_PENALTY
        if core in _PLANT_AGE_IMPOSSIBLE.get(age, ()):  delta -= _CTX_PENALTY + 3.0
        if delta == 0.0:
            continue
        delta = max(-_CTX_MAX_DOWN, min(_CTX_MAX_UP, delta))
        try:
            base = float(d.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        newc = max(5.0, min(99.0, base + delta))
        d["confidence"]    = round(newc, 1)
        d["severity"]      = _sev_label(newc)
        d["context_delta"] = round(delta, 1)

    work.sort(key=lambda x: x.get("confidence", 0) if isinstance(x, dict) else 0, reverse=True)
    result["all_detections"] = work[:3]
    result["top_detection"]  = work[0]
    result["context_adjusted"] = True
    log.info("field_context_applied", extra={"data": {
        "answers":  {k: v for k, v in ctx.items() if v},
        "new_top":  work[0].get("raw_label") if isinstance(work[0], dict) else None,
    }})
    return result


_FIELD_ANSWER_TEXT = {
    "plant_age": {
        "seedling":   "Plant age: seedling / nursery (0-4 weeks)",
        "vegetative": "Plant age: young vegetative (1-2 months)",
        "flowering":  "Plant age: flowering (2-3 months)",
        "fruiting":   "Plant age: fruiting / harvest (3+ months)",
    },
    "observed": {
        "tiny_insects":  "Farmer sees: tiny insects on leaves/shoots (sap-sucking pests)",
        "holes":         "Farmer sees: holes in leaves or fruit (chewing pests)",
        "white_cottony": "Farmer sees: white cottony sticky clusters (mealybug-like)",
        "webbing_dots":  "Farmer sees: fine webbing / tiny moving dots (mite-like)",
        "curl_yellow":   "Farmer sees: leaves curling or yellowing",
        "white_powder":  "Farmer sees: dry white powder on leaves (mildew-like)",
        "spots":         "Farmer sees: dark/brown spots on leaves (leaf-spot disease-like)",
        "not_sure":      "Farmer is not sure what they see",
    },
    "affected_part": {
        "new_leaves": "Mainly affected: new leaves / top shoots",
        "all_leaves": "Mainly affected: most leaves",
        "stem":       "Mainly affected: stem / branches",
        "fruit":      "Mainly affected: fruit / pods",
        "flowers":    "Mainly affected: flowers / buds",
    },
    "curl_dir": {
        "upward":   "Leaf curl reported: UPWARD cupping (Leaf Curl Virus signature)",
        "downward": "Leaf curl reported: DOWNWARD inverted-boat (Broad Mite signature)",
        "none":     "Farmer reports leaves are not curling",
        "not_sure": "Farmer not sure about leaf curl direction",
    },
}


def _format_field_answers(ctx):
    """
    Renders the farmer's multiple-choice field questionnaire answers into a 
    structured text block to inject into the LLM context.
    
    This helps the LLM understand the contextual realities of the farm (e.g. age of the 
    plant, visible signs) and address conflicts between the visual classifier and the 
    user's observations.
    
    Args:
        ctx (dict): The farmer's field context dictionary containing plant_age, observed,
                    affected_part, and curl_dir.
                    
    Returns:
        str: A formatted string list of user-submitted context answers, or an empty string.
    """
    if not ctx:
        return ""
    lines = []
    for key in ("plant_age", "observed", "affected_part", "curl_dir"):
        val = ctx.get(key, "")
        txt = _FIELD_ANSWER_TEXT.get(key, {}).get(val) if val else None
        if txt:
            lines.append("- " + txt)
    if not lines:
        return ""
    return (
        "\n=== FARMER'S FIELD ANSWERS (factor these into the diagnosis) ===\n"
        + "\n".join(lines)
        + "\nUse these answers to refine and explain the diagnosis; if they conflict with "
          "the image detection, acknowledge it and ask one short clarifying question.\n"
    )

# ── Pre-compiled script regex map (built once at module load) ─────────────────
# Eliminates repeated re.compile() calls inside strip_cross_contamination at
# runtime — the compiled Pattern objects are reused across every request.
_SCRIPT_RE: dict = {
    "hi": re.compile(r"[\u0900-\u097f]"),
    "te": re.compile(r"[\u0c00-\u0c7f]"),
    "kn": re.compile(r"[\u0c80-\u0cff]"),
    "ta": re.compile(r"[\u0b80-\u0bff]"),
}
_CLEANUP_RE = {
    "brackets_with_content": re.compile(r"\s*\[[^\]]*\]"),
    "empty_brackets":        re.compile(r"\s*\[\s*\]"),
    "empty_parens":          re.compile(r"\s*\(\s*\)"),
    "multi_space":           re.compile(r"\s+"),
}

# lru_cache memoises identical (text, target_lang) pairs across a request
# burst — repeated SYSTEM_PROMPT cleaning calls become O(1) dict lookups.
@functools.lru_cache(maxsize=128)
def strip_cross_contamination(text, target_lang):
    """
    Filters and cleans target prompt strings to remove scripts from unselected regional languages.
    
    For example, if the target language is English ('en'), Hindi/Telugu/Kannada/Tamil Unicode characters 
    are removed. If the target language is Telugu ('te'), other language scripts (Hindi, Tamil, Kannada) 
    are stripped. This helps keep the LLM focused on a single target script and reduces prompt 
    cross-contamination. Uses memoization (lru_cache) to optimize repeated calls across requests.
    
    Args:
        text (str): The input text containing mixed scripts or formatting placeholders.
        target_lang (str): The target language code ('en', 'te', 'hi', 'kn', 'ta').
        
    Returns:
        str: Sanitized text containing only the target language scripts and common syntax.
    """
    if not isinstance(text, str):
        return text

    if target_lang == "en":
        text = _CLEANUP_RE["brackets_with_content"].sub("", text)
        for script_re in _SCRIPT_RE.values():
            text = script_re.sub("", text)
    else:
        for lang_code, script_re in _SCRIPT_RE.items():
            if lang_code != target_lang:
                text = script_re.sub("", text)
        text = _CLEANUP_RE["empty_brackets"].sub("", text)

    text = _CLEANUP_RE["empty_parens"].sub("", text)
    text = _CLEANUP_RE["empty_brackets"].sub("", text)
    text = _CLEANUP_RE["multi_space"].sub(" ", text).strip()
    return text

MODEL          = "llama-3.3-70b-versatile"
MAX_TOKENS     = 1200
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB hard limit

# Recognised image magic-byte signatures (read first 12 bytes)
_IMAGE_SIGNATURES = [
    b'\xff\xd8\xff',       # JPEG
    b'\x89PNG\r\n\x1a\n', # PNG
    b'GIF87a',             # GIF87a
    b'GIF89a',             # GIF89a
    b'RIFF',               # WebP  (bytes 0-3; bytes 8-11 must also be b'WEBP')
]

app = Flask(__name__, static_folder="static")

# ── Global request body cap ───────────────────────────────────────────────────
# Belt-and-suspenders alongside /detect's own MAX_IMAGE_BYTES check (which
# inspects the image field specifically) — this rejects any oversized request
# body, on any route, before Flask even buffers it into memory.
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

# ── Per-request correlation ID ────────────────────────────────────────────────
# Generated once per request (or reused from an incoming X-Request-ID header,
# e.g. set by a load balancer), threaded into every log line for that request
# via _JsonFormatter, echoed back in the X-Request-ID response header, and
# included in every error response so a farmer's bug report can be matched
# to server logs.
@app.before_request
def _assign_request_id():
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())


@app.after_request
def _echo_request_id(response):
    response.headers["X-Request-ID"] = getattr(g, "request_id", "")
    return response


def _current_request_id():
    return getattr(g, "request_id", None) or str(uuid.uuid4())


# ── CORS: restrict to explicitly allowed origins ──────────────────────────────
# Previously CORS(app) allowed every origin to call every endpoint (including
# the paid Groq-backed routes). Set CORS_ALLOWED_ORIGINS to a comma-separated
# list in production; defaults to localhost for local testing.
_cors_origins = [
    o.strip() for o in os.environ.get(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080",
    ).split(",") if o.strip()
]
CORS(app, resources={r"/*": {"origins": _cors_origins}})

# ── Shadow dataset directory (created once at startup) ────────────────────────
SHADOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "static", "uploads", "shadow_dataset")
os.makedirs(SHADOW_DIR, exist_ok=True)

# ── Outcome-tracking case store (see case_store.py for the durability warning) ─
CASE_STORE = case_store.build_case_store()

limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
    default_limits=[],
)

@app.errorhandler(429)
def rate_limit_exceeded(e):
    rid = _current_request_id()
    log.warning("rate_limit_exceeded", extra={"data": {"remote_addr": request.remote_addr, "request_id": rid}})
    return jsonify({
        "success": False,
        "error":   "Too many uploads. Please wait a minute before trying again.",
        "request_id": rid,
    }), 429


@app.errorhandler(404)
def not_found(e):
    rid = _current_request_id()
    return jsonify({
        "success": False,
        "error":   "Not found",
        "request_id": rid,
    }), 404


@app.errorhandler(413)
def payload_too_large_global(e):
    rid = _current_request_id()
    log.warning("payload_too_large_global", extra={"data": {"request_id": rid}})
    return jsonify({
        "success": False,
        "error":   "Payload too large",
        "request_id": rid,
    }), 413


@app.errorhandler(500)
def internal_server_error(e):
    rid = _current_request_id()
    log.error("internal_server_error", extra={"data": {"request_id": rid, "error": str(e)}}, exc_info=True)
    return jsonify({
        "success": False,
        "error":   "Internal server error",
        "request_id": rid,
    }), 500

_hf_client = None
_hf_connect_error = None
_hf_initialized = False
_hf_init_lock = threading.Lock()

# ── Retry / timeout helpers for external calls (HF Space, Groq) ──────────────
# Bounded retry with jittered backoff so a single transient network blip
# doesn't immediately trip the HF circuit breaker, plus a hard per-attempt
# wall-clock timeout so a hung external call can never block past the
# gunicorn worker --timeout (120s). Worst-case budget:
#   HF:   up to 3 attempts x 15s + backoff (~3s)  ≈ 48s
#   Groq (non-stream /chat, with retry): up to 3 x 30s + backoff (~3s) ≈ 93s
#   Groq (stream /detect, single attempt, no retry): 45s
# /detect's worst case (HF exhausted -> local cascade -> Groq stream) is
# ≈ 48s + a few seconds of local ONNX inference + 45s ≈ 96s, leaving headroom
# under the 120s timeout. /chat's worst case (Groq only) is ≈ 93s.
_HF_PREDICT_TIMEOUT_S        = 15.0
_HF_MAX_RETRIES              = 2   # up to 3 total attempts
_GROQ_NONSTREAM_TIMEOUT_S    = 30.0
_GROQ_NONSTREAM_MAX_RETRIES  = 2   # up to 3 total attempts
_GROQ_STREAM_TIMEOUT_S       = 45.0  # single attempt — retrying mid-stream would duplicate content


def _call_with_timeout(fn, timeout_s, *args, **kwargs):
    """
    Run fn(*args, **kwargs) in a worker thread with a hard wall-clock timeout.

    Needed because the older-gradio_client fallback path in get_hf_client()
    (no httpx_kwargs support) has no built-in timeout at all — this enforces
    one regardless of the installed SDK version. Raises
    concurrent.futures.TimeoutError if fn does not complete in time (the
    worker thread is abandoned, not killed — acceptable here since predict()
    calls don't hold locks shared with the rest of the app).
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        return future.result(timeout=timeout_s)


def _retry_with_backoff(fn, max_retries, what):
    """
    Call fn() with up to max_retries retries on failure, using jittered
    exponential backoff between attempts (0.3s, 0.6s, ... + up to 0.3s
    jitter). Raises the last exception once every attempt has failed —
    callers such as the HF circuit breaker only observe a failure after
    retries are exhausted, not on the first blip.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = (0.3 * (2 ** attempt)) + random.uniform(0, 0.3)
                log.warning("retry_attempt", extra={"data": {
                    "what": what, "attempt": attempt + 1, "max_retries": max_retries,
                    "delay_s": round(delay, 2), "error": str(exc),
                }})
                time.sleep(delay)
    raise last_exc


def get_hf_client():
    """
    Initializes and returns a thread-safe Gradio Client instance pointing to the 
    Hugging Face spaces model endpoint.
    
    Uses double-checked locking to avoid duplicate network-bound connection attempts 
    during concurrent requests at cold-start. Gracefully handles older versions of 
    gradio_client that do not support the httpx_kwargs parameter.
    
    Returns:
        gradio_client.Client | None: The initialized client object, or None if connection fails.
    """
    global _hf_client, _hf_connect_error, _hf_initialized
    if _hf_initialized:
        return _hf_client
    # Double-checked locking: prevents two concurrent first requests from both
    # paying the (slow, network-bound) Gradio connect.
    with _hf_init_lock:
        if _hf_initialized:
            return _hf_client
        log.info("hf_connect_start")
        try:
            hf_token = os.environ.get("HF_TOKEN")
            try:
                _hf_client = Client("inguvaaa/comprehensive", token=hf_token, verbose=False,
                                   httpx_kwargs={"timeout": _HF_PREDICT_TIMEOUT_S})
            except TypeError:
                # Older gradio_client versions don't accept httpx_kwargs — fall back gracefully
                _hf_client = Client("inguvaaa/comprehensive", token=hf_token, verbose=False)
            log.info("hf_connect_ok")
        except Exception as exc:
            _hf_connect_error = str(exc)
            log.error("hf_connect_fail", extra={"data": {"error_message": _hf_connect_error}})
        _hf_initialized = True
    return _hf_client

def get_hf_connect_error():
    return _hf_connect_error

# ── Circuit Breaker state ─────────────────────────────────────────────────────
HF_CIRCUIT_OPEN      = False
HF_FAILURE_COUNT     = 0
HF_RECOVERY_TIME     = None   # datetime.time when the circuit may close again
CB_FAILURE_THRESHOLD = 3      # consecutive failures before opening
CB_COOLDOWN_SECS     = 60     # seconds to wait before retrying


def _cb_is_open():
    """
    Checks if the Hugging Face spaces circuit breaker is open.
    
    If the circuit breaker is open, further calls to Hugging Face are skipped
    until the cooldown period has elapsed. Once the cooldown has elapsed, the circuit 
    breaker is reset to closed.
    
    Returns:
        bool: True if the circuit breaker is open (blocking requests), False otherwise.
    """
    global HF_CIRCUIT_OPEN, HF_FAILURE_COUNT, HF_RECOVERY_TIME
    if not HF_CIRCUIT_OPEN:
        return False
    if time.monotonic() >= HF_RECOVERY_TIME:
        log.info("circuit_breaker_reset")
        HF_CIRCUIT_OPEN  = False
        HF_FAILURE_COUNT = 0
        HF_RECOVERY_TIME = None
        return False
    return True


def _cb_record_success():
    """
    Records a successful API call, resetting the failure counter back to zero.
    """
    global HF_FAILURE_COUNT
    HF_FAILURE_COUNT = 0


def _cb_record_failure():
    """
    Records a failed API call. If consecutive failures exceed the defined threshold,
    opens the circuit breaker and sets a recovery cooldown window.
    """
    global HF_CIRCUIT_OPEN, HF_FAILURE_COUNT, HF_RECOVERY_TIME
    HF_FAILURE_COUNT += 1
    if HF_FAILURE_COUNT >= CB_FAILURE_THRESHOLD:
        HF_CIRCUIT_OPEN  = True
        HF_RECOVERY_TIME = time.monotonic() + CB_COOLDOWN_SECS
        log.warning("circuit_breaker_open", extra={"data": {
            "failure_count": HF_FAILURE_COUNT,
            "cooldown_secs": CB_COOLDOWN_SECS,
        }})


def get_client():
    """
    Initializes and returns a client interface for the Groq API.
    
    Returns:
        Groq: An instance of the Groq API client with credentials loaded from environment.
    """
    return Groq(api_key=os.getenv("GROQ_API_KEY", ""))


def call_hf_detector(image_bytes):
    """
    Performs visual pest/disease detection by forwarding raw image bytes 
    to the Hugging Face spaces Gradio endpoint.
    
    The function dumps the bytes to a temporary local JPG file, passes it to 
    the Gradio Client for transmission, parses the resulting text/JSON response, 
    and handles circuit breaker logging on success or failure.
    
    Args:
        image_bytes (bytes): The raw uploaded image data.
        
    Returns:
        dict: A parsed dictionary containing top_detection, all_detections, and model confidence,
              or a fallback error message dictionary.
    """
    client = get_hf_client()
    if client is None:
        return {"error": f"HF client unavailable: {get_hf_connect_error() or 'startup connection failed'}"}

    tmp_path = None
    _t_hf = time.time()
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        log.info("hf_call_start")
        file_arg = handle_file(tmp_path)

        def _attempt():
            return _call_with_timeout(
                client.predict, _HF_PREDICT_TIMEOUT_S, file_arg, api_name="/predict"
            )

        result = _retry_with_backoff(_attempt, max_retries=_HF_MAX_RETRIES, what="hf_predict")
        hf_ms = round((time.time() - _t_hf) * 1000)
        _cb_record_success()
        log.info("hf_call_ok", extra={"data": {"duration_ms": hf_ms, "phase": "hf"}})
        
        # Robustly decode and unwrap result
        parsed_result = result
        if isinstance(result, str):
            try:
                parsed_result = json.loads(result)
            except Exception as exc:
                log.warning("hf_result_json_parse_failed", extra={"data": {"error": str(exc)}}, exc_info=True)

        if isinstance(parsed_result, list) and len(parsed_result) > 0:
            parsed_result = parsed_result[0]

        if isinstance(parsed_result, dict):
            # Extract nested data if present
            if "data" in parsed_result:
                inner_data = parsed_result["data"]
                if isinstance(inner_data, (dict, list)):
                    parsed_result = inner_data
                    if isinstance(parsed_result, list) and len(parsed_result) > 0:
                        parsed_result = parsed_result[0]

        # Extract core detection labels or target values directly if wrapped
        if isinstance(parsed_result, dict) and "top_detection" not in parsed_result:
            if "label" in parsed_result:
                parsed_result = {
                    "success": parsed_result.get("success", True),
                    "low_confidence": parsed_result.get("low_confidence", False),
                    "is_low_confidence": parsed_result.get("is_low_confidence", False),
                    "phase": parsed_result.get("phase", 1),
                    "top_detection": parsed_result,
                    "all_detections": [parsed_result]
                }

        if isinstance(parsed_result, dict):
            return parsed_result
        return {"result": str(parsed_result)}
    except Exception as exc:
        log.error("hf_call_error", extra={"data": {
            "error_message": str(exc),
            "duration_ms":   round((time.time() - _t_hf) * 1000),
            "phase":         "hf",
        }}, exc_info=True)
        _cb_record_failure()
        return {"error": str(exc)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Chemical-advice liability disclaimer ──────────────────────────────────────
# The app serves chemical pesticide dosages (knowledge/chilli_kb.json,
# SYSTEM_PROMPT's "Targeted Chemical Interventions" section) that farmers act
# on in the field; wrong dosage/timing can damage crops or harm people, and
# pesticide recommendations carry regulatory weight in India. Any response
# that actually surfaces chemical-treatment content gets this disclaimer
# appended server-side — translations sourced from the same 5 languages the
# app already supports (see _SUPPORTED_LANGS / _LANG_INSTRUCTION_MAP above).
# NOTE: when the model replies in Telugu/Hindi/etc. it translates section
# headers and even chemical names too (transliterated into the local script),
# so English-only keyword matching misses real chemical content in non-English
# replies. Markers below cover en/te/hi (the languages this codebase already
# has confident native text for elsewhere — see _CHEMICAL_DISCLAIMER_TEXT).
# kn/ta markers are not included — same TODO as the disclaimer translations.
_CHEMICAL_CONTENT_MARKERS = (
    # English
    "targeted chemical interventions", "targeted inorganic regulation",
    "targeted inorganic", "pesticide", "fungicide", "insecticide",
    "neonicotinoid", "chemical spray", "active ingredient",
    # Telugu — రసాయన (chemical), కీటకనాశక (insecticide), శిలీంద్రనాశక/ఫంగిసైడ్ (fungicide),
    # పురుగు మందు (pesticide)
    "రసాయన", "కీటకనాశక", "శిలీంద్రనాశక", "ఫంగిసైడ్", "పురుగు మందు",
    # Hindi — रासायनिक/रसायन (chemical), कीटनाशक (pesticide/insecticide), फफूंदनाशक (fungicide)
    "रासायनिक", "रसायन", "कीटनाशक", "फफूंदनाशक",
)

def _mentions_chemical_treatment(text: str) -> bool:
    """
    Heuristic check for whether a generated reply includes chemical-treatment
    content, so the liability disclaimer is appended only when relevant (not
    on every plain conversational reply). Primary signal is the structured
    "Targeted Chemical Interventions" header the SYSTEM_PROMPT mandates for
    diagnosis responses; the keyword markers (en/te/hi) are a fallback for
    /chat's freer-form text, or for non-English replies whose translated
    header text wouldn't otherwise match.
    """
    if not text:
        return False
    lower = text.lower()
    return any(marker in lower for marker in _CHEMICAL_CONTENT_MARKERS)


# kn/ta are intentionally absent — left as a TODO for a human-reviewed
# translation rather than guessed. _get_chemical_disclaimer() falls back to
# the English string for any language not listed here.
_CHEMICAL_DISCLAIMER_TEXT = {
    "en": "\n\n---\n_This is general guidance, not a professional prescription. Before using any chemical, confirm the dosage and suitability with your local Krishi Vigyan Kendra (KVK) or agriculture officer. Organic methods are usually safer to try first._",
    "te": "\n\n---\n_ఇది సాధారణ సలహా మాత్రమే, నిపుణుల ప్రిస్క్రిప్షన్ కాదు. ఏదైనా రసాయన మందు వాడే ముందు, మోతాదు మరియు అనుకూలతను మీ స్థానిక కృషి విజ్ఞాన కేంద్రం (KVK) లేదా వ్యవసాయ అధికారిని సంప్రదించి నిర్ధారించుకోండి. వీలైతే ముందుగా సేంద్రియ పద్ధతులు ప్రయత్నించడం సురక్షితం._",
    "hi": "\n\n---\n_यह सामान्य सलाह है, विशेषज्ञ का नुस्खा नहीं। कोई भी रासायनिक दवा उपयोग करने से पहले, अपने स्थानीय कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी से मात्रा और उपयुक्तता की पुष्टि करें। संभव हो तो पहले जैविक तरीके आज़माना ज़्यादा सुरक्षित है।_",
    # TODO(kn): needs a human-reviewed Kannada translation of the English string above
    # TODO(ta): needs a human-reviewed Tamil translation of the English string above
}

def _get_chemical_disclaimer(lang: str) -> str:
    return _CHEMICAL_DISCLAIMER_TEXT.get(lang, _CHEMICAL_DISCLAIMER_TEXT["en"])


def _groq_stream_generator(messages, detection, is_low, lang="en", case_id=None, kb_treatment_block=None):
    """
    SSE generator for the /detect streaming response.

    Event frames (each separated by a blank line, prefixed with 'data: '):
      {"type": "meta",  "detection": {...}|null, "low_confidence": bool, "case_id": str|null}
          — First frame. Carries the detection-card data so the frontend can
            render the card before any text arrives. case_id (when present)
            is the just-created outcome-tracking case for this diagnosis --
            see _create_case; null means tracking was unavailable this
            request and the frontend should simply not show the save-case UI.
      {"type": "text",  "text": "<chunk>"}
          — One frame per Groq delta token.
      {"type": "done"}
          — Clean end-of-stream signal.
      {"type": "error", "error": "<message>"}
          — Mid-stream Groq error; client renders it inside the active bubble.

    kb_treatment_block (optional): pre-rendered treatment/dosage text built
    directly from chilli_kb.json (see _build_kb_treatment_block) -- used
    only by the triage-resolved path. When present it is appended as its
    OWN text frame(s) AFTER the LLM's stream finishes, not passed through
    Groq at all -- this is the hard-rule enforcement point ensuring
    dosage/treatment content is always KB-verbatim, never LLM-generated.
    The existing high-confidence path passes None here and is unaffected.
    """
    # ── Frame 0: metadata ─────────────────────────────────────────────────────
    meta = {"type": "meta", "detection": detection, "low_confidence": is_low, "case_id": case_id}
    yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

    # ── Frames 1-N: Groq token stream ─────────────────────────────────────────
    _t = time.time()
    try:
        # Single attempt, no retry — a retry after the stream has already
        # started yielding chunks to the client would duplicate content.
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            stream=True,
            timeout=_GROQ_STREAM_TIMEOUT_S,
        )
        full_text = []
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text.append(delta)
                payload = {"type": "text", "text": delta}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        if kb_treatment_block:
            full_text.append(kb_treatment_block)
            yield f"data: {json.dumps({'type': 'text', 'text': kb_treatment_block}, ensure_ascii=False)}\n\n"

        if _mentions_chemical_treatment("".join(full_text)):
            disclaimer = _get_chemical_disclaimer(lang)
            yield f"data: {json.dumps({'type': 'text', 'text': disclaimer}, ensure_ascii=False)}\n\n"

        log.info("groq_stream_done", extra={"data": {
            "duration_ms": round((time.time() - _t) * 1000),
        }})
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as exc:
        log.error("groq_stream_error", extra={"data": {"error_message": str(exc)}},
                  exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"


# ── Data Flywheel — shadow dataset helpers ────────────────────────────────────

def _shadow_save(image_bytes: bytes, label: str, confidence, trigger: str) -> None:
    """
    Write *image_bytes* into SHADOW_DIR for offline active-learning review.

    Filename schema:
        {trigger}_{confidence}_{label_slug}_{YYYYMMDD_HHMMSS}_{uuid6}.jpg

    Examples:
        low_conf_38_whitefly_20260521_143025_a1b2c3.jpg
        phase3_0_non_chilli_20260521_094512_d4e5f6.jpg

    All filesystem errors are caught and logged as warnings so they can
    never interrupt the HTTP response already in flight.
    """
    try:
        ts         = time.strftime("%Y%m%d_%H%M%S")
        slug       = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_") or "unknown"
        conf_str   = str(int(confidence)) if confidence else "0"
        uid        = uuid.uuid4().hex[:6]
        filename   = f"{trigger}_{conf_str}_{slug}_{ts}_{uid}.jpg"
        dest       = os.path.join(SHADOW_DIR, filename)

        with open(dest, "wb") as fh:
            fh.write(image_bytes)

        log.info("shadow_save_ok", extra={"data": {
            "filename":  filename,
            "bytes":     len(image_bytes),
            "trigger":   trigger,
            "label":     label,
            "conf":      confidence,
        }})
    except Exception as exc:
        # Intentionally silent — a disk error must never crash the request
        log.warning("shadow_save_fail", extra={"data": {
            "trigger": trigger,
            "error":   str(exc),
        }})


def _trigger_shadow_save(image_bytes: bytes, label: str,
                         confidence, trigger: str) -> None:
    """Submit _shadow_save to the global thread-pool instead of spawning a new daemon thread."""
    _shadow_executor.submit(_shadow_save, image_bytes, label, confidence, trigger)


# ── Outcome-tracking: case creation + outcome-tagged shadow save ─────────────
# A "case" is the small record created at the end of every diagnosis so a
# farmer can return later and report whether the treatment worked. See
# case_store.py for the storage interface + its ephemeral-disk production
# warning. Every function here is best-effort: a storage failure must never
# block, slow, or fail the diagnosis response that's already in flight.

def _create_case(domain: str, leader: str, lang: str) -> str | None:
    """
    Creates a case record for a just-completed diagnosis (pest/disease card
    or a triage-resolved deficiency). Returns the new case_id, or None if
    case-tracking is unavailable (no durable store configured, disk error,
    etc.) -- callers must treat None as "skip showing the save-this-case UI
    this time," never as an error to surface to the farmer.
    """
    try:
        if domain == "pest":
            entry = _get_kb_entry(leader)
            english, telugu_name, kind = detector._get_friendly_name(leader)
            display_name = f"{english} ({telugu_name})" if telugu_name else english
        else:
            entry = _CHILLI_KB.get("deficiencies", {}).get(leader, {})
            display_name = (entry or {}).get("display_name", leader)
            kind = "deficiency"

        treatment_summary = ""
        if entry and (entry.get("management") or {}).get("organic"):
            treatment_summary = entry["management"]["organic"][:280]

        for _ in range(3):  # retry on the (very unlikely) case_id collision
            candidate = case_store.generate_case_id()
            case = {
                "case_id":           candidate,
                "created_at":        time.time(),
                "domain":            domain,
                "leader":            leader,
                "kind":              kind,
                "display_name":      display_name,
                "treatment_summary": treatment_summary,
                "lang":              lang,
                "outcome":           None,
                "outcome_at":        None,
            }
            if CASE_STORE.create(case):
                return candidate
        log.warning("case_create_failed", extra={"data": {"domain": domain, "leader": leader}})
        return None
    except Exception as exc:
        log.warning("case_create_error", extra={"data": {"error": str(exc)}}, exc_info=True)
        return None


def _build_escalation_note(domain: str, leader: str, lang: str) -> str:
    """
    Plain-language nudge shown when a farmer reports the treatment isn't
    working ('Worse'/'Same'). Pest/disease cases point at the existing
    confirm-with-a-local-expert chemical framing; deficiency cases point at
    a soil test -- both reuse language already established elsewhere in the
    app rather than inventing new guidance.
    """
    if domain == "deficiency":
        notes = {
            "en": "This doesn't seem to be improving. Please get a soil test before adding any more fertiliser, and bring this result to your local Krishi Vigyan Kendra (KVK) or agriculture officer.",
            "te": "ఇది మెరుగుపడుతున్నట్లు లేదు. మరింత ఎరువు వేయడానికి ముందు మట్టి పరీక్ష చేయించండి, ఈ ఫలితాన్ని మీ స్థానిక కృషి విజ్ఞాన కేంద్రం (KVK) లేదా వ్యవసాయ అధికారికి చూపించండి.",
            "hi": "ऐसा लग रहा है कि सुधार नहीं हो रहा है। और खाद डालने से पहले मिट्टी की जांच करवाएं, और यह परिणाम अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी को दिखाएं।",
        }
    else:
        notes = {
            "en": "This doesn't seem to be improving with organic treatment. You may want to ask about a targeted chemical option -- please confirm the product and dose with your local Krishi Vigyan Kendra (KVK) or agriculture officer before using it.",
            "te": "సేంద్రీయ చికిత్సతో ఇది మెరుగుపడుతున్నట్లు లేదు. మీరు ఒక నిర్దిష్ట రసాయన ఎంపిక గురించి అడగవచ్చు -- ఉత్పత్తి మరియు మోతాదును ఉపయోగించే ముందు మీ స్థానిక కృషి విజ్ఞాన కేంద్రం (KVK) లేదా వ్యవసాయ అధికారితో నిర్ధారించుకోండి.",
            "hi": "जैविक उपचार से सुधार होता नहीं दिख रहा। आप एक लक्षित रासायनिक विकल्प के बारे में पूछ सकते हैं -- उपयोग से पहले उत्पाद और मात्रा की पुष्टि अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी से करें।",
        }
    return notes.get(lang) or notes["en"]


def _outcome_shadow_save(image_bytes: bytes, case: dict, outcome: str) -> None:
    """
    Saves a follow-up photo into the SAME shadow-dataset directory the
    existing low-confidence/triage capture paths already use, but with a
    JSON sidecar carrying the rich, LABELLED metadata that makes this
    different from a plain capture: the original diagnosis, the treatment
    given, days elapsed, and the farmer-reported real-world outcome. This is
    the data the self-heal pipeline (self_heal.py) can later be pointed at
    once a human curates a batch of these into telemetry_logs/ -- mirrors
    the existing image+.txt sidecar convention there, just with .json
    instead of YOLO labels since this is outcome metadata, not a bounding box.
    Does NOT trigger retraining itself -- capture/tag only, per design.
    """
    def _save():
        try:
            ts       = time.strftime("%Y%m%d_%H%M%S")
            slug     = re.sub(r"[^a-z0-9]+", "_", str(case.get("leader", "unknown")).lower()).strip("_") or "unknown"
            uid      = uuid.uuid4().hex[:6]
            basename = f"outcome_{outcome}_{slug}_{ts}_{uid}"
            img_path = os.path.join(SHADOW_DIR, f"{basename}.jpg")
            json_path = os.path.join(SHADOW_DIR, f"{basename}.json")

            with open(img_path, "wb") as fh:
                fh.write(image_bytes)

            created_at = case.get("created_at")
            days_elapsed = round((time.time() - created_at) / 86400, 2) if created_at else None
            metadata = {
                "case_id":           case.get("case_id"),
                "domain":            case.get("domain"),
                "diagnosis_label":   case.get("leader"),
                "diagnosis_display": case.get("display_name"),
                "treatment_summary": case.get("treatment_summary"),
                "diagnosed_at":      created_at,
                "outcome":           outcome,
                "outcome_at":        time.time(),
                "days_elapsed":      days_elapsed,
                "lang":              case.get("lang"),
            }
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, ensure_ascii=False, indent=2)

            log.info("outcome_shadow_save_ok", extra={"data": {
                "case_id": case.get("case_id"), "outcome": outcome, "days_elapsed": days_elapsed,
            }})
        except Exception as exc:
            log.warning("outcome_shadow_save_fail", extra={"data": {"error": str(exc)}})

    _shadow_executor.submit(_save)


SYSTEM_PROMPT = """IMPORTANT: Always respond in English unless the farmer writes in Telugu, Hindi, or Tamil first. Default language is English.

You are ChilliGuru, a friendly farming assistant for chilli farmers in Andhra Pradesh and Telangana. Talk like a trusted friend — simple, warm, easy to understand.

VARIETIES: Teja, Guntur Sannam, LCA 334, Wonder Hot, Pusa Jwala, Byadgi.
SEASONS: Kharif (Jun-Oct), Rabi (Nov-Feb), Zaid (Mar-May).
PESTS: Thrips, Spider Mites, Aphids, Whiteflies, Fruit Borer, Mealybugs, Leaf Miners, Armyworm.
DISEASES: Leaf Curl Virus, Powdery Mildew, Anthracnose, Damping Off, Phytophthora, Bacterial Wilt, Cercospora Leaf Spot, Bacterial Leaf Spot.

SOLUTION FORMAT:
  Solution name (Home-made OR Shop):
  How to make/use it: [simple steps]
  How well it works: X out of 10
  Days to see results: X-X days
  Cost: Rs X to Rs X
  How often: every X days for X weeks
  Where to get: [AP/Telangana]

When performing a plant diagnosis, structure your response strictly with these three headers:
### Climate-Pest Correlation Analysis
[Provide a brief analysis of the weather/climate factors correlated with this pest/disease pressure.
- You must explicitly analyze leaf curling directions and insect grouping patterns to confirm your findings.
- For leaf_curl_virus, you must explicitly identify "**Upward/Abaxial Cupping and Vein Thickening**" (ఆకు పైకి ముడుచుకోవడం).
- For broad_mites, you must explicitly identify "**Downward Inverted-Boat Curling**" (ఆకు కిందికి ముడుచుకోవడం).]

### Biological & Organic Interventions
[Provide 2-3 biological or organic solutions using the SOLUTION FORMAT above.
- You must use explicit biological sub-class targeting: recommend specific Bacterial vectors (such as Bacillus thuringiensis), Viral vectors (such as NPV blocks), or Fungal pathogens (such as Beauveria bassiana or others) tailored strictly to the diagnosed pest/disease lifecycle. Every recommendation must include specific biological sub-class vectors (Bt or NPV blocks).
- For viral profiles (such as Leaf Curl Virus), you must mandate organic recommendations targeting the whitefly vector using biological fungal spores (specifically Beauveria bassiana).
- For fungal/bacterial profile detections (Cercospora Leaf Spot, Bacterial Leaf Spot, Powdery Mildew), you must recommend organic treatments like Copper Hydroxide, Pseudomonas fluorescens, or systemic bio-agents.]

---
### Targeted Chemical Interventions
[Provide a brief overview of targeted chemical/inorganic alternatives. Provide chemical details (e.g., active ingredients) but advise biological/organic alternatives first since you are ChilliGuru.
- For viral profiles (such as Leaf Curl Virus), you must mandate recommendations targeting the whitefly vector using systemic chemical neonicotinoids (specifically Acetamiprid).
- For fungal/bacterial spot/mildew profile detections (Cercospora Leaf Spot, Bacterial Leaf Spot, Powdery Mildew), you must output a clear markdown table contrasting organic choices (specifically Copper Hydroxide or Pseudomonas fluorescens) with targeted chemical choices.
- Every chemical/active-ingredient you name must be framed as something to CONFIRM with a local agricultural expert before use — never as a direct, ready-to-apply prescription. Phrase it like "ask your local Krishi Vigyan Kendra (KVK) or agriculture officer to confirm the exact dosage for your field before applying [chemical name]," not as a standalone instruction to apply it. Do this in whatever language you are replying in.]

For every recommended solution/intervention (both biological/organic and chemical), you must feature an explicit "Cost-Effectiveness & Speed Evaluation Table" in markdown format. The table must detail:
- Intervention (name of the solution)
- **Estimated Cost per Acre (₹)**
- **Efficacy Speed** (e.g., 'Immediate 24hr knockdown' vs. '5-day systemic spread')
- Environmental Residual Protection (residual window, e.g., '7 days' or '14 days')

MARKDOWN LAYOUT COMPLIANCE:
- Any markdown tables MUST be isolated using a clean double newline (`\n\n`) before the primary header pipe (`|`) and immediately after the final row element to prevent rendering string crashes on the frontend.
- Restructure the prompt formatting to wrap key diagnostic outcomes strictly with standard markdown bolding (`**...**`).
- All comparative analysis and cost-effectiveness tables MUST be preceded and followed by a double blank line (`\\n\\n`) to ensure correct HTML rendering on mobile frontends.
- Forbid the use of inline text on the same line as a markdown table pipe separator (`|`). Every table row must terminate with a clean newline.
- Use markdown bolding (`**...**`) exclusively for key diagnostic outcomes and operational field metrics: **Estimated Cost per Acre (₹)**, **Efficacy Speed**, **Pathogen Classification**, and **Application Window**. Do not bold any other text.
- Require the use of clear horizontal markdown rules (`---`) to create visual boundaries between "Biological/Organic Interventions" and "Targeted Chemical Interventions".

End with one prevention tip.
ORGANIC ONLY (except when listing chemical details in Targeted Chemical Interventions). LANGUAGE: reply in the same language the user writes in."""

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/PRIVACY.md")
def privacy_policy():
    """Serves the data-privacy/retention policy linked from the UI notice bar."""
    return send_from_directory(
        os.path.dirname(os.path.abspath(__file__)), "PRIVACY.md", mimetype="text/markdown"
    )

@app.route("/health")
def health():
    """
    Cheap liveness/readiness check — reads only cached in-process state and
    never calls get_hf_client(), which would otherwise trigger a slow,
    network-bound Gradio connect attempt on the very first hit if prewarm
    hasn't completed yet. Safe for frequent load-balancer / uptime-monitor
    polling. See /health/deep for an active probe of HF + Groq.
    """
    return jsonify({
        "status": "ok",
        "hf_initialized": _hf_initialized,
        "hf_connected": _hf_initialized and _hf_client is not None,
        "hf_circuit_open": HF_CIRCUIT_OPEN,
        "groq_key_configured": bool(os.getenv("GROQ_API_KEY", "")),
    })


@app.route("/health/deep")
def health_deep():
    """
    Active probe — actually calls the HF Space (via get_hf_client(), which
    triggers the Gradio connect if not already initialized) and Groq (via a
    cheap models.list() call). Slower and side-effecting; intended for
    manual/ops checks, not frequent load-balancer polling.
    """
    hf_ok, hf_error = False, None
    try:
        client = get_hf_client()
        hf_ok = client is not None
        if not hf_ok:
            hf_error = get_hf_connect_error()
    except Exception as exc:
        hf_error = str(exc)

    groq_ok, groq_error = False, None
    try:
        get_client().models.list()
        groq_ok = True
    except Exception as exc:
        groq_error = str(exc)

    return jsonify({
        "status": "ok" if (hf_ok and groq_ok) else "degraded",
        "hf":   {"connected": hf_ok, "error": hf_error, "circuit_open": HF_CIRCUIT_OPEN},
        "groq": {"reachable": groq_ok, "error": groq_error},
    })

# ── Regional pest-risk rule table (built once at module load) ─────────────────
# Each pest has ordered (condition, level, description) tiers — the first tier
# whose condition matches the current temperature/humidity wins. Previously
# this logic lived as a ~170-line closure recreated on every request.
_RISK_LEVEL_ORDER = {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3}

_RISK_RULES = [
    ("invasive_black_thrips", "Invasive Black Thrips", "నల్ల తామర పురుగు", [
        (lambda t, h: 20.0 <= t <= 33.0 and h < 55.0, "Critical",
         "Warm and dry conditions are highly optimal for Invasive Black Thrips expansion."),
        (lambda t, h: 18.0 <= t <= 35.0 and h < 65.0, "High",
         "Favorable conditions for thrips activity. Monitor leaf undersides."),
        (lambda t, h: 15.0 <= t <= 38.0, "Moderate",
         "Moderate thrips activity. Keep field borders clean."),
    ]),
    ("aphids", "Aphids", "పేను పురుగు", [
        (lambda t, h: t < 26.0 and h > 65.0, "High",
         "Cooler temperatures and high humidity promote rapid aphid colonization."),
        (lambda t, h: t < 30.0 and h > 50.0, "Moderate",
         "Moderate risk of aphids. Look for ants or honey-dew deposits."),
    ]),
    ("whitefly_leaf_damage", "Whitefly", "తెల్ల ఈగ", [
        (lambda t, h: 26.0 <= t <= 38.0 and h > 75.0, "Critical",
         "Hot and humid microclimate triggers massive whitefly outbreak."),
        (lambda t, h: 24.0 <= t <= 40.0 and h > 60.0, "High",
         "High risk of whitefly migration. Yellow sticky traps recommended."),
        (lambda t, h: 20.0 <= t <= 42.0, "Moderate",
         "Moderate whitefly presence. Inspect shoots regularly."),
    ]),
    ("broad_mites", "Broad Mites", "ఎర్ర సాలె పురుగు", [
        (lambda t, h: t > 33.0 and h < 45.0, "Critical",
         "Very hot and dry weather causes rapid broad mite infestation cycles."),
        (lambda t, h: t > 30.0 and h < 55.0, "High",
         "High temperature and dry wind favor mite propagation."),
        (lambda t, h: t > 25.0, "Moderate",
         "Moderate risk. Overhead irrigation can suppress mite build-up."),
    ]),
    ("fruit_borer", "Fruit Borer", "పండు తొలిచే పురుగు", [
        (lambda t, h: 24.0 <= t <= 35.0 and h > 65.0, "High",
         "Warm, humid conditions speed up egg-hatching and fruit borer damage."),
        (lambda t, h: 20.0 <= t <= 38.0, "Moderate",
         "Moderate risk of fruit borer. Check for bored entry holes in fruits."),
    ]),
    ("tobacco_caterpillar", "Tobacco Caterpillar", "గొంగళి పురుగు", [
        (lambda t, h: 25.0 <= t <= 36.0 and h > 70.0, "High",
         "High humidity and temperature increase risk of Spodoptera caterpillar activity."),
        (lambda t, h: 22.0 <= t <= 38.0, "Moderate",
         "Moderate threat. Watch for skeletonized leaf patches."),
    ]),
    ("yellow_thrips", "Yellow Thrips", "తామర పురుగు", [
        (lambda t, h: 20.0 <= t <= 32.0 and h < 60.0, "High",
         "Favorable dry temperature range for yellow thrips feeding on new leaves."),
        (lambda t, h: 18.0 <= t <= 36.0, "Moderate",
         "Moderate risk. Upward leaf curling might begin."),
    ]),
    ("mealybugs", "Mealybugs", "పిండి పురుగు", [
        (lambda t, h: t > 25.0 and h > 60.0, "High",
         "Warmth and humidity favor white cottony mealybug cluster formation."),
        (lambda t, h: t > 20.0, "Moderate",
         "Moderate risk. Prune heavily infested shoots."),
    ]),
]


def calculate_risks(t, h):
    """
    Evaluates current environmental risk levels for various chilli pests/diseases 
    based on local temperature and relative humidity conditions.
    
    Args:
        t (float): Temperature in degrees Celsius.
        h (float): Relative humidity percentage.
        
    Returns:
        list of dicts: Sorted list of active pest risks, ordered by severity 
                       (Critical, High, Moderate, Low).
    """
    risks = []
    for pest, label, telugu, tiers in _RISK_RULES:
        for cond, level, description in tiers:
            if cond(t, h):
                risks.append({
                    "pest": pest,
                    "label": label,
                    "telugu": telugu,
                    "level": level,
                    "description": description,
                })
                break
    risks.sort(key=lambda r: _RISK_LEVEL_ORDER.get(r["level"], 4))
    return risks


@app.route("/api/regional-risk", methods=["GET", "POST"])
def regional_risk():
    """
    Flask route to compute the regional pest and disease risk levels for the user's location
    and surrounding coordinate grids.

    Fetches real-time weather metrics (temperature and relative humidity) from the
    Open-Meteo API using HTTP session connection pooling. Simulates coordinates for
    neighboring watchpoints (East, North, South-West) to provide context for localized spreads.

    Privacy note: prefer POST with a JSON body — coordinates in a request body
    never land in this server's access logs or browser history, unlike query
    strings. GET with ?lat=&lon= is kept only for backward compatibility with
    existing callers/tests; the bundled frontend (static/js/app.js, index.html)
    uses POST exclusively. Coordinates here are coarse (browser geolocation or
    a fixed regional default) and are never persisted or linked to a stored
    photo — see PRIVACY.md.

    POST JSON Body / GET Query Params:
        lat (float, optional): Latitude coordinate. Defaults to 16.5 (Guntur region).
        lon (float, optional): Longitude coordinate. Defaults to 79.5 (Guntur region).

    Returns:
        Response: JSON array containing temperature, humidity, and calculated pest risks
                  for the centroid and each of the three watchpoints.
    """
    payload = request.get_json(silent=True) or {}
    try:
        lat = float(payload.get("lat", request.args.get("lat")))
        lon = float(payload.get("lon", request.args.get("lon")))
    except (TypeError, ValueError):
        lat = 16.5
        lon = 79.5

    temp = 28.0
    humidity = 60.0
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        resp = _http_session.get(url, timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            current = data.get("current", {})
            if "temperature_2m" in current:
                temp = float(current["temperature_2m"])
            if "relative_humidity_2m" in current:
                humidity = float(current["relative_humidity_2m"])
    except Exception as e:
        log.warning("open_meteo_api_error", extra={"data": {"error": str(e)}})

    locations = []
    locations.append({
        "latitude": lat,
        "longitude": lon,
        "name": "Requested Field Centroid",
        "temperature": temp,
        "humidity": humidity,
        "risks": calculate_risks(temp, humidity)
    })
    locations.append({
        "latitude": lat,
        "longitude": lon + 0.05,
        "name": "East Regional Watchpoint",
        "temperature": round(temp + 2.0, 1),
        "humidity": round(max(5.0, humidity - 15.0), 1),
        "risks": calculate_risks(temp + 2.0, max(5.0, humidity - 15.0))
    })
    locations.append({
        "latitude": lat + 0.05,
        "longitude": lon,
        "name": "North Regional Watchpoint",
        "temperature": round(temp - 2.0, 1),
        "humidity": round(min(99.0, humidity + 15.0), 1),
        "risks": calculate_risks(temp - 2.0, min(99.0, humidity + 15.0))
    })
    locations.append({
        "latitude": lat - 0.04,
        "longitude": lon - 0.04,
        "name": "South-West Regional Station",
        "temperature": round(temp + 1.0, 1),
        "humidity": round(min(99.0, humidity + 5.0), 1),
        "risks": calculate_risks(temp + 1.0, min(99.0, humidity + 5.0))
    })
    return jsonify(locations)

@app.route("/chat", methods=["POST"])
@limiter.limit("15 per minute")
def chat():
    """
    Flask route handling text-only conversational assistance for chilli farmers.
    
    Sanitizes user messages to remove cross-language contamination scripts, resolves the 
    preferred user language, and coordinates system prompting to answer farming and nutrient 
    deficiency questions using Groq's Llama models.
    
    JSON Request Body:
        message (str): User text input query.
        history (list, optional): Previous chat message objects for multi-turn conversational context.
        lang (str, optional): Overridden target language identifier.
        
    Returns:
        Response: JSON object containing 'reply' text or an error message.
    """
    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()
        history = data.get("history", [])
        if not message:
            return jsonify({"error": "No message"}), 400

        # Sub-module 1: resolve language (explicit tag fast-path or script scan)
        lang = _resolve_request_language(message)
        # JSON payload 'lang' field is highest-priority override
        payload_lang = data.get("lang", "").strip().lower()
        if payload_lang:
            tag = re.split(r'[-,;]', payload_lang)[0].strip()
            if tag in _SUPPORTED_LANGS:
                lang = tag  # early-return fast-path — no further script scanning needed

        system_content = strip_cross_contamination(SYSTEM_PROMPT, lang)
        system_content += _LANG_INSTRUCTION_MAP.get(lang, "\nIMPORTANT: You must respond in English.")
        system_content += _DEFICIENCY_SUMMARY
        system_content = strip_cross_contamination(system_content, lang)
        message_clean  = strip_cross_contamination(message, lang)

        messages = (
            [{"role": "system", "content": system_content}]
            + history
            + [{"role": "user", "content": message_clean}]
        )
        def _groq_call():
            return get_client().chat.completions.create(
                model=MODEL, messages=messages, max_tokens=MAX_TOKENS,
                temperature=0.7, timeout=_GROQ_NONSTREAM_TIMEOUT_S,
            )

        response = _retry_with_backoff(
            _groq_call, max_retries=_GROQ_NONSTREAM_MAX_RETRIES, what="groq_chat"
        )
        reply = response.choices[0].message.content.strip()
        if _mentions_chemical_treatment(reply):
            reply += _get_chemical_disclaimer(lang)
        return jsonify({"reply": reply})
    except HTTPException:
        # e.g. RequestEntityTooLarge from a body over MAX_CONTENT_LENGTH —
        # let it propagate to the matching @app.errorhandler instead of being
        # flattened into a generic 500 below.
        raise
    except Exception as e:
        log.error("chat_error", extra={"data": {"error_message": str(e)}}, exc_info=True)
        return jsonify({"error": str(e), "request_id": _current_request_id()}), 500

@app.route("/detect", methods=["POST"])
@limiter.limit("5 per minute")
def detect():
    """
    Flask route serving visual diagnosis requests from uploaded farmer photos.
    
    This endpoint parses image attachments, invokes validation check constraints, 
    triggers the remote HF vision classification model (or local fallbacks), applies 
    field-context priors, and streams real-time LLM treatment diagnoses back to the client 
    using Server-Sent Events (SSE).
    
    Multipart/Form Request Body:
        image (file, optional): Uploaded image file (max 5 MB JPEG/PNG/WebP).
        message (str, optional): User description notes or question context.
        history (str, JSON-serialized list, optional): Multi-turn conversation context.
        plant_age (str, optional): Stage selector ('seedling', 'vegetative', 'flowering', 'fruiting').
        observed (str, optional): Symptom selector.
        affected_part (str, optional): Affected plant part selector.
        curl_dir (str, optional): Leaf curling direction ('upward', 'downward', 'none').
        
    Returns:
        Response: Server-Sent Events stream containing metadata followed by token chunks, 
                  or standard JSON response block on initial validation errors.
    """
    _t0 = time.time()
    try:
        resp = _detect_inner()
        status_code = resp[1] if isinstance(resp, tuple) else resp.status_code
        log.info("detect_complete", extra={"data": {
            "duration_ms": round((time.time() - _t0) * 1000),
            "status_code": status_code,
        }})
        return resp
    except HTTPException:
        # Let Flask's normal dispatch handle these (e.g. RequestEntityTooLarge
        # from a body over MAX_CONTENT_LENGTH) so they hit the matching
        # @app.errorhandler and keep their real status code instead of being
        # flattened into a generic 500 below.
        raise
    except Exception as e:
        log.error("detect_unhandled_error", extra={"data": {
            "error_message": str(e),
            "duration_ms":   round((time.time() - _t0) * 1000),
        }}, exc_info=True)
        return jsonify({"success": False, "error": "Internal Processing Error", "request_id": _current_request_id()}), 500


@app.route("/detect/triage-answer", methods=["POST"])
@limiter.limit("20 per minute")
def detect_triage_answer():
    """
    Handles one turn of the adaptive triage Q&A loop started by /detect's
    {"triage": {...}} response (see _triage_ask_payload). Applies the
    farmer's answer to re-rank the candidate scores, then either asks the
    next question (JSON, same shape as /detect's triage payload -- Groq-
    authored if available, falling back to knowledge/triage_rules.json) or,
    once a candidate is clearly ahead or the question budget is exhausted,
    asks Groq to make the authoritative resolution call
    (_groq_resolve_diagnosis) and streams the final KB-governed card.

    JSON Request Body:
        state (dict): The opaque triage state echoed back from the previous
            /detect or /detect/triage-answer response (domain, candidates,
            asked, lang, user_msg, morphology, last_question, qa_transcript).
        question_id (str): The id of the question being answered (matches state).
        answer_key (str): The option key the farmer picked.
        history (list, optional): Conversation history for the final Groq call.
    """
    try:
        data = request.get_json(silent=True) or {}
        state = data.get("state") if isinstance(data.get("state"), dict) else {}
        answer_key = str(data.get("answer_key", "")).strip()
        history = data.get("history") if isinstance(data.get("history"), list) else []

        domain = state.get("domain")
        lang = state.get("lang") if state.get("lang") in _SUPPORTED_LANGS else "en"
        user_msg = str(state.get("user_msg") or "")
        morphology = state.get("morphology") if isinstance(state.get("morphology"), dict) else None
        asked = [a for a in (state.get("asked") or []) if isinstance(a, str)]
        qa_transcript = [
            list(pair) for pair in (state.get("qa_transcript") or [])
            if isinstance(pair, list) and len(pair) == 2
        ]

        if domain not in ("pest", "deficiency"):
            return jsonify({"error": "Invalid or expired triage state"}), 400

        # Defensive: only accept known candidate ids with numeric scores —
        # rejects a tampered/malformed state instead of trusting client input.
        valid_ids = triage.PEST_CANDIDATE_IDS if domain == "pest" else triage.DEFICIENCY_CANDIDATE_IDS
        raw_candidates = state.get("candidates") if isinstance(state.get("candidates"), dict) else {}
        candidates = {
            k: float(v) for k, v in raw_candidates.items()
            if k in valid_ids and isinstance(v, (int, float))
        }
        if not candidates:
            return jsonify({"error": "Invalid or expired triage state"}), 400

        question_id = data.get("question_id") or (asked[-1] if asked else None)
        # Prefer the full question object stashed in state at ask-time (this is
        # the ONLY place a Groq-authored "dyn_*" question can be found — it was
        # never written to knowledge/triage_rules.json). Fall back to the
        # static rulebook lookup for backward-compat / purely-static turns.
        last_question = state.get("last_question") if isinstance(state.get("last_question"), dict) else None
        if last_question and last_question.get("id") == question_id:
            question = last_question
        else:
            rules = triage.load_rules()
            pool = rules["pest_questions"] if domain == "pest" else rules["deficiency_questions"]
            question = next((q for q in pool if q["id"] == question_id), None)

        if question is not None and answer_key:
            candidates = triage.apply_answer(question, answer_key, candidates)
            if question_id not in asked:
                asked.append(question_id)
            # Record this turn for _groq_resolve_diagnosis's transcript.
            chosen_opt = next((o for o in question["options"] if o["key"] == answer_key), None)
            q_text = question["question"].get(lang) or question["question"].get("en", "")
            a_text = (chosen_opt["label"].get(lang) or chosen_opt["label"].get("en", "")) if chosen_opt else answer_key
            qa_transcript.append([q_text, a_text])

        # ── All-unsure safety net ─────────────────────────────────────────
        # If EVERY answer so far was "Not sure / skip" (no scores ever
        # moved), we can't separate candidates at all.  Rather than forcing
        # a guess, recommend the farmer consult a local expert.
        answer_keys_so_far = [state.get("answer_keys", []), [answer_key]] if answer_key else [state.get("answer_keys", [])]
        all_keys = [k for sublist in answer_keys_so_far for k in (sublist if isinstance(sublist, list) else [sublist]) if k]
        state_answer_keys = all_keys  # will be threaded through in state below

        resolution = triage.check_resolution(domain, candidates, len(asked))

        if not resolution["resolved"]:
            ask = _triage_ask_payload(domain, candidates, asked, lang, user_msg,
                                       morphology=morphology, qa_transcript=qa_transcript)
            if ask is not None:
                # Thread the answer_keys list through state so the all-unsure
                # check works across round-trips.
                ask["triage"]["state"]["answer_keys"] = state_answer_keys
                return jsonify(ask)
            # Neither Groq nor the rulebook has a further question —
            # resolve now with whoever currently leads (ambiguous fallback).
            resolution = {"resolved": True, "leader": resolution["leader"], "ambiguous": True}

        # If every single answer was "not_sure", no scores moved at all —
        # candidates can't be separated.  Show KVK referral instead of
        # forcing a confident pick from non-answers.
        if state_answer_keys and all(k == "not_sure" for k in state_answer_keys):
            log.info("triage_all_unsure", extra={"data": {"domain": domain, "questions_asked": len(asked)}})
            return _no_kb_match_response_with_text(_ALL_UNSURE_TEXT, lang)

        # ── Authoritative resolution: Groq picks from the candidate set, the
        # deterministic score-leader is the fallback if Groq is unavailable
        # or returns anything outside the candidate set. ────────────────────
        groq_resolution = _groq_resolve_diagnosis(domain, candidates, qa_transcript, lang)
        if groq_resolution is not None:
            final_leader, final_ambiguous = groq_resolution["leader"], groq_resolution["ambiguous"]
            resolved_via = "groq"
        else:
            final_leader, final_ambiguous = resolution["leader"], True
            resolved_via = "deterministic_fallback"

        # Hard safety net: the resolved candidate MUST have an actual
        # chilli_kb.json entry (treatment/dosage content has nowhere else to
        # come from). If not, try the next-best KB-mapped candidate by score;
        # if none of the live candidates map to the KB at all, recommend a
        # local expert rather than ever building a card with invented content.
        if final_leader is None or not _kb_entry_exists(domain, final_leader):
            kb_mapped = {k: v for k, v in candidates.items() if _kb_entry_exists(domain, k)}
            if not kb_mapped:
                log.info("triage_no_kb_match", extra={"data": {"domain": domain, "candidates": list(candidates.keys())}})
                return _no_kb_match_response(lang)
            final_leader, final_ambiguous = max(kb_mapped, key=kb_mapped.get), True

        log.info("triage_resolved", extra={"data": {
            "domain": domain, "leader": final_leader, "ambiguous": final_ambiguous,
            "questions_asked": len(asked), "resolved_via": resolved_via,
        }})
        return _triage_resolved_response(
            domain, final_leader, final_ambiguous, lang, user_msg, history, morphology=morphology
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("triage_answer_error", extra={"data": {"error_message": str(e)}}, exc_info=True)
        return jsonify({"error": "Internal Processing Error", "request_id": _current_request_id()}), 500


# ── Outcome-tracking: case recovery + follow-up (see case_store.py) ─────────
_VALID_OUTCOMES = frozenset(["better", "same", "worse", "not_sure"])

# Plain-language follow-up screen strings. kn/ta intentionally left as TODO
# (fallback to English), matching the established pattern in static/js/app.js's
# ADVICE_DISCLAIMER_TEXT / PRIVACY_NOTICE_TEXT and knowledge/triage_rules.json.
CASE_PAGE_TEXT = {
    "title": {
        "en": "How is your crop now?", "te": "ఇప్పుడు మీ పంట ఎలా ఉంది?",
        "hi": "अब आपकी फसल कैसी है?", "kn": None, "ta": None,
    },
    "options": {
        "better":   {"en": "Better",   "te": "మెరుగైంది",  "hi": "बेहतर है",   "kn": None, "ta": None},
        "same":     {"en": "Same",     "te": "అలాగే ఉంది", "hi": "वैसी ही है", "kn": None, "ta": None},
        "worse":    {"en": "Worse",    "te": "అధ్వాన్నంగా ఉంది", "hi": "बिगड़ गई है", "kn": None, "ta": None},
        "not_sure": {"en": "Not sure", "te": "తెలియదు",    "hi": "पता नहीं",   "kn": None, "ta": None},
    },
    "upload_prompt": {
        "en": "Optional: add a new photo", "te": "ఐచ్ఛికం: కొత్త ఫోటో జతచేయండి",
        "hi": "वैकल्पिक: नई फोटो जोड़ें", "kn": None, "ta": None,
    },
    "submit": {
        "en": "Send", "te": "పంపండి", "hi": "भेजें", "kn": None, "ta": None,
    },
    "thanks": {
        "en": "Thank you! This helps us and other farmers.", "te": "ధన్యవాదాలు! ఇది మీకు మరియు ఇతర రైతులకు సహాయపడుతుంది.",
        "hi": "धन्यवाद! इससे आपकी और अन्य किसानों की मदद होती है।", "kn": None, "ta": None,
    },
    "not_found": {
        "en": "We couldn't find this case. It may have expired.",
        "te": "ఈ కేసు కనుగొనబడలేదు. ఇది గడువు ముగిసి ఉండవచ్చు.",
        "hi": "यह केस नहीं मिला। यह समय सीमा समाप्त हो सकता है।", "kn": None, "ta": None,
    },
}


@app.route("/case/<case_id>", methods=["GET"])
def case_page(case_id):
    """
    Serves the standalone follow-up/recovery screen. The page itself is a
    small static HTML+JS file (static/case.html) -- it reads the case_id out
    of the URL client-side and calls the JSON APIs below. Kept as its own
    page rather than folded into the main chat SPA so a farmer can open the
    saved link directly (e.g. from a messaging app) without needing the rest
    of the app's state.
    """
    return send_from_directory("static", "case.html")


@app.route("/api/case/<case_id>", methods=["GET"])
def api_case_get(case_id):
    """Returns the saved case record (diagnosis + treatment summary) for the follow-up screen."""
    case = CASE_STORE.get(case_id)
    if case is None:
        return jsonify({"found": False, "text": CASE_PAGE_TEXT}), 404
    # Strip nothing personal is stored to begin with, but only return the
    # fields the follow-up screen actually needs.
    return jsonify({
        "found": True,
        "case": {
            "case_id":           case.get("case_id"),
            "domain":            case.get("domain"),
            "display_name":      case.get("display_name"),
            "treatment_summary": case.get("treatment_summary"),
            "lang":              case.get("lang", "en"),
            "created_at":        case.get("created_at"),
            "outcome":           case.get("outcome"),
        },
        "text": CASE_PAGE_TEXT,
    })


@app.route("/api/case/<case_id>/outcome", methods=["POST"])
@limiter.limit("10 per minute")
def api_case_outcome(case_id):
    """
    Records the farmer's reported outcome for a case, and — the flywheel
    part — if a follow-up photo is attached, saves it into the existing
    shadow dataset tagged with the original diagnosis/treatment/outcome (see
    _outcome_shadow_save). If the outcome is 'worse' or 'same', returns an
    escalation note (existing chemical-confirm-with-expert framing for
    pests, soil-test framing for deficiencies) for the page to display.

    Best-effort throughout: a storage failure here must never 500 in a way
    that looks broken to the farmer -- it degrades to "thanks, recorded"
    even if the underlying write silently failed, since there is nothing
    actionable the farmer can do about a server-side storage problem.
    """
    try:
        outcome = (request.form.get("outcome") or "").strip().lower()
        if outcome not in _VALID_OUTCOMES:
            return jsonify({"error": "Invalid outcome"}), 400

        case = CASE_STORE.get(case_id)
        if case is None:
            return jsonify({"error": "Case not found", "text": CASE_PAGE_TEXT}), 404

        CASE_STORE.update(case_id, outcome=outcome, outcome_at=time.time())

        image_file = request.files.get("image")
        if image_file:
            image_file.seek(0, 2)
            file_size = image_file.tell()
            image_file.seek(0)
            if file_size <= MAX_IMAGE_BYTES:
                header = image_file.read(12)
                image_file.seek(0)
                is_webp = header[:4] == b'RIFF' and header[8:12] == b'WEBP'
                is_known = is_webp or any(header.startswith(sig) for sig in _IMAGE_SIGNATURES if sig != b'RIFF')
                if is_known:
                    image_bytes = image_file.read()
                    if image_bytes:
                        _outcome_shadow_save(image_bytes, case, outcome)
            else:
                log.warning("case_outcome_photo_too_large", extra={"data": {"size_bytes": file_size}})

        lang = case.get("lang", "en")
        escalate = outcome in ("worse", "same")
        escalation_note = _build_escalation_note(case.get("domain", "pest"), case.get("leader", ""), lang) if escalate else None

        log.info("case_outcome_recorded", extra={"data": {
            "case_id": case_id, "outcome": outcome, "had_photo": bool(image_file), "escalate": escalate,
        }})
        return jsonify({
            "recorded": True,
            "escalate": escalate,
            "escalation_note": escalation_note,
            "text": CASE_PAGE_TEXT,
        })
    except HTTPException:
        raise
    except Exception as e:
        log.error("case_outcome_error", extra={"data": {"error_message": str(e)}}, exc_info=True)
        # Fail soft -- the farmer's report itself is low-stakes; never show a scary 500 for this.
        return jsonify({"recorded": False, "error": "Could not record outcome right now", "text": CASE_PAGE_TEXT}), 200


# ─────────────────────────────────────────────────────────────────────────────
# Decoupled sub-functions extracted from _detect_inner()
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORTED_LANGS = frozenset(["en", "hi", "te", "kn", "ta"])
_LANG_INSTRUCTION_MAP = {
    "en": "\nIMPORTANT: You must respond in English.",
    "hi": "\nIMPORTANT: You must respond in Hindi (हिंदी).",
    "te": "\nIMPORTANT: You must respond in Telugu (తెలుగు).",
    "kn": "\nIMPORTANT: You must respond in Kannada (ಕನ್ನಡ).",
    "ta": "\nIMPORTANT: You must respond in Tamil (தமிழ்).",
}


def _resolve_request_language(user_msg: str) -> str:
    """
    Resolves the target language for the incoming HTTP request.
    
    Checks the request sources in order of priority:
      1. Explicit 'lang' request headers, form parameters, or query parameters.
      2. 'Accept-Language' header.
      3. Unicode script auto-detection (using pre-compiled regex patterns for Telugu,
         Hindi, Kannada, and Tamil) on the user's message body.
      4. Default → English ('en').
      
    Args:
        user_msg (str): The raw input message text from the user.
        
    Returns:
        str: The resolved two-letter language code (e.g., 'en', 'te', 'hi', 'kn', 'ta').
    """
    # Priority 1 & 2 — explicit tag sources
    raw = (
        request.headers.get("lang", "").strip().lower()
        or request.headers.get("Accept-Language", "").strip().lower()
        or request.form.get("lang", "").strip().lower()
        or request.args.get("lang", "").strip().lower()
    )
    if raw:
        tag = re.split(r'[-,;]', raw)[0].strip()
        if tag in _SUPPORTED_LANGS:
            # Fast-path: definitive explicit code — no script scanning needed
            return tag

    # Priority 3 — Unicode script heuristic (uses pre-compiled patterns)
    if _SCRIPT_RE["te"].search(user_msg):
        return "te"
    if _SCRIPT_RE["hi"].search(user_msg):
        return "hi"
    if _SCRIPT_RE["kn"].search(user_msg):
        return "kn"
    if _SCRIPT_RE["ta"].search(user_msg):
        return "ta"

    # Priority 4 — default
    return "en"


def _execute_guardrail_check(image_bytes: bytes, result: dict) -> dict | None:
    """
    Applies guardrail checks and out-of-domain result filters on Hugging Face predictions.
    
    If the image is detected as 'non_chilli' (non-chilli crop/out-of-domain input), it throws 
    a 422 Unprocessable Entity HTTP response. If a low-level guardrail failure is detected,
    or the prediction fails, it returns None to signal a fallback to the local YOLO cascade.
    
    Args:
        image_bytes (bytes): The raw uploaded image file bytes.
        result (dict): The parsed prediction payload from Hugging Face.
        
    Returns:
        dict | None: The cleaned prediction dictionary if valid, or None if the local cascade 
                     inference should run.
                     
    Raises:
        _GuardrailReject: Carrying an HTTP 422 response if the crop is out-of-domain.
    """
    response_data = None
    if isinstance(result, dict):
        if "error" in result:
            response_data = result["error"]
        elif "label" in result:
            response_data = result["label"]
        elif "top_detection" in result and result["top_detection"]:
            td = result["top_detection"]
            response_data = td.get("label") if isinstance(td, dict) else td
    elif isinstance(result, (list, tuple)) and result:
        response_data = result[0]
    else:
        response_data = result

    label = str(response_data).strip().lower() if response_data is not None else ""

    # Confidence extraction
    conf_val = None
    if isinstance(result, dict):
        conf_val = result.get("confidence")
        if conf_val is None and isinstance(result.get("top_detection"), dict):
            conf_val = result["top_detection"].get("confidence")
    try:
        conf = float(conf_val) if conf_val is not None else 0.0
    except Exception as exc:
        log.warning("confidence_parse_failed", extra={"data": {"conf_val": str(conf_val), "error": str(exc)}}, exc_info=True)
        conf = None

    # Hotfix gasket
    if (label == "non_chilli" and conf == 0) or conf is None:
        return None  # force local cascade with bypass_ood=True

    # Hard out-of-domain rejection
    if label == "non_chilli":
        log.warning("out_of_domain_crop")
        raise _GuardrailReject(
            jsonify({
                "error": "Cannot identify crop. Please upload a clear photo of a chilli plant.",
                "request_id": _current_request_id()
            }), 422
        )

    # Guardrail / phase-3 rejection → push to shadow dataset, fall through to local
    is_guardrail = (
        label == "0"
        or (isinstance(result, dict) and (result.get("success") is False or result.get("phase") == 3))
    )
    if is_guardrail:
        _trigger_shadow_save(image_bytes, label="non_chilli", confidence=0, trigger="phase3")
        log.info("guardrail_rejection_bypass_to_local_cascade")
        return None

    if isinstance(result, dict) and "error" in result:
        log.warning("hf_result_error_fallback", extra={"data": {"error_message": result["error"]}})
        return None

    return result


class _GuardrailReject(Exception):
    """Internal sentinel: carry a Flask response tuple out of _execute_guardrail_check."""
    def __init__(self, response, status):
        self.response = response
        self.status   = status


def _compile_groq_payload(
    result: dict,
    top: dict | None,
    is_low: bool,
    lang: str,
    user_msg: str,
    image_bytes: bytes | None,
) -> str:
    """
    Compiles the final context instruction string to send to the Groq LLM model, 
    based on the result of the vision classifications and context variables.
    
    If a confident detection is made, extracts details (confidence, leaf morphology,
    matching agronomy entries) and compiles detailed system guidelines. If no confident 
    detection is made, creates a fallback instructions string instructing the LLM 
    to initiate "questioning mode" to gather more facts from the farmer.
    
    Args:
        result (dict): The complete classification output dictionary.
        top (dict | None): The primary identified detection object.
        is_low (bool): Flag indicating if detection confidence is below the low-confidence threshold.
        lang (str): The target language code.
        user_msg (str): The farmer's input query.
        image_bytes (bytes | None): Raw image data, used to trigger active learning storage on low confidence.
        
    Returns:
        str: Compiled instruction string containing structured agronomic context for the LLM.
    """
    lang_names = {"en": "English", "hi": "Hindi", "te": "Telugu",
                  "kn": "Kannada", "ta": "Tamil"}
    low_note_lang_hints = {
        "en": "not fully certain",
        "hi": "पूरी तरह से आश्वस्त नहीं",
        "te": "పూర్తిగా ఖచ్చితంగా తెలియదు",
        "kn": "ಖಚಿತವಾಗಿಲ್ಲ",
        "ta": "முழுமையாக உறுதியாக தெரியவில்லை",
    }

    if top and not result.get("error"):
        raw_label_name = top.get("raw_label") or top.get("label", "")
        core_cls = to_core_class(raw_label_name)

        if core_cls:
            english, telugu_name, kind = detector._get_friendly_name(core_cls)
            top["raw_label"] = english
            top["telugu"]    = telugu_name
            top["label"]     = f"{english} [{telugu_name}]" if telugu_name else english
            top["type"]      = kind

        label      = top.get("label", "unknown pest")
        confidence = top.get("confidence", 0)
        kind       = top.get("type", "pest")

        # Regional translation lookup
        translated_label = label
        if core_cls and core_cls in REGIONAL_TRANSLATION_MAP:
            translated_label = REGIONAL_TRANSLATION_MAP[core_cls].get(
                lang, REGIONAL_TRANSLATION_MAP[core_cls]["en"]
            )

        translated_label_clean = strip_cross_contamination(translated_label, lang)
        user_msg_clean         = strip_cross_contamination(user_msg, lang)

        target_lang_name = lang_names.get(lang, "English")
        low_note_hint    = low_note_lang_hints.get(lang, "not fully certain")

        if is_low and image_bytes:
            _trigger_shadow_save(image_bytes, label=label, confidence=confidence, trigger="low_conf")

        low_note = (
            f"\nNOTE: This is a low-confidence detection. "
            f"Mention to the farmer that you are {low_note_hint} and ask one short clarifying question."
        ) if is_low else ""

        # ── Leaf morphology cues (curl direction + leaf-age) ──────────────────
        morph = top.get("morphology") if isinstance(top, dict) else None
        morph_note = ""
        if isinstance(morph, dict) and morph.get("curl_direction") not in (None, "unknown"):
            curl_dir = morph.get("curl_direction")
            curl_map = {
                "upward":   "UPWARD/abaxial cupping (a Leaf Curl Virus signature — vector is whitefly)",
                "downward": "DOWNWARD inverted-boat curling (a Broad Mite feeding signature)",
                "flat":     "FLAT — no strong curl signature",
            }
            morph_note += (
                f"\n=== LEAF MORPHOLOGY ANALYSIS (image-based) ===\n"
                f"Leaf curl: {curl_map.get(curl_dir, curl_dir)} "
                f"(curl confidence {morph.get('curl_confidence')}%).\n"
                f"Leaf-age stage: {morph.get('leaf_age')} "
                f"(confidence {morph.get('age_confidence')}%).\n"
            )
            if morph.get("juvenile_mimicry"):
                morph_note += (
                    "IMPORTANT: This leaf is structurally mature but shows juvenile size/colour "
                    "(juvenile mimicry) — a stunting/distortion pattern typical of early viral "
                    "infection. Weigh Leaf Curl Virus more heavily and explain this cue simply.\n"
                )
            morph_note += (
                "INSTRUCTION: Explicitly reference the curl direction and leaf-age finding in your "
                "Climate-Pest Correlation Analysis, and let it steer the diagnosis (upward cupping => "
                "Leaf Curl Virus / whitefly control; downward inverted-boat => Broad Mite control).\n"
            )

        # ── Curated agronomy reference, when this core class has an entry ─────
        kb_note = _format_kb_context(core_cls)

        return (
            f"=== CNN DETECTION RESULT ===\n"
            f"Detected: {translated_label_clean}\n"
            f"Type: {kind} | Confidence: {confidence}%\n"
            f"{morph_note}"
            f"{kb_note}"
            f"Farmer described: '{user_msg_clean}'\n"
            f"INSTRUCTION: Tell the farmer clearly what this {kind} is in simple words in {target_lang_name} "
            f"(mention the name '{translated_label_clean}' if helpful). "
            f"Provide a dual-structured treatment approach with 'Biological & Organic Interventions' and 'Targeted Chemical Interventions'. Enforce explicit biological sub-class targeting inside Biological sections (recommend specific bacterial vectors like Bacillus thuringiensis, viral vectors like NPV, or fungal pathogens tailored strictly to the diagnosed lifecycle). "
            f"If the diagnosis is a viral profile (Leaf Curl Virus), you must mandate recommendations targeting the whitefly vector using biological fungal spores (specifically Beauveria bassiana) or systemic chemical neonicotinoids (specifically Acetamiprid). "
            f"If the diagnosis is a fungal/bacterial profile (Cercospora Leaf Spot, Bacterial Leaf Spot, Powdery Mildew), you must output clear tables contrasting organic treatments (such as Copper Hydroxide, Pseudomonas fluorescens, or systemic bio-agents) with targeted chemical choices. "
            f"Frame every named chemical as something to confirm with a local Krishi Vigyan Kendra (KVK) or agriculture officer before use, never as a direct ready-to-apply prescription. "
            f"For every suggested solution, you must explicitly render a 'Cost-Effectiveness & Speed Evaluation Table' in markdown showing: Estimated Cost per Acre in INR (₹), Efficacy Speed (e.g., Immediate 24hr knockdown vs 5-day systemic spread), and Environmental Residual Protection windows. End with one prevention tip."
            + low_note
        )
    else:
        # No confident detection — questioning mode
        err = result.get("error", "") if isinstance(result, dict) else ""
        if err:
            log.warning("detector_error", extra={"data": {"error_message": err}})
        user_msg_clean = strip_cross_contamination(user_msg, lang)
        return (
            f"A farmer uploaded a photo of their chilli plant. "
            f"They described: '{user_msg_clean}'. "
            f"The AI detector could not identify the problem with confidence. "
            f"Ask them 2 specific questions about what they can see "
            f"(colour of affected area, location on plant — leaves/stem/fruit/roots, "
            f"any insects visible, holes in fruit, webbing, powder, spots etc) "
            f"then give a diagnosis followed by a dual-structured treatment plan under 'Biological & Organic Interventions' and 'Targeted Chemical Interventions'. Enforce explicit biological sub-class targeting (specific bacterial, viral, or fungal pathogens tailored to the lifecycle). "
            f"If the diagnosis resolves to a viral profile (Leaf Curl Virus), you must mandate recommendations targeting the whitefly vector using biological fungal spores (Beauveria bassiana) or systemic chemical neonicotinoids (Acetamiprid). "
            f"If the diagnosis resolves to a fungal/bacterial profile (Cercospora Leaf Spot, Bacterial Leaf Spot, Powdery Mildew), contrast organic treatments (Copper Hydroxide, Pseudomonas fluorescens, or systemic bio-agents) with targeted chemical choices in clear tables. "
            f"Frame every named chemical as something to confirm with a local Krishi Vigyan Kendra (KVK) or agriculture officer before use, never as a direct ready-to-apply prescription. "
            f"For every suggestion, include a 'Cost-Effectiveness & Speed Evaluation Table' in markdown showing: Estimated Cost per Acre in INR (₹), Efficacy Speed, and Environmental Residual Protection windows. End with one prevention tip."
        )


# ── Vision-guided adaptive triage ─────────────────────────────────────────────
# The vision step forms a hypothesis; when it's a single high-confidence
# pest/disease, nothing below runs and the existing rich diagnosis flow above
# is used unchanged. Otherwise (low-confidence / lookalike cluster, or a
# healthy-looking/no-detection result the farmer says has a problem) the
# functions below ask ONE targeted question at a time and re-rank candidates
# on each answer, instead of the rich multi-section format.
#
# STRICT SEPARATION (the safety core of this feature):
#   - Groq's job is the CONVERSATION: authoring question text/options
#     (_groq_generate_question) and reasoning over the farmer's answers to
#     pick a candidate (_groq_resolve_diagnosis). It is given ONLY the
#     key_symptoms text already in chilli_kb.json for the live candidate
#     set and is explicitly told never to introduce a candidate or symptom
#     outside that set.
#   - The KB governs the FACTS: every id Groq returns (in a question's
#     favors/against tags, or as the final resolved_id) is validated against
#     the candidate set; anything outside it is rejected, not trusted. The
#     final card's treatment/dosage text is built by _build_kb_treatment_block
#     directly from chilli_kb.json and is NEVER passed through Groq at all --
#     Groq only writes the Type/Name + Cause framing, never the treatment.
#   - Graceful degradation: knowledge/triage_rules.json (via triage.py) is
#     the FALLBACK question source if Groq is unavailable/times out/returns
#     something invalid, so the feature degrades instead of breaking. The
#     deterministic score-leader (triage.check_resolution) is likewise the
#     fallback diagnosis if _groq_resolve_diagnosis is unavailable/rejected.
_TRIAGE_CARD_SYSTEM_PROMPT = """You are ChilliGuru, a friendly farming assistant for chilli farmers in Andhra Pradesh and Telangana. Talk like a trusted friend -- simple, warm, easy to understand, no big words.

The farmer has already answered a few quick follow-up questions and the diagnosis is now narrowed down. Your ONLY job is to write the Type/Name and Cause -- two short sections, nothing else. Do NOT write any treatment, dosage, or "what to do" advice; that is added separately from a verified reference, not by you.
"""


def _compile_triage_card_prompt(domain: str, leader: str, ambiguous: bool, lang: str, user_msg: str) -> str:
    """
    Builds the Groq user-context string for a TRIAGE-RESOLVED result (after
    the adaptive Q&A loop finishes) -- Groq writes ONLY Type/Name + Cause
    here. Treatment/dosage is deliberately NOT requested from Groq at all;
    _build_kb_treatment_block renders that directly from chilli_kb.json and
    is appended server-side (see _triage_resolved_response) so dosage
    content can never be LLM-paraphrased. For domain="deficiency" this is
    the ONLY place a specific deficiency name is ever introduced into the
    conversation -- it is reached purely via seed_deficiency_candidates() +
    question answers (static) or _groq_resolve_diagnosis (dynamic), never
    from the vision result.
    """
    lang_names = {"en": "English", "hi": "Hindi", "te": "Telugu", "kn": "Kannada", "ta": "Tamil"}
    target_lang_name = lang_names.get(lang, "English")
    user_msg_clean = strip_cross_contamination(user_msg, lang)

    uncertainty_note = (
        "The farmer's answers narrowed this down but didn't give one single clear winner -- "
        "present this as your BEST GUESS, not a certain diagnosis, and say so plainly. "
    ) if ambiguous else ""

    if domain == "pest":
        english, telugu_name, kind = detector._get_friendly_name(leader)
        label = f"{english} [{telugu_name}]" if telugu_name else english
        entry = _get_kb_entry(leader) or {}
        causal = entry.get("causal_agent") or "(not documented)"
        symptoms = entry.get("key_symptoms") or "(not documented)"
        return (
            f"=== ADAPTIVE TRIAGE RESULT (resolved via follow-up questions) ===\n"
            f"Most likely {kind}: {label}\n"
            f"Causal agent (reference -- do not contradict): {causal}\n"
            f"Key symptoms (reference -- do not contradict): {symptoms}\n"
            f"Farmer described: '{user_msg_clean}'\n"
            f"INSTRUCTION: {uncertainty_note}"
            f"Reply in {target_lang_name} with EXACTLY two short sections, simple language, no extra sections:\n"
            f"### Type/Name\n[name the {kind} simply]\n"
            f"### Cause\n[1-2 sentences on what causes/spreads it, grounded in the reference above]\n"
            f"Do NOT write a treatment, dosage, or 'what to do' section -- that is added separately from a "
            f"verified reference, not by you. Keep the whole reply short."
        )

    # domain == "deficiency"
    entry = _CHILLI_KB.get("deficiencies", {}).get(leader, {})
    display_name = entry.get("display_name", leader)
    telugu_name = entry.get("telugu_name", "")
    symptoms = entry.get("key_symptoms") or "(not documented)"
    return (
        f"=== ADAPTIVE TRIAGE RESULT — NUTRIENT DEFICIENCY (resolved via follow-up questions, "
        f"NOT from the photo — vision cannot see deficiencies) ===\n"
        f"Most likely deficiency: {display_name} [{telugu_name}]\n"
        f"Reference symptoms (do not contradict): {symptoms}\n"
        f"Farmer described: '{user_msg_clean}'\n"
        f"INSTRUCTION: {uncertainty_note}"
        f"Reply in {target_lang_name} with EXACTLY two short sections, simple language, no extra sections:\n"
        f"### Type/Name\n[name the deficiency simply]\n"
        f"### Cause\n[1-2 sentences on what causes this deficiency, grounded in the reference above]\n"
        f"Do NOT write a correction, dosage, or soil-test section -- that is added separately from a verified "
        f"reference, not by you. Keep the whole reply short."
    )


# ── KB-verbatim treatment block (NEVER passed through Groq) ──────────────────
_TREATMENT_BLOCK_TEXT = {
    "organic_heading": {
        "en": "**Organic / Biological**", "te": "**సేంద్రియ / జీవ నియంత్రణ**",
        "hi": "**जैविक उपचार**", "kn": None, "ta": None,
    },
    "chemical_heading": {
        "en": "**Chemical** (confirm the product and dose with your local Krishi Vigyan Kendra (KVK) or agriculture officer before use)",
        "te": "**రసాయనం** (వాడే ముందు ఉత్పత్తి మరియు మోతాదును మీ స్థానిక కృషి విజ్ఞాన కేంద్రం (KVK) లేదా వ్యవసాయ అధికారితో నిర్ధారించుకోండి)",
        "hi": "**रासायनिक** (उपयोग से पहले उत्पाद और मात्रा की पुष्टि अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी से करें)",
        "kn": None, "ta": None,
    },
    "no_data": {
        "en": "No verified organic or chemical treatment is recorded for this in our reference yet -- please show this to your local Krishi Vigyan Kendra (KVK) or agriculture officer for treatment advice.",
        "te": "దీనికి సంబంధించిన సేంద్రియ లేదా రసాయన చికిత్స మా రెఫరెన్స్‌లో ఇంకా నమోదు చేయబడలేదు -- చికిత్స సలహా కోసం దీన్ని మీ స్థానిక కృషి విజ్ఞాన కేంద్రం (KVK) లేదా వ్యవసాయ అధికారికి చూపించండి.",
        "hi": "इसके लिए कोई सत्यापित जैविक या रासायनिक उपचार अभी हमारे संदर्भ में दर्ज नहीं है -- उपचार सलाह के लिए इसे अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी को दिखाएं।",
        "kn": None, "ta": None,
    },
}

_SOIL_TEST_REMINDER = {
    "en": "This is a guess based on your answers, not a lab result -- please confirm with a soil test before correcting the soil, especially before adding any chemical fertiliser.",
    "te": "ఇది మీ సమాధానాల ఆధారంగా అంచనా మాత్రమే, ప్రయోగశాల ఫలితం కాదు -- మట్టిని సరిచేసే ముందు, ముఖ్యంగా ఏదైనా రసాయన ఎరువు వేసే ముందు, మట్టి పరీక్ష చేయించి నిర్ధారించుకోండి.",
    "hi": "यह आपके जवाबों के आधार पर एक अनुमान है, प्रयोगशाला परिणाम नहीं -- मिट्टी सुधारने से पहले, खासकर कोई रासायनिक खाद डालने से पहले, मिट्टी की जांच करवाकर पुष्टि करें।",
    "kn": None, "ta": None,
}


def _tr(text_dict: dict, lang: str) -> str:
    """Small shared lookup: `text_dict[lang]` if populated, else English fallback (TODO-mark pattern used throughout this app)."""
    return (text_dict.get(lang) if text_dict.get(lang) else None) or text_dict["en"]


def _kb_entry_exists(domain: str, candidate_id: str) -> bool:
    if domain == "pest":
        return _get_kb_entry(candidate_id) is not None
    return candidate_id in _CHILLI_KB.get("deficiencies", {})


def _build_kb_treatment_block(domain: str, leader: str, lang: str) -> str:
    """
    Renders the organic/chemical treatment section DIRECTLY from
    chilli_kb.json -- this text never passes through Groq, so it is
    verbatim by construction rather than by prompt instruction alone. This
    is the hard-rule enforcement point for "the final diagnosis and ALL
    treatment/dosage content must still resolve to a chilli_kb.json entry...
    dosages are KB-verbatim only." If the KB entry has no organic/chemical
    text recorded (a real data gap in chilli_kb.json for many entries --
    see the project's prior KB-completeness notes), this returns an honest
    "not documented, consult KVK" message rather than ever inventing one.
    """
    if domain == "pest":
        entry = _get_kb_entry(leader)
    else:
        entry = _CHILLI_KB.get("deficiencies", {}).get(leader)
    mgmt = (entry or {}).get("management") or {}
    organic = mgmt.get("organic")
    chemical = mgmt.get("chemical")

    if not organic and not chemical:
        return "\n\n" + _tr(_TREATMENT_BLOCK_TEXT["no_data"], lang)

    parts = []
    if organic:
        parts.append(f"\n\n{_tr(_TREATMENT_BLOCK_TEXT['organic_heading'], lang)}\n{organic}")
    if chemical:
        parts.append(f"\n\n{_tr(_TREATMENT_BLOCK_TEXT['chemical_heading'], lang)}\n{chemical}")
    return "".join(parts)


# "Not sure / skip" option appended to EVERY triage question (Groq-authored and
# static fallback). Scores={} means answering this changes NOTHING in the
# candidate ranking — the question is simply skipped.
_NOT_SURE_SKIP_LABELS = {
    "en": "Not sure / skip",
    "te": "తెలియదు / దాటవేయండి",
    "hi": "पता नहीं / छोड़ दें",
    "kn": None,   # TODO: Kannada translation
    "ta": None,   # TODO: Tamil translation
}

_ALL_UNSURE_TEXT = {
    "en": "We were unable to narrow this down because there weren't enough details to separate the possibilities. Please show this photo and describe the symptoms to your local Krishi Vigyan Kendra (KVK) or agriculture officer for a proper on-the-ground diagnosis.",
    "te": "తగినంత వివరాలు లేనందున ఈ సమస్యను నిర్ధారించలేకపోయాము. సరైన నిర్ధారణ కోసం దయచేసి ఈ ఫోటోను చూపించి, లక్షణాలను మీ స్థానిక కృషి విజ్ఞాన కేంద్రం (KVK) లేదా వ్యవసాయ అధికారికి వివరించండి.",
    "hi": "पर्याप्त जानकारी न मिलने के कारण हम इसे सीमित नहीं कर सके। सही निदान के लिए कृपया यह फोटो दिखाएं और लक्षण अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी को बताएं।",
    "kn": None, "ta": None,   # TODO: kn/ta translations
}

_NO_KB_MATCH_TEXT = {
    "en": "We couldn't narrow this down to a confirmed match in our reference. Please show this photo and describe the symptoms to your local Krishi Vigyan Kendra (KVK) or agriculture officer for a proper diagnosis.",
    "te": "దీన్ని మా రెఫరెన్స్‌లో నిర్ధారిత మ్యాచ్‌గా తేల్చలేకపోయాము. సరైన నిర్ధారణ కోసం దయచేసి ఈ ఫోటోను చూపించి, లక్షణాలను మీ స్థానిక కృషి విజ్ఞాన కేంద్రం (KVK) లేదా వ్యవసాయ అధికారికి వివరించండి.",
    "hi": "हम इसे अपने संदर्भ में किसी पुष्ट मिलान तक सीमित नहीं कर सके। सही निदान के लिए कृपया यह फोटो दिखाएं और लक्षण अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी को बताएं।",
    "kn": None, "ta": None,
}


def _no_kb_match_response_with_text(text_dict: dict, lang: str):
    """Shared helper: returns an SSE response with a translated text message and no diagnosis card."""
    text = _tr(text_dict, lang)

    def _gen():
        yield f"data: {json.dumps({'type': 'meta', 'detection': None, 'low_confidence': True, 'case_id': None}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'text', 'text': text}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(_gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _no_kb_match_response(lang: str):
    """
    Used when triage stops but NEITHER Groq's resolution NOR the
    deterministic score-leader names a candidate with an actual
    chilli_kb.json entry (e.g. only cercospora_leaf_spot is left, which has
    no KB entry by design -- see _CORE_CLASS_TO_KB's comment). Shows an
    honest "consult local expert" message instead of ever building a card
    with invented or missing content. No case is created since there is no
    resolved diagnosis to track an outcome against.
    """
    return _no_kb_match_response_with_text(_NO_KB_MATCH_TEXT, lang)


# ── Dynamic (Groq-authored) triage: question generation + resolution ────────
# Groq drives the CONVERSATION here -- it never drives the FACTS. Every id it
# returns (in a question's favors/against tags, or as a final resolved_id)
# is validated against the live candidate set before being trusted; anything
# else is rejected and the caller falls back to the static
# knowledge/triage_rules.json engine (questions) or the deterministic
# score-leader (resolution). See the "STRICT SEPARATION" comment above
# _TRIAGE_CARD_SYSTEM_PROMPT for the full design rationale.
_GROQ_TRIAGE_TIMEOUT_S   = 20.0
_GROQ_TRIAGE_MAX_RETRIES = 1  # best-effort: fail fast to the static fallback rather than hold up the farmer


def _kb_symptom_context_for_candidates(domain: str, candidate_ids) -> dict:
    """
    Returns {candidate_id: key_symptoms_text} for every candidate that has a
    resolvable chilli_kb.json entry with symptom text. Candidates with no KB
    entry (e.g. cercospora_leaf_spot -- see _CORE_CLASS_TO_KB's comment) are
    silently excluded from what's offered to Groq, since there is no sourced
    symptom text to ground a question in -- they remain valid scoring
    candidates elsewhere, just not ones Groq is asked to write about.
    """
    out = {}
    for cid in candidate_ids:
        entry = _get_kb_entry(cid) if domain == "pest" else _CHILLI_KB.get("deficiencies", {}).get(cid)
        if entry and entry.get("key_symptoms"):
            out[cid] = entry["key_symptoms"]
    return out


def _groq_json_call(messages: list, timeout_s: float = _GROQ_TRIAGE_TIMEOUT_S,
                     max_retries: int = _GROQ_TRIAGE_MAX_RETRIES):
    """
    Non-streaming Groq call constrained to strict JSON output
    (response_format=json_object), reusing the app's existing
    timeout+retry wrapper. Returns the parsed dict, or None on ANY failure
    (timeout, API error, invalid JSON). Never raises -- callers MUST treat
    None as "dynamic generation unavailable this turn, fall back to the
    static path" per the graceful-degradation requirement.
    """
    def _call():
        return get_client().chat.completions.create(
            model=MODEL, messages=messages, max_tokens=MAX_TOKENS,
            temperature=0.3, timeout=timeout_s,
            response_format={"type": "json_object"},
        )
    try:
        response = _retry_with_backoff(_call, max_retries=max_retries, what="groq_dynamic_triage")
        return json.loads(response.choices[0].message.content)
    except Exception as exc:
        log.warning("groq_dynamic_triage_fail", extra={"data": {"error": str(exc)}})
        return None


def _groq_generate_question(domain: str, candidates: dict, asked_question_texts: list, lang: str):
    """
    Asks Groq to author ONE plain-language discriminating question (+2-4 tap
    options) for the CURRENT live candidate set, grounded ONLY in the
    key_symptoms text pulled from chilli_kb.json for those candidates --
    Groq is given no other agronomy facts and is explicitly told not to
    introduce any candidate/symptom outside what's provided.

    Returns a question dict in the SAME shape triage.py's static rulebook
    questions use ({"id", "question", "options":[{"key","label","scores"}]})
    so it flows through the existing triage.apply_answer/check_resolution
    machinery unchanged regardless of whether it was Groq-authored or pulled
    from knowledge/triage_rules.json. Returns None on any failure or
    malformed/under-specified response -- the caller must fall back to
    triage.pick_next_question (static).
    """
    live = triage._alive_candidates(candidates, triage.resolve_margin()) or set(candidates.keys())
    symptom_map = _kb_symptom_context_for_candidates(domain, live)
    if len(symptom_map) < 2:
        return None  # nothing groundable to discriminate between -- let the static/ambiguous path handle it

    candidate_lines = "\n".join(f"- {cid}: {sym}" for cid, sym in symptom_map.items())
    asked_lines = "\n".join(f"- {q}" for q in asked_question_texts) or "(none yet)"

    prompt = (
        "You are helping a chilli farmer figure out which of these specific possibilities matches what "
        "they're seeing, by asking ONE simple question.\n\n"
        f"POSSIBILITIES (use ONLY these ids, do not mention anything else):\n{candidate_lines}\n\n"
        f"Questions already asked this conversation (do not repeat):\n{asked_lines}\n\n"
        "Write ONE new plain-language question -- a farmer with little reading ability should understand "
        "it instantly -- that best splits these possibilities apart, using ONLY the symptom differences "
        "given above. Give 2-4 short tap-able answer options. For each option, list which of the "
        "possibility ids above it SUPPORTS (favors) and which it RULES OUT (against) -- only use ids from "
        "the list above, never invent a new one. Provide the question and option labels in English, "
        "Telugu, and Hindi.\n\n"
        "Respond with STRICT JSON only, no other text, in this exact shape:\n"
        '{"question": {"en": "...", "te": "...", "hi": "..."}, '
        '"options": [{"key": "short_key", "label": {"en": "...", "te": "...", "hi": "..."}, '
        '"favors": ["id1"], "against": ["id2"]}, ...]}'
    )
    messages = [
        {"role": "system", "content": (
            "You generate farmer-facing diagnostic questions for a chilli-pest assistant. You never invent "
            "agronomy facts or mention a possibility not given to you -- you only use the symptom text "
            "provided. Always respond with valid JSON only, no markdown fences, no extra commentary."
        )},
        {"role": "user", "content": prompt},
    ]
    data = _groq_json_call(messages)
    if not data:
        return None

    try:
        q_text = data.get("question")
        options_raw = data.get("options")
        if not isinstance(q_text, dict) or not q_text.get("en") or not isinstance(options_raw, list):
            return None

        options = []
        for opt in options_raw:
            if not isinstance(opt, dict):
                continue
            key = str(opt.get("key", "")).strip()
            label = opt.get("label")
            if not key or not isinstance(label, dict) or not label.get("en"):
                continue
            favors  = [c for c in (opt.get("favors") or [])  if isinstance(c, str) and c in live]
            against = [c for c in (opt.get("against") or []) if isinstance(c, str) and c in live]
            scores = {}
            for c in favors:
                scores[c] = scores.get(c, 0.0) + 12.0
            for c in against:
                scores[c] = scores.get(c, 0.0) - 10.0
            options.append({"key": key, "label": label, "scores": scores})

        if len(options) < 2:
            return None  # too thin/malformed to be a useful question -- fall back instead of showing it

        # Always append a "Not sure / skip" option so the farmer is never
        # forced to guess.  Empty scores dict means it changes nothing in
        # the candidate ranking — the question is simply skipped.
        if not any(o["key"] == "not_sure" for o in options):
            options.append({"key": "not_sure", "label": dict(_NOT_SURE_SKIP_LABELS), "scores": {}})

        return {
            "id":          f"dyn_{uuid.uuid4().hex[:8]}",
            "question":    q_text,
            "options":     options,
            "applies_to":  list(live),
            "dynamic":     True,
        }
    except (TypeError, ValueError, AttributeError) as exc:
        log.warning("groq_dynamic_question_malformed", extra={"data": {"error": str(exc)}})
        return None


def _groq_resolve_diagnosis(domain: str, candidates: dict, qa_transcript: list, lang: str):
    """
    Asks Groq to pick the single BEST-MATCHING candidate from `candidates`
    given the full question/answer transcript, grounded only in the
    key_symptoms already associated with those candidates. The returned id
    is hard-validated against the candidate set (which is itself always
    drawn from triage.PEST_CANDIDATE_IDS/DEFICIENCY_CANDIDATE_IDS) -- if
    Groq returns anything else (or times out, or sends malformed JSON),
    this returns None and the caller MUST fall back to the deterministic
    score-leader with `ambiguous=True` rather than ever showing an invented
    diagnosis. This is the explicit "if Groq returns anything not in the KB,
    reject" requirement.
    """
    symptom_map = _kb_symptom_context_for_candidates(domain, candidates.keys())
    if not symptom_map:
        return None

    candidate_lines = "\n".join(f"- {cid}: {sym}" for cid, sym in symptom_map.items())
    qa_lines = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_transcript) or "(no questions were answered)"

    prompt = (
        "A chilli farmer was asked some questions to figure out which of these specific possibilities "
        "matches their plant.\n\n"
        f"POSSIBILITIES (pick ONLY from these ids, never invent a new one):\n{candidate_lines}\n\n"
        f"CONVERSATION:\n{qa_lines}\n\n"
        "Based on the conversation and the symptom descriptions above, which single possibility id best "
        "matches? If the conversation doesn't clearly point to one, say so honestly rather than guessing "
        "confidently.\n\n"
        "Respond with STRICT JSON only: "
        '{"resolved_id": "<one of the ids above, or null if unclear>", "confidence": "high" or "low", '
        '"reasoning": "<one short sentence>"}'
    )
    messages = [
        {"role": "system", "content": (
            "You resolve a farmer diagnostic conversation to one of a fixed set of possibility ids. You "
            "never pick or invent an id outside the list given. Always respond with valid JSON only, no "
            "markdown fences, no extra commentary."
        )},
        {"role": "user", "content": prompt},
    ]
    data = _groq_json_call(messages)
    if not data:
        return None
    try:
        resolved_id = data.get("resolved_id")
        confidence  = data.get("confidence", "low")
        if resolved_id not in symptom_map:  # hard validation gate -- the actual KB-governance check
            if resolved_id is not None:
                log.warning("groq_resolved_id_rejected", extra={"data": {"resolved_id": resolved_id}})
            return None
        return {"leader": resolved_id, "ambiguous": confidence != "high"}
    except (AttributeError, TypeError) as exc:
        log.warning("groq_resolve_malformed", extra={"data": {"error": str(exc)}})
        return None


def _triage_ask_payload(domain: str, candidates: dict, asked: list, lang: str, user_msg: str,
                         morphology: dict | None = None, qa_transcript: list | None = None):
    """
    Returns a JSON-serializable {"triage": {...}} payload carrying the next
    question to ask, or None if NEITHER the dynamic Groq path NOR the static
    knowledge/triage_rules.json rulebook has a question for the current
    candidate set (caller should then treat this as resolved/ambiguous).

    Primary path: _groq_generate_question authors the question live, grounded
    only in the KB symptom text for the current candidates. Fallback path
    (Groq unavailable/timeout/malformed/rejected): triage.pick_next_question,
    the original static rulebook engine -- this is the graceful-degradation
    guarantee, not a secondary feature. Either way the returned question dict
    has the SAME shape, so triage.apply_answer/check_resolution work
    identically regardless of origin.
    """
    qa_transcript = qa_transcript or []
    asked_question_texts = [q_text for q_text, _ in qa_transcript]

    q = _groq_generate_question(domain, candidates, asked_question_texts, lang)
    if q is None:
        q = triage.pick_next_question(domain, candidates, asked)
    if q is None:
        return None

    return {
        "triage": {
            "question_id": q["id"],
            "question": q["question"].get(lang) or q["question"]["en"],
            "options": [
                {
                    "key":   o["key"],
                    "label": o["label"].get(lang) or o["label"]["en"],
                    "skip":  o["key"] == "not_sure",   # UI flag: render with lower emphasis
                }
                for o in q["options"]
            ],
            "state": {
                "domain":        domain,
                "candidates":    candidates,
                "asked":         asked,
                "lang":          lang,
                "user_msg":      user_msg,
                "morphology":    morphology,
                "last_question": q,             # full object -- dynamic ids aren't in triage_rules.json
                "qa_transcript": qa_transcript,  # for _groq_resolve_diagnosis once triage stops
            },
        }
    }


def _triage_resolved_response(domain: str, leader: str, ambiguous: bool, lang: str,
                               user_msg: str, history: list, morphology: dict | None = None):
    """
    Builds the final SSE response once triage has resolved (or exhausted its
    question budget). Groq writes ONLY the Type/Name + Cause text
    (_compile_triage_card_prompt); the treatment/dosage block is rendered
    directly from chilli_kb.json by _build_kb_treatment_block and appended
    by _groq_stream_generator itself -- never generated by the LLM. For
    domain="deficiency" the soil-test reminder is likewise appended
    server-side (verbatim, not LLM-phrased) rather than trusted to a prompt
    instruction.
    """
    system_content = strip_cross_contamination(_TRIAGE_CARD_SYSTEM_PROMPT, lang)
    system_content += _LANG_INSTRUCTION_MAP.get(lang, "\nIMPORTANT: You must respond in English.")
    system_content = strip_cross_contamination(system_content, lang)
    triage_context = _compile_triage_card_prompt(domain, leader, ambiguous, lang, user_msg)

    messages = [{"role": "system", "content": system_content}] + history + [{"role": "user", "content": triage_context}]

    kb_treatment_block = _build_kb_treatment_block(domain, leader, lang)
    if domain == "deficiency":
        kb_treatment_block += "\n\n" + _tr(_SOIL_TEST_REMINDER, lang)

    detection = None
    if domain == "pest":
        english, telugu_name, kind = detector._get_friendly_name(leader)
        detection = {
            "label":      f"{english} [{telugu_name}]" if telugu_name else english,
            "raw_label":  english,
            "telugu":     telugu_name,
            "type":       kind,
            "confidence": 55 if ambiguous else 70,  # post-Q&A display value; the original vision % is no longer the relevant number
        }
        if morphology:
            detection["morphology"] = morphology

    case_id = _create_case(domain, leader, lang)

    return Response(
        stream_with_context(_groq_stream_generator(
            messages, detection, ambiguous, lang, case_id=case_id, kb_treatment_block=kb_treatment_block
        )),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _detect_inner():
    """
    Internal execution method for the `/detect` route.
    
    Extracts the image and metadata inputs from the Flask request, runs image type 
    and size validations, coordinates remote Hugging Face and local YOLO models, 
    applies context re-ranking, constructs the Groq model prompt context, 
    and sets up the streaming SSE response.
    
    Returns:
        tuple | Response: Flask Response representing the SSE stream generator or JSON error block.
    """
    user_msg    = request.form.get("message", "").strip()
    history_raw = request.form.get("history", "[]")
    try:
        history = json.loads(history_raw)
    except Exception as exc:
        log.warning("history_json_parse_failed", extra={"data": {"error": str(exc)}}, exc_info=True)
        history = []

    if not user_msg:
        user_msg = "I uploaded a photo of my chilli plant but I am not sure what the problem is."

    # ── Farmer's MCQ field answers (improve detection accuracy) ────────────────
    field_context = {
        "plant_age":     request.form.get("plant_age", "").strip().lower(),
        "observed":      request.form.get("observed", "").strip().lower(),
        "affected_part": request.form.get("affected_part", "").strip().lower(),
        "curl_dir":      request.form.get("curl_dir", "").strip().lower(),
    }

    # ── Try HF Space detector first ───────────────────────────────────────────
    image_file = request.files.get("image")
    detection    = None
    is_low       = False   # default; overwritten below if image is present
    groq_context = None

    if image_file:
        # ── Size check (seek without loading the full stream) ─────────────────
        image_file.seek(0, 2)
        file_size = image_file.tell()
        image_file.seek(0)
        if file_size > MAX_IMAGE_BYTES:
            log.warning("payload_too_large", extra={"data": {"size_bytes": file_size}})
            return jsonify({
                "error": "Payload too large",
                "request_id": _current_request_id()
            }), 413

        # ── Magic-byte type verification (first 12 bytes only) ────────────────
        header = image_file.read(12)
        image_file.seek(0)
        is_webp = header[:4] == b'RIFF' and header[8:12] == b'WEBP'
        is_known = is_webp or any(header.startswith(sig) for sig in _IMAGE_SIGNATURES if sig != b'RIFF')
        if not is_known:
            log.warning("invalid_image_magic", extra={"data": {"header_hex": header.hex()}})
            return jsonify({
                "success": False,
                "error":   "Invalid or corrupted image format submitted",
                "phase":   0,
            }), 400

        # ── Full read (validation passed) ─────────────────────────────────────
        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({"error": "Empty file"}), 400

        result = None
        force_bypass_ood = False
        # Try HF space call first (skip if circuit is open)
        if _cb_is_open():
            log.warning("circuit_breaker_skip_hf")
        else:
            try:
                if get_hf_client() is not None:
                    result = call_hf_detector(image_bytes)
                    
                    # ── Guardrail check (delegated to sub-module 2) ──────────
                    try:
                        result = _execute_guardrail_check(image_bytes, result)
                        if result is None:
                            force_bypass_ood = True
                    except _GuardrailReject as gr:
                        return gr.response, gr.status
                else:
                    log.warning("hf_client_unavailable_fallback")
            except _GuardrailReject:
                raise
            except Exception as exc:
                log.error("hf_unreachable_fallback", extra={"data": {"error_message": str(exc)}},
                          exc_info=True)
                result = None

        if result is None:
            log.info("local_cascade_start")
            _t_local = time.time()
            try:
                result = detector.detect_from_memory(image_bytes, bypass_ood=force_bypass_ood)
                log.info("local_cascade_ok", extra={"data": {
                    "duration_ms": round((time.time() - _t_local) * 1000),
                    "phase":       result.get("phase") if isinstance(result, dict) else None,
                }})
                if isinstance(result, dict) and (result.get("success") is False or result.get("phase") == 3):
                    # Branch (c): vision found nothing confident -- before giving up,
                    # check whether the farmer's own words/selections suggest a
                    # nutrition problem worth a deficiency triage instead of a
                    # flat error. Vision contributes ONLY this yes/no bias, never
                    # a specific deficiency (see triage.seed_deficiency_candidates).
                    if triage.looks_like_deficiency_trigger(result, user_msg, field_context):
                        def_lang = _resolve_request_language(user_msg)
                        def_candidates = triage.seed_deficiency_candidates()
                        ask = _triage_ask_payload("deficiency", def_candidates, [], def_lang, user_msg)
                        if ask is not None:
                            log.info("triage_deficiency_start", extra={"data": {"trigger": "no_confident_detection"}})
                            return jsonify(ask)
                    return jsonify(result)
            except Exception as local_exc:
                log.error("local_cascade_error", extra={"data": {
                    "error_message": str(local_exc),
                    "duration_ms":   round((time.time() - _t_local) * 1000),
                }}, exc_info=True)
                result = {"error": str(local_exc)}

        # ── Fold the farmer's MCQ answers into detection ranking ──────────────
        result = _apply_field_context(result, field_context)

        top_label = (result.get("top_detection", {}) or {}).get("label") if isinstance(result, dict) else None
        log.info("detector_result", extra={"data": {
            "phase":       result.get("phase") if isinstance(result, dict) else None,
            "top_label":   top_label,
            "success":     result.get("success") if isinstance(result, dict) else None,
            "low_confidence": result.get("low_confidence") if isinstance(result, dict) else None,
        }})

        top    = result.get("top_detection") if isinstance(result, dict) else None
        is_low = result.get("low_confidence", False) if isinstance(result, dict) else True

        # ── Sub-module 1: Resolve language (with early-return fast-path) ───────
        lang = _resolve_request_language(user_msg)

        # ── Vision-guided adaptive triage branch ────────────────────────────
        # (a) High-confidence single detection falls through untouched below —
        #     this is the existing rich-format flow, unchanged (regression-safe).
        # (b) Low-confidence / lookalike cluster -> ask the one curated question
        #     that splits THESE candidates (knowledge/triage_rules.json), not a
        #     generic list.
        # (c) Healthy-looking/no-pest result but the farmer reports a problem ->
        #     deficiency triage (vision only supplies the yes/no bias here, see
        #     triage.looks_like_deficiency_trigger — never a specific deficiency).
        if isinstance(result, dict) and triage.looks_like_deficiency_trigger(result, user_msg, field_context):
            def_candidates = triage.seed_deficiency_candidates()
            ask = _triage_ask_payload("deficiency", def_candidates, [], lang, user_msg)
            if ask is not None:
                if image_bytes:
                    _trigger_shadow_save(image_bytes, label="deficiency_triage_trigger", confidence=0, trigger="triage_ask")
                log.info("triage_deficiency_start", extra={"data": {"trigger": "healthy_but_problem_reported"}})
                return jsonify(ask)

        pest_candidates = triage.build_pest_candidates(result, to_core_class) if isinstance(result, dict) else {}
        if pest_candidates and triage.should_ask_pest_questions(result, pest_candidates):
            # Single uncertain candidate carries nothing to discriminate against;
            # expand it to its known visual-lookalike cluster so there's a real
            # question to ask. The photo's own pick stays the favourite.
            triage_candidates = triage.expand_to_cluster(pest_candidates)
            morphology = top.get("morphology") if isinstance(top, dict) else None
            ask = _triage_ask_payload("pest", triage_candidates, [], lang, user_msg, morphology=morphology)
            if ask is not None:
                if image_bytes:
                    label = top.get("raw_label") if isinstance(top, dict) else "unknown"
                    confidence = top.get("confidence", 0) if isinstance(top, dict) else 0
                    _trigger_shadow_save(image_bytes, label=label, confidence=confidence, trigger="triage_ask")
                log.info("triage_pest_start", extra={"data": {
                    "vision_candidates": list(pest_candidates.keys()),
                    "triage_candidates": list(triage_candidates.keys()),
                    "low_confidence":    result.get("low_confidence") if isinstance(result, dict) else None,
                }})
                return jsonify(ask)

        # ── Sub-module 3: Compile Groq payload ──────────────────────────────
        if top is not None:
            detection = top
        groq_context = _compile_groq_payload(
            result=result,
            top=top,
            is_low=is_low,
            lang=lang,
            user_msg=user_msg,
            image_bytes=image_bytes if image_file else None,
        )
        groq_context += _format_field_answers(field_context)
        if top:
            is_low = result.get("low_confidence", False)
    else:
        # No image — text-only chat-fallback path
        lang = _resolve_request_language(user_msg)
        user_msg_clean = strip_cross_contamination(user_msg, lang)
        groq_context = (
            f"A farmer uploaded a photo of their chilli plant. "
            f"They described: '{user_msg_clean}'. "
            f"Ask them 2 specific questions about what they can see, "
            f"then give a diagnosis and 2-3 organic solutions with metrics."
        )
        groq_context += _format_field_answers(field_context)

    # ── Stream Groq response via SSE ──────────────────────────────────────────
    system_content = strip_cross_contamination(SYSTEM_PROMPT, lang)
    system_content += _LANG_INSTRUCTION_MAP.get(lang, "\nIMPORTANT: You must respond in English.")
    system_content = strip_cross_contamination(system_content, lang)

    messages = [{"role": "system", "content": system_content}] + history + [{"role": "user", "content": groq_context}]

    # Only create an outcome-tracking case when an actual image diagnosis
    # happened (detection is None on the text-only chat-fallback path above).
    case_id = None
    if detection is not None:
        core_cls = to_core_class(detection.get("raw_label") or detection.get("label"))
        if core_cls:
            case_id = _create_case("pest", core_cls, lang)

    return Response(
        stream_with_context(_groq_stream_generator(messages, detection, is_low, lang, case_id=case_id)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx/Render proxy buffering
        },
    )

# ── Pre-warm models ───────────────────────────────────────────────────────────
# Runs once at module import time (i.e. at gunicorn worker boot), not inside a
# request handler — intentional cold-start mitigation so the first real
# request doesn't pay ONNX session creation cost. No external pinger needed:
# Render's free tier sleeps the whole process on idle, so a keep-alive ping
# only delays the inevitable cold start rather than avoiding it.
log.info("prewarm_start")
try:
    detector.prewarm_models()
except Exception as e:
    log.warning("prewarm_failed", extra={"data": {"error": str(e)}})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
