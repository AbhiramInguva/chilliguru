"""
triage.py — vision-guided adaptive triage engine.

The vision cascade (detector.py / the remote HF Space) takes one cursory look
and produces a hypothesis (top label + confidence + candidate list). This
module decides, from that hypothesis alone, whether any follow-up question is
worth asking, and if so picks the SINGLE question that best splits the
current candidate set -- never a fixed generic list. Each answer re-ranks the
candidates; the loop stops as soon as one is clearly ahead or a per-domain
question budget is exhausted.

HARD RULE: nutrient deficiencies are never inferred from the photo. See
seed_deficiency_candidates() -- every deficiency candidate starts at the same
score; only question answers (apply_answer) can move that score. Vision may
only set the boolean "this looks like a nutrition issue, not a pest" bias via
looks_like_deficiency_trigger(), never a specific deficiency id.

All question/option text is sourced from knowledge/triage_rules.json, which
in turn is sourced only from knowledge/chilli_kb.json's existing symptom
text -- this module does not invent any new symptom-to-cause mapping.
"""
import json
import os
import re

_TRIAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_RULES_PATH = os.path.join(_TRIAGE_DIR, "knowledge", "triage_rules.json")

PEST_CANDIDATE_IDS = {
    "aphids", "whitefly_leaf_damage", "fruit_borer", "tobacco_caterpillar",
    "yellow_thrips", "broad_mites", "invasive_black_thrips", "mealybugs",
    "leaf_curl_virus", "bacterial_leaf_spot", "cercospora_leaf_spot", "powdery_mildew",
}
DEFICIENCY_CANDIDATE_IDS = {
    "iron", "manganese", "zinc", "nitrogen", "molybdenum",
    "magnesium", "boron", "calcium", "phosphorous", "potassium", "copper",
}

DEFICIENCY_SEED_SCORE = 50.0
LOOKALIKE_MARGIN = 15.0  # percentage points -- mirrors model_info.json's margin_threshold (0.15)

_rules_cache = None


def load_rules():
    global _rules_cache
    if _rules_cache is None:
        with open(_RULES_PATH, encoding="utf-8") as f:
            _rules_cache = json.load(f)
    return _rules_cache


def max_questions(domain):
    return load_rules().get("max_questions", {}).get(domain, 2)


def resolve_margin():
    return float(load_rules().get("resolve_margin", 12.0))


# ── Branch (a)/(b): pest/disease candidate-set construction ───────────────────

def build_pest_candidates(result, to_core_class_fn):
    """
    Builds an initial {core_class: confidence} dict from result['all_detections'],
    using the app's existing to_core_class() resolver (passed in to avoid a
    circular import — app.py already owns that mapping). Returns {} if there
    is nothing resolvable.
    """
    candidates = {}
    dets = result.get("all_detections") if isinstance(result, dict) else None
    if not isinstance(dets, list):
        top = result.get("top_detection") if isinstance(result, dict) else None
        dets = [top] if isinstance(top, dict) else []
    for d in dets:
        if not isinstance(d, dict):
            continue
        core = to_core_class_fn(d.get("raw_label") or d.get("label"))
        if core and core in PEST_CANDIDATE_IDS and core not in candidates:
            try:
                candidates[core] = float(d.get("confidence", 0))
            except (TypeError, ValueError):
                candidates[core] = 0.0
    return candidates


def is_lookalike_or_low(result, candidates):
    """
    True if detector.py's own low_confidence flag is set, OR the top two
    candidates are within LOOKALIKE_MARGIN of each other (a "lookalike
    cluster" rather than a single clear winner). Both signals are read from
    existing detector.py/model_info.json values -- no new threshold invented.
    """
    if isinstance(result, dict) and result.get("low_confidence"):
        return True
    if len(candidates) >= 2:
        ranked = sorted(candidates.values(), reverse=True)
        if ranked[0] - ranked[1] < LOOKALIKE_MARGIN:
            return True
    return False


# ── Branch (c): deficiency trigger (vision bias only, never a specific id) ───

_YELLOWING_KEYWORDS = (
    # English
    "yellow", "yellowing", "pale", "chloros", "discolor", "discolour",
    "bleach", "whitish leaves", "fading",
    # Telugu — పసుపు (yellow), పాలిపోవడం (paling/fading)
    "పసుపు", "పాలిపో",
    # Hindi — पीला/पीलापन (yellow/yellowing)
    "पीला", "पीलापन",
)


def looks_like_deficiency_trigger(result, user_msg, field_context):
    """
    True if vision found no confident pest/disease (or explicitly "healthy")
    AND the farmer's own words/selections suggest a problem exists. This is
    the ONLY signal vision is allowed to contribute toward a deficiency
    diagnosis -- a yes/no bias, never a specific deficiency. The actual
    deficiency identity always comes from seed_deficiency_candidates() +
    question answers below.
    """
    top = result.get("top_detection") if isinstance(result, dict) else None
    raw_label = ""
    if isinstance(top, dict):
        raw_label = str(top.get("raw_label") or top.get("label") or "").lower()
    no_confident_pest = (
        result.get("success") is False
        or result.get("phase") == 3
        or "healthy" in raw_label
        or top is None
    ) if isinstance(result, dict) else True

    if not no_confident_pest:
        return False

    user_reports_problem = field_context.get("observed") == "curl_yellow"
    if not user_reports_problem and user_msg:
        lower_msg = user_msg.lower()
        user_reports_problem = any(kw in lower_msg for kw in _YELLOWING_KEYWORDS)
    return user_reports_problem


def seed_deficiency_candidates():
    """
    All 11 deficiency ids start at an EQUAL score. Vision contributes nothing
    here by construction -- only apply_answer() (driven by the farmer's own
    answers, sourced from knowledge/triage_rules.json) can move these scores
    apart. This is the hard-rule enforcement point: there is no code path
    that lets a vision/result value influence which specific deficiency leads.
    """
    return {k: DEFICIENCY_SEED_SCORE for k in DEFICIENCY_CANDIDATE_IDS}


# ── Question selection (information gain over the current candidate set) ─────

def _alive_candidates(candidates, margin):
    """Candidates within `margin` of the current leader — the set a question must split."""
    if not candidates:
        return set()
    leader_score = max(candidates.values())
    return {k for k, v in candidates.items() if leader_score - v < margin}


def pick_next_question(domain, candidates, asked_ids):
    """
    Returns the next question dict (id/question/options/applies_to) to ask,
    or None if no remaining question applies to the current live candidate
    set. Picks the first not-yet-asked question (in rulebook order) whose
    applies_to overlaps >=2 of the currently-tied leading candidates --
    i.e. the question most likely to actually split the live ambiguity,
    rather than a fixed generic list.
    """
    rules = load_rules()
    pool = rules["pest_questions"] if domain == "pest" else rules["deficiency_questions"]
    margin = resolve_margin()
    alive = _alive_candidates(candidates, margin)

    for q in pool:
        if q["id"] in asked_ids:
            continue
        applies_to = q.get("applies_to")
        if applies_to is None:
            # Entry question (e.g. def_leaf_age) — always eligible first.
            return q
        if len(set(applies_to) & alive) >= 2:
            return q
    return None


def apply_answer(question, answer_key, candidates):
    """Returns a NEW candidates dict with the chosen option's score deltas applied."""
    updated = dict(candidates)
    option = next((o for o in question["options"] if o["key"] == answer_key), None)
    if option is None:
        return updated
    for core_cls, delta in option.get("scores", {}).items():
        if core_cls in updated:
            updated[core_cls] = updated[core_cls] + delta
    return updated


def check_resolution(domain, candidates, asked_count):
    """
    Returns {"resolved": bool, "leader": id, "ambiguous": bool}.
    Resolved when the leader is ahead of the runner-up by >= resolve_margin,
    OR the per-domain question budget (max_questions) is exhausted -- in
    which case `ambiguous=True` signals the caller to present the
    most-likely candidate with a caveat (soil-test note for deficiencies)
    rather than forcing a confident pick.
    """
    if not candidates:
        return {"resolved": False, "leader": None, "ambiguous": True}
    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    leader, leader_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else -1e9
    clear_lead = (leader_score - runner_up_score) >= resolve_margin()
    budget_exhausted = asked_count >= max_questions(domain)
    if clear_lead:
        return {"resolved": True, "leader": leader, "ambiguous": False}
    if budget_exhausted:
        return {"resolved": True, "leader": leader, "ambiguous": True}
    return {"resolved": False, "leader": leader, "ambiguous": False}
