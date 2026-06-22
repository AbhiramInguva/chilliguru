"""
validate_kb.py — sanity-checks the agronomy knowledge base.

Checks:
  1. knowledge/chilli_kb.json: every entry in deficiencies/pests/diseases has
     the required fields (id, display_name, telugu_name, category,
     key_symptoms, causal_agent, management.organic, management.chemical).
  2. knowledge/kb_class_map.json: every one of the 15 classes in
     model_info.json is present, and its kb_ref (when not null) resolves to
     a real entry in chilli_kb.json.

Exits non-zero if any required field is missing. Unmapped model classes and
KB entries with no model class are expected/reported, not treated as errors
(see kb_class_map.json's own notes on why Healthy/Mealybug are unmapped).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FIELDS = [
    "id", "display_name", "telugu_name", "category",
    "key_symptoms", "causal_agent", "management",
]
REQUIRED_MANAGEMENT_FIELDS = ["organic", "chemical"]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_kb_entries(kb):
    errors = []
    for section in ("deficiencies", "pests", "diseases"):
        entries = kb.get(section)
        if entries is None:
            errors.append(f"chilli_kb.json missing top-level section '{section}'")
            continue
        for entry_id, entry in entries.items():
            loc = f"{section}.{entry_id}"
            for field in REQUIRED_FIELDS:
                if field not in entry:
                    errors.append(f"{loc}: missing required field '{field}'")
            mgmt = entry.get("management")
            if isinstance(mgmt, dict):
                for field in REQUIRED_MANAGEMENT_FIELDS:
                    if field not in mgmt:
                        errors.append(f"{loc}: management missing '{field}' key")
            elif "management" in entry:
                errors.append(f"{loc}: 'management' must be an object")
            if entry.get("id") != entry_id:
                errors.append(f"{loc}: id field '{entry.get('id')}' != dict key '{entry_id}'")
    return errors


def validate_class_map(kb, class_map, model_info):
    errors = []
    unmapped = []
    model_classes = model_info.get("classes", [])
    mapped_classes = class_map.get("classes", {})

    for cls in model_classes:
        if cls not in mapped_classes:
            errors.append(f"kb_class_map.json: model class '{cls}' is not present at all")
            continue
        kb_ref = mapped_classes[cls].get("kb_ref")
        if kb_ref is None:
            unmapped.append(cls)
            continue
        section, _, entry_id = kb_ref.partition(".")
        if not kb.get(section, {}).get(entry_id):
            errors.append(
                f"kb_class_map.json: '{cls}' -> kb_ref '{kb_ref}' does not "
                f"resolve to any entry in chilli_kb.json"
            )

    extra = [cls for cls in mapped_classes if cls not in model_classes]
    if extra:
        errors.append(f"kb_class_map.json has classes not in model_info.json: {extra}")

    return errors, unmapped, len(model_classes)


def main():
    kb = load_json(ROOT / "knowledge" / "chilli_kb.json")
    class_map = load_json(ROOT / "knowledge" / "kb_class_map.json")
    model_info = load_json(ROOT / "model_info.json")

    entry_errors = validate_kb_entries(kb)
    map_errors, unmapped, total_classes = validate_class_map(kb, class_map, model_info)

    print(f"chilli_kb.json: {sum(len(kb.get(s, {})) for s in ('deficiencies', 'pests', 'diseases'))} entries checked")
    print(f"kb_class_map.json: {total_classes} model classes checked")

    if unmapped:
        print(f"\nUnmapped model classes ({len(unmapped)}/{total_classes}) — no KB-grounded advice available:")
        for cls in unmapped:
            note = class_map["classes"][cls].get("notes", "")
            print(f"  - {cls}: {note}")

    all_errors = entry_errors + map_errors
    if all_errors:
        print(f"\n{len(all_errors)} ERROR(S):")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
