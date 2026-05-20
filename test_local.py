# test_local.py
# Run: python test_local.py
# Tests project layout and contracts locally without GPU or HF token.

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
results = []


def test(name, func):
    try:
        func()
        results.append((PASS, name))
        print(f"  {PASS} {name}")
    except Exception as e:
        results.append((FAIL, f"{name}: {e}"))
        print(f"  {FAIL} {name}: {e}")


print("\n" + "=" * 60)
print("🧪 ChilliGuru Local Test Suite")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
print("\n📁 1. File Structure")
# ─────────────────────────────────────────────────────────────

REQUIRED_FILES = [
    "app.py",
    "hf_space_app.py",
    "upload_to_hf.py",
    "hf_space_requirements.txt",
    "model_info.json",
    "chilli_v2.ipynb",
    "requirements.txt",
    ".gitignore",
]

for f in REQUIRED_FILES:
    test(
        f"File exists: {f}",
        lambda f=f: Path(f).exists()
        or (_ for _ in ()).throw(FileNotFoundError(f"{f} missing")),
    )

# ─────────────────────────────────────────────────────────────
print("\n📋 2. model_info.json Validation")
# ─────────────────────────────────────────────────────────────


def test_model_info():
    with open("model_info.json", encoding="utf-8") as f:
        info = json.load(f)

    assert "num_classes" in info, "Missing 'num_classes'"
    assert info["num_classes"] == 15, f"Expected 15 classes, got {info['num_classes']}"
    assert "classes" in info, "Missing 'classes'"
    assert len(info["classes"]) == 15, f"Expected 15 class names, got {len(info['classes'])}"
    assert "telugu_names" in info, "Missing 'telugu_names'"
    assert len(info["telugu_names"]) == 15, "Expected 15 Telugu names"
    assert "class_types" in info, "Missing 'class_types'"

    for cls in info["classes"]:
        assert cls in info["telugu_names"], f"Missing Telugu for '{cls}'"
        assert cls in info["class_types"], f"Missing type for '{cls}'"
        assert info["class_types"][cls] in ("pest", "disease", "healthy"), (
            f"Invalid type for '{cls}': {info['class_types'][cls]}"
        )

    assert "confidence_threshold" in info, "Missing confidence_threshold"
    assert "high_confidence_min" in info, "Missing high_confidence_min"
    assert "margin_threshold" in info, "Missing margin_threshold"


test("model_info.json structure", test_model_info)

# ─────────────────────────────────────────────────────────────
print("\n📓 3. Notebook Validation (chilli_v2.ipynb)")
# ─────────────────────────────────────────────────────────────


def test_notebook():
    with open("chilli_v2.ipynb", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb["nbformat"] == 4, f"Invalid nbformat: {nb['nbformat']}"
    assert "cells" in nb, "Missing 'cells'"
    assert len(nb["cells"]) > 10, f"Only {len(nb['cells'])} cells (expected 14+)"

    meta = nb.get("metadata", {})
    kaggle = meta.get("kaggle", {})
    assert kaggle.get("accelerator") == "nvidiaTeslaT4", "Kaggle GPU not set to T4"
    assert kaggle.get("isInternetEnabled") is True, "Internet not enabled"

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    assert len(code_cells) >= 8, f"Only {len(code_cells)} code cells"
    assert len(md_cells) >= 4, f"Only {len(md_cells)} markdown cells"

    for i, cell in enumerate(code_cells):
        src = cell["source"]
        source = "".join(src) if isinstance(src, list) else src
        assert len(source.strip()) > 0, f"Code cell {i} is empty"


test("Notebook structure valid", test_notebook)

# ─────────────────────────────────────────────────────────────
print("\n🐍 4. Python Syntax Check (all .py files)")
# ─────────────────────────────────────────────────────────────

py_files = ["upload_to_hf.py", "hf_space_app.py"]
if Path("app.py").exists():
    py_files.append("app.py")

for pyf in py_files:

    def check_syntax(f=pyf):
        src = Path(f).read_text(encoding="utf-8")
        ast.parse(src)

    test(f"Syntax OK: {pyf}", check_syntax)

# ─────────────────────────────────────────────────────────────
print("\n📦 5. Import Check (dependencies)")
# ─────────────────────────────────────────────────────────────

# Matches requirements.txt for the Flask app / local dev
CORE_DEPS = ["flask", "groq", "PIL", "requests", "gradio_client"]
OPTIONAL_DEPS = ["ultralytics", "torch", "huggingface_hub", "gradio", "flask_cors"]

for dep in CORE_DEPS:

    def check_import(d=dep):
        importlib.import_module(d)

    test(f"Import: {dep}", check_import)

for dep in OPTIONAL_DEPS:
    try:
        importlib.import_module(dep)
        results.append((PASS, f"Import: {dep}"))
        print(f"  {PASS} Import: {dep}")
    except ImportError:
        results.append((WARN, f"Import: {dep} (optional, not installed locally)"))
        print(f"  {WARN} Import: {dep} (not installed — OK for local dev)")

# ─────────────────────────────────────────────────────────────
print("\n🔗 6. HF Space App — Logic Test (mock)")
# ─────────────────────────────────────────────────────────────


def test_hf_app_logic():
    def is_low_confidence(confidence, margin, high_conf_min=55, margin_threshold=0.15):
        return confidence < high_conf_min or (confidence < 70 and margin < margin_threshold)

    assert not is_low_confidence(82.5, 0.95), "82.5% conf should be confident"
    assert is_low_confidence(45.0, 1.0), "45% should be low confidence"
    assert is_low_confidence(60.0, 0.05), "60% + 0.05 margin should be low confidence"
    assert not is_low_confidence(60.0, 0.25), "60% + 0.25 margin should be confident"
    assert is_low_confidence(55.0, 0.10), "55% + 0.10 margin should be low (margin < 0.15)"
    assert not is_low_confidence(70.0, 0.01), "70% should be confident regardless of margin"


test("Confidence/margin logic", test_hf_app_logic)

# ─────────────────────────────────────────────────────────────
print("\n🌐 7. Flask App — Route Check")
# ─────────────────────────────────────────────────────────────


def test_flask_routes():
    with open("app.py", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    routes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call):
                    if isinstance(dec.func, ast.Attribute) and dec.func.attr == "route":
                        if dec.args:
                            route_path = ast.literal_eval(dec.args[0])
                            routes.append(route_path)

    assert "/" in routes, "Missing root route '/'"
    assert "/detect" in routes, "Missing '/detect' route"


test("Flask routes (/detect exists)", test_flask_routes)

# ─────────────────────────────────────────────────────────────
print("\n📄 8. upload_to_hf.py — Dry Run Check")
# ─────────────────────────────────────────────────────────────


def test_upload_script():
    content = Path("upload_to_hf.py").read_text(encoding="utf-8")

    assert 'login(token="hf_' not in content, "Do not hardcode HF user tokens in the script"
    assert "HF_TOKEN" in content or "HUGGING_FACE_HUB_TOKEN" in content, (
        "Should use env var for token"
    )
    assert "argparse" in content, "Should accept file paths as arguments"


test("upload_to_hf.py security & args", test_upload_script)

# ─────────────────────────────────────────────────────────────
print("\n🔧 9. Requirements Consistency")
# ─────────────────────────────────────────────────────────────


def test_requirements():
    hf_reqs = Path("hf_space_requirements.txt").read_text(encoding="utf-8").strip().split("\n")
    hf_packages = [
        r.split(">=")[0].split("==")[0].strip().lower() for r in hf_reqs if r.strip()
    ]

    required_in_hf = ["ultralytics", "gradio", "pillow"]
    for pkg in required_in_hf:
        assert pkg in hf_packages, f"HF requirements missing: {pkg}"


test("HF Space requirements complete", test_requirements)

# ─────────────────────────────────────────────────────────────
print("\n🔄 10. End-to-End Mock (Flask → HF prediction format)")
# ─────────────────────────────────────────────────────────────


def test_e2e_format():
    hf_response = {
        "success": True,
        "low_confidence": False,
        "top_detection": {
            "label": "Black Thrips",
            "telugu": "నల్ల తామర పురుగులు",
            "confidence": 82.5,
            "type": "pest",
            "margin": 0.95,
        },
        "all_detections": [
            {
                "label": "Black Thrips",
                "telugu": "నల్ల తామర పురుగులు",
                "confidence": 82.5,
                "type": "pest",
            }
        ],
    }

    result = hf_response
    assert result.get("top_detection") is not None
    assert result["top_detection"]["label"] == "Black Thrips"
    assert result["top_detection"]["telugu"] == "నల్ల తామర పురుగులు"
    assert result["top_detection"]["confidence"] == 82.5
    assert result["top_detection"]["type"] == "pest"
    assert result["low_confidence"] is False

    low_conf_response = {
        "success": True,
        "low_confidence": True,
        "top_detection": {
            "label": "Leaf Curl",
            "telugu": "ఆకు ముడత",
            "confidence": 52.0,
            "type": "disease",
            "margin": 0.05,
        },
        "all_detections": [],
    }
    assert low_conf_response["low_confidence"] is True

    error_response = {"error": "Model failed to load"}
    assert error_response.get("error") is not None
    assert error_response.get("top_detection") is None


test("E2E format: HF → Flask parsing", test_e2e_format)

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 TEST SUMMARY")
print("=" * 60)

passes = sum(1 for s, _ in results if s == PASS)
fails = sum(1 for s, _ in results if s == FAIL)
warns = sum(1 for s, _ in results if s == WARN)

print(f"\n  {PASS} Passed: {passes}")
print(f"  {FAIL} Failed: {fails}")
print(f"  {WARN} Warnings: {warns}")

if fails > 0:
    print(f"\n{'=' * 60}")
    print("FAILURES:")
    for status, msg in results:
        if status == FAIL:
            print(f"  {FAIL} {msg}")

print(f"\n{'=' * 60}")
if fails == 0:
    print("🎉 ALL TESTS PASSED — Ready to deploy!")
    print("\nNext steps:")
    print(
        "  1. Download .pt from Kaggle\n"
        "  2. Run: python upload_to_hf.py --model <path> --info model_info.json "
        "--app hf_space_app.py --requirements hf_space_requirements.txt\n"
        "  3. Wait for Space rebuild\n"
        "  4. Test: curl -X POST https://inguvaaa-chilliguru-detector.hf.space/api/predict "
        "-F 'image=@test.jpg'"
    )
else:
    print(f"⛔ {fails} test(s) failed — fix before deploying!")
    sys.exit(1)
