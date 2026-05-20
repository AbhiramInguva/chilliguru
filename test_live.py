#!/usr/bin/env python3
"""
Smoke-test the live Hugging Face Space after deploy.

Usage:
  pip install gradio_client requests pillow
  python3 test_live.py [path/to/image.jpg]

If no image path is given, tries common filenames in the repo, then generates a tiny JPEG.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import requests

SPACE_URL = "https://inguvaaa-chilliguru-detector.hf.space"
MAX_RETRIES = 5
RETRY_DELAY = 30


def space_http_ok() -> bool:
    try:
        resp = requests.get(SPACE_URL.rstrip("/") + "/", timeout=15)
        return resp.status_code == 200
    except OSError:
        return False


def resolve_test_image(argv: list[str]) -> Path | None:
    if len(argv) > 1:
        p = Path(argv[1]).expanduser()
        return p if p.is_file() else None

    for name in ("chilliwhiteflies-300x225.jpg", "images.jpeg", "test.jpg", "test_chilli.jpg"):
        p = Path(name)
        if p.is_file():
            return p
    return None


def make_dummy_jpeg(path: Path) -> None:
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(180, 40, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    path.write_bytes(buf.getvalue())


def main() -> int:
    print("🔍 Testing live HF Space deployment")
    print(f"   URL: {SPACE_URL}\n")

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"   Attempt {attempt}/{MAX_RETRIES}...", end=" ", flush=True)
        if space_http_ok():
            print("✅ Space HTTP is up")
            break
        print(f"not ready, waiting {RETRY_DELAY}s...")
        time.sleep(RETRY_DELAY)
    else:
        print("❌ Space did not respond after retries")
        return 1

    test_img = resolve_test_image(sys.argv)
    use_temp = False
    if test_img is None:
        test_img = Path("_deploy_smoke_test.jpg")
        make_dummy_jpeg(test_img)
        use_temp = True
        print(f"\n   Using generated dummy image: {test_img}")

    print(f"\n   Predicting via Gradio API: {test_img}")
    try:
        from gradio_client import Client, handle_file

        client = Client(SPACE_URL, verbose=False)
        result = client.predict(image=handle_file(str(test_img)), api_name="/predict")
    except ImportError:
        print("   ⚠️  gradio_client not installed. pip install gradio_client")
        return 0
    except Exception as exc:
        print(f"   ❌ Prediction failed: {exc}")
        return 1
    finally:
        if use_temp and test_img.exists():
            try:
                test_img.unlink()
            except OSError:
                pass

    print(f"   Response: {result}")

    if not isinstance(result, dict):
        print("   ⚠️  Non-dict response (check Space API version)")
        return 0

    if result.get("error"):
        print(f"   ❌ Error: {result['error']}")
        return 1

    if result.get("top_detection"):
        det = result["top_detection"]
        print("\n   ✅ Detection response OK")
        print(f"      Label:       {det.get('label', 'N/A')}")
        print(f"      Telugu:      {det.get('telugu', 'N/A')}")
        print(f"      Confidence:  {det.get('confidence', 'N/A')}")
        print(f"      Type:        {det.get('type', 'N/A')}")
        print(f"      Low conf:    {result.get('low_confidence', 'N/A')}")
    elif result.get("low_confidence"):
        print("   ⚠️  Low confidence (model running; image may be ambiguous)")
    else:
        print(f"   ⚠️  Unexpected dict shape: keys={list(result.keys())}")

    print("\n" + "=" * 50)
    print("🎉 Live deployment check finished")
    print(f"   Space: {SPACE_URL}")
    print(f"   API:   {SPACE_URL}/api/predict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
