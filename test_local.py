import io
import json
import traceback

from PIL import Image

import app as app_module


def _print_result(name, ok, details=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if details:
        print(f"       {details}")


def _make_dummy_jpeg_bytes():
    image = Image.new("RGB", (32, 32), color=(200, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _safe_json(resp):
    try:
        return resp.get_json(silent=True)
    except Exception:
        return None


def test_get_root(client):
    name = "GET / returns 200 and serves frontend"
    try:
        resp = client.get("/")
        ok = resp.status_code == 200 and "text/html" in (resp.content_type or "")
        _print_result(name, ok, f"status={resp.status_code}, content_type={resp.content_type}")
    except Exception:
        _print_result(name, False, traceback.format_exc())


def test_health(client):
    name = "GET /health returns 200 with JSON"
    try:
        resp = client.get("/health")
        body = _safe_json(resp)
        ok = resp.status_code == 200 and isinstance(body, dict)
        _print_result(name, ok, f"status={resp.status_code}, json={json.dumps(body, indent=2, default=str)}")
    except Exception:
        _print_result(name, False, traceback.format_exc())


def test_detect_real_image(client):
    name = "POST /detect with real image-like JPEG bytes"
    try:
        jpeg_bytes = _make_dummy_jpeg_bytes()
        data = {
            "message": "Please detect chilli issue from this image",
            "history": "[]",
            "image": (io.BytesIO(jpeg_bytes), "dummy.jpg"),
        }
        resp = client.post("/detect", data=data, content_type="multipart/form-data")
        body = _safe_json(resp)
        is_json = isinstance(body, dict)
        no_html = "text/html" not in (resp.content_type or "")
        ok = is_json and no_html
        _print_result(name, ok, f"status={resp.status_code}, content_type={resp.content_type}")
        print("       JSON response:", json.dumps(body, indent=2, default=str))
    except Exception:
        _print_result(name, False, traceback.format_exc())


def test_detect_no_image(client):
    name = "POST /detect with no image returns JSON error/safe response"
    try:
        resp = client.post(
            "/detect",
            data={"message": "No image sent", "history": "[]"},
            content_type="multipart/form-data",
        )
        body = _safe_json(resp)
        is_json = isinstance(body, dict)
        no_html = "text/html" not in (resp.content_type or "")
        ok = is_json and no_html
        _print_result(name, ok, f"status={resp.status_code}, content_type={resp.content_type}")
        print("       JSON response:", json.dumps(body, indent=2, default=str))
    except Exception:
        _print_result(name, False, traceback.format_exc())


def test_detect_invalid_file(client):
    name = "POST /detect with invalid file (.txt) handled gracefully"
    try:
        bogus = b"this is not an image"
        data = {
            "message": "Test invalid upload",
            "history": "[]",
            "image": (io.BytesIO(bogus), "not_image.txt"),
        }
        resp = client.post("/detect", data=data, content_type="multipart/form-data")
        body = _safe_json(resp)
        is_json = isinstance(body, dict)
        no_html = "text/html" not in (resp.content_type or "")
        ok = is_json and no_html
        _print_result(name, ok, f"status={resp.status_code}, content_type={resp.content_type}")
        print("       JSON response:", json.dumps(body, indent=2, default=str))
    except Exception:
        _print_result(name, False, traceback.format_exc())


def main():
    app = app_module.app
    app.testing = True

    print("Running local Flask endpoint tests...\n")
    with app.test_client() as client:
        test_get_root(client)
        test_health(client)
        test_detect_real_image(client)
        test_detect_no_image(client)
        test_detect_invalid_file(client)

    print("\nDone.")


if __name__ == "__main__":
    main()
