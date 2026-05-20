import tempfile
import time
import traceback

from PIL import Image


def make_dummy_jpeg(path):
    image = Image.new("RGB", (64, 64), color=(20, 180, 20))
    image.save(path, format="JPEG")


def main():
    total_start = time.perf_counter()
    print("Starting Hugging Face Space connectivity test...\n")

    try:
        import_start = time.perf_counter()
        from gradio_client import Client, handle_file
        import_elapsed = time.perf_counter() - import_start
        print(f"gradio_client import: OK ({import_elapsed:.2f}s)")
    except Exception as exc:
        print(f"gradio_client import: FAILED - {exc}")
        print(traceback.format_exc())
        return 1

    client = None
    space_candidates = ["inguvaaa-chilliguru-detector", "inguvaaa/chilliguru-detector"]
    for space_name in space_candidates:
        try:
            connect_start = time.perf_counter()
            client = Client(space_name, verbose=False)
            connect_elapsed = time.perf_counter() - connect_start
            print(f"HF Space connected successfully via '{space_name}' ({connect_elapsed:.2f}s)")
            break
        except Exception as exc:
            print(f"HF Space connection attempt failed for '{space_name}' - {exc}")

    if client is None:
        print("HF Space connection FAILED for all known IDs.")
        return 1

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        make_dummy_jpeg(tmp_path)
        print(f"Dummy JPEG generated at: {tmp_path}")

        predict_start = time.perf_counter()
        raw_response = client.predict(image=handle_file(tmp_path), api_name="/predict")
        predict_elapsed = time.perf_counter() - predict_start
        print(f"Prediction request completed in {predict_elapsed:.2f}s")
        print("Raw response:")
        print(raw_response)
    except Exception as exc:
        print(f"Prediction FAILED - {exc}")
        print(traceback.format_exc())
        return 1
    finally:
        if tmp_path:
            try:
                import os

                os.unlink(tmp_path)
            except Exception:
                pass

    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal connect + prediction flow time: {total_elapsed:.2f}s")
    print("HF test script finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
