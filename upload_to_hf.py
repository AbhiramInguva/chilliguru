#!/usr/bin/env python3
"""
Upload chilli_pest_v2.pt, model_info.json, and optionally app.py / requirements.txt
to the Hugging Face Space inguvaaa/chilliguru-detector.

Usage:
  export HF_TOKEN=hf_...   # https://huggingface.co/settings/tokens (write access)
  pip install huggingface_hub
  python upload_to_hf.py --model ./chilli_pest_v2.pt --info ./model_info.json

Optional:
  python upload_to_hf.py --model ./chilli_pest_v2.pt --info ./model_info.json \\
    --app ./hf_space_app.py --requirements ./hf_space_requirements.txt

Or log in once: huggingface-cli login
Then omit HF_TOKEN (library uses cached credentials).
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload files to HF Space chilliguru-detector")
    parser.add_argument("--repo", default="inguvaaa/chilliguru-detector", help="HF Space repo id")
    parser.add_argument("--model", default="chilli_pest_v2.pt", help="Local path to .pt model")
    parser.add_argument("--info", default="model_info.json", help="Local path to model_info.json")
    parser.add_argument("--app", default=None, help="Local path to Gradio app.py to upload as app.py")
    parser.add_argument(
        "--requirements",
        default=None,
        help="Local path to requirements.txt for the Space",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    else:
        # Uses ~/.cache/huggingface/token if you ran `huggingface-cli login`
        login(add_to_git_credential=False)

    api = HfApi()
    repo_id = args.repo
    repo_type = "space"

    def upload(local: str, remote: str, msg: str) -> None:
        if not os.path.isfile(local):
            print(f"ERROR: file not found: {local}", file=sys.stderr)
            raise SystemExit(2)
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=msg,
        )
        print(f"OK: uploaded {remote} from {local}")

    upload(
        args.model,
        "chilli_pest_v2.pt",
        "Add YOLOv8m V2 model (15 classes, merged)",
    )
    upload(
        args.info,
        "model_info.json",
        "Add model_info.json (Telugu names, class types, thresholds)",
    )

    if args.app:
        upload(args.app, "app.py", "Update app.py for V2 model and confidence logic")

    if args.requirements:
        upload(args.requirements, "requirements.txt", "Update Space requirements.txt")

    print(f"\nDone. Space files: https://huggingface.co/spaces/{repo_id}/tree/main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
