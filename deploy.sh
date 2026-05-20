#!/usr/bin/env bash
# deploy.sh — Full deployment pipeline (local tests → HF Space upload → live smoke test)
# Usage: ./deploy.sh [path_to_model.pt]
#
# Requires: HF_TOKEN (write) for upload_to_hf.py
# Optional: KAGGLE_KERNEL=owner/slug  + kaggle CLI to auto-download weights when no .pt path

set -euo pipefail

echo "🌶️ ChilliGuru Deployment Pipeline"
echo "=================================="

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "❌ Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) first: export HF_TOKEN=hf_xxx"
  exit 1
fi

MODEL_PATH="${1:-chilli_pest_v2.pt}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "📥 Model not found at: $MODEL_PATH"
  if command -v kaggle >/dev/null 2>&1 && [ -n "${KAGGLE_KERNEL:-}" ]; then
    echo "   Downloading kernel output: $KAGGLE_KERNEL → ./kaggle_output/"
    mkdir -p ./kaggle_output
    kaggle kernels output "$KAGGLE_KERNEL" -p ./kaggle_output/
    MODEL_PATH="./kaggle_output/chilli_pest_v2.pt"
  else
    echo "❌ Model file missing."
    echo "   Pass a path: ./deploy.sh /path/to/chilli_pest_v2.pt"
    echo "   Or set KAGGLE_KERNEL=owner/notebook-slug and install kaggle CLI + ~/.kaggle/kaggle.json"
    exit 1
  fi
fi

echo "✅ Model: $MODEL_PATH ($(du -h "$MODEL_PATH" | cut -f1))"

echo ""
echo "🧪 Running local tests..."
python3 test_local.py

echo ""
echo "🚀 Uploading to HF Space..."
python3 upload_to_hf.py \
  --model "$MODEL_PATH" \
  --info model_info.json \
  --app hf_space_app.py \
  --requirements hf_space_requirements.txt

echo ""
echo "⏳ Waiting for Space to rebuild (60s)..."
sleep 60

echo ""
echo "🔍 Testing live endpoint..."
python3 test_live.py

echo ""
echo "=================================="
echo "🎉 Deployment complete!"
