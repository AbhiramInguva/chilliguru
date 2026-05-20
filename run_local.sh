#!/usr/bin/env bash
set -euo pipefail

if [[ -d "venv" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
  echo "Activated virtualenv: venv"
elif [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  echo "Activated virtualenv: .venv"
else
  echo "No virtualenv found (venv/.venv). Using current Python environment."
fi

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running HF Space connectivity test..."
python test_hf_connection.py

echo "Starting Flask app with gunicorn..."
echo "Open http://localhost:5000 and try uploading a chilli photo"
gunicorn app:app --timeout 120 --workers 1 --bind 0.0.0.0:5000
