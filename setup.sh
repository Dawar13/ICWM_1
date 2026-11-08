#!/usr/bin/env bash
# Creates a virtual environment inside this folder. No sudo, no system-wide changes.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
# If the machine already has a CUDA build of torch in another environment, you may point PYTHON at it instead.
python -c "import torch" 2>/dev/null || pip install torch
pip install -r requirements.txt
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU 0: {torch.cuda.get_device_name(0)}, {free/1e9:.1f} GB free of {total/1e9:.1f} GB")
EOF
echo "Activate with: source .venv/bin/activate"
