#!/usr/bin/env bash
# Pinned runtime for the file-scope bot + the A5000 pipeline path (one 24 GB A5000).
# The bot loads Qwen3-4B-Instruct-2507 in 4-bit (bitsandbytes) with SDPA attention;
# this is inference only — no training/SFT stack (no trl / peft / datasets / flash-attn).
# The venv dir keeps its historical name (.venv-qlora) because the docs and pipeline
# commands reference that path.
set -euo pipefail

python3.11 -m venv .venv-qlora
# shellcheck disable=SC1091
source .venv-qlora/bin/activate

python -m pip install --upgrade pip wheel packaging

# torch 2.6 (cu124)
python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.0"

python -m pip install \
  "transformers==4.57.6" \
  "accelerate==1.14.0" \
  "bitsandbytes==0.49.2" \
  "safetensors>=0.6"

echo
echo "=== A5000-only ==="
echo "The pipeline pins the A5000 by UUID for you (pipeline/device_guard.py, fail-closed)."
echo "For a bare 'scope_bot.py' run, select it manually — plain CUDA_VISIBLE_DEVICES=0 picks the A6000 here:"
nvidia-smi -L
echo "  export CUDA_DEVICE_ORDER=PCI_BUS_ID"
echo "  export CUDA_VISIBLE_DEVICES=<the GPU-UUID of the RTX A5000 from the list above>"
