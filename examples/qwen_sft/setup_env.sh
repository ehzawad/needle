#!/usr/bin/env bash
# Pinned, known-compatible stack for resumable QLoRA SFT of Qwen3-4B on one 24 GB A5000.
# The council found the current shell's stack is unsafe for this run:
#   - TRL 0.17 drops completion_mask under packing -> silently trains on the prompt.
#   - transformers >=4.56 won't reload optimizer state under torch <2.6 (resume breaks).
#   - Python 3.10 site-packages; the project wants >=3.11.
# Use a clean venv with these versions.
set -euo pipefail

python3.11 -m venv .venv-qlora
# shellcheck disable=SC1091
source .venv-qlora/bin/activate

python -m pip install --upgrade pip wheel packaging ninja

# torch 2.6 (cu124) — required so transformers can reload optimizer/scheduler state on resume
python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.0"

python -m pip install \
  "transformers==4.57.6" \
  "trl==0.23.1" \
  "peft==0.19.1" \
  "accelerate==1.14.0" \
  "datasets==4.4.2" \
  "bitsandbytes==0.49.2" \
  "safetensors>=0.6"

# FlashAttention 2 (optional but recommended; lowers activation memory)
MAX_JOBS=8 python -m pip install "flash-attn==2.8.3.post1" --no-build-isolation || \
  echo "flash-attn build failed — set attn_implementation='sdpa' and packing=False in train_qlora.py"

echo
echo "=== select the A5000 by UUID (bare CUDA_VISIBLE_DEVICES=0 picks the A6000 here) ==="
nvidia-smi -L
echo "Then, before training:"
echo "  export CUDA_DEVICE_ORDER=PCI_BUS_ID"
echo "  export CUDA_VISIBLE_DEVICES=<the GPU-UUID of the RTX A5000 from the line above>"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
