"""Resumable QLoRA SFT for Qwen3-4B refusal-aware fine-tuning — one 24 GB A5000.

Council-reconciled, VRAM-measured config (peak ~15.98 GiB reserved on this A5000,
~7.5 GiB headroom). Trains on the completion span only (answer OR refusal), so the
model learns to answer in-scope questions generatively and abstain on OOD.

CRITICAL env pitfalls this script guards against:
  * bare CUDA_VISIBLE_DEVICES=0 selects the A6000 (CUDA enum order) -> assert A5000.
  * TRL < 0.23 drops completion_mask under packing -> silently trains on the prompt.
    We assert the loss mask before the first optimizer step.
  * transformers >= 4.56 won't reload optimizer state under torch < 2.6 -> use torch 2.6.
See setup_env.sh for the pinned, known-compatible stack. Select the A5000 by UUID:
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<A5000 uuid from `nvidia-smi -L`>

Usage:
  SMOKE=1 STOP_AFTER_STEP=1 OUTPUT_DIR=outputs/smoke python train_qlora.py   # 1 step, save
  SMOKE=1 RESUME=auto      OUTPUT_DIR=outputs/smoke python train_qlora.py     # resume -> step 2
  OUTPUT_DIR=outputs/qwen3-4b-refusal python train_qlora.py                   # full run
  RESUME=auto OUTPUT_DIR=outputs/qwen3-4b-refusal python train_qlora.py       # resume after a stop
"""
from __future__ import annotations

import json
import os
import re
import signal
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MAX_LENGTH = 1024
SYSTEM_PROMPT = (
    "You are the support assistant. Answer only when the request is within your approved "
    "support topics and you can answer from approved facts; rephrase those facts naturally. "
    "If the request is outside those topics or you are uncertain, say briefly that you cannot "
    "reliably help. Do not guess."
)

SMOKE = os.getenv("SMOKE", "0") == "1"
STOP_AFTER_STEP = int(os.getenv("STOP_AFTER_STEP", "0"))
OUT = Path(os.getenv("OUTPUT_DIR", "outputs/qwen3-4b-refusal-smoke" if SMOKE else "outputs/qwen3-4b-refusal-qlora"))

# --- guard the GPU: exactly one visible, and it must be the 24 GB A5000 ---
assert torch.cuda.device_count() == 1, f"pin exactly one GPU (got {torch.cuda.device_count()})"
_gpu = torch.cuda.get_device_name(0)
assert "A5000" in _gpu, f"wrong GPU selected: {_gpu} — select the A5000 by UUID"
assert torch.cuda.is_bf16_supported()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
assert tokenizer.eos_token == "<|im_end|>"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.uint8,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, quantization_config=bnb_config,
    dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    device_map={"": 0}, low_cpu_mem_usage=True,
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# --- data: prompt/completion chat records (completion = answer OR refusal) ---
dataset = load_dataset("json", data_files={
    "train": os.getenv("TRAIN_FILE", "data/sft/train.jsonl"),
    "validation": os.getenv("VAL_FILE", "data/sft/validation.jsonl"),
})

def _ensure_chat(row):
    if "prompt" in row and "completion" in row:
        return row
    return {  # accept a flat {question, response} form too
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["question"]}],
        "completion": [{"role": "assistant", "content": row["response"]}],
    }
dataset = dataset.map(_ensure_chat)

# never let right-truncation silently drop part of an answer/refusal
for split in ("train", "validation"):
    for i, row in enumerate(dataset[split]):
        n = len(tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True, add_generation_prompt=False))
        if n > MAX_LENGTH:
            raise ValueError(f"{split}[{i}] = {n} tokens > MAX_LENGTH={MAX_LENGTH}; shorten it, don't truncate the answer.")

if SMOKE:
    dataset["train"] = dataset["train"].select(range(min(32, len(dataset["train"]))))
    dataset["validation"] = dataset["validation"].select(range(min(8, len(dataset["validation"]))))

args = SFTConfig(
    output_dir=str(OUT), overwrite_output_dir=False,
    max_length=MAX_LENGTH, packing=True, packing_strategy="bfd", eval_packing=False,
    completion_only_loss=True, assistant_only_loss=False, eos_token="<|im_end|>",
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=16, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True, fp16=False, tf32=True,
    optim="paged_adamw_8bit", learning_rate=1e-4, weight_decay=0.0, max_grad_norm=1.0,
    lr_scheduler_type="cosine", warmup_ratio=0.05, num_train_epochs=3.0, max_steps=2 if SMOKE else -1,
    logging_steps=1 if SMOKE else 5, logging_first_step=True,
    eval_strategy="steps", eval_steps=1 if SMOKE else 25, prediction_loss_only=True,
    save_strategy="steps", save_steps=1 if SMOKE else 25, save_total_limit=2 if SMOKE else 3,
    save_safetensors=True, save_only_model=False,
    seed=42, data_seed=42, dataloader_num_workers=0, ignore_data_skip=False,
    restore_callback_states_from_checkpoint=True, report_to=[],
)


class CompleteCheckpointMarker(TrainerCallback):
    """Write _SUCCESS only after a checkpoint is fully saved (survives a kill mid-write)."""
    def on_save(self, a, state, control, **kw):
        ckpt = Path(a.output_dir) / f"checkpoint-{state.global_step}"
        tmp, ok = ckpt / "_SUCCESS.tmp", ckpt / "_SUCCESS"
        with tmp.open("w") as f:
            f.write(f"{state.global_step}\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, ok)
        return control


_stop = {"flag": False}
def _request_stop(sig, frame):
    _stop["flag"] = True
signal.signal(signal.SIGTERM, _request_stop)
signal.signal(signal.SIGINT, _request_stop)


class GracefulStop(TrainerCallback):
    """SIGTERM/SIGINT (or STOP_AFTER_STEP) -> checkpoint at the next completed step, then stop."""
    def on_step_end(self, a, state, control, **kw):
        if _stop["flag"] or (STOP_AFTER_STEP and state.global_step >= STOP_AFTER_STEP):
            control.should_save = True
            control.should_training_stop = True
        return control


def _ckpt_num(p):
    m = re.fullmatch(r"checkpoint-(\d+)", p.name)
    return int(m.group(1)) if m else -1


def _valid_ckpt(path: Path) -> bool:
    REQ = {"adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
           "rng_state.pth", "trainer_state.json", "_SUCCESS"}
    try:
        if not REQ.issubset(p.name for p in path.iterdir()):
            return False
        state = json.loads((path / "trainer_state.json").read_text())
        if state["global_step"] != _ckpt_num(path):
            return False
        with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as f:
            for k in f.keys():
                f.get_tensor(k)  # catch a truncated safetensors
        torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
        torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True)
        torch.load(path / "rng_state.pth", map_location="cpu", weights_only=False)
        return True
    except Exception as e:
        print(f"ignoring incomplete checkpoint {path}: {e}")
        return False


def _latest_complete(output_dir: Path) -> str:
    for c in sorted(output_dir.glob("checkpoint-*"), key=_ckpt_num, reverse=True):
        if _valid_ckpt(c):
            return str(c)
    raise RuntimeError(f"no complete checkpoint in {output_dir}")


trainer = SFTTrainer(
    model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["validation"],
    processing_class=tokenizer, peft_config=lora_config,
    callbacks=[CompleteCheckpointMarker(), GracefulStop()],
)

# assert loss is masked to the completion (catches the TRL packing/masking failure)
packed = trainer.train_dataset[0]
assert "completion_mask" in packed, "completion_mask missing — TRL too old? use trl>=0.23.1"
col = trainer.data_collator([packed])
labels = col["labels"].reshape(-1)
mask = torch.tensor(packed["completion_mask"], dtype=torch.bool).reshape(-1)
assert torch.all(labels[~mask] == -100) and torch.any(labels[mask] != -100)
print("trainable completion text:", tokenizer.decode(labels[labels != -100].tolist())[:200])

_spec = os.getenv("RESUME", "")
resume = _latest_complete(OUT) if _spec == "auto" else (_spec or None)

torch.cuda.reset_peak_memory_stats()
trainer.train(resume_from_checkpoint=resume)
peak = torch.cuda.max_memory_reserved() / 2**30
print(f"peak reserved={peak:.2f} GiB")
if SMOKE and peak > 18.0:
    raise RuntimeError(f"smoke peak {peak:.2f} GiB exceeds the 18 GiB gate")

trainer.save_model(str(OUT / "final-adapter"))
tokenizer.save_pretrained(OUT / "final-adapter")
trainer.save_state()
print(f"done -> {OUT/'final-adapter'}")
