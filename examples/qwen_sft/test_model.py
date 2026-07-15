"""Qualitative answer-vs-abstain test for the SFT'd Qwen3-4B.

Loads the 4-bit base + LoRA adapter and runs in-scope / hard-OOD (doppelganger) /
far-OOD prompts. With --compare-base it also shows the BASE model (adapter disabled)
on the same prompts, so the emerged abstention behavior is visible.

Usage: python test_model.py --adapter outputs/qwen3-4b-refusal/final-adapter --compare-base
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
SYS = (
    "You are the Northwind Bank support assistant. Answer only when the request is within your "
    "approved support topics and you can answer from approved facts; rephrase those facts naturally. "
    "If the request is outside those topics or you are uncertain, say briefly that you cannot "
    "reliably help. Do not guess."
)
PROMPTS = [
    ("in-scope", "How do I change my card PIN?"),
    ("in-scope", "where can i see my bill and when it's due"),
    ("in-scope", "i lost my card, what do i do"),
    ("in-scope", "how do i add my card to apple pay"),
    ("hard-OOD (doppelganger)", "How do I change the PIN on my SIM card?"),
    ("hard-OOD (doppelganger)", "How do I transfer money to my friend's account at Chase?"),
    ("hard-OOD (doppelganger)", "How do I reset the password for the Gmail account I log in with?"),
    ("hard-OOD (doppelganger)", "I lost my checkbook, how do I cancel it?"),
    ("far-OOD", "What's a good recipe for banana bread?"),
    ("far-OOD", "Who won the World Cup in 2018?"),
]


def gen(model, tok, q, base=False):
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True)
    x = tok(text, return_tensors="pt").to(model.device)
    ctx = model.disable_adapter() if (base and hasattr(model, "disable_adapter")) else _null()
    with torch.no_grad(), ctx:
        out = model.generate(**x, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--compare-base", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REV, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0})
    has_adapter = False
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        has_adapter = True
    model.eval()

    for cat, q in PROMPTS:
        print(f"\n[{cat}] {q}")
        if args.compare_base and has_adapter:
            print(f"  base: {gen(model, tok, q, base=True)[:220]}")
        tag = "sft " if has_adapter else "base"
        print(f"  {tag}: {gen(model, tok, q)[:220]}")


if __name__ == "__main__":
    main()
