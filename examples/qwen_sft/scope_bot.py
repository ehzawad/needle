"""File-defined-scope RAG bot — the council-recommended architecture (NO domain SFT).

The domain lives in a FILE (the scope cards). Every query:
  1) scope gate: a TRAINING-FREE structured LLM judge reads the query + the cards
     (includes/excludes) and decides ANSWER / CLARIFY / ABSTAIN — it runs BEFORE
     generation, so a scope-wrong-but-grounded answer can't slip through.
  2) if ANSWER: generate the reply grounded ONLY in the selected card's approved facts.
     if CLARIFY: ask which of the candidate meanings.  if ABSTAIN: refuse.
Edit the cards file → the bot's domain changes, no retraining. At 16 cards all cards
fit the judge context; at 150-500, add frozen retrieval to shortlist (see RAG_SCOPE.md).

Uses the local Qwen3-4B-Instruct-2507 (4-bit) for both the judge and the generator.
Run with the pinned venv: .venv-qlora/bin/python scope_bot.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
REV = "cdbee75f17c01a7cc42f958dc650907174af0554"

JUDGE_SYS = (
    "You are a strict SCOPE GATE. You are given the ONLY supported topics (cards), each with what it "
    "INCLUDES and EXCLUDES, and a user question. Decide, using the cards as the sole authority:\n"
    "- ANSWER: exactly one card's supported goal directly covers the requested action AND object AND "
    "actor/state, and NO excluded item matches the request.\n"
    "- CLARIFY: the request is plausibly in-scope but a scope-defining detail is missing (e.g. which PIN), "
    "with a small set of candidate cards.\n"
    "- ABSTAIN: the request matches an EXCLUDED item of a card, or no card covers it.\n"
    "A high topical similarity is NOT enough — check the excluded list. Output ONLY compact JSON: "
    '{"disposition":"ANSWER|CLARIFY|ABSTAIN","card_id":"<id or null>","candidates":["<id>",...],"reason":"<short>"}'
)
GEN_SYS = (
    "You are the Northwind Bank support assistant. Answer the user's question using ONLY the APPROVED FACTS "
    "below. Rephrase them naturally and conversationally; add no fact, number, step, or claim that is not in "
    "the approved facts. If the facts don't fully cover it, say what you can and note the limit."
)
REFUSAL = "I'm sorry, that's outside the topics I can help with, so I won't guess."


def cards_block(cards):
    out = []
    for c in cards:
        inc = "; ".join(c.get("included", []))
        exc = "; ".join(c.get("excluded", []))
        out.append(f"- {c['intent_id']}: {c['supported_goal']}\n    INCLUDES: {inc}\n    EXCLUDES: {exc}")
    return "\n".join(out)


class ScopeBot:
    def __init__(self, kb="seed16/cards.json"):
        self.cards = json.loads(Path(kb).read_text())
        self.by_id = {c["intent_id"]: c for c in self.cards}
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REV, quantization_config=bnb, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0}).eval()

    def _chat(self, sys, user, max_new=256):
        text = self.tok.apply_chat_template(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        x = self.tok(text, return_tensors="pt").to(self.m.device)
        with torch.no_grad():
            out = self.m.generate(**x, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def gate(self, query):
        raw = self._chat(JUDGE_SYS, f"CARDS:\n{cards_block(self.cards)}\n\nUSER QUESTION: {query}", max_new=200)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            d = json.loads(m.group(0)) if m else {}
        except Exception:
            d = {}
        disp = d.get("disposition", "ABSTAIN")
        if disp not in ("ANSWER", "CLARIFY", "ABSTAIN"):
            disp = "ABSTAIN"
        return {"disposition": disp, "card_id": d.get("card_id"),
                "candidates": d.get("candidates", []), "reason": d.get("reason", "")}

    def respond(self, query):
        g = self.gate(query)
        if g["disposition"] == "ANSWER" and g["card_id"] in self.by_id:
            facts = "\n".join(f"- {f}" for f in self.by_id[g["card_id"]].get("key_facts", []))
            reply = self._chat(GEN_SYS, f"APPROVED FACTS ({g['card_id']}):\n{facts}\n\nUSER QUESTION: {query}", max_new=200)
            return {"disposition": "ANSWER", "card": g["card_id"], "reply": reply, "reason": g["reason"]}
        if g["disposition"] == "CLARIFY":
            cands = [c for c in g["candidates"] if c in self.by_id][:3] or list(self.by_id)[:0]
            opts = " or ".join(self.by_id[c]["supported_goal"].rstrip(".").lower() for c in cands) if cands else "which topic you mean"
            return {"disposition": "CLARIFY", "reply": f"Just to make sure — do you mean {opts}?", "reason": g["reason"]}
        return {"disposition": "ABSTAIN", "reply": REFUSAL, "reason": g["reason"]}


DEMO = [
    ("in-scope", "How do I change my Northwind card PIN?"),
    ("in-scope", "where can i see my bill and when it's due"),
    ("in-scope", "i lost my card, what do i do"),
    ("hard-OOD (explicit doppelganger)", "How do I change the PIN on my SIM card?"),
    ("hard-OOD (explicit doppelganger)", "How do I transfer money to my friend's account at Chase?"),
    ("identical-text ambiguity", "How do I change my PIN?"),
    ("far-OOD", "What's a good recipe for banana bread?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="seed16/cards.json")
    ap.add_argument("--query", default=None)
    ap.add_argument("--interactive", action="store_true", help="type any query and see the live decision")
    args = ap.parse_args()
    bot = ScopeBot(args.kb)

    if args.interactive:
        print("Type a question (Ctrl-D to quit). The gate decision + the model's reason are shown live.")
        import sys
        for line in sys.stdin:
            q = line.strip()
            if not q:
                continue
            g = bot.gate(q)
            r = bot.respond(q)
            print(f"  gate: {g['disposition']} card={g['card_id']}  reason: {g['reason']}")
            print(f"  bot : {r['reply']}\n")
        return

    items = [("query", args.query)] if args.query else DEMO
    for cat, q in items:
        r = bot.respond(q)
        print(f"\n[{cat}] {q}")
        print(f"  -> {r['disposition']}" + (f" (card={r.get('card')})" if r.get("card") else "") + f"  [{r['reason'][:70]}]")
        print(f"     {r['reply']}")


if __name__ == "__main__":
    main()
