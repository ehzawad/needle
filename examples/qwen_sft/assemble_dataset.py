"""Assemble train/val SFT JSONL from the data-gen workflow's VERIFIED output.

Input: a JSON file with the workflow return { cards, intents, per_intent: [
  { intent, approved_in_scope:[{question,answer}], approved_hard_neg:[{question}],
    approved_far:[{question}], notes } ] }.
Writes OUT/: cards.json, train.jsonl, validation.jsonl (prompt/completion),
seed_annotated.jsonl (with intent+type), manifest.json.

Usage: python assemble_dataset.py --result seed16/result.json --out seed16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are the Northwind Bank support assistant. Answer only when the request is within your "
    "approved support topics and you can answer from approved facts; rephrase those facts naturally. "
    "If the request is outside those topics or you are uncertain, say briefly that you cannot "
    "reliably help. Do not guess."
)
REFUSALS = [
    "I can't reliably answer that within the topics I support, so I won't guess.",
    "That request is outside the banking topics I'm able to help with.",
    "I don't have a supported answer for that, so I'd rather not guess.",
    "I'm not confident that falls within my supported scope, so I can't answer it reliably.",
    "That's outside what I can help with here; I don't want to give you a wrong answer.",
    "I'm not able to help with that one — it's not among my supported topics.",
]


def rec(q, completion, intent, kind):
    return {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
            "completion": [{"role": "assistant", "content": completion}], "intent": intent, "example_type": kind}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", default="seed16")
    ap.add_argument("--val-per-intent", type=int, default=2)
    args = ap.parse_args()
    data = json.loads(Path(args.result).read_text())
    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

    train, val, ref_i, empties = [], [], 0, []
    for pi in data["per_intent"]:
        intent = pi["intent"]
        insc = pi.get("approved_in_scope") or []
        hard = pi.get("approved_hard_neg") or []
        far = pi.get("approved_far") or []
        if not insc:
            empties.append(intent); continue
        vh = args.val_per_intent
        for j, qa in enumerate(insc):
            (val if j < vh else train).append(rec(qa["question"], qa["answer"], intent, "answer"))
        for j, hn in enumerate(hard):
            r = rec(hn["question"], REFUSALS[ref_i % len(REFUSALS)], intent, "hard_refusal"); ref_i += 1
            (val if j < 1 else train).append(r)
        for j, f in enumerate(far):
            r = rec(f["question"], REFUSALS[ref_i % len(REFUSALS)], intent, "far_refusal"); ref_i += 1
            (val if j < 1 else train).append(r)

    (OUT / "cards.json").write_text(json.dumps(data.get("cards", []), indent=2))
    _write(OUT / "seed_annotated.jsonl", train + val)
    _write(OUT / "train.jsonl", [{"prompt": r["prompt"], "completion": r["completion"]} for r in train])
    _write(OUT / "validation.jsonl", [{"prompt": r["prompt"], "completion": r["completion"]} for r in val])
    ct, cv = _counts(train), _counts(val)
    (OUT / "manifest.json").write_text(json.dumps({
        "intents": data.get("intents", []), "n_intents_with_data": len(data["per_intent"]) - len(empties),
        "empty_intents": empties, "train": ct, "validation": cv,
        "train_total": len(train), "val_total": len(val), "needs_full_human_review": True,
    }, indent=2))
    print(f"train={len(train)} {ct}\nval={len(val)} {cv}\nempty_intents={empties}")


def _counts(rows):
    c = {}
    for r in rows:
        c[r["example_type"]] = c.get(r["example_type"], 0) + 1
    return c


def _write(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
