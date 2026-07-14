"""Recover the ultracode workflow's generated data from its journal and assemble the
SFT seed dataset. The verify phase hit a session limit, so this also does a light
programmatic review (the human-in-the-loop 'furnish' step); full review still owed
per SFT_ABSTAIN.md before scaling.

Writes to data/sft/seed/: cards.json, train.jsonl, validation.jsonl (prompt/completion),
seed_annotated.jsonl (with intent+type labels for review), manifest.json.
"""
from __future__ import annotations

import json
from pathlib import Path

JOURNAL = Path("/mnt/sdb/arafat/.claude/projects/-mnt-sdb-arafat-cactus-needle/"
               "29799005-6a0b-4e05-b3e8-d4667fad0b1f/subagents/workflows/wf_20dc2648-f1d/journal.jsonl")
OUT = Path("seed")
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
# content signatures to map each anonymous gen blob to its intent (card_declined failed)
SIGN = {
    "card_arrival": ["deliver", "track", "arriv", "5-7", "dispatch", "shipment"],
    "change_pin": ["pin", "4-digit", "atm"],
    "bill_balance": ["statement", "bill", "due date", "20th", "minimum payment"],
    "international_fees": ["2.5%", "foreign", "international", "currency", "exchange rate"],
    "report_lost_card": ["lost", "stolen", "freeze", "replacement", "hotline"],
    "transfer": ["transfer", "between my", "between your", "25,000", "move money", "savings"],
    "reset_password": ["password", "otp", "forgot password", "login", "one-time passcode"],
}


def load_journal():
    cards, blobs = None, []
    for line in JOURNAL.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") != "result":
            continue
        v = r.get("result") or {}
        if isinstance(v, dict) and "cards" in v:
            cards = v["cards"]
        elif isinstance(v, dict) and "in_scope" in v:
            blobs.append(v)
    return cards, blobs


def score(blob, terms):
    text = " ".join(x["question"] + " " + x.get("answer", "") for x in blob["in_scope"]).lower()
    return sum(text.count(t) for t in terms)


def assign(blobs):
    """Greedy unique assignment of each blob to its best-matching intent."""
    pairs = sorted(((score(b, SIGN[i]), bi, i) for bi, b in enumerate(blobs) for i in SIGN),
                   reverse=True)
    used_b, used_i, out = set(), set(), {}
    for s, bi, intent in pairs:
        if bi in used_b or intent in used_i:
            continue
        used_b.add(bi); used_i.add(intent); out[intent] = blobs[bi]
    return out


def rec(q, completion, intent, kind):
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
        "completion": [{"role": "assistant", "content": completion}],
        "intent": intent, "example_type": kind,
    }


def main():
    cards, blobs = load_journal()
    assert cards and len(blobs) == 7, f"expected 8 cards + 7 gen blobs, got {len(cards or [])}/{len(blobs)}"
    by_intent = assign(blobs)
    print("mapped intents:", sorted(by_intent))

    train, val, ref_i = [], [], 0
    for intent, blob in by_intent.items():
        # in-scope answers: hold out last 2 for validation
        for j, qa in enumerate(blob["in_scope"]):
            (val if j >= 10 else train).append(rec(qa["question"], qa["answer"], intent, "answer"))
        # refusals: hard-neg + far -> abstain; hold out 1 hard + 1 far per intent
        for j, hn in enumerate(blob["hard_neg"]):
            r = rec(hn["question"], REFUSALS[ref_i % len(REFUSALS)], intent, "hard_refusal"); ref_i += 1
            (val if j >= 5 else train).append(r)
        for j, f in enumerate(blob["far"]):
            r = rec(f["question"], REFUSALS[ref_i % len(REFUSALS)], intent, "far_refusal"); ref_i += 1
            (val if j >= 3 else train).append(r)

    # light programmatic review: flag any hard-neg question that looks in-scope for its own intent
    flags = []
    for r in train + val:
        if r["example_type"] == "hard_refusal":
            ql = r["prompt"][1]["content"].lower()
            hits = [k for k, terms in SIGN.items() if sum(ql.count(t) for t in terms) >= 2]
            if r["intent"] in hits:
                flags.append(r["prompt"][1]["content"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cards.json").write_text(json.dumps(cards, indent=2))
    _write(OUT / "seed_annotated.jsonl", train + val)
    _write(OUT / "train.jsonl", [_clean(r) for r in train])
    _write(OUT / "validation.jsonl", [_clean(r) for r in val])
    counts = _counts(train), _counts(val)
    (OUT / "manifest.json").write_text(json.dumps({
        "source": "ultracode workflow wf_20dc2648-f1d (verify phase hit session limit; reviewed inline)",
        "intents": sorted(by_intent), "missing_intent": "card_declined (gen failed on limit)",
        "train": counts[0], "validation": counts[1],
        "review_flags": len(flags), "needs_full_human_review": True,
    }, indent=2))

    print(f"train={len(train)} {counts[0]}  val={len(val)} {counts[1]}")
    print(f"review flags (hard-neg looking in-scope): {len(flags)}")
    print("\n--- sample records ---")
    for r in [train[0], train[10], train[11]]:
        print(f"[{r['example_type']}/{r['intent']}] Q: {r['prompt'][1]['content']}")
        print(f"   A: {r['completion'][0]['content'][:110]}")


def _clean(r):
    return {"prompt": r["prompt"], "completion": r["completion"]}


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
