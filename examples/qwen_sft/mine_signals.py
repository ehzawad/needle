"""Turn conversation logs into weak (query -> correct-disposition) labels.

This is the user's insight made mechanical: the NEXT user turn tells us whether the
PRIOR gate decision was right. For every gate decision we inspect the following user
turn. The taxonomy and its guardrails follow the Codex Council reconciliation:

  gate ANSWER  + correction  -> decide the TARGET from the ORIGINAL query, not the
                                correction: original already named an OOD thing ->
                                LEAK (should ABSTAIN); original was generic -> should
                                CLARIFY (another user could have meant the in-scope one).
  gate ANSWER  + explicit thanks/success                 -> ACCEPT (card choice ok).
  gate ANSWER  + silence / end of session                -> CENSORED: NO label.
  gate CLARIFY + user just supplies the missing slot      -> JUSTIFIED: NO label.
  gate CLARIFY + explicit annoyance, OR the ORIGINAL query already resolves a
                 candidate's discriminator in-scope (deterministic)  -> OVER_CLARIFY.
  gate ABSTAIN + explicit rephrase into a clearly in-scope ask         -> OVER_ABSTAIN.

Key guardrails baked in (all were council findings):
  - silence is not acceptance;
  - a terse valid clarification answer is not over-clarification;
  - OOD vocabulary is derived from the CARDS (scope_policy), never a hardcoded regex,
    so it tracks card edits;
  - every ANSWER-expanding label (OVER_CLARIFY / OVER_ABSTAIN) is forced to human
    review and can never auto-enter the exemplar bank.

Heuristic miner; an optional `llm_label` hook can adjudicate the uncertain cases.
Conservative by design: when in doubt, emit low confidence or no label.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from feedback_log import read_sessions
from scope_policy import query_hits_ood_cue, resolve_discriminator

OUT_PATH = Path("logs/mined_labels.jsonl")
CARDS_PATH = "seed16/cards.json"

CORRECTION = re.compile(
    r"\b(no+|nope|not (that|it|what)|that'?s not|thats not|wrong|actually|i meant|"
    r"i said|isn'?t what|you misunderstood|not my|i asked (about|for))\b", re.I)
ACCEPT = re.compile(
    r"\b(thanks|thank you|thx|got it|perfect|great|awesome|that works|that worked|"
    r"makes sense|appreciate|cheers|solved|fixed it|exactly what i needed)\b", re.I)
ANNOYED = re.compile(
    r"\b(obviously|of course|as i said|i already said|already told you|clearly|"
    r"like i said|i told you)\b", re.I)
REPHRASE = re.compile(
    r"\b(i mean|i'?m asking|im asking|in other words|specifically|to clarify|"
    r"what i meant|i'?m talking about)\b", re.I)

# labels that would EXPAND answering are always human-reviewed (council: asymmetric safety)
ANSWER_EXPANDING = {"OVER_CLARIFY", "OVER_ABSTAIN"}


def _next_user_turn(turns: list[dict], i: int) -> dict | None:
    for j in range(i + 1, len(turns)):
        if turns[j].get("query"):
            return turns[j]
    return None


def _clarify_was_unnecessary(query: str, candidates: list[str], by_id: dict) -> str | None:
    """Deterministic over-clarify evidence: does the ORIGINAL query already resolve a
    candidate card's discriminator IN_SCOPE? If so, the clarify added friction.
    Returns the resolved card_id, else None."""
    for c in candidates:
        card = by_id.get(c)
        if not card:
            continue
        for d in card.get("required_discriminators", []):
            status, _ = resolve_discriminator(query, d)
            if status == "IN_SCOPE":
                return c
    return None


def mine(sessions: dict[str, list[dict]], cards: list[dict], llm_label=None) -> list[dict]:
    by_id = {c["intent_id"]: c for c in cards}
    labels: list[dict] = []
    for sid, turns in sessions.items():
        for i, t in enumerate(turns):
            disp = t.get("disposition")
            q = t.get("query", "")
            nxt = _next_user_turn(turns, i)
            nxt_q = (nxt or {}).get("query", "") if nxt else ""

            signal = mined_disp = mined_card = conf = None
            related: list[str] = []

            if disp == "ANSWER":
                if nxt and CORRECTION.search(nxt_q):
                    if query_hits_ood_cue(q, cards):          # original itself was OOD -> we leaked
                        signal, mined_disp, conf = "LEAK", "ABSTAIN", 0.9
                    else:                                      # original was ambiguous -> ask
                        signal, mined_disp, conf = "UNDER_CLARIFY", "CLARIFY", 0.85
                        related = [t.get("card_id")] if t.get("card_id") else []
                elif nxt and ACCEPT.search(nxt_q):
                    signal, mined_disp, mined_card, conf = "ACCEPT", "ANSWER", t.get("card_id"), 0.8
                # silence / terminal turn -> CENSORED: emit no label

            elif disp == "CLARIFY":
                resolved = _clarify_was_unnecessary(q, t.get("candidates", []), by_id)
                if nxt and (ANNOYED.search(nxt_q) or resolved):
                    signal, mined_disp = "OVER_CLARIFY", "ANSWER"
                    mined_card = resolved or (t.get("candidates") or [None])[0]
                    conf = 0.6 if resolved else 0.5
                    related = t.get("candidates", [])
                # else: user simply supplied the slot -> JUSTIFIED clarify, no label

            elif disp == "ABSTAIN":
                if nxt and REPHRASE.search(nxt_q) and not query_hits_ood_cue(nxt_q, cards):
                    signal, mined_disp, conf = "OVER_ABSTAIN", "REVIEW", 0.3

            if signal is None:
                continue
            rec = {
                "session_id": sid, "turn": t.get("turn"), "query": q,
                "gate_disposition": disp, "gate_card": t.get("card_id"),
                "signal": signal, "mined_disposition": mined_disp, "mined_card": mined_card,
                "related_card_ids": related, "confidence": conf, "evidence": nxt_q,
                "kb_version": t.get("kb_version"),
                "needs_review": conf < 0.6 or signal in ANSWER_EXPANDING,
            }
            if llm_label is not None:
                rec = llm_label(rec, t, nxt) or rec
            labels.append(rec)
    return labels


def main():
    cards = json.loads(Path(CARDS_PATH).read_text())
    sessions = read_sessions()
    labels = mine(sessions, cards)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in labels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_sig: dict[str, int] = {}
    for r in labels:
        by_sig[r["signal"]] = by_sig.get(r["signal"], 0) + 1
    print(f"mined {len(labels)} weak labels from {len(sessions)} sessions -> {OUT_PATH}")
    for k, v in sorted(by_sig.items()):
        print(f"  {k:14} {v}")
    print(f"  needs human review: {sum(r['needs_review'] for r in labels)}")


if __name__ == "__main__":
    main()
