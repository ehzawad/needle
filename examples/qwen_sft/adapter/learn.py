"""Turn reviewed weak labels into a gate EXEMPLAR BANK (the cheap, training-free
adapter the council favors first).

Why an exemplar bank before a trained head:
  - instant: no GPU training loop; edit-and-serve.
  - file-edit-safe: an exemplar is pinned to a kb_version; when cards.json changes,
    exemplars from an older kb_version are quarantined, not silently trusted.
  - auditable: every exemplar traces to a real conversation + a human approval.
A trained safe-to-answer head (see FEEDBACK_PIPELINE.md) is the escalation IF the
bank provably cannot close the gap — this module deliberately keeps that door open by
emitting the same (query -> disposition) records a head would train on.

Input : logs/mined_labels.jsonl  (from mine_signals.py)
Filter: keep confidence >= MIN_CONF AND approved (needs_review == False OR a human
        set review == "approve"); drop anything whose kb_version != current.
Output: exemplars.json  — a small, per-card-capped few-shot bank the gate injects.
"""
from __future__ import annotations

import json
from pathlib import Path

from feedback_log import kb_version

LABELS = Path("logs/mined_labels.jsonl")
MIN_CONF = 0.7          # high-confidence signals only
PER_CARD_CAP = 3        # avoid gate-prompt bloat; newest-approved win
GLOBAL_CAP = 40


def load_labels(path=LABELS) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def approved(rec: dict) -> bool:
    if rec.get("review") == "reject":
        return False
    if rec.get("needs_review"):
        return rec.get("review") == "approve"
    return True


def build(kb="seed16/cards.json") -> list[dict]:
    cur = kb_version(kb)
    labels = load_labels()
    kept: list[dict] = []
    for r in labels:
        if r.get("confidence", 0) < MIN_CONF:
            continue
        if not approved(r):
            continue
        if r.get("kb_version") != cur:
            continue  # stale or UNSTAMPED: made under a different/unknown scope file -> quarantine
        disp = r.get("mined_disposition")
        if disp not in ("ANSWER", "CLARIFY", "ABSTAIN"):
            continue
        kept.append({
            "query": r["query"],
            "disposition": disp,
            "card_id": r.get("mined_card"),
            "related_card_ids": r.get("related_card_ids", []),
            "why": r.get("signal"),
            "source_session": r.get("session_id"),
        })
    # de-dup by (query, disposition); cap per card and globally, newest-first
    seen, per_card, bank = set(), {}, []
    for r in reversed(kept):
        key = (r["query"].strip().lower(), r["disposition"])
        if key in seen:
            continue
        seen.add(key)
        # CLARIFY/ABSTAIN have no answer card; bucket them by their related card so
        # boundary exemplars are not all crammed into one capped null bucket.
        cid = r["card_id"] or (r.get("related_card_ids") or ["_global"])[0]
        if per_card.get(cid, 0) >= PER_CARD_CAP:
            continue
        per_card[cid] = per_card.get(cid, 0) + 1
        bank.append(r)
        if len(bank) >= GLOBAL_CAP:
            break
    return bank
