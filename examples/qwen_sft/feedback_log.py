"""Conversation telemetry — the real-world training signal for the scope gate.

Every served turn is appended to a JSONL log. The POINT of the log is that a
LATER turn in the same session labels the PRIOR gate decision for free:
  - a correction after an ANSWER  ("no, my SIM")     -> we should have CLARIFIED
  - a trivial pick after a CLARIFY ("obviously card") -> we should have ANSWERED
  - a thanks / new topic after ANSWER                 -> the ANSWER was right
  - a rephrase-into-scope after ABSTAIN               -> we over-abstained
`mine_signals.py` turns those follow-ups into weak (query -> correct disposition)
labels, which feed the exemplar bank / gate-head. This is the user's insight made
concrete: production chat IS the label source, not synthetic data.

Schema is intentionally small and stable so it survives card-file edits; the KB
version (a hash of cards.json) is stamped on every record so a label can be tied
to the exact scope file that produced it.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("logs/conversations.jsonl")


def kb_version(kb_path="seed16/cards.json") -> str:
    """Short hash of the scope file, so labels are pinned to the KB that made them."""
    try:
        return hashlib.sha1(Path(kb_path).read_bytes()).hexdigest()[:12]
    except FileNotFoundError:
        return "unknown"


class TurnLogger:
    """Append-only per-turn logger. One JSON object per served turn."""

    def __init__(self, path: Path | str = LOG_PATH, kb_path="seed16/cards.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.kb = kb_version(kb_path)

    def log(self, session_id: str, turn: int, query: str, gate: dict, reply: str,
            shortlist: list[str] | None = None, extra: dict | None = None) -> dict:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": session_id,
            "turn": turn,
            "query": query,
            "disposition": gate.get("disposition"),
            "card_id": gate.get("card_id"),
            "candidates": gate.get("candidates", []),
            "reason": gate.get("reason", ""),
            "reply": reply,
            "kb_version": self.kb,
            "shortlist": shortlist or [],
        }
        if extra:
            rec.update(extra)
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


def read_sessions(path: Path | str = LOG_PATH) -> dict[str, list[dict]]:
    """Group logged turns into ordered conversations keyed by session_id."""
    path = Path(path)
    sessions: dict[str, list[dict]] = {}
    if not path.exists():
        return sessions
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        sessions.setdefault(rec.get("session_id", "?"), []).append(rec)
    for turns in sessions.values():
        turns.sort(key=lambda r: r.get("turn", 0))
    return sessions
