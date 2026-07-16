"""Frozen-file behavioral regressions for scope_bot.py and eval50.py.

Imports the real scope_bot module (which imports torch) but never LOADS the model —
these exercise pure methods (`_load_exemplars`, the `respond`->`_respond_from_gate`
dispatch) with lightweight fakes, so they stay CPU-fast and GPU-free.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scope_bot import ScopeBot  # noqa: E402


class LoadExemplarsQuarantineTest(unittest.TestCase):
    """A bank is trusted ONLY when its kb_version exactly matches the current scope
    file. Missing/null/stale versions must be quarantined (fail closed)."""

    def _bank(self, tmp: str, kb_version) -> Path:
        p = Path(tmp) / "exemplars.json"
        p.write_text(json.dumps({"kb_version": kb_version,
                                 "exemplars": [{"query": "q", "disposition": "CLARIFY", "card_id": None}]}))
        return p

    def test_matching_version_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kb = Path(tmp) / "cards.json"
            kb.write_text("[]")
            cur = hashlib.sha1(kb.read_bytes()).hexdigest()[:12]
            got = ScopeBot._load_exemplars(None, str(self._bank(tmp, cur)), str(kb))
            self.assertEqual(len(got), 1)

    def test_null_and_wrong_version_are_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kb = Path(tmp) / "cards.json"
            kb.write_text("[]")
            for bad in (None, "deadbeef0000", "not-the-hash"):
                got = ScopeBot._load_exemplars(None, str(self._bank(tmp, bad)), str(kb))
                self.assertEqual(got, [], msg=f"kb_version={bad!r} must quarantine")


class RespondGatesOnceTest(unittest.TestCase):
    """The interactive double-gate fix: respond() must call the model gate exactly once."""

    def test_respond_calls_gate_exactly_once(self) -> None:
        calls = {"gate": 0}
        fake = types.SimpleNamespace()

        def _gate(q: str) -> dict:
            calls["gate"] += 1
            return {"disposition": "ABSTAIN", "card_id": None, "candidates": [], "reason": "x"}

        fake.gate = _gate
        fake._respond_from_gate = lambda q, g: {"disposition": g["disposition"], "reply": "", "reason": g["reason"]}
        ScopeBot.respond(fake, "hello")
        self.assertEqual(calls["gate"], 1)


class Eval50UnderClarifyTest(unittest.TestCase):
    """under-clarify must count ambiguous ANSWER (a guess), never ambiguous ABSTAIN."""

    def test_source_uses_answer_predicate_not_total_minus_clarify(self) -> None:
        src = (_ROOT / "eval50.py").read_text(encoding="utf-8")
        self.assertNotIn("amb_total - amb_clarify", src)
        self.assertIn('c == "ambiguous" and d == "ANSWER"', src)


if __name__ == "__main__":
    unittest.main()
