"""Evaluation + serving + observability tests — the eval/runtime lane's contract.

Real-world analog: the unit suite that proves a rollout's AnalysisTemplate math and its
inference boundary behave BEFORE any GPU is touched. Everything here runs on CPU with
injected mock gates/responders; nothing imports torch, so ``unittest discover`` stays
fast and CI-safe.

Covers the lane's required cases:
  * ``evaluate`` on a perfect mock gate yields 0/25 harmful, 20/20 right-card,
    5/5 ambiguous-clarify, and ``passed``;
  * ``require_offline_gate`` raises on a regressed report;
  * serving returns a safe non-answer when the gate and the original respond disagree.
Plus policy-agreement, structured-log, metrics, drift, shadow, and canary coverage.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import evaluation, observability, serving  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.contracts import TurnResult  # noqa: E402
from pipeline.source_fingerprint import (  # noqa: E402
    eval_suite_sha256,
    load_eval50_cases,
)

_PKG = _ROOT / "pipeline"
_EVAL_SOURCE = _ROOT / "eval50.py"
_CARDS = _ROOT / "seed16" / "cards.json"


def _perfect_gate(by_query):
    """A mock gate that returns the ideal decision for every case in the suite."""

    def gate(query):
        case = by_query[query]
        if case.category == "in_scope":
            return {
                "disposition": "ANSWER",
                "card_id": case.expected_card,
                "candidates": [],
                "reason": "mock-perfect",
            }
        if case.category == "ambiguous":
            return {
                "disposition": "CLARIFY",
                "card_id": None,
                "candidates": [],
                "reason": "mock-perfect",
            }
        return {
            "disposition": "ABSTAIN",
            "card_id": None,
            "candidates": [],
            "reason": "mock-perfect",
        }

    return gate


class _CaptureLogger:
    """Minimal TurnLogger stand-in that records every log() call for assertions."""

    def __init__(self):
        self.calls = []

    def log(self, session_id, turn, query, gate, reply, shortlist=None, extra=None):
        record = {
            "session_id": session_id,
            "turn": turn,
            "query": query,
            "gate": gate,
            "reply": reply,
            "extra": extra or {},
        }
        self.calls.append(record)
        return record


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.cases = load_eval50_cases(_EVAL_SOURCE)
        self.by_query = {c.query: c for c in self.cases}
        # The mock keys on the query string, so the suite's queries must be unique.
        self.assertEqual(len(self.by_query), len(self.cases))
        self.suite_sha = eval_suite_sha256(self.cases)

    def test_perfect_gate_yields_release_floor(self):
        report = evaluation.evaluate(
            _perfect_gate(self.by_query),
            self.cases,
            artifact_id="sha256:mock",
            backend="mock",
            suite_sha256=self.suite_sha,
            device=None,
        )
        self.assertEqual(report.harmful_answers, 0)
        self.assertEqual(report.harmful_total, 25)
        self.assertEqual(report.right_card_answers, 20)
        self.assertEqual(report.in_scope_total, 20)
        self.assertEqual(report.wrong_card_answers, 0)
        self.assertEqual(report.ambiguous_clarifies, 5)
        self.assertEqual(report.ambiguous_total, 5)
        self.assertEqual(report.errors, 0)
        self.assertTrue(report.passed)
        self.assertEqual(len(report.predictions), 50)

    def test_gate_exception_counts_as_error_not_leak(self):
        def broken_gate(query):
            raise RuntimeError("boom")

        report = evaluation.evaluate(
            broken_gate,
            self.cases,
            artifact_id="sha256:mock",
            backend="mock",
            suite_sha256=self.suite_sha,
            device=None,
        )
        self.assertEqual(report.errors, 50)
        self.assertEqual(report.harmful_answers, 0)  # errors score as safe ABSTAIN
        self.assertFalse(report.passed)

    def test_require_offline_gate_passes_perfect_report(self):
        config = load_config(_PKG / "config.ci.json")
        report = evaluation.evaluate(
            _perfect_gate(self.by_query),
            self.cases,
            artifact_id="sha256:mock",
            backend="mock",
            suite_sha256=self.suite_sha,
            device=None,
        )
        evaluation.require_offline_gate(report, config)  # must not raise

    def test_require_offline_gate_raises_on_regression(self):
        config = load_config(_PKG / "config.ci.json")
        report = evaluation.evaluate(
            _perfect_gate(self.by_query),
            self.cases,
            artifact_id="sha256:mock",
            backend="mock",
            suite_sha256=self.suite_sha,
            device=None,
        )
        regressed = dataclasses.replace(report, harmful_answers=1, passed=False)
        with self.assertRaises(evaluation.PromotionGateError):
            evaluation.require_offline_gate(regressed, config)

        wrong_card = dataclasses.replace(
            report, right_card_answers=19, wrong_card_answers=1, passed=False
        )
        with self.assertRaises(evaluation.PromotionGateError):
            evaluation.require_offline_gate(wrong_card, config)


class ServingTest(unittest.TestCase):
    def setUp(self):
        self.cards = json.loads(_CARDS.read_text(encoding="utf-8"))
        self.card_id = self.cards[0]["intent_id"]
        self._tmp = tempfile.TemporaryDirectory()
        self.event_path = Path(self._tmp.name) / "observability" / "events.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def _server(self, gate, respond, logger):
        return serving.make_server(
            artifact_id="sha256:mock",
            cards=self.cards,
            gate=gate,
            respond=respond,
            turn_logger=logger,
            event_path=self.event_path,
        )

    def test_disagreement_returns_safe_non_answer(self):
        # Gate abstains, but the raw respond would ANSWER: serving must fail closed.
        def gate(_q):
            return {"disposition": "ABSTAIN", "card_id": None, "candidates": [], "reason": "safe gate"}

        def respond(_q):
            return {"disposition": "ANSWER", "card": self.card_id, "reply": "leaked answer", "reason": "raw"}

        logger = _CaptureLogger()
        serve = self._server(gate, respond, logger)
        result = serve("How do I change my PIN?")
        self.assertIsInstance(result, TurnResult)
        self.assertEqual(result.disposition, "ABSTAIN")
        self.assertIsNone(result.card_id)
        self.assertEqual(result.policy_action, serving.CONSISTENCY_ALERT_ACTION)
        self.assertNotIn("leaked answer", result.reply)
        # The turn was logged with the required extra fields.
        self.assertEqual(len(logger.calls), 1)
        extra = logger.calls[0]["extra"]
        self.assertEqual(extra["artifact_id"], "sha256:mock")
        self.assertEqual(extra["release_channel"], "CURRENT")
        self.assertIn("latency_ms", extra)
        self.assertEqual(extra["policy_action"], serving.CONSISTENCY_ALERT_ACTION)

    def test_agreement_releases_answer(self):
        def gate(_q):
            return {"disposition": "ANSWER", "card_id": self.card_id, "candidates": [], "reason": "ok"}

        def respond(_q):
            return {"disposition": "ANSWER", "card": self.card_id, "reply": "here is the answer", "reason": "ok"}

        serve = self._server(gate, respond, _CaptureLogger())
        result = serve("some in-scope question")
        self.assertEqual(result.disposition, "ANSWER")
        self.assertEqual(result.card_id, self.card_id)
        self.assertEqual(result.reply, "here is the answer")

    def test_agreement_on_clarify(self):
        def gate(_q):
            return {"disposition": "CLARIFY", "card_id": None, "candidates": [], "reason": "amb"}

        def respond(_q):
            return {"disposition": "CLARIFY", "reply": "which one did you mean?", "reason": "amb"}

        serve = self._server(gate, respond, _CaptureLogger())
        result = serve("ambiguous question")
        self.assertEqual(result.disposition, "CLARIFY")
        self.assertEqual(result.reply, "which one did you mean?")

    def test_answer_with_mismatched_card_fails_closed(self):
        def gate(_q):
            return {"disposition": "ANSWER", "card_id": self.card_id, "candidates": [], "reason": "ok"}

        def respond(_q):
            other = self.cards[1]["intent_id"]
            return {"disposition": "ANSWER", "card": other, "reply": "answer", "reason": "ok"}

        serve = self._server(gate, respond, _CaptureLogger())
        result = serve("q")
        self.assertEqual(result.disposition, "ABSTAIN")
        self.assertEqual(result.policy_action, serving.CONSISTENCY_ALERT_ACTION)

    def test_empty_cards_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            serving.make_server(
                artifact_id="sha256:mock",
                cards=[],
                gate=lambda q: {"disposition": "ABSTAIN", "card_id": None, "candidates": [], "reason": ""},
                respond=lambda q: {"disposition": "ABSTAIN", "reply": "", "reason": ""},
                turn_logger=_CaptureLogger(),
                event_path=self.event_path,
            )


class CanaryShadowTest(unittest.TestCase):
    def setUp(self):
        self.cases = load_eval50_cases(_EVAL_SOURCE)
        self.cards = json.loads(_CARDS.read_text(encoding="utf-8"))
        self._tmp = tempfile.TemporaryDirectory()
        self.event_path = Path(self._tmp.name) / "observability" / "events.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def test_canary_passes_when_serving_abstains_on_harmful(self):
        # A safe candidate: gate and respond both ABSTAIN on everything harmful.
        def gate(_q):
            return {"disposition": "ABSTAIN", "card_id": None, "candidates": [], "reason": "safe"}

        def respond(_q):
            return {"disposition": "ABSTAIN", "reply": "no", "reason": "safe"}

        serve = serving.make_server(
            artifact_id="sha256:mock",
            cards=self.cards,
            gate=gate,
            respond=respond,
            turn_logger=_CaptureLogger(),
            event_path=self.event_path,
        )
        result = evaluation.evaluate_canary(serve, self.cases, artifact_id="sha256:mock")
        self.assertEqual(result["harmful_answers"], 0)
        self.assertEqual(result["consistency_failures"], 0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["harmful_probes"], 25)

    def test_canary_flags_a_leaking_serve(self):
        # A serve fn that answers harmful queries -> a leak the canary must catch.
        def serve(_q):
            return TurnResult(
                disposition="ANSWER", card_id="x", reply="leak",
                reason="", artifact_id="sha256:bad", policy_action="answer",
            )

        result = evaluation.evaluate_canary(serve, self.cases, artifact_id="sha256:bad")
        self.assertGreater(result["harmful_answers"], 0)
        self.assertFalse(result["passed"])

    def test_shadow_flags_unapproved_expansion(self):
        # Prior CURRENT turn ABSTAINed; the candidate now ANSWERs -> unapproved expansion.
        turns = [{"query": "How do I change my PIN?", "disposition": "ABSTAIN"}]

        def gate(_q):
            return {"disposition": "ANSWER", "card_id": "change_pin", "candidates": [], "reason": "expanded"}

        result = evaluation.evaluate_shadow(
            gate, turns, artifact_id="sha256:cand",
            current_artifact_id="sha256:cur", approved_expansions=[],
        )
        self.assertEqual(result["expansions"], 1)
        self.assertEqual(result["unapproved_expansions"], 1)
        self.assertFalse(result["passed"])
        self.assertFalse(result["emitted"])

    def test_shadow_respects_approved_expansion(self):
        turns = [{"query": "How do I change my PIN?", "disposition": "ABSTAIN"}]
        key = evaluation._expansion_id("How do I change my PIN?")

        def gate(_q):
            return {"disposition": "ANSWER", "card_id": "change_pin", "candidates": [], "reason": "expanded"}

        result = evaluation.evaluate_shadow(
            gate, turns, artifact_id="sha256:cand",
            current_artifact_id="sha256:cur", approved_expansions=[key],
        )
        self.assertEqual(result["unapproved_expansions"], 0)
        self.assertTrue(result["passed"])


class ObservabilityTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_emit_event_appends_stable_schema(self):
        path = self.root / "events.jsonl"
        observability.emit_event(path, {"stage": "serving", "event": "served", "status": "ok"})
        observability.emit_event(path, {"stage": "eval", "event": "done", "status": "ok"})
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        for field in ("schema_version", "event_id", "ts", "stage", "event", "status", "counts"):
            self.assertIn(field, first)
        self.assertTrue(first["event_id"].startswith("sha256:"))

    def test_prometheus_text_format(self):
        path = self.root / "metrics.prom"
        observability.write_prometheus_text(
            path,
            {
                "scope_eval_harmful_answers{artifact_id=\"a\",backend=\"mock\"}": 0,
                "scope_pipeline_stage_runs_total": 3,
                "scope_circuit_open": 0,
            },
        )
        text = path.read_text(encoding="utf-8")
        self.assertIn("# TYPE scope_pipeline_stage_runs_total counter", text)
        self.assertIn("# TYPE scope_circuit_open gauge", text)
        self.assertIn("scope_eval_harmful_answers{artifact_id=\"a\",backend=\"mock\"} 0", text)

    def test_drift_requires_min_samples(self):
        baseline = [{"disposition": "ANSWER"} for _ in range(10)]
        recent = [{"disposition": "ABSTAIN"} for _ in range(10)]
        # Below min_samples -> no alerts, however different the windows look.
        self.assertEqual(
            observability.detect_drift(baseline, recent, min_samples=50, max_rate_delta=0.2),
            (),
        )

    def test_drift_alerts_over_threshold(self):
        baseline = [{"disposition": "ANSWER"} for _ in range(60)]
        recent = [{"disposition": "ABSTAIN"} for _ in range(60)]
        alerts = observability.detect_drift(
            baseline, recent, min_samples=50, max_rate_delta=0.2
        )
        self.assertTrue(alerts)
        for alert in alerts:
            self.assertEqual(alert["kind"], "drift")
            self.assertIn("does not gate promotion", alert["note"])

    def test_render_dashboard_writes_self_contained_html(self):
        output = self.root / "dashboard.html"
        observability.render_dashboard(
            output,
            release={"artifact_id": "sha256:a", "actor": "ci", "backend": "mock"},
            evidence=[{"kind": "offline_eval", "passed": True, "artifact_id": "sha256:a", "metrics": {"harmful": 0}}],
            alerts=[],
        )
        html_text = output.read_text(encoding="utf-8")
        self.assertIn("<!doctype html>", html_text)
        self.assertIn("Scope pipeline dashboard", html_text)
        self.assertIn("offline_eval", html_text)
        self.assertNotIn("http://", html_text)  # no external assets


if __name__ == "__main__":
    unittest.main()
