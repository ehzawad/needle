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
import socket
import sys
import tempfile
import threading
import unittest
import urllib.request
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

    def test_ambiguous_abstain_is_not_a_guess(self):
        # An ambiguous case scored ABSTAIN is a safe non-answer, NOT a guess: it must
        # yield ambiguous_answers == 0 (only an ambiguous ANSWER counts).
        def abstain_ambiguous(query):
            case = self.by_query[query]
            if case.category == "in_scope":
                return {"disposition": "ANSWER", "card_id": case.expected_card,
                        "candidates": [], "reason": "x"}
            # ambiguous + everything else -> ABSTAIN
            return {"disposition": "ABSTAIN", "card_id": None, "candidates": [], "reason": "x"}

        report = evaluation.evaluate(
            abstain_ambiguous, self.cases, artifact_id="sha256:mock", backend="mock",
            suite_sha256=self.suite_sha, device=None,
        )
        self.assertEqual(report.ambiguous_answers, 0)
        self.assertEqual(report.ambiguous_clarifies, 0)
        self.assertEqual(report.ambiguous_total, 5)

    def test_ambiguous_answer_is_counted_as_a_guess(self):
        # An ambiguous case scored ANSWER is a guess and must increment ambiguous_answers.
        def answer_ambiguous(query):
            case = self.by_query[query]
            if case.category in ("in_scope", "ambiguous"):
                card = case.expected_card or self.by_query[
                    next(c.query for c in self.cases if c.category == "in_scope")
                ].expected_card
                return {"disposition": "ANSWER", "card_id": card, "candidates": [], "reason": "x"}
            return {"disposition": "ABSTAIN", "card_id": None, "candidates": [], "reason": "x"}

        report = evaluation.evaluate(
            answer_ambiguous, self.cases, artifact_id="sha256:mock", backend="mock",
            suite_sha256=self.suite_sha, device=None,
        )
        self.assertEqual(report.ambiguous_answers, 5)
        self.assertEqual(report.ambiguous_clarifies, 0)

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

    def test_offline_gate_enforces_ambiguous_answers_max(self):
        # ambiguous_answers is now an EXPLICIT metric (an ambiguous case scored ANSWER).
        # Clarifies at the min (5) but one ambiguous case ANSWERED -> a floor miss, even
        # though the clarifies-min check alone would pass.
        config = load_config(_PKG / "config.ci.json")
        metrics = {
            "harmful_answers": 0, "harmful_total": 25, "right_card_answers": 20,
            "in_scope_total": 20, "wrong_card_answers": 0, "ambiguous_clarifies": 5,
            "ambiguous_answers": 1, "ambiguous_total": 6, "errors": 0,
        }
        reasons = evaluation.offline_gate_reasons(metrics, config.promotion)
        self.assertTrue(any("ambiguous_answers" in r for r in reasons), reasons)

    def test_offline_gate_fails_closed_without_explicit_ambiguous_answers(self):
        # A metrics mapping that predates the explicit field must NOT clear the floor:
        # ambiguous_answers is fail-closed to a violating value when absent.
        config = load_config(_PKG / "config.ci.json")
        metrics = {
            "harmful_answers": 0, "harmful_total": 25, "right_card_answers": 20,
            "in_scope_total": 20, "wrong_card_answers": 0, "ambiguous_clarifies": 5,
            "ambiguous_total": 5, "errors": 0,  # no ambiguous_answers key
        }
        reasons = evaluation.offline_gate_reasons(metrics, config.promotion)
        self.assertTrue(any("ambiguous_answers" in r for r in reasons), reasons)

    def test_offline_checkers_parity(self):
        # The DAG's _check_offline and evaluation.offline_gate_reasons must agree on every
        # input so the two offline entrypoints cannot silently diverge.
        from pipeline import dag

        config = load_config(_PKG / "config.ci.json")
        p = config.promotion
        metric_sets = [
            {"harmful_answers": 0, "harmful_total": 25, "right_card_answers": 20,
             "in_scope_total": 20, "wrong_card_answers": 0, "ambiguous_clarifies": 5,
             "ambiguous_answers": 0, "ambiguous_total": 5, "errors": 0},  # clean
            {"harmful_answers": 1, "harmful_total": 25, "right_card_answers": 20,
             "in_scope_total": 20, "wrong_card_answers": 0, "ambiguous_clarifies": 5,
             "ambiguous_answers": 0, "ambiguous_total": 5, "errors": 0},  # a leak
            {"harmful_answers": 0, "harmful_total": 25, "right_card_answers": 19,
             "in_scope_total": 20, "wrong_card_answers": 1, "ambiguous_clarifies": 5,
             "ambiguous_answers": 0, "ambiguous_total": 5, "errors": 0},  # wrong card
            {"harmful_answers": 0, "harmful_total": 25, "right_card_answers": 20,
             "in_scope_total": 20, "wrong_card_answers": 0, "ambiguous_clarifies": 5,
             "ambiguous_answers": 1, "ambiguous_total": 6, "errors": 0},  # ambiguous answered
            {"harmful_answers": 0, "harmful_total": 25, "right_card_answers": 20,
             "in_scope_total": 20, "wrong_card_answers": 0, "ambiguous_clarifies": 5,
             "ambiguous_answers": 0, "ambiguous_total": 5, "errors": 3},  # gate errors
            {},  # everything missing -> fail closed
        ]
        for metrics in metric_sets:
            self.assertEqual(
                dag._check_offline(metrics, p),
                evaluation.offline_gate_reasons(metrics, p),
                f"offline checkers diverged for {metrics}",
            )


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

    def test_malformed_card_id_fails_closed(self):
        # A list-valued card id must NOT reach ``g_card in by_id`` (that would raise
        # TypeError and escape the fail-closed boundary). It degrades to a safe refusal.
        def gate(_q):
            return {"disposition": "ANSWER", "card_id": ["not", "a", "string"],
                    "candidates": [], "reason": "x"}

        def respond(_q):
            return {"disposition": "ANSWER", "card": self.card_id, "reply": "answer", "reason": "ok"}

        serve = self._server(gate, respond, _CaptureLogger())
        result = serve("q")  # must not raise
        self.assertEqual(result.disposition, "ABSTAIN")
        self.assertEqual(result.policy_action, serving.CONSISTENCY_ALERT_ACTION)

    def test_malformed_respond_types_fail_closed(self):
        # A respond whose reply/card are wrong types must not crash or leak.
        def gate(_q):
            return {"disposition": "ANSWER", "card_id": self.card_id, "candidates": [], "reason": "ok"}

        def respond(_q):
            return {"disposition": "ANSWER", "card": ["x"], "reply": {"not": "a string"}, "reason": 5}

        serve = self._server(gate, respond, _CaptureLogger())
        result = serve("q")  # must not raise
        self.assertEqual(result.disposition, "ABSTAIN")
        self.assertEqual(result.policy_action, serving.CONSISTENCY_ALERT_ACTION)


class ServingCircuitTest(unittest.TestCase):
    """The serving circuit breaker is ARTIFACT-SCOPED, not a global block."""

    def setUp(self):
        self.cards = json.loads(_CARDS.read_text(encoding="utf-8"))
        self.card_id = self.cards[0]["intent_id"]
        self._tmp = tempfile.TemporaryDirectory()
        self.state = Path(self._tmp.name)
        self.event_path = self.state / "observability" / "events.jsonl"
        self.circuit_path = self.state / "circuit.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _server(self, artifact_id):
        def gate(_q):
            return {"disposition": "ANSWER", "card_id": self.card_id, "candidates": [], "reason": "ok"}

        def respond(_q):
            return {"disposition": "ANSWER", "card": self.card_id, "reply": "here", "reason": "ok"}

        return serving.make_server(
            artifact_id=artifact_id, cards=self.cards, gate=gate, respond=respond,
            turn_logger=_CaptureLogger(), event_path=self.event_path,
        )

    def test_no_circuit_serves(self):
        serve = self._server("sha256:good")
        self.assertEqual(serve("q").disposition, "ANSWER")

    def test_circuit_open_against_other_artifact_still_serves(self):
        # Breaker names artifact B; a server for A must keep serving (not a global block).
        self.circuit_path.write_text(
            '{"open": true, "bad_artifact_id": "sha256:bad"}', encoding="utf-8")
        serve = self._server("sha256:good")
        result = serve("q")
        self.assertEqual(result.disposition, "ANSWER")
        self.assertNotEqual(result.policy_action, serving.CIRCUIT_OPEN_ACTION)

    def test_circuit_open_against_this_artifact_blocks(self):
        self.circuit_path.write_text(
            '{"open": true, "bad_artifact_id": "sha256:bad"}', encoding="utf-8")
        serve = self._server("sha256:bad")
        result = serve("q")
        self.assertEqual(result.disposition, "ABSTAIN")
        self.assertEqual(result.policy_action, serving.CIRCUIT_OPEN_ACTION)

    def test_open_circuit_without_bad_id_fails_closed(self):
        self.circuit_path.write_text('{"open": true}', encoding="utf-8")
        serve = self._server("sha256:good")
        result = serve("q")
        self.assertEqual(result.disposition, "ABSTAIN")
        self.assertEqual(result.policy_action, serving.CIRCUIT_OPEN_ACTION)

    def test_malformed_circuit_fails_closed(self):
        self.circuit_path.write_text("{not valid json", encoding="utf-8")
        serve = self._server("sha256:good")
        result = serve("q")
        self.assertEqual(result.disposition, "ABSTAIN")
        self.assertEqual(result.policy_action, serving.CIRCUIT_OPEN_ACTION)

    def test_explicitly_resolved_circuit_serves(self):
        self.circuit_path.write_text(
            '{"open": false, "bad_artifact_id": "sha256:good"}', encoding="utf-8")
        serve = self._server("sha256:good")
        self.assertEqual(serve("q").disposition, "ANSWER")


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

    def test_shadow_fails_on_gate_error(self):
        # A gate that raises every turn never expands, but errors>0 -> must FAIL the stage
        # (it previously passed because passed was set solely from unapproved==0).
        turns = [
            {"query": "How do I change my PIN?", "disposition": "ABSTAIN"},
            {"query": "What is my balance?", "disposition": "ANSWER"},
        ]

        def gate(_q):
            raise RuntimeError("gate exploded")

        result = evaluation.evaluate_shadow(
            gate, turns, artifact_id="sha256:cand",
            current_artifact_id="sha256:cur", approved_expansions=[],
        )
        self.assertEqual(result["errors"], 2)
        self.assertEqual(result["unapproved_expansions"], 0)
        self.assertFalse(result["passed"])

    def test_canary_fails_on_malformed_serve_result(self):
        # A serve that returns a non-TurnResult (no valid disposition) is an error: scored
        # as a safe non-answer (never a leak) but the stage must FAIL.
        def serve(_q):
            return {"not": "a turnresult"}

        result = evaluation.evaluate_canary(serve, self.cases, artifact_id="sha256:bad")
        self.assertGreater(result["errors"], 0)
        self.assertEqual(result["harmful_answers"], 0)
        self.assertFalse(result["passed"])


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

    def test_load_drift_alerts_from_configured_log(self):
        # A >=50/>=50 configured-log case: 60 older ANSWER + 60 newer ABSTAIN turns exceed
        # the 0.2 disposition-rate delta, so the shared loader returns real alerts. A None
        # path (unconfigured, e.g. CI) returns [].
        log = self.root / "conversations.jsonl"
        lines = []
        for i in range(120):
            disp = "ANSWER" if i < 60 else "ABSTAIN"
            lines.append(json.dumps({
                "session_id": f"s{i}", "turn": 1, "query": "q", "disposition": disp,
                "card_id": None, "ts": f"2026-07-16T00:00:00.{i:03d}",
            }))
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        alerts = observability.load_drift_alerts(log, min_samples=50, max_rate_delta=0.2)
        self.assertTrue(alerts)
        self.assertTrue(all(a["kind"] == "drift" for a in alerts))
        self.assertEqual(
            observability.load_drift_alerts(None, min_samples=50, max_rate_delta=0.2), [])

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


class HttpServingTest(unittest.TestCase):
    """HTTP front-door tests: session identity/sequencing, /gate one-log, malformed
    Content-Length. Proves the ThreadingHTTPServer boundary is safe and correct."""

    def setUp(self):
        self.cards = json.loads(_CARDS.read_text(encoding="utf-8"))
        self.card_id = self.cards[0]["intent_id"]
        self._tmp = tempfile.TemporaryDirectory()
        self.event_path = Path(self._tmp.name) / "observability" / "events.jsonl"
        self.logger = _CaptureLogger()

    def tearDown(self):
        self._tmp.cleanup()

    def _start(self, gate=None, respond=None):
        def _gate(_q):
            return {"disposition": "ANSWER", "card_id": self.card_id,
                    "candidates": ["a", "b"], "reason": "ok", "policy_action": "answer"}

        def _respond(_q):
            return {"disposition": "ANSWER", "card": self.card_id,
                    "reply": "here is the answer", "reason": "ok"}

        gate = gate or _gate
        respond = respond or _respond
        recorder = serving.TurnRecorder(
            turn_logger=self.logger, event_path=self.event_path,
            artifact_id="sha256:mock", default_session="serve-sha256:mock")
        serve = serving.make_server(
            artifact_id="sha256:mock", cards=self.cards, gate=gate, respond=respond,
            turn_logger=self.logger, event_path=self.event_path, recorder=recorder)
        logged_gate = serving.make_logged_gate(gate, recorder)
        from pipeline.serving_http import build_http_server
        server = build_http_server(
            "127.0.0.1", 0, serve=serve, gate=logged_gate, recorder=recorder,
            artifact_id="sha256:mock",
            healthz=lambda: {"ok": True, "artifact_id": "sha256:mock", "device": None})
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def _stop():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.addCleanup(_stop)
        host, port = server.server_address
        self.base = f"http://{host}:{port}"
        self.addr = (host, port)
        return server

    def _post(self, route, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base + route, data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_sessions_are_isolated_and_sequenced(self):
        self._start()
        _, a1 = self._post("/respond", {"query": "q1", "session_id": "A"})
        _, a2 = self._post("/respond", {"query": "q2", "session_id": "A"})
        _, b1 = self._post("/respond", {"query": "q1", "session_id": "B"})
        self.assertEqual((a1["session_id"], a1["turn"]), ("A", 1))
        self.assertEqual((a2["session_id"], a2["turn"]), ("A", 2))
        # B is a SEPARATE conversation: its first turn is 1, not 3.
        self.assertEqual((b1["session_id"], b1["turn"]), ("B", 1))
        # An anonymous request gets its own generated session, never merged into A/B.
        _, anon = self._post("/respond", {"query": "q"})
        self.assertTrue(anon["session_id"].startswith("sess-"))
        self.assertEqual(anon["turn"], 1)
        self.assertNotIn(anon["session_id"], ("A", "B"))

    def test_turns_are_monotonic_across_gate_and_respond(self):
        self._start()
        _, g = self._post("/gate", {"query": "q1", "session_id": "S"})
        _, r = self._post("/respond", {"query": "q2", "session_id": "S"})
        self.assertEqual(g["turn"], 1)
        self.assertEqual(r["turn"], 2)  # one shared per-session counter

    def test_gate_and_respond_each_log_exactly_one_record(self):
        self._start()
        # One /gate request -> exactly one complete gate-only record.
        _, g = self._post("/gate", {"query": "q1", "session_id": "S"})
        self.assertEqual(len(self.logger.calls), 1)
        rec = self.logger.calls[0]
        self.assertEqual(rec["reply"], "")  # gate-only: empty reply, no generation
        self.assertEqual(rec["gate"]["disposition"], "ANSWER")
        self.assertEqual(rec["gate"]["card_id"], self.card_id)
        self.assertEqual(rec["gate"]["candidates"], ["a", "b"])
        self.assertEqual(rec["extra"]["endpoint"], "gate")
        self.assertEqual(rec["extra"]["artifact_id"], "sha256:mock")
        self.assertEqual(rec["extra"]["release_channel"], "CURRENT")
        self.assertIn("latency_ms", rec["extra"])
        self.assertEqual(rec["extra"]["policy_action"], "answer")
        self.assertEqual(g["disposition"], "ANSWER")
        # One /respond request -> exactly one more record (no double HTTP-layer log).
        self._post("/respond", {"query": "q2", "session_id": "S"})
        self.assertEqual(len(self.logger.calls), 2)
        self.assertEqual(self.logger.calls[1]["extra"]["endpoint"], "respond")
        self.assertEqual(self.logger.calls[1]["reply"], "here is the answer")

    def test_malformed_content_length_returns_400(self):
        self._start()
        host, port = self.addr
        sock = socket.create_connection((host, port), timeout=5)
        try:
            body = b'{"query": "hi"}'
            request = (
                b"POST /respond HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: not-a-number\r\n"
                b"Connection: close\r\n"  # close after the 400 so the unread body isn't reparsed
                b"\r\n" + body
            )
            sock.sendall(request)
            status_line = sock.recv(256).decode("utf-8", "replace").split("\r\n", 1)[0]
        finally:
            sock.close()
        self.assertIn("400", status_line)


if __name__ == "__main__":
    unittest.main()
