"""Serving boundary with a dual-decision safety cross-check — fail closed.

Real-world analog: a KServe / Triton inference boundary that separates the serving
CONTRACT from the model implementation, wrapped in a consistency guard. The model is
never imported here; the candidate's behaviour arrives through two injected callables —
an external, policy-wrapped SAFE gate and the original ``ScopeBot.respond`` — and this
module only RELEASES a reply when the two agree. That is the council's no-fork
guarantee made operational: the frozen bot is not edited, it is cross-examined.

The serving algorithm is frozen (do not reorder):

  1. Check the circuit breaker and candidate integrity; if the circuit is open, refuse.
  2. Run the injected, policy-wrapped gate.
  3. Run the original ``respond``.
  4. Release an ANSWER only if BOTH say ANSWER with the same, valid card.
  5. Release a clarification / refusal only when the two dispositions agree.
  6. On any exception or mismatch, emit a consistency alert and return a safe non-answer.
  7. Log the turn via the injected ``TurnLogger``, adding ``artifact_id`` /
     ``release_channel`` / ``latency_ms`` / ``policy_action``.

Full hash verification of the candidate's blobs is the CALLER's job (done via the
registry before a server is built); step 1's integrity check here is the light,
serving-local structural assertion plus the circuit-breaker read.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipeline.contracts import GateFn, RespondFn, ServeFn, TurnResult
from pipeline.observability import emit_event

__all__ = [
    "CONSISTENCY_ALERT_ACTION",
    "CIRCUIT_OPEN_ACTION",
    "SAFE_REFUSAL",
    "make_server",
]

# The reply a fail-closed non-answer carries. Kept local: importing scope_bot's REFUSAL
# would drag in torch, which this control-plane-adjacent module must never do.
SAFE_REFUSAL = "I'm sorry, I can't answer that safely right now, so I won't guess."
_CLARIFY_FALLBACK = "Just to make sure I help with the right thing — could you clarify what you mean?"

# policy_action markers that observability/canary key on.
CONSISTENCY_ALERT_ACTION = "consistency_alert"
CIRCUIT_OPEN_ACTION = "circuit_open"

# Serving always serves the CURRENT release; make_server takes no channel argument, so
# this is the honest default stamped on every logged turn.
_DEFAULT_CHANNEL = "CURRENT"

_DISPOSITIONS = frozenset({"ANSWER", "CLARIFY", "ABSTAIN"})


def _state_root_from_event_path(event_path: Path) -> Path:
    """``<state>/observability/events.jsonl`` -> ``<state>`` (the registry state root)."""
    return Path(event_path).resolve().parent.parent


def _circuit_open(event_path: Path) -> tuple[bool, Mapping[str, Any] | None]:
    """Read ``<state>/circuit.json`` and decide, fail-closed, whether serving is disabled.

    The circuit file exists only after a trip (the release controller writes it BEFORE
    rolling ``CURRENT`` back so serving fails closed). Its mere presence therefore means
    'open' unless it is explicitly marked closed/resolved. An unreadable file is treated
    as open.
    """
    circuit = _state_root_from_event_path(event_path) / "circuit.json"
    if not circuit.exists():
        return False, None
    try:
        data = json.loads(circuit.read_text(encoding="utf-8"))
    except Exception:
        return True, {"error": "unreadable circuit.json"}
    if not isinstance(data, Mapping):
        return True, {"error": "malformed circuit.json"}
    explicitly_closed = (
        data.get("open") is False
        or data.get("state") == "closed"
        or bool(data.get("resolved_at"))
    )
    return (not explicitly_closed), data


def make_server(
    *,
    artifact_id: str,
    cards: Sequence[Mapping[str, Any]],
    gate: GateFn,
    respond: RespondFn,
    turn_logger: object,
    event_path: Path,
) -> ServeFn:
    """Build the serving function for a verified candidate.

    The returned ``ServeFn`` takes a single query and returns a :class:`TurnResult`,
    running the frozen 7-step algorithm on every call and logging each turn.
    """
    # Step 1 (construction-time integrity): a candidate with no usable cards cannot
    # validate any answer, so its answers would always fail closed anyway. Refuse to
    # build a server on a structurally empty card set.
    by_id: dict[str, Mapping[str, Any]] = {}
    for card in cards:
        if isinstance(card, Mapping) and isinstance(card.get("intent_id"), str):
            by_id[card["intent_id"]] = card
    if not by_id:
        raise ValueError(
            "serving: candidate integrity check failed — no valid cards to serve"
        )

    event_path = Path(event_path)
    session_id = f"serve-{artifact_id}"
    state = {"turn": 0}

    def _safe_nonanswer(*, reason: str, policy_action: str) -> TurnResult:
        return TurnResult(
            disposition="ABSTAIN",
            card_id=None,
            reply=SAFE_REFUSAL,
            reason=reason,
            artifact_id=artifact_id,
            policy_action=policy_action,
        )

    def _finish(
        query: str,
        turn: int,
        result: TurnResult,
        t0: float,
        *,
        status: str,
        event: str,
        error_code: str | None,
    ) -> None:
        latency_ms = int((time.monotonic() - t0) * 1000)
        gate_view = {
            "disposition": result.disposition,
            "card_id": result.card_id,
            "candidates": [],
            "reason": result.reason,
        }
        extra = {
            "artifact_id": artifact_id,
            "release_channel": _DEFAULT_CHANNEL,
            "latency_ms": latency_ms,
            "policy_action": result.policy_action,
        }
        try:  # logging must never break serving.
            turn_logger.log(session_id, turn, query, gate_view, result.reply, extra=extra)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            emit_event(
                event_path,
                {
                    "stage": "serving",
                    "event": event,
                    "status": status,
                    "artifact_id": artifact_id,
                    "duration_ms": latency_ms,
                    "counts": {"disposition": result.disposition},
                    "error_code": error_code,
                },
            )
        except Exception:
            pass

    def serve(query: str) -> TurnResult:
        state["turn"] += 1
        turn = state["turn"]
        t0 = time.monotonic()

        # Step 1: circuit breaker + integrity — fail closed when open.
        is_open, _circuit = _circuit_open(event_path)
        if is_open:
            result = _safe_nonanswer(
                reason="circuit is open; serving is disabled",
                policy_action=CIRCUIT_OPEN_ACTION,
            )
            _finish(
                query, turn, result, t0,
                status="alert", event="circuit_open", error_code="circuit_open",
            )
            return result

        # Step 2: injected, policy-wrapped safe gate.
        try:
            gate_decision = gate(query)
            if not isinstance(gate_decision, Mapping):
                raise TypeError("gate did not return a mapping")
            g_disp = gate_decision.get("disposition")
            g_card = gate_decision.get("card_id")
            g_reason = gate_decision.get("reason", "") or ""
            g_action = gate_decision.get("policy_action")
            if g_disp not in _DISPOSITIONS:
                g_disp, g_card = "ABSTAIN", None
        except Exception as exc:
            result = _safe_nonanswer(
                reason=f"gate error: {type(exc).__name__}: {exc}",
                policy_action=CONSISTENCY_ALERT_ACTION,
            )
            _finish(
                query, turn, result, t0,
                status="alert", event="serving_consistency_alert",
                error_code="gate_error",
            )
            return result

        # Step 3: the ORIGINAL ScopeBot.respond (returns {"disposition","card","reply"}).
        try:
            respond_decision = respond(query)
            if not isinstance(respond_decision, Mapping):
                raise TypeError("respond did not return a mapping")
            r_disp = respond_decision.get("disposition")
            r_card = respond_decision.get("card")
            r_reply = respond_decision.get("reply", "") or ""
            r_reason = respond_decision.get("reason", "") or ""
        except Exception as exc:
            result = _safe_nonanswer(
                reason=f"respond error: {type(exc).__name__}: {exc}",
                policy_action=CONSISTENCY_ALERT_ACTION,
            )
            _finish(
                query, turn, result, t0,
                status="alert", event="serving_consistency_alert",
                error_code="respond_error",
            )
            return result

        # Step 4: release an ANSWER only if BOTH agree on the same, valid card.
        if (
            g_disp == "ANSWER"
            and r_disp == "ANSWER"
            and g_card is not None
            and g_card == r_card
            and g_card in by_id
        ):
            result = TurnResult(
                disposition="ANSWER",
                card_id=g_card,
                reply=r_reply,
                reason=g_reason or r_reason,
                artifact_id=artifact_id,
                policy_action=g_action or "answer",
            )
            _finish(
                query, turn, result, t0,
                status="ok", event="served", error_code=None,
            )
            return result

        # Step 5: release clarify / refuse only when dispositions agree.
        if g_disp == r_disp and g_disp in ("CLARIFY", "ABSTAIN"):
            if g_disp == "CLARIFY":
                reply = r_reply or _CLARIFY_FALLBACK
            else:
                reply = r_reply or SAFE_REFUSAL
            result = TurnResult(
                disposition=g_disp,
                card_id=None,
                reply=reply,
                reason=g_reason or r_reason,
                artifact_id=artifact_id,
                policy_action=g_action or g_disp.lower(),
            )
            _finish(
                query, turn, result, t0,
                status="ok", event="served", error_code=None,
            )
            return result

        # Step 6: disagreement (incl. gate ABSTAIN vs respond ANSWER, or different/invalid
        # cards) -> consistency alert + safe non-answer. This is the fail-closed core.
        result = _safe_nonanswer(
            reason=(
                f"serving consistency check failed: gate={g_disp}/{g_card} "
                f"respond={r_disp}/{r_card}"
            ),
            policy_action=CONSISTENCY_ALERT_ACTION,
        )
        _finish(
            query, turn, result, t0,
            status="alert", event="serving_consistency_alert",
            error_code="consistency_mismatch",
        )
        return result

    return serve
