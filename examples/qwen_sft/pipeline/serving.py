"""Serving boundary with a dual-decision safety cross-check — fail closed.

Real-world analog: a KServe / Triton inference boundary that separates the serving
CONTRACT from the model implementation, wrapped in a consistency guard. The model is
never imported here; the candidate's behaviour arrives through two injected callables —
an external, policy-wrapped SAFE gate and the original ``ScopeBot.respond`` — and this
module only RELEASES a reply when the two agree. That is the council's no-fork
guarantee made operational: the frozen bot is not edited, it is cross-examined.

The serving algorithm is frozen (do not reorder):

  1. Check the circuit breaker and candidate integrity; refuse if the (artifact-scoped)
     circuit is open against this artifact.
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
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pipeline.contracts import GateFn, RespondFn, ServeFn, TurnResult
from pipeline.observability import emit_event

__all__ = [
    "CONSISTENCY_ALERT_ACTION",
    "CIRCUIT_OPEN_ACTION",
    "SAFE_REFUSAL",
    "TurnRecorder",
    "make_logged_gate",
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


def _norm_id(value: object | None) -> str | None:
    """Normalize an artifact id to canonical ``sha256:<hex>`` form (``None`` passes)."""
    if value is None:
        return None
    text = str(value)
    return text if text.startswith("sha256:") else f"sha256:{text}"


def _circuit_blocks(
    event_path: Path, artifact_id: str
) -> tuple[bool, Mapping[str, Any] | None]:
    """Read ``<state>/circuit.json`` and decide, fail-closed, whether THIS artifact serves.

    The circuit breaker is artifact-scoped, not a global kill switch: after a trip rolls
    ``CURRENT`` from a bad artifact B back to a good artifact A, the breaker names B, so a
    server for B must fail closed while a server for A keeps serving. The file exists only
    after a trip (written BEFORE ``CURRENT`` is rolled back), so its presence means 'open'
    unless explicitly marked closed/resolved. It blocks fail-closed when: it is
    unreadable/malformed, OR it is open with no valid ``bad_artifact_id`` (we cannot prove
    this artifact is the safe one), OR its ``bad_artifact_id`` equals ``artifact_id``.
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
    if explicitly_closed:
        return False, data
    bad = data.get("bad_artifact_id")
    if not isinstance(bad, str) or not bad:
        # Open but unattributable -> block fail-closed.
        return True, data
    return (_norm_id(bad) == _norm_id(artifact_id)), data


class TurnRecorder:
    """Thread-safe per-session turn counter + single logging point for a served turn.

    One recorder is shared by ``/gate`` and ``/respond`` so their turns interleave into
    ONE monotonically increasing sequence per session, and every served turn (gate-only
    or full respond) is persisted exactly once through the same injected ``TurnLogger``.
    Anonymous HTTP requests each get their own session id at the HTTP boundary; they are
    never merged into one global session here. Safe under ``ThreadingHTTPServer``: the
    counter map is mutated only under a lock.
    """

    def __init__(self, *, turn_logger: object, event_path: Path, artifact_id: str,
                 default_session: str) -> None:
        self._lock = threading.Lock()
        self._turns: dict[str, int] = {}
        self.turn_logger = turn_logger
        self.event_path = Path(event_path)
        self.artifact_id = artifact_id
        self.default_session = default_session

    def next_turn(self, session_id: str) -> int:
        """Return the next monotonic turn number for ``session_id`` (thread-safe)."""
        with self._lock:
            n = self._turns.get(session_id, 0) + 1
            self._turns[session_id] = n
            return n

    def record(self, *, session_id: str, turn: int, query: str,
               gate_view: Mapping[str, Any], reply: str, latency_ms: int,
               policy_action: str | None, endpoint: str, event: str, status: str,
               error_code: str | None) -> None:
        """Persist one served turn (log + observability event). Never raises: telemetry
        must not break serving."""
        extra = {
            "artifact_id": self.artifact_id,
            "release_channel": _DEFAULT_CHANNEL,
            "latency_ms": latency_ms,
            "policy_action": policy_action,
            "endpoint": endpoint,
        }
        try:  # logging must never break serving.
            self.turn_logger.log(  # type: ignore[attr-defined]
                session_id, turn, query, dict(gate_view), reply, extra=extra)
        except Exception:
            pass
        try:
            emit_event(
                self.event_path,
                {
                    "stage": "serving",
                    "event": event,
                    "status": status,
                    "artifact_id": self.artifact_id,
                    "duration_ms": latency_ms,
                    "counts": {"disposition": gate_view.get("disposition"),
                               "endpoint": endpoint},
                    "error_code": error_code,
                },
            )
        except Exception:
            pass


def make_logged_gate(gate: GateFn, recorder: TurnRecorder) -> Callable[..., Any]:
    """Wrap a policy-wrapped gate so ``/gate`` logs exactly one gate-only record.

    Calls the model gate EXACTLY once (never ``respond`` — ``/gate`` must not generate),
    persists the full disposition/card/candidates/reason with an empty reply, the artifact
    id, CURRENT channel, latency, policy_action, and ``endpoint="gate"`` through the shared
    recorder, and emits a ``gate_served`` event. Returns the raw gate decision unchanged.
    """
    def logged_gate(query: str, session_id: str | None = None,
                    turn: int | None = None) -> Any:
        sid = session_id or recorder.default_session
        t = turn if turn is not None else recorder.next_turn(sid)
        t0 = time.monotonic()
        decision = gate(query)  # the ONE model gate call; no generation
        latency_ms = int((time.monotonic() - t0) * 1000)
        if isinstance(decision, Mapping):
            reason = decision.get("reason", "")
            raw_candidates = decision.get("candidates") or []
            gate_view = {
                "disposition": decision.get("disposition"),
                "card_id": decision.get("card_id"),
                "candidates": list(raw_candidates) if isinstance(raw_candidates, (list, tuple)) else [],
                "reason": reason if isinstance(reason, str) else str(reason),
            }
            raw_action = decision.get("policy_action")
            policy_action = raw_action if isinstance(raw_action, str) and raw_action else None
        else:
            gate_view = {"disposition": None, "card_id": None, "candidates": [], "reason": ""}
            policy_action = None
        recorder.record(
            session_id=sid, turn=t, query=query, gate_view=gate_view, reply="",
            latency_ms=latency_ms, policy_action=policy_action, endpoint="gate",
            event="gate_served", status="ok", error_code=None,
        )
        return decision

    return logged_gate


def make_server(
    *,
    artifact_id: str,
    cards: Sequence[Mapping[str, Any]],
    gate: GateFn,
    respond: RespondFn,
    turn_logger: object,
    event_path: Path,
    recorder: TurnRecorder | None = None,
) -> ServeFn:
    """Build the serving function for a verified candidate.

    The returned ``ServeFn`` takes a query (and, from the HTTP layer, an optional
    ``session_id`` / ``turn``) and returns a :class:`TurnResult`, running the frozen
    7-step algorithm on every call and logging each turn EXACTLY once through the shared
    :class:`TurnRecorder`. When no recorder is injected (canary / direct callers) it owns
    a private recorder keyed on a single default session, preserving the single-arg
    ``serve(query)`` contract.
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
    default_session = f"serve-{artifact_id}"
    if recorder is None:
        recorder = TurnRecorder(
            turn_logger=turn_logger, event_path=event_path,
            artifact_id=artifact_id, default_session=default_session,
        )

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
        session_id: str,
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
        recorder.record(
            session_id=session_id, turn=turn, query=query, gate_view=gate_view,
            reply=result.reply, latency_ms=latency_ms,
            policy_action=result.policy_action, endpoint="respond",
            event=event, status=status, error_code=error_code,
        )

    def serve(query: str, session_id: str | None = None,
              turn: int | None = None) -> TurnResult:
        sid = session_id or recorder.default_session
        turn = turn if turn is not None else recorder.next_turn(sid)
        t0 = time.monotonic()

        # Step 1: circuit breaker + integrity — fail closed when open against THIS artifact.
        is_open, _circuit = _circuit_blocks(event_path, artifact_id)
        if is_open:
            result = _safe_nonanswer(
                reason="circuit is open; serving is disabled",
                policy_action=CIRCUIT_OPEN_ACTION,
            )
            _finish(
                query, sid, turn, result, t0,
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
            g_reason = gate_decision.get("reason", "")
            g_action = gate_decision.get("policy_action")
            # Validate TYPES before any comparison: a non-string card id (e.g. a list)
            # must never reach ``g_card in by_id`` — that would raise TypeError and escape
            # the fail-closed boundary. Any malformed field degrades to a safe value.
            if g_disp not in _DISPOSITIONS:
                g_disp, g_card = "ABSTAIN", None
            if not isinstance(g_card, str) or not g_card:
                g_card = None
            if not isinstance(g_reason, str):
                g_reason = ""
            if not isinstance(g_action, str) or not g_action:
                g_action = None
        except Exception as exc:
            result = _safe_nonanswer(
                reason=f"gate error: {type(exc).__name__}: {exc}",
                policy_action=CONSISTENCY_ALERT_ACTION,
            )
            _finish(
                query, sid, turn, result, t0,
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
            r_reply = respond_decision.get("reply", "")
            r_reason = respond_decision.get("reason", "")
            # Validate TYPES before comparison. An invalid disposition is set to None so it
            # cannot accidentally AGREE with the gate (every gate disposition is a valid
            # member by now) — the turn then falls to the consistency-alert safe refusal.
            if r_disp not in _DISPOSITIONS:
                r_disp, r_card = None, None
            if not isinstance(r_card, str) or not r_card:
                r_card = None
            if not isinstance(r_reply, str):
                r_reply = ""
            if not isinstance(r_reason, str):
                r_reason = ""
        except Exception as exc:
            result = _safe_nonanswer(
                reason=f"respond error: {type(exc).__name__}: {exc}",
                policy_action=CONSISTENCY_ALERT_ACTION,
            )
            _finish(
                query, sid, turn, result, t0,
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
                query, sid, turn, result, t0,
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
                query, sid, turn, result, t0,
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
            query, sid, turn, result, t0,
            status="alert", event="serving_consistency_alert",
            error_code="consistency_mismatch",
        )
        return result

    return serve
