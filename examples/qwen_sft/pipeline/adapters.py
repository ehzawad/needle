"""Injection adapters around the frozen bot, policy, miner, and exemplar builder.

Real-world analog: a KServe / Triton *inference adapter* (a thin predictor shim that
exposes a stable gate/respond contract while the model implementation stays untouched)
combined with a Repository/DAO wrapper over the existing data tools. Nothing here
edits ``scope_bot.py``, ``scope_policy.py``, ``mine_signals.py`` or ``adapter/learn.py``
— every model-, policy-, mining-, and build-touching import is DEFERRED into the
function body so this module stays import-light and, critically, torch-free at module
scope. The only sanctioned place a real Qwen load happens is inside ``load_real_bot``
(and even then it is launched exclusively by the guarded GPU worker).

Two seams matter:

  * the *gate* seam — ``policy_wrapped_gate`` and ``make_mock_gate`` produce the exact
    same :data:`~pipeline.contracts.GateFn` contract whether a real model or a
    deterministic CI stub sits behind it, and BOTH pass through the deterministic,
    downgrade-only ``scope_policy.enforce_policy`` so a malformed or adversarial
    decision can only ever get SAFER (ANSWER -> CLARIFY/ABSTAIN), never riskier.
  * the *build* seam — ``mine_existing`` / ``build_existing`` call the existing feedback
    miner and exemplar builder rather than reproducing their filtering, staging the
    inputs under a temporary cwd so the builder's relative-path contract is honored.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipeline.contracts import EvalCase, GateDecision, GateFn, RespondFn

_DISPOSITIONS = ("ANSWER", "CLARIFY", "ABSTAIN")

# Deterministic CI mapping from an eval category to the disposition a correct gate
# should return. In-scope answers carry the case's expected card so coverage means
# "answered from the RIGHT card"; every out-of-scope / adversarial category abstains
# (safe), and ambiguous clarifies. This reproduces 0/25 harmful, 20/20 right-card,
# 5/5 ambiguous with no model in the loop.
_CATEGORY_DISPOSITION = {
    "in_scope": "ANSWER",
    "hard_ood": "ABSTAIN",
    "far_ood": "ABSTAIN",
    "ambiguous": "CLARIFY",
    "adversarial": "ABSTAIN",
}


def normalize_decision(raw: Mapping[str, Any]) -> GateDecision:
    """Clamp any raw gate output to the frozen :class:`GateDecision` shape.

    Fail-closed: an unknown/missing disposition becomes ``ABSTAIN``; a card id is kept
    only for an ``ANSWER`` (a CLARIFY/ABSTAIN carries no answered card); candidates are
    coerced to a list of strings; the reason is stringified. Tolerant of a non-mapping
    input so a caller wrapping a flaky gate never explodes here.
    """
    data: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    disposition = data.get("disposition")
    if disposition not in _DISPOSITIONS:
        disposition = "ABSTAIN"
    raw_card = data.get("card_id")
    card_id = raw_card if (disposition == "ANSWER" and isinstance(raw_card, str) and raw_card) else None
    raw_candidates = data.get("candidates", [])
    if isinstance(raw_candidates, (list, tuple)):
        candidates = [c for c in raw_candidates if isinstance(c, str)]
    else:
        candidates = []
    reason = data.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    decision: GateDecision = {
        "disposition": disposition,
        "card_id": card_id,
        "candidates": candidates,
        "reason": reason,
    }
    return decision


def policy_wrapped_gate(raw_gate: GateFn, cards: Sequence[Mapping[str, Any]]) -> GateFn:
    """Wrap a raw gate so its decision is enforced by the deterministic scope policy.

    The returned :data:`GateFn` calls ``raw_gate(query)``, passes the proposal through
    the frozen, downgrade-only ``scope_policy.enforce_policy`` (so a bare-PIN ANSWER
    becomes CLARIFY and a SIM-PIN ANSWER becomes ABSTAIN), then normalizes to the
    frozen shape. ANY exception in the raw gate or the policy fails closed to ABSTAIN.
    """
    # Deferred, torch-free import: scope_policy is pure stdlib. Kept out of module scope
    # so an AST scan of the pipeline package sees no top-level model/scope import here.
    from scope_policy import enforce_policy

    by_id = {
        c["intent_id"]: c
        for c in cards
        if isinstance(c, Mapping) and isinstance(c.get("intent_id"), str)
    }

    def _gate(query: str) -> GateDecision:
        try:
            proposed = raw_gate(query)
            enforced = enforce_policy(query, dict(proposed), by_id)
            return normalize_decision(enforced)
        except Exception as exc:  # fail closed — a broken gate can only ever abstain
            return normalize_decision(
                {"disposition": "ABSTAIN", "reason": f"gate-error: failed closed ({exc})"}
            )

    return _gate


def make_mock_gate(cases: Sequence[EvalCase]) -> GateFn:
    """Build a DETERMINISTIC, model-free gate keyed by exact query text.

    Each :class:`EvalCase` implies the disposition a correct gate returns (in-scope ->
    ANSWER from its expected card, ambiguous -> CLARIFY, everything OOD/adversarial ->
    ABSTAIN). The returned gate reproduces the recorded 0/25 harmful, 20/20 right-card,
    5/5 ambiguous result in CI with no GPU. An unrecognized query fails closed to a
    safe ABSTAIN. When composed under :func:`policy_wrapped_gate`, the in-scope answers
    survive the policy unchanged (their discriminators resolve in-scope), so the
    end-to-end CI metrics are identical.
    """
    table: dict[str, GateDecision] = {}
    for case in cases:
        disposition = _CATEGORY_DISPOSITION.get(case.category, "ABSTAIN")
        card_id = case.expected_card if disposition == "ANSWER" else None
        table[case.query] = normalize_decision(
            {
                "disposition": disposition,
                "card_id": card_id,
                "candidates": [],
                "reason": f"mock:{case.category}",
            }
        )

    def _gate(query: str) -> GateDecision:
        decision = table.get(query)
        if decision is None:
            return normalize_decision(
                {"disposition": "ABSTAIN", "reason": "mock: unknown query (fail closed)"}
            )
        # Return a fresh copy so a caller mutating the result cannot poison the table.
        return dict(decision)  # type: ignore[return-value]

    return _gate


def load_real_bot(candidate_dir: Path) -> object:
    """Load the frozen ``ScopeBot`` from a registered candidate's cards + exemplars.

    REAL-ONLY: this is the single place a real Qwen model is loaded, so the scope_bot
    import (which pulls torch/transformers) is deferred into the body. It is reached
    only from the guarded GPU worker, whose environment has already pinned the A5000.
    Accepts either the artifact directory (``.../files/cards.json``) or the files dir
    directly, so it is decoupled from exactly how the worker addresses the candidate.
    """
    from scope_bot import ScopeBot  # deferred: loads the model — GPU worker only

    root = Path(candidate_dir)

    def _pick(name: str) -> Path:
        direct = root / name
        return direct if direct.exists() else root / "files" / name

    cards_path = _pick("cards.json")
    exemplar_path = _pick("exemplars.json")
    return ScopeBot(
        kb=str(cards_path),
        use_exemplars=True,
        exemplar_path=str(exemplar_path),
    )


def real_gate(bot: object, cards: Sequence[Mapping[str, Any]]) -> GateFn:
    """Wrap a real ``ScopeBot.gate`` through the scope policy.

    REAL-ONLY seam. ``bot.gate`` already applies ``enforce_policy`` internally; wrapping
    it again is idempotent because the policy is strictly downgrade-only, and it keeps
    the serving/eval contract identical to the mock path (both are policy-wrapped).
    """
    return policy_wrapped_gate(bot.gate, cards)  # type: ignore[attr-defined]


def real_respond(bot: object) -> RespondFn:
    """Expose the frozen ``ScopeBot.respond`` as the serving :data:`RespondFn`.

    REAL-ONLY seam. Serving cross-checks this original grounded response against the
    injected, policy-wrapped gate and fails closed on disagreement (see serving).
    """
    return bot.respond  # type: ignore[attr-defined]


def mine_existing(logs_path: Path, cards_path: Path) -> list[dict[str, Any]]:
    """Turn conversation logs into weak labels via the EXISTING ``mine_signals.mine``.

    Deferred imports keep the miner + its log reader (both torch-free stdlib) out of
    module scope. No taxonomy or guardrail is reimplemented here — this only feeds the
    existing miner the grouped sessions and the current cards.
    """
    from feedback_log import read_sessions
    from mine_signals import mine

    cards = json.loads(Path(cards_path).read_text(encoding="utf-8"))
    sessions = read_sessions(Path(logs_path))
    return mine(sessions, cards)


def build_existing(cards_path: Path, merged_labels_path: Path, staging_dir: Path) -> bytes:
    """Build the exemplar bank via the EXISTING ``adapter.learn.build`` in a staged cwd.

    The builder reads ``seed16/cards.json`` and ``logs/mined_labels.jsonl`` relative to
    the current working directory and applies its own confidence/approval/kb-version
    filtering. We do NOT copy that filtering: we stage the two inputs under a temporary
    tree, run the builder from there (``contextlib.chdir``), and serialize the same
    ``{"kb_version", "exemplars"}`` document ``adapter.learn`` would write, so the
    frozen ``ScopeBot`` can load the result unchanged.
    """
    import contextlib
    import shutil

    from adapter import learn
    from feedback_log import kb_version

    staging = Path(staging_dir)
    (staging / "seed16").mkdir(parents=True, exist_ok=True)
    (staging / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cards_path, staging / "seed16" / "cards.json")
    shutil.copyfile(merged_labels_path, staging / "logs" / "mined_labels.jsonl")

    with contextlib.chdir(staging):
        bank = learn.build(kb="seed16/cards.json")
        stamped_version = kb_version("seed16/cards.json")

    document = {"kb_version": stamped_version, "exemplars": bank}
    return (json.dumps(document, indent=2) + "\n").encode("utf-8")
