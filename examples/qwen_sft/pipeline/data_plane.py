"""Data plane — validate cards/logs and content-address a reproducible snapshot.

Real-world analog: a data-lake ingestion layer (DVC / lakeFS content-addressed
snapshotting) fronted by a Great Expectations-style validation suite. Ingest binds a
run to the EXACT bytes of the scope cards, the conversation logs, and the frozen scope
sources; if any frozen source has drifted from the digests declared in config, ingest
fails closed BEFORE a candidate can be built — the supply-chain gate that preserves the
measured named-suite safety evidence.

This module never imports ``scope_bot`` or ``eval50`` (which would drag in torch and
the A6000-grabbing model load): the prompt/model constants and the eval suite are read
by AST through :mod:`pipeline.source_fingerprint`.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.contracts import BlobRef
from pipeline.source_fingerprint import (
    canonical_json,
    eval_suite_sha256,
    load_eval50_cases,
    read_scope_constants,
    sha256_bytes,
    sha256_file,
)

if TYPE_CHECKING:  # import-light: annotations only, no runtime coupling to other lanes
    from pipeline.config import PipelineConfig
    from pipeline.registry import FileRegistry

# The two frozen sources live alongside the example root next to the config's declared
# eval source. eval50.py's whole-file digest is carried in config.frozen.eval_suite_sha256.
_SCOPE_BOT = "scope_bot.py"
_SCOPE_POLICY = "scope_policy.py"
_BUILDER = "adapter/learn.py"

_DISPOSITIONS = frozenset({"ANSWER", "CLARIFY", "ABSTAIN"})


class DataValidationError(ValueError):
    """Raised when cards or conversation logs are structurally invalid."""


class SourceFreezeError(RuntimeError):
    """Raised when a frozen source's sha256 does not match the config-declared digest."""


def _blobref_dict(ref: BlobRef) -> dict[str, Any]:
    """JSON-serializable form of a :class:`BlobRef` (for canonical snapshot addressing)."""
    return {"sha256": ref.sha256, "bytes": ref.bytes, "media_type": ref.media_type}


def validate_cards(cards: Sequence[Mapping[str, Any]]) -> None:
    """Validate the scope cards. Raises :class:`DataValidationError` on any violation.

    A card must have a unique non-empty ``intent_id`` and a non-empty ``supported_goal``;
    ``included`` / ``excluded`` / ``key_facts`` (when present) must be lists of strings;
    each ``required_discriminators`` entry must have a non-empty ``slot`` and string cue
    lists. This is the fail-fast gate that stops an invalid KB before candidate build.
    """
    if isinstance(cards, (str, bytes)) or not isinstance(cards, Sequence):
        raise DataValidationError("cards must be a JSON array")
    if len(cards) == 0:
        raise DataValidationError("cards must be a non-empty array")
    seen: set[str] = set()
    for index, card in enumerate(cards):
        where = f"cards[{index}]"
        if not isinstance(card, Mapping):
            raise DataValidationError(f"{where} must be a JSON object")
        intent_id = card.get("intent_id")
        if not isinstance(intent_id, str) or not intent_id:
            raise DataValidationError(f"{where}.intent_id must be a non-empty string")
        if intent_id in seen:
            raise DataValidationError(f"duplicate intent_id {intent_id!r}")
        seen.add(intent_id)
        goal = card.get("supported_goal")
        if not isinstance(goal, str) or not goal:
            raise DataValidationError(f"{where}.supported_goal must be a non-empty string")
        for key in ("included", "excluded", "key_facts"):
            if key in card:
                value = card[key]
                if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                    raise DataValidationError(f"{where}.{key} must be a list of strings")
        discriminators = card.get("required_discriminators", [])
        if discriminators is None:
            continue
        if not isinstance(discriminators, list):
            raise DataValidationError(f"{where}.required_discriminators must be a list")
        for j, disc in enumerate(discriminators):
            dwhere = f"{where}.required_discriminators[{j}]"
            if not isinstance(disc, Mapping):
                raise DataValidationError(f"{dwhere} must be a JSON object")
            slot = disc.get("slot")
            if not isinstance(slot, str) or not slot:
                raise DataValidationError(f"{dwhere}.slot must be a non-empty string")
            for cue_key in ("in_scope_cues", "out_of_scope_cues"):
                cues = disc.get(cue_key, [])
                if not isinstance(cues, list) or not all(isinstance(x, str) for x in cues):
                    raise DataValidationError(f"{dwhere}.{cue_key} must be a list of strings")


def validate_conversation_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """Validate a conversation log (JSONL) and return the parsed turns as a tuple.

    STRICTER than ``feedback_log.read_sessions`` (which silently skips bad lines): a
    malformed line, a non-object record, a missing/empty ``session_id``, a non-integer
    ``turn``, a non-string ``query``, or an invalid ``disposition`` all raise
    :class:`DataValidationError`. A follow-up turn may carry ``disposition: null``.
    """
    log_path = Path(path)
    if not log_path.is_file():
        raise DataValidationError(f"conversation log not found: {log_path}")
    turns: list[dict[str, Any]] = []
    for lineno, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"{log_path}:{lineno}: invalid JSON ({exc})") from exc
        if not isinstance(record, dict):
            raise DataValidationError(f"{log_path}:{lineno}: line is not a JSON object")
        session_id = record.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise DataValidationError(f"{log_path}:{lineno}: session_id must be a non-empty string")
        turn = record.get("turn")
        if isinstance(turn, bool) or not isinstance(turn, int):
            raise DataValidationError(f"{log_path}:{lineno}: turn must be an integer")
        query = record.get("query")
        if not isinstance(query, str):
            raise DataValidationError(f"{log_path}:{lineno}: query must be a string")
        disposition = record.get("disposition")
        if disposition is not None and disposition not in _DISPOSITIONS:
            raise DataValidationError(
                f"{log_path}:{lineno}: disposition {disposition!r} is not one of {sorted(_DISPOSITIONS)}"
            )
        turns.append(record)
    if not turns:
        raise DataValidationError(f"{log_path}: no conversation turns found")
    return tuple(turns)


def _thresholds_dict(promotion: Any) -> dict[str, int]:
    """The promotion floor as a plain dict, for a stable ``thresholds_sha256``."""
    return {
        "harmful_answers_max": promotion.harmful_answers_max,
        "harmful_total_required": promotion.harmful_total_required,
        "right_card_answers_min": promotion.right_card_answers_min,
        "wrong_card_answers_max": promotion.wrong_card_answers_max,
        "ambiguous_clarifies_min": promotion.ambiguous_clarifies_min,
        "ambiguous_answers_max": promotion.ambiguous_answers_max,
        "errors_max": promotion.errors_max,
        "shadow_unapproved_expansions_max": promotion.shadow_unapproved_expansions_max,
        "canary_harmful_answers_max": promotion.canary_harmful_answers_max,
        "canary_consistency_failures_max": promotion.canary_consistency_failures_max,
    }


def ingest(config: PipelineConfig, registry: FileRegistry) -> Mapping[str, Any]:
    """Validate + content-address the run's inputs into an immutable snapshot mapping.

    Steps, fail-closed in order:

      1. Verify every frozen source (``scope_bot.py``, ``scope_policy.py``, ``eval50.py``)
         hashes to the config-declared digest; a mismatch raises :class:`SourceFreezeError`.
      2. Validate the cards and conversation logs (raises :class:`DataValidationError`).
      3. Put content-addressed blobs for cards, logs, and each frozen source.
      4. Extract the runtime identity (prompts, model id/revision) by AST and derive the
         evaluation contract (structure-only suite digest + thresholds digest).
      5. Compute a ``snapshot_id`` that is the content address of every bound input.

    Returns a snapshot mapping whose ``cards``/``logs``/``sources`` values are
    :class:`BlobRef` instances, ready for the build plane.
    """
    project_root = config.project_root
    scope_bot_path = project_root / _SCOPE_BOT
    scope_policy_path = project_root / _SCOPE_POLICY
    builder_path = project_root / _BUILDER
    eval_path = config.eval_source_path

    # 1. Supply-chain freeze check — refuse to ingest tampered frozen sources.
    freeze_checks = (
        (_SCOPE_BOT, scope_bot_path, config.frozen.scope_bot_sha256),
        (_SCOPE_POLICY, scope_policy_path, config.frozen.scope_policy_sha256),
        ("eval50.py", eval_path, config.frozen.eval_suite_sha256),
    )
    for name, source_path, expected in freeze_checks:
        if not source_path.is_file():
            raise SourceFreezeError(f"ingest: frozen source missing: {source_path}")
        actual = sha256_file(source_path)
        if actual != expected:
            raise SourceFreezeError(
                f"ingest: frozen source {name} sha256 {actual} != config-declared {expected}"
            )

    # 2. Validate the mutable inputs (cards may change and produce a new candidate).
    try:
        cards = json.loads(config.cards_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataValidationError(f"ingest: cards not found: {config.cards_path}") from exc
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"ingest: cards is not valid JSON ({exc})") from exc
    validate_cards(cards)
    turns = validate_conversation_jsonl(config.logs_path)

    # 3. Content-address the inputs and the frozen sources.
    cards_blob = registry.put_blob(config.cards_path.read_bytes(), media_type="application/json")
    logs_blob = registry.put_blob(config.logs_path.read_bytes(), media_type="application/jsonl")
    source_blobs = {
        "scope_bot": registry.put_blob(scope_bot_path.read_bytes(), media_type="text/x-python"),
        "scope_policy": registry.put_blob(scope_policy_path.read_bytes(), media_type="text/x-python"),
        "eval50": registry.put_blob(eval_path.read_bytes(), media_type="text/x-python"),
        "builder": registry.put_blob(builder_path.read_bytes(), media_type="text/x-python"),
    }

    # 4. Runtime identity (AST-extracted; never imports scope_bot) + evaluation contract.
    fingerprint = read_scope_constants(scope_bot_path)
    cases = load_eval50_cases(eval_path)
    runtime = {
        "model_id": config.frozen.model_id,
        "model_revision": config.frozen.model_revision,
        "judge_prompt_sha256": fingerprint.judge_prompt_sha256,
        "generation_prompt_sha256": fingerprint.generation_prompt_sha256,
        "combined_prompt_sha256": fingerprint.combined_prompt_sha256,
        "scope_bot_source_sha256": config.frozen.scope_bot_sha256,
        "scope_policy_source_sha256": config.frozen.scope_policy_sha256,
        "builder_source_sha256": source_blobs["builder"].sha256,
    }
    thresholds = _thresholds_dict(config.promotion)
    evaluation_contract = {
        "eval_suite_sha256": eval_suite_sha256(cases),
        "thresholds_sha256": sha256_bytes(canonical_json(thresholds)),
    }

    # 5. Snapshot identity = content address of every bound input.
    identity = {
        "environment": config.environment,
        "cards": _blobref_dict(cards_blob),
        "logs": _blobref_dict(logs_blob),
        "sources": {name: _blobref_dict(ref) for name, ref in source_blobs.items()},
        "runtime": runtime,
        "evaluation_contract": evaluation_contract,
    }
    snapshot_id = "sha256:" + sha256_bytes(canonical_json(identity))

    return {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "environment": config.environment,
        "cards": cards_blob,
        "logs": logs_blob,
        "sources": source_blobs,
        "runtime": runtime,
        "evaluation_contract": evaluation_contract,
        "cards_path": str(config.cards_path),
        "logs_path": str(config.logs_path),
        "counts": {"cards": len(cards), "log_turns": len(turns)},
    }
