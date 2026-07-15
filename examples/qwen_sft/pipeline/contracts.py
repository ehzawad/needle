"""Frozen cross-module types for the pipeline package.

Every module imports its shared types from HERE so the fan-out implementer lanes
compose without integration drift. This file is pure types — no behavior, no I/O,
no torch. Owned by the foundation lane; treated as frozen by all others.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

Disposition = Literal["ANSWER", "CLARIFY", "ABSTAIN"]
Backend = Literal["mock", "real"]
Channel = Literal["SHADOW", "CANARY", "CURRENT"]


class GateDecision(TypedDict):
    disposition: Disposition
    card_id: str | None
    candidates: list[str]
    reason: str


GateFn = Callable[[str], GateDecision]
RespondFn = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True)
class EvalCase:
    category: Literal["in_scope", "hard_ood", "far_ood", "ambiguous", "adversarial"]
    query: str
    expected_card: str | None = None


@dataclass(frozen=True)
class BlobRef:
    sha256: str
    bytes: int
    media_type: str


@dataclass(frozen=True)
class DeviceReport:
    uuid: str
    name: str
    logical_index: int
    visible_count: int
    torch_version: str


@dataclass(frozen=True)
class StageResult:
    stage: str
    cache_key: str
    status: Literal["success", "cached", "blocked", "failed"]
    outputs: Mapping[str, str]
    metrics: Mapping[str, int | float | str | bool]


@dataclass(frozen=True)
class EvalReport:
    artifact_id: str
    backend: Backend
    suite_sha256: str
    harmful_answers: int
    harmful_total: int
    right_card_answers: int
    in_scope_total: int
    wrong_card_answers: int
    ambiguous_clarifies: int
    ambiguous_total: int
    errors: int
    predictions: tuple[Mapping[str, Any], ...]
    device: DeviceReport | None
    passed: bool


@dataclass(frozen=True)
class TurnResult:
    disposition: Disposition
    card_id: str | None
    reply: str
    reason: str
    artifact_id: str
    policy_action: str | None


ServeFn = Callable[[str], TurnResult]
