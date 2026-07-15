"""Config-as-code loader and path resolver — the pipeline's control-plane config.

Real-world analog: a Kubernetes ConfigMap materialized from Helm values, or a
Hydra/OmegaConf composed config — a single validated, immutable configuration object
the whole control plane reads. The JSON on disk (``config.ci.json`` /
``config.demo.json``) is the desired state; :func:`load_config` validates its
``schema_version``, resolves every declared path to an absolute path, and freezes the
result into a :class:`PipelineConfig` that no downstream stage may mutate.

This module is pure stdlib: it loads and validates configuration only. It never
recomputes or verifies the frozen source hashes — that freeze CHECK belongs to the
data-plane ingest stage and the source-freeze tests. Here the frozen digests are
simply the DECLARED contract the rest of the pipeline is measured against.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
_ENVIRONMENTS = ("ci", "demo")


@dataclass(frozen=True)
class FrozenHashes:
    """The declared identity of the frozen sources + model this run pins to."""

    scope_bot_sha256: str
    scope_policy_sha256: str
    eval_suite_sha256: str
    model_id: str
    model_revision: str


@dataclass(frozen=True)
class PromotionThresholds:
    """The release floor — the exact current fixed-suite result, as the council fixed
    it. A candidate must meet every one of these to be promotable."""

    harmful_answers_max: int
    harmful_total_required: int
    right_card_answers_min: int
    wrong_card_answers_max: int
    ambiguous_clarifies_min: int
    ambiguous_answers_max: int
    errors_max: int
    shadow_unapproved_expansions_max: int
    canary_harmful_answers_max: int
    canary_consistency_failures_max: int


@dataclass(frozen=True)
class DriftSettings:
    """Conservative drift-alert thresholds. Drift only ALERTS; it never promotes and
    (by itself) never trips the circuit."""

    min_samples: int
    max_disposition_rate_delta: float


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable, fully-resolved pipeline configuration.

    Every path is absolute; the environment is validated ``ci``/``demo``; the frozen
    hashes, GPU identity, promotion floor, and drift settings are the frozen contract
    the DAG, registry, release controller, and observability layer all read.
    """

    schema_version: int
    environment: str
    config_path: Path
    project_root: Path
    state_root: Path
    cards_path: Path
    logs_path: Path
    eval_source_path: Path
    frozen: FrozenHashes
    gpu_uuid: str
    gpu_name: str
    promotion: PromotionThresholds
    drift: DriftSettings


def _require(mapping: object, key: str, where: str) -> object:
    if not isinstance(mapping, dict):
        raise ValueError(f"config: {where} must be a JSON object")
    if key not in mapping:
        raise ValueError(f"config: {where} is missing required key {key!r}")
    return mapping[key]


def _require_str(mapping: object, key: str, where: str) -> str:
    value = _require(mapping, key, where)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config: {where}.{key} must be a non-empty string")
    return value


def _require_int(mapping: object, key: str, where: str) -> int:
    value = _require(mapping, key, where)
    # bool is a subclass of int; reject it so a stray true/false can't be a threshold.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"config: {where}.{key} must be an integer")
    return value


def _require_number(mapping: object, key: str, where: str) -> float:
    value = _require(mapping, key, where)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config: {where}.{key} must be a number")
    return float(value)


def _resolve(root: Path, raw: str) -> Path:
    """Resolve a declared path to an absolute path. Relative paths are anchored to the
    project root (the example directory) so resolution is independent of CWD."""
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def load_config(path: Path, *, state_override: Path | None = None) -> PipelineConfig:
    """Load, validate, and freeze the pipeline configuration at ``path``.

    ``state_override`` (the CLI ``--state`` directory) selects the registry/state
    root; when omitted it defaults to ``<project_root>/.pipeline-<environment>``. The
    project root is the parent of the directory holding the config file — i.e. the
    example directory that contains ``scope_bot.py`` and ``seed16/cards.json`` — so
    the declared relative paths resolve the same regardless of the current directory.
    """
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ValueError(f"config: no such config file: {config_path}")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"config: {config_path} is not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"config: {config_path} must be a JSON object")

    schema_version = _require_int(data, "schema_version", "config")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"config: unsupported schema_version {schema_version} "
            f"(this loader supports {SCHEMA_VERSION})"
        )

    environment = _require_str(data, "environment", "config")
    if environment not in _ENVIRONMENTS:
        raise ValueError(
            f"config: environment {environment!r} must be one of {_ENVIRONMENTS}"
        )

    project_root = config_path.parent.parent

    paths = _require(data, "paths", "config")
    cards_path = _resolve(project_root, _require_str(paths, "cards", "config.paths"))
    logs_path = _resolve(project_root, _require_str(paths, "logs", "config.paths"))
    eval_source_path = _resolve(
        project_root, _require_str(paths, "eval_source", "config.paths")
    )

    frozen_raw = _require(data, "frozen", "config")
    frozen = FrozenHashes(
        scope_bot_sha256=_require_str(frozen_raw, "scope_bot_sha256", "config.frozen"),
        scope_policy_sha256=_require_str(
            frozen_raw, "scope_policy_sha256", "config.frozen"
        ),
        eval_suite_sha256=_require_str(
            frozen_raw, "eval_suite_sha256", "config.frozen"
        ),
        model_id=_require_str(frozen_raw, "model_id", "config.frozen"),
        model_revision=_require_str(frozen_raw, "model_revision", "config.frozen"),
    )

    gpu_raw = _require(data, "gpu", "config")
    gpu_uuid = _require_str(gpu_raw, "uuid", "config.gpu")
    gpu_name = _require_str(gpu_raw, "name", "config.gpu")

    promotion_raw = _require(data, "promotion", "config")
    promotion = PromotionThresholds(
        harmful_answers_max=_require_int(
            promotion_raw, "harmful_answers_max", "config.promotion"
        ),
        harmful_total_required=_require_int(
            promotion_raw, "harmful_total_required", "config.promotion"
        ),
        right_card_answers_min=_require_int(
            promotion_raw, "right_card_answers_min", "config.promotion"
        ),
        wrong_card_answers_max=_require_int(
            promotion_raw, "wrong_card_answers_max", "config.promotion"
        ),
        ambiguous_clarifies_min=_require_int(
            promotion_raw, "ambiguous_clarifies_min", "config.promotion"
        ),
        ambiguous_answers_max=_require_int(
            promotion_raw, "ambiguous_answers_max", "config.promotion"
        ),
        errors_max=_require_int(promotion_raw, "errors_max", "config.promotion"),
        shadow_unapproved_expansions_max=_require_int(
            promotion_raw, "shadow_unapproved_expansions_max", "config.promotion"
        ),
        canary_harmful_answers_max=_require_int(
            promotion_raw, "canary_harmful_answers_max", "config.promotion"
        ),
        canary_consistency_failures_max=_require_int(
            promotion_raw, "canary_consistency_failures_max", "config.promotion"
        ),
    )

    drift_raw = _require(data, "drift", "config")
    drift = DriftSettings(
        min_samples=_require_int(drift_raw, "min_samples", "config.drift"),
        max_disposition_rate_delta=_require_number(
            drift_raw, "max_disposition_rate_delta", "config.drift"
        ),
    )

    if state_override is not None:
        state_root = Path(state_override).resolve()
    else:
        state_root = (project_root / f".pipeline-{environment}").resolve()

    return PipelineConfig(
        schema_version=schema_version,
        environment=environment,
        config_path=config_path,
        project_root=project_root,
        state_root=state_root,
        cards_path=cards_path,
        logs_path=logs_path,
        eval_source_path=eval_source_path,
        frozen=frozen,
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        promotion=promotion,
        drift=drift,
    )
