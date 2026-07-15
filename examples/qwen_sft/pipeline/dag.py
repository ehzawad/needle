"""Resumable stage runner — the pipeline's orchestrator and lineage recorder.

Real-world analog: an Airflow / Dagster DAG run, reduced to files and a single process.
Stages execute in a fixed order, each emitting an immutable :class:`~pipeline.contracts.
StageResult`; each stage is content-addressed by a ``cache_key`` over its stage name,
version, input hashes, and the relevant config, so a re-run REUSES a prior success only
when every referenced output still verifies (idempotent, resumable lineage). Per-stage
results and a run summary are written under ``runs/<run_id>/`` — the audit trail Airflow
would keep in its metadata DB.

The DAG is the ONLY place the two backends diverge:

  * ``backend == "mock"``  — model-free CI. A deterministic mock gate
    (:func:`~pipeline.adapters.make_mock_gate`), always policy-wrapped, produces the
    offline / shadow / canary evidence entirely in-process. No torch, no GPU.
  * ``backend == "real"`` — every model-touching action (offline + shadow + canary) is
    computed by ONE guarded child launched via
    :func:`~pipeline.device_guard.launch_gpu_worker` running ``-m pipeline.gpu_worker``.
    This control process never imports torch/transformers/scope_bot.

Safety wiring the council fixed lives here too: a mined ``LEAK`` (or high-confidence
``UNDER_CLARIFY``) in traffic served by the CURRENT release trips the circuit and rolls
back BEFORE any new candidate is considered; an offline-gate miss, a shadow expansion,
or a canary leak each stops the run with ``CURRENT`` unchanged.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pipeline import evaluation, observability, release, serving
from pipeline.adapters import make_mock_gate, policy_wrapped_gate
from pipeline.contracts import Backend, BlobRef, GateFn, RespondFn, StageResult
from pipeline.data_plane import ingest
from pipeline.build_plane import (
    build_candidate,
    mine_labels,
    write_review_queue,
)
from pipeline.registry import FileRegistry, RegistryError
from pipeline.source_fingerprint import (
    canonical_json,
    eval_suite_sha256,
    load_eval50_cases,
    sha256_bytes,
    sha256_file,
)

if TYPE_CHECKING:  # import-light: only for type checkers, no runtime coupling.
    from pipeline.config import PipelineConfig

__all__ = ["run_pipeline", "verify_run", "PipelineBlocked"]

# Stage identifiers, in mandatory order. Versions bump when a stage's semantics change,
# invalidating its cache without touching another stage.
STAGE_ORDER: tuple[str, ...] = (
    "INGEST",
    "MINE",
    "REVIEW_QUEUE",
    "BUILD_CANDIDATE",
    "REGISTER",
    "OFFLINE_EVAL",
    "SHADOW",
    "CANARY",
    "PROMOTE",
)
_STAGE_VERSION = "1"

# A mined UNDER_CLARIFY at/above this confidence is treated as a safety regression on the
# CURRENT release (matches mine_signals' 0.85 for an under-clarify). LEAK always trips.
_UNDER_CLARIFY_TRIP_CONF = 0.85

# The guarded real-worker gets a generous ceiling: one 4B model load + 50-case eval.
_WORKER_TIMEOUT_S = 1800

_EVIDENCE_BEGIN = "===EVIDENCE_JSON_BEGIN==="
_EVIDENCE_END = "===EVIDENCE_JSON_END==="


class PipelineBlocked(RuntimeError):
    """Raised internally when a gate stage blocks the run; carries the blocking result."""

    def __init__(self, result: StageResult, reason: str) -> None:
        super().__init__(reason)
        self.result = result
        self.reason = reason


class _MemoryTurnLogger:
    """No-side-effects TurnLogger stand-in for mock canary/serving probes.

    Matches ``feedback_log.TurnLogger.log`` structurally so the serving wrapper logs each
    probe, but keeps records in memory so synthetic probes never pollute the label source.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def log(self, session_id: str, turn: int, query: str, gate: dict, reply: str,
            shortlist: list | None = None, extra: dict | None = None) -> dict:
        record: dict[str, Any] = {
            "session_id": session_id, "turn": turn, "query": query,
            "disposition": gate.get("disposition"), "card_id": gate.get("card_id"),
            "reply": reply,
        }
        if extra:
            record.update(extra)
        self.records.append(record)
        return record


# ---------------------------------------------------------------------------
# Small JSON / blob helpers.
# ---------------------------------------------------------------------------


def _blobref(value: Mapping[str, Any] | BlobRef) -> BlobRef:
    if isinstance(value, BlobRef):
        return value
    return BlobRef(sha256=value["sha256"], bytes=int(value["bytes"]),
                   media_type=value["media_type"])


def _json_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an ingest snapshot (with :class:`BlobRef` leaves) to a plain JSON dict."""
    def conv(value: Any) -> Any:
        if isinstance(value, BlobRef):
            return {"sha256": value.sha256, "bytes": value.bytes, "media_type": value.media_type}
        if isinstance(value, Mapping):
            return {str(k): conv(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [conv(v) for v in value]
        return value
    return {str(k): conv(v) for k, v in snapshot.items()}


def _parse_jsonl(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            rows.append(record)
    return rows


def _cache_key(stage: str, version: str, inputs: Mapping[str, Any]) -> str:
    """Content id of a stage invocation: stage name + version + canonical input hashes."""
    return sha256_bytes(
        (stage + version).encode("utf-8")
        + canonical_json({"inputs": inputs})
    )


def _stage_dict(result: StageResult) -> dict[str, Any]:
    return {
        "stage": result.stage,
        "cache_key": result.cache_key,
        "status": result.status,
        "outputs": dict(result.outputs),
        "metrics": dict(result.metrics),
    }


# ---------------------------------------------------------------------------
# Run context — carries registry, config, backend, and lazily-gathered evidence.
# ---------------------------------------------------------------------------


class _RunContext:
    def __init__(self, config: PipelineConfig, registry: FileRegistry, *,
                 backend: Backend, actor: str, run_id: str) -> None:
        self.config = config
        self.registry = registry
        self.backend = backend
        self.actor = actor
        self.run_id = run_id
        self.stages_dir = registry.runs_dir / run_id / "stages"
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self.event_path = registry.observability_dir / "events.jsonl"
        self.results: dict[str, StageResult] = {}
        self.payloads: dict[str, dict[str, Any]] = {}
        self._bundles: dict[str, dict[str, Any]] = {}
        self.integrity_failures = 0

    # -- observability -----------------------------------------------------

    def emit(self, stage: str, event: str, status: str, *, cache_key: str | None = None,
             artifact_id: str | None = None, duration_ms: int | None = None,
             counts: Mapping[str, Any] | None = None, error_code: str | None = None) -> None:
        try:
            observability.emit_event(self.event_path, {
                "run_id": self.run_id, "stage": stage, "event": event, "status": status,
                "backend": self.backend, "cache_key": cache_key, "artifact_id": artifact_id,
                "duration_ms": duration_ms, "counts": dict(counts or {}),
                "error_code": error_code,
            })
        except Exception:
            pass

    # -- resumable stage cache --------------------------------------------

    def _stage_path(self, stage: str) -> Path:
        return self.stages_dir / f"{stage}.json"

    def _load_cached(self, stage: str, cache_key: str,
                     verify: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
        path = self._stage_path(stage)
        if not path.is_file():
            return None
        try:
            saved = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if saved.get("cache_key") != cache_key:
            return None
        if saved.get("status") not in ("success", "cached"):
            return None
        payload = saved.get("payload", {})
        try:
            if not verify(payload):
                return None
        except Exception:
            return None
        return saved

    def run_stage(
        self,
        stage: str,
        *,
        inputs: Mapping[str, Any],
        compute: Callable[[], tuple[Mapping[str, str], Mapping[str, Any], dict[str, Any]]],
        verify: Callable[[dict[str, Any]], bool],
    ) -> dict[str, Any]:
        """Execute (or reuse) one stage. ``compute`` returns (outputs, metrics, payload)."""
        cache_key = _cache_key(stage, _STAGE_VERSION, inputs)
        t0 = time.monotonic()
        cached = self._load_cached(stage, cache_key, verify)
        if cached is not None:
            result = StageResult(stage=stage, cache_key=cache_key, status="cached",
                                 outputs=cached["outputs"], metrics=cached["metrics"])
            self.results[stage] = result
            self.payloads[stage] = cached.get("payload", {})
            self.emit(stage, "stage", "cached", cache_key=cache_key,
                      artifact_id=cached["outputs"].get("artifact_id"),
                      duration_ms=int((time.monotonic() - t0) * 1000),
                      counts=cached["metrics"])
            return self.payloads[stage]

        outputs, metrics, payload = compute()
        result = StageResult(stage=stage, cache_key=cache_key, status="success",
                             outputs=dict(outputs), metrics=dict(metrics))
        self.results[stage] = result
        self.payloads[stage] = payload
        self._persist(stage, result, payload)
        self.emit(stage, "stage", "success", cache_key=cache_key,
                  artifact_id=dict(outputs).get("artifact_id"),
                  duration_ms=int((time.monotonic() - t0) * 1000), counts=metrics)
        return payload

    def block(self, stage: str, cache_key: str, *, metrics: Mapping[str, Any],
              outputs: Mapping[str, str], payload: dict[str, Any], reason: str,
              error_code: str) -> None:
        result = StageResult(stage=stage, cache_key=cache_key, status="blocked",
                             outputs=dict(outputs), metrics=dict(metrics))
        self.results[stage] = result
        self.payloads[stage] = payload
        self._persist(stage, result, payload)
        self.emit(stage, "stage", "blocked", cache_key=cache_key,
                  artifact_id=dict(outputs).get("artifact_id"), counts=metrics,
                  error_code=error_code)
        raise PipelineBlocked(result, reason)

    def _persist(self, stage: str, result: StageResult, payload: dict[str, Any]) -> None:
        body = _stage_dict(result)
        body["payload"] = payload
        body["run_id"] = self.run_id
        self._stage_path(stage).write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # -- evidence gathering (mock in-process, real via the guarded worker) --

    def evidence_bundle(self, artifact_id: str, *, snapshot: Mapping[str, Any],
                        cases: Sequence[Any], suite_sha256: str,
                        current_artifact_id: str | None,
                        approved: Sequence[str]) -> dict[str, Any]:
        if artifact_id in self._bundles:
            return self._bundles[artifact_id]
        if self.backend == "mock":
            bundle = self._mock_bundle(artifact_id, snapshot, cases, suite_sha256,
                                       current_artifact_id, approved)
        else:
            bundle = self._real_bundle(artifact_id, cases, suite_sha256,
                                       snapshot, current_artifact_id, approved)
        self._bundles[artifact_id] = bundle
        return bundle

    def _mock_bundle(self, artifact_id: str, snapshot: Mapping[str, Any],
                     cases: Sequence[Any], suite_sha256: str,
                     current_artifact_id: str | None,
                     approved: Sequence[str]) -> dict[str, Any]:
        cards = json.loads(self.registry.read_blob(_blobref(snapshot["cards"])).decode("utf-8"))
        wrapped = policy_wrapped_gate(make_mock_gate(cases), cards)

        offline = evaluation.evaluate(
            wrapped, cases, artifact_id=artifact_id, backend="mock",
            suite_sha256=suite_sha256, device=None,
        )

        logs = _parse_jsonl(self.registry.read_blob(_blobref(snapshot["logs"])))
        eligible = _eligible_shadow_turns(logs, current_artifact_id)
        shadow = evaluation.evaluate_shadow(
            wrapped, eligible, artifact_id=artifact_id,
            current_artifact_id=current_artifact_id, approved_expansions=approved,
        )

        respond = _mock_respond(wrapped)
        serve = serving.make_server(
            artifact_id=artifact_id, cards=cards, gate=wrapped, respond=respond,
            turn_logger=_MemoryTurnLogger(), event_path=self.event_path,
        )
        canary = evaluation.evaluate_canary(serve, cases, artifact_id=artifact_id)

        return {
            "offline_eval": evaluation.eval_report_to_dict(offline),
            "shadow": dict(shadow), "canary": dict(canary), "device": None,
        }

    def _real_bundle(self, artifact_id: str, cases: Sequence[Any], suite_sha256: str,
                     snapshot: Mapping[str, Any], current_artifact_id: str | None,
                     approved: Sequence[str]) -> dict[str, Any]:
        # Model-touching evidence is produced EXCLUSIVELY by the guarded child; this
        # process never imports torch. We stage the worker's inputs under the run dir.
        from pipeline.device_guard import launch_gpu_worker  # torch-free at import

        work = self.registry.runs_dir / self.run_id / "worker"
        work.mkdir(parents=True, exist_ok=True)
        logs = _parse_jsonl(self.registry.read_blob(_blobref(snapshot["logs"])))
        eligible = _eligible_shadow_turns(logs, current_artifact_id)
        shadow_path = work / "shadow_turns.jsonl"
        shadow_path.write_text(
            "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in eligible),
            encoding="utf-8",
        )
        approved_path = work / "approved.json"
        approved_path.write_text(json.dumps(list(approved)), encoding="utf-8")
        out_path = work / "evidence_bundle.json"

        argv = [
            "-m", "pipeline.gpu_worker",
            "--candidate-dir", str(self.registry.artifact_dir(artifact_id)),
            "--eval-source", str(self.config.eval_source_path),
            "--artifact-id", artifact_id,
            "--suite-sha256", suite_sha256,
            "--shadow-turns", str(shadow_path),
            "--approved-expansions", str(approved_path),
            "--event-path", str(self.event_path),
            "--out", str(out_path),
        ]
        if current_artifact_id:
            argv += ["--current-artifact-id", current_artifact_id]

        completed = launch_gpu_worker(argv, cwd=self.config.project_root,
                                      timeout_s=_WORKER_TIMEOUT_S)
        if completed.returncode != 0:
            raise RuntimeError(
                f"gpu_worker exited {completed.returncode}: "
                f"{(completed.stderr or '').strip()[-600:]}"
            )
        bundle = _parse_worker_bundle(completed.stdout, out_path)
        if bundle is None:
            raise RuntimeError("gpu_worker produced no parseable evidence bundle")
        return bundle


def _eligible_shadow_turns(logs: Sequence[Mapping[str, Any]],
                           current_artifact_id: str | None) -> list[dict[str, Any]]:
    """Shadow replays ONLY prior traffic served by the CURRENT release. On a first run
    (no CURRENT, seed logs untagged) this is empty, so shadow is trivially safe."""
    if not current_artifact_id:
        return []
    return [dict(t) for t in logs if t.get("artifact_id") == current_artifact_id]


def _mock_respond(gate: GateFn) -> RespondFn:
    """A model-free ``respond`` that mirrors the (policy-wrapped) gate, so the serving
    cross-check agrees in CI exactly as the real ``ScopeBot.respond`` would when correct."""
    def respond(query: str) -> Mapping[str, Any]:
        decision = gate(query)
        disposition = decision.get("disposition")
        card = decision.get("card_id")
        if disposition == "ANSWER" and card:
            return {"disposition": "ANSWER", "card": card,
                    "reply": f"[mock grounded answer from card {card}]",
                    "reason": decision.get("reason", "")}
        if disposition == "CLARIFY":
            return {"disposition": "CLARIFY", "card": None,
                    "reply": "[mock clarify] which of these did you mean?",
                    "reason": decision.get("reason", "")}
        return {"disposition": "ABSTAIN", "card": None,
                "reply": "[mock refusal] outside supported scope",
                "reason": decision.get("reason", "")}
    return respond


def _parse_worker_bundle(stdout: str, out_path: Path) -> dict[str, Any] | None:
    if out_path.is_file():
        try:
            data = json.loads(out_path.read_text("utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    if _EVIDENCE_BEGIN in stdout and _EVIDENCE_END in stdout:
        chunk = stdout.split(_EVIDENCE_BEGIN, 1)[1].split(_EVIDENCE_END, 1)[0].strip()
        try:
            data = json.loads(chunk)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Offline release-floor check (mirrors evaluation.require_offline_gate on a dict).
# ---------------------------------------------------------------------------


def _check_offline(metrics: Mapping[str, Any], promotion: Any) -> list[str]:
    reasons: list[str] = []
    if int(metrics.get("harmful_answers", 1)) > promotion.harmful_answers_max:
        reasons.append(f"harmful_answers={metrics.get('harmful_answers')}")
    if int(metrics.get("harmful_total", 0)) < promotion.harmful_total_required:
        reasons.append(f"harmful_total={metrics.get('harmful_total')} < {promotion.harmful_total_required}")
    if int(metrics.get("right_card_answers", 0)) < promotion.right_card_answers_min:
        reasons.append(f"right_card_answers={metrics.get('right_card_answers')} < {promotion.right_card_answers_min}")
    if int(metrics.get("wrong_card_answers", 1)) > promotion.wrong_card_answers_max:
        reasons.append(f"wrong_card_answers={metrics.get('wrong_card_answers')}")
    if int(metrics.get("ambiguous_clarifies", 0)) < promotion.ambiguous_clarifies_min:
        reasons.append(f"ambiguous_clarifies={metrics.get('ambiguous_clarifies')} < {promotion.ambiguous_clarifies_min}")
    if int(metrics.get("errors", 1)) > promotion.errors_max:
        reasons.append(f"errors={metrics.get('errors')}")
    return reasons


def _offline_metric_view(offline: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("harmful_answers", "harmful_total", "right_card_answers", "in_scope_total",
            "wrong_card_answers", "ambiguous_clarifies", "ambiguous_total", "errors")
    return {k: offline.get(k) for k in keys}


# ---------------------------------------------------------------------------
# Circuit-trip detection on mined signals tied to the CURRENT release.
# ---------------------------------------------------------------------------


def _mined_safety_trip(labels: Sequence[Mapping[str, Any]],
                       served_by_current: set[tuple[Any, Any]]) -> Mapping[str, Any] | None:
    """Return the first LEAK / high-confidence UNDER_CLARIFY whose turn was served by the
    CURRENT release, or ``None``. This is what opens the circuit + rolls back."""
    for label in labels:
        key = (label.get("session_id"), label.get("turn"))
        if key not in served_by_current:
            continue
        signal = label.get("signal")
        conf = label.get("confidence") or 0.0
        if signal == "LEAK" or (signal == "UNDER_CLARIFY" and conf >= _UNDER_CLARIFY_TRIP_CONF):
            return label
    return None


# ---------------------------------------------------------------------------
# Run id + observability rollup.
# ---------------------------------------------------------------------------


def _run_id(config: PipelineConfig, backend: Backend) -> str:
    seed = canonical_json({
        "environment": config.environment,
        "backend": backend,
        "config": str(config.config_path),
    })
    return "run-" + sha256_bytes(seed)[:16]


def _channel_record(registry: FileRegistry, channel: str) -> dict[str, Any] | None:
    path = registry.channel_path(channel)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _circuit_open(registry: FileRegistry) -> bool:
    path = registry.circuit_path
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return bool(isinstance(data, dict) and data.get("open"))


def _write_observability(ctx: _RunContext, *, artifact_id: str | None,
                         evidence: Sequence[Mapping[str, Any]],
                         offline_metrics: Mapping[str, Any] | None,
                         canary_metrics: Mapping[str, Any] | None) -> None:
    registry = ctx.registry
    stage_runs = len(ctx.results)
    stage_failures = sum(1 for r in ctx.results.values() if r.status == "blocked")
    metrics: dict[str, int | float] = {
        "scope_pipeline_stage_runs_total": stage_runs,
        "scope_pipeline_stage_failures_total": stage_failures,
        "scope_registry_integrity_failures_total": ctx.integrity_failures,
        "scope_circuit_open": 1 if _circuit_open(registry) else 0,
        "scope_drift_alerts_total": 0,
    }
    if artifact_id and offline_metrics is not None:
        label = f'{{artifact_id="{artifact_id}",backend="{ctx.backend}"}}'
        metrics[f"scope_eval_harmful_answers{label}"] = int(offline_metrics.get("harmful_answers", 0))
        metrics[f"scope_eval_wrong_card_answers{label}"] = int(offline_metrics.get("wrong_card_answers", 0))
        ambiguous_answers = int(offline_metrics.get("ambiguous_total", 0)) - int(
            offline_metrics.get("ambiguous_clarifies", 0))
        metrics[f"scope_eval_ambiguous_answers{label}"] = ambiguous_answers
    if artifact_id and canary_metrics is not None:
        clabel = f'{{artifact_id="{artifact_id}"}}'
        metrics[f"scope_serving_consistency_failures_total{clabel}"] = int(
            canary_metrics.get("consistency_failures", 0))
    try:
        observability.write_prometheus_text(registry.observability_dir / "metrics.prom", metrics)
    except Exception:
        pass

    current = _channel_record(registry, "CURRENT") or {"channel": "CURRENT", "artifact_id": None}
    release_view = dict(current)
    release_view.setdefault("channel", "CURRENT")
    release_view["backend"] = ctx.backend
    try:
        observability.render_dashboard(
            registry.observability_dir / "dashboard.html",
            release=release_view, evidence=list(evidence), alerts=[],
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The DAG.
# ---------------------------------------------------------------------------


def run_pipeline(
    config: PipelineConfig,
    *,
    backend: Backend,
    promote_current: bool,
    actor: str,
) -> Mapping[str, Any]:
    """Run the fixed stage DAG and return an immutable run summary mapping.

    Stages: INGEST -> MINE -> REVIEW_QUEUE -> BUILD_CANDIDATE -> REGISTER -> OFFLINE_EVAL
    -> SHADOW -> CANARY -> (optional) PROMOTE. On a mined safety regression against the
    CURRENT release the circuit is tripped and the run stops. On any gate miss the run is
    ``blocked`` and ``CURRENT`` is left unchanged.
    """
    if backend not in ("mock", "real"):
        raise ValueError(f"run_pipeline: backend must be 'mock' or 'real', got {backend!r}")
    registry = FileRegistry(config.state_root, environment=config.environment)  # type: ignore[arg-type]
    run_id = _run_id(config, backend)
    ctx = _RunContext(config, registry, backend=backend, actor=actor, run_id=run_id)
    ctx.emit("run", "run", "started")

    current_before = release.resolve_channel(registry, "CURRENT")
    cases = load_eval50_cases(config.eval_source_path)
    suite_sha256 = eval_suite_sha256(cases)

    status = "evaluated"
    circuit_tripped = False
    artifact_id: str | None = None
    evidence_ids: dict[str, str] = {}
    offline_metrics: dict[str, Any] | None = None
    canary_metrics: dict[str, Any] | None = None
    evidence_records: list[Mapping[str, Any]] = []
    block_reason: str | None = None

    try:
        # --- INGEST -------------------------------------------------------
        def _do_ingest() -> tuple[Mapping[str, str], Mapping[str, Any], dict[str, Any]]:
            snapshot = _json_snapshot(ingest(config, registry))
            outputs = {
                "snapshot_id": snapshot["snapshot_id"],
                "cards": snapshot["cards"]["sha256"],
                "logs": snapshot["logs"]["sha256"],
            }
            metrics = dict(snapshot.get("counts", {}))
            return outputs, metrics, {"snapshot": snapshot}

        def _verify_ingest(payload: dict[str, Any]) -> bool:
            snap = payload.get("snapshot", {})
            registry.read_blob(_blobref(snap["cards"]))
            registry.read_blob(_blobref(snap["logs"]))
            return True

        ingest_payload = ctx.run_stage(
            "INGEST",
            inputs={
                "cards_file": sha256_file(config.cards_path),
                "logs_file": sha256_file(config.logs_path),
                "environment": config.environment,
                "frozen": {
                    "scope_bot": config.frozen.scope_bot_sha256,
                    "scope_policy": config.frozen.scope_policy_sha256,
                    "eval_suite": config.frozen.eval_suite_sha256,
                },
            },
            compute=_do_ingest, verify=_verify_ingest,
        )
        snapshot = ingest_payload["snapshot"]

        # --- MINE ---------------------------------------------------------
        def _do_mine() -> tuple[Mapping[str, str], Mapping[str, Any], dict[str, Any]]:
            labels_blob = mine_labels(snapshot, registry)
            labels = _parse_jsonl(registry.read_blob(labels_blob))
            signals: dict[str, int] = {}
            for label in labels:
                sig = str(label.get("signal"))
                signals[sig] = signals.get(sig, 0) + 1
            outputs = {"labels_sha256": labels_blob.sha256}
            metrics = {"mined": len(labels),
                       "needs_review": sum(1 for x in labels if x.get("needs_review")),
                       **{f"signal_{k}": v for k, v in signals.items()}}
            payload = {
                "labels": {"sha256": labels_blob.sha256, "bytes": labels_blob.bytes,
                           "media_type": labels_blob.media_type},
                "labels_list": labels,
            }
            return outputs, metrics, payload

        def _verify_mine(payload: dict[str, Any]) -> bool:
            registry.read_blob(_blobref(payload["labels"]))
            return True

        mine_payload = ctx.run_stage(
            "MINE",
            inputs={"snapshot_id": snapshot["snapshot_id"],
                    "cards": snapshot["cards"]["sha256"],
                    "logs": snapshot["logs"]["sha256"]},
            compute=_do_mine, verify=_verify_mine,
        )
        labels_blob = _blobref(mine_payload["labels"])
        mined_labels = mine_payload.get("labels_list") or _parse_jsonl(
            registry.read_blob(labels_blob))

        # Circuit feedback: a LEAK / high-confidence under-clarify in traffic SERVED by
        # the current release trips the breaker and rolls back before anything else.
        logs = _parse_jsonl(registry.read_blob(_blobref(snapshot["logs"])))
        served_by_current = {
            (t.get("session_id"), t.get("turn"))
            for t in logs if current_before and t.get("artifact_id") == current_before
        }
        trip_label = _mined_safety_trip(mined_labels, served_by_current)
        if current_before and trip_label is not None:
            trip_evidence = registry.write_evidence("shadow", {
                "artifact_id": current_before, "backend": backend, "passed": False,
                "metrics": {"signal": trip_label.get("signal"),
                            "confidence": trip_label.get("confidence")},
                "note": "mined safety regression on CURRENT",
            })
            circuit = release.trip_circuit(
                registry, bad_artifact_id=current_before,
                evidence_id=trip_evidence["evidence_id"],
                reason=f"mined {trip_label.get('signal')} on current release",
            )
            circuit_tripped = True
            status = "circuit_tripped"
            block_reason = f"circuit tripped: mined {trip_label.get('signal')} on CURRENT"
            ctx.emit("MINE", "circuit_trip", "alert", artifact_id=current_before,
                     error_code="mined_safety_regression",
                     counts={"rolled_back_to": circuit.get("rolled_back_to")})
            raise PipelineBlocked(ctx.results["MINE"], block_reason)

        # --- REVIEW_QUEUE -------------------------------------------------
        reviews_path = registry.reviews_path

        def _reviews_sha() -> str:
            if reviews_path.is_file():
                return sha256_bytes(reviews_path.read_bytes())
            return sha256_bytes(b"")

        def _do_review() -> tuple[Mapping[str, str], Mapping[str, Any], dict[str, Any]]:
            # Idempotent: only needs-review labels are queued, and only once. The set of
            # queued items is a pure function of the (immutable) labels blob; human
            # DECISIONS are separate records that BUILD_CANDIDATE keys on, not this stage.
            pending = write_review_queue(labels_blob, registry, reviews_path)
            return ({"labels_sha256": labels_blob.sha256, "reviews_sha256": _reviews_sha()},
                    {"pending": pending},
                    {"labels": {"sha256": labels_blob.sha256, "bytes": labels_blob.bytes,
                                "media_type": labels_blob.media_type}, "pending": pending})

        ctx.run_stage(
            "REVIEW_QUEUE",
            inputs={"labels_sha256": labels_blob.sha256},
            compute=_do_review, verify=_verify_mine,
        )

        # Approved answer-expansions a human signed off on (empty in CI). Kept for the
        # shadow gate; a real reviewer adds them via `pipeline review set`.
        approved: list[str] = []

        # --- BUILD_CANDIDATE ---------------------------------------------
        parent = current_before

        def _do_build() -> tuple[Mapping[str, str], Mapping[str, Any], dict[str, Any]]:
            manifest = build_candidate(
                snapshot, labels_blob, reviews_path, registry, parent_artifact_id=parent)
            aid = manifest["artifact_id"]
            return ({"artifact_id": aid},
                    {"artifact_id": aid, "parent": parent or "none"},
                    {"artifact_id": aid, "manifest_sha256": manifest.get("manifest_sha256")})

        def _verify_candidate(payload: dict[str, Any]) -> bool:
            registry.verify_candidate(payload["artifact_id"])
            return True

        build_payload = ctx.run_stage(
            "BUILD_CANDIDATE",
            inputs={"snapshot_id": snapshot["snapshot_id"],
                    "labels_sha256": labels_blob.sha256,
                    "reviews_sha256": _reviews_sha(), "parent": parent or ""},
            compute=_do_build, verify=_verify_candidate,
        )
        artifact_id = build_payload["artifact_id"]

        # --- REGISTER (verify the immutable candidate) --------------------
        def _do_register() -> tuple[Mapping[str, str], Mapping[str, Any], dict[str, Any]]:
            registry.verify_candidate(artifact_id)
            return ({"artifact_id": artifact_id}, {"verified": True},
                    {"artifact_id": artifact_id, "verified": True})

        ctx.run_stage(
            "REGISTER", inputs={"artifact_id": artifact_id},
            compute=_do_register, verify=_verify_candidate,
        )

        # --- OFFLINE_EVAL -------------------------------------------------
        thresholds_sha = snapshot["evaluation_contract"]["thresholds_sha256"]

        def _do_offline() -> tuple[Mapping[str, str], Mapping[str, Any], dict[str, Any]]:
            bundle = ctx.evidence_bundle(
                artifact_id, snapshot=snapshot, cases=cases, suite_sha256=suite_sha256,
                current_artifact_id=current_before, approved=approved)
            offline = bundle["offline_eval"]
            reasons = _check_offline(offline, config.promotion)
            predictions_sha = sha256_bytes(canonical_json(offline.get("predictions", [])))
            body = {
                "artifact_id": artifact_id, "backend": backend, "suite_sha256": suite_sha256,
                "metrics": _offline_metric_view(offline),
                "predictions_sha256": predictions_sha,
                "passed": not reasons, "device": bundle.get("device"),
            }
            evidence = registry.write_evidence("offline_eval", body)
            payload = {"evidence_id": evidence["evidence_id"], "passed": not reasons,
                       "reasons": reasons, "metrics": _offline_metric_view(offline),
                       "record": dict(evidence)}
            return ({"evidence_id": evidence["evidence_id"]},
                    {**_offline_metric_view(offline), "passed": not reasons}, payload)

        def _verify_evidence(payload: dict[str, Any]) -> bool:
            registry.load_evidence(payload["evidence_id"])
            return True

        offline_payload = ctx.run_stage(
            "OFFLINE_EVAL",
            inputs={"artifact_id": artifact_id, "suite_sha256": suite_sha256,
                    "backend": backend, "thresholds_sha256": thresholds_sha},
            compute=_do_offline, verify=_verify_evidence,
        )
        evidence_ids["offline"] = offline_payload["evidence_id"]
        offline_metrics = offline_payload["metrics"]
        evidence_records.append(registry.load_evidence(offline_payload["evidence_id"]))
        if not offline_payload["passed"]:
            block_reason = "offline eval below release floor: " + "; ".join(
                offline_payload.get("reasons", []))
            ctx.block("OFFLINE_EVAL", ctx.results["OFFLINE_EVAL"].cache_key,
                      metrics=ctx.results["OFFLINE_EVAL"].metrics,
                      outputs={"artifact_id": artifact_id,
                               "evidence_id": offline_payload["evidence_id"]},
                      payload=offline_payload, reason=block_reason,
                      error_code="offline_gate")

        # --- SHADOW -------------------------------------------------------
        def _do_shadow() -> tuple[Mapping[str, str], Mapping[str, Any], dict[str, Any]]:
            bundle = ctx.evidence_bundle(
                artifact_id, snapshot=snapshot, cases=cases, suite_sha256=suite_sha256,
                current_artifact_id=current_before, approved=approved)
            shadow = bundle["shadow"]
            body = {
                "artifact_id": artifact_id, "backend": backend,
                "current_artifact_id": current_before,
                "metrics": {"replayed": shadow.get("replayed", 0),
                            "expansions": shadow.get("expansions", 0),
                            "approved_expansions": shadow.get("approved_expansions", 0),
                            "unapproved_expansions": shadow.get("unapproved_expansions", 0),
                            "errors": shadow.get("errors", 0)},
                "passed": bool(shadow.get("passed")),
            }
            evidence = registry.write_evidence("shadow", body)
            payload = {"evidence_id": evidence["evidence_id"],
                       "passed": bool(shadow.get("passed")),
                       "metrics": body["metrics"], "record": dict(evidence)}
            return ({"evidence_id": evidence["evidence_id"]},
                    {**body["metrics"], "passed": body["passed"]}, payload)

        shadow_payload = ctx.run_stage(
            "SHADOW",
            inputs={"artifact_id": artifact_id, "backend": backend,
                    "current": current_before or "", "logs": snapshot["logs"]["sha256"],
                    "approved": sorted(approved)},
            compute=_do_shadow, verify=_verify_evidence,
        )
        evidence_ids["shadow"] = shadow_payload["evidence_id"]
        evidence_records.append(registry.load_evidence(shadow_payload["evidence_id"]))
        if not shadow_payload["passed"]:
            # Fail: clear SHADOW, CURRENT unchanged.
            block_reason = "shadow replay found unapproved answer-expansions"
            ctx.block("SHADOW", ctx.results["SHADOW"].cache_key,
                      metrics=ctx.results["SHADOW"].metrics,
                      outputs={"artifact_id": artifact_id,
                               "evidence_id": shadow_payload["evidence_id"]},
                      payload=shadow_payload, reason=block_reason,
                      error_code="shadow_expansion")
        # Passed: stage the SHADOW channel at the candidate.
        try:
            release.set_channel(registry, "SHADOW", artifact_id,
                                evidence_ids=[shadow_payload["evidence_id"]], actor=actor)
        except Exception:
            pass

        # --- CANARY -------------------------------------------------------
        def _do_canary() -> tuple[Mapping[str, str], Mapping[str, Any], dict[str, Any]]:
            bundle = ctx.evidence_bundle(
                artifact_id, snapshot=snapshot, cases=cases, suite_sha256=suite_sha256,
                current_artifact_id=current_before, approved=approved)
            canary = bundle["canary"]
            body = {
                "artifact_id": artifact_id, "backend": backend,
                "metrics": {"probes_total": canary.get("probes_total", 0),
                            "harmful_answers": canary.get("harmful_answers", 0),
                            "consistency_failures": canary.get("consistency_failures", 0),
                            "errors": canary.get("errors", 0),
                            "smoke_answered": canary.get("smoke_answered", False)},
                "passed": bool(canary.get("passed")),
            }
            evidence = registry.write_evidence("canary", body)
            payload = {"evidence_id": evidence["evidence_id"],
                       "passed": bool(canary.get("passed")),
                       "metrics": body["metrics"], "record": dict(evidence)}
            return ({"evidence_id": evidence["evidence_id"]},
                    {**body["metrics"], "passed": body["passed"]}, payload)

        canary_payload = ctx.run_stage(
            "CANARY",
            inputs={"artifact_id": artifact_id, "backend": backend,
                    "suite_sha256": suite_sha256},
            compute=_do_canary, verify=_verify_evidence,
        )
        evidence_ids["canary"] = canary_payload["evidence_id"]
        canary_metrics = canary_payload["metrics"]
        evidence_records.append(registry.load_evidence(canary_payload["evidence_id"]))
        try:
            release.set_channel(registry, "CANARY", artifact_id,
                                evidence_ids=[canary_payload["evidence_id"]], actor=actor)
        except Exception:
            pass
        if not canary_payload["passed"]:
            # Fail: leave CURRENT unchanged. The candidate is quarantined (never promoted);
            # the genuine circuit+rollback path is the mined-regression case on CURRENT.
            block_reason = "canary serving probes leaked or disagreed"
            ctx.block("CANARY", ctx.results["CANARY"].cache_key,
                      metrics=ctx.results["CANARY"].metrics,
                      outputs={"artifact_id": artifact_id,
                               "evidence_id": canary_payload["evidence_id"]},
                      payload=canary_payload, reason=block_reason,
                      error_code="canary_leak")

        # --- PROMOTE (optional) ------------------------------------------
        if promote_current:
            def _do_promote() -> tuple[Mapping[str, str], Mapping[str, Any], dict[str, Any]]:
                record = release.promote(
                    registry, artifact_id,
                    offline_evidence_id=evidence_ids["offline"],
                    shadow_evidence_id=evidence_ids["shadow"],
                    canary_evidence_id=evidence_ids["canary"], actor=actor)
                return ({"artifact_id": artifact_id,
                         "release_id": str(record.get("release_id"))},
                        {"promoted": True},
                        {"artifact_id": artifact_id, "record": dict(record)})

            def _verify_promote(payload: dict[str, Any]) -> bool:
                return release.resolve_channel(registry, "CURRENT") == payload.get("artifact_id")

            try:
                ctx.run_stage(
                    "PROMOTE",
                    inputs={"artifact_id": artifact_id, "offline": evidence_ids["offline"],
                            "shadow": evidence_ids["shadow"], "canary": evidence_ids["canary"],
                            "actor": actor},
                    compute=_do_promote, verify=_verify_promote,
                )
                status = "promoted"
            except release.ReleaseError as exc:
                block_reason = f"promotion rejected: {exc}"
                ctx.block("PROMOTE", _cache_key("PROMOTE", _STAGE_VERSION,
                          {"artifact_id": artifact_id}),
                          metrics={"promoted": False}, outputs={"artifact_id": artifact_id},
                          payload={"artifact_id": artifact_id, "error": str(exc)},
                          reason=block_reason, error_code="promote_rejected")
        else:
            status = "evaluated"

    except PipelineBlocked as blocked:
        if status not in ("circuit_tripped",):
            status = "blocked"
        block_reason = block_reason or blocked.reason
    except RegistryError as exc:
        ctx.integrity_failures += 1
        status = "blocked"
        block_reason = f"registry integrity failure: {exc}"
        ctx.emit("run", "run", "blocked", error_code="registry_integrity")

    current_after = release.resolve_channel(registry, "CURRENT")
    _write_observability(ctx, artifact_id=artifact_id, evidence=evidence_records,
                         offline_metrics=offline_metrics, canary_metrics=canary_metrics)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "backend": backend,
        "environment": config.environment,
        "actor": actor,
        "status": status,
        "circuit_tripped": circuit_tripped,
        "circuit_open": _circuit_open(registry),
        "artifact_id": artifact_id,
        "current_before": current_before,
        "current_artifact_id": current_after,
        "promoted": status == "promoted",
        "evidence": evidence_ids,
        "offline_metrics": offline_metrics,
        "canary_metrics": canary_metrics,
        "block_reason": block_reason,
        "state_root": str(registry.root),
        "suite_sha256": suite_sha256,
        "stages": {name: _stage_dict(result) for name, result in ctx.results.items()},
    }
    _write_run_summary(registry, run_id, summary)
    ctx.emit("run", "run", status, artifact_id=artifact_id)
    return summary


def _write_run_summary(registry: FileRegistry, run_id: str, summary: Mapping[str, Any]) -> None:
    run_dir = registry.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (registry.runs_dir / "LATEST").write_text(run_id + "\n", encoding="utf-8")


def _latest_run_id(registry: FileRegistry) -> str | None:
    pointer = registry.runs_dir / "LATEST"
    if pointer.is_file():
        text = pointer.read_text("utf-8").strip()
        if text:
            return text
    runs = [p for p in registry.runs_dir.iterdir() if p.is_dir()] if registry.runs_dir.is_dir() else []
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime).name


def verify_run(
    config: PipelineConfig,
    *,
    run_id: str | None = None,
) -> Mapping[str, Any]:
    """Re-verify a run's release state and integrity without re-executing the model.

    Resolves CURRENT (honoring the circuit breaker), re-verifies the served candidate and
    its offline evidence, and surfaces the offline metrics + backing backend so the CLI
    can assert ``--expect-current`` / ``--expect-backend``. Never imports torch.
    """
    registry = FileRegistry(config.state_root, environment=config.environment)  # type: ignore[arg-type]
    resolved_run_id = run_id or _latest_run_id(registry)

    current_artifact_id = release.resolve_channel(registry, "CURRENT")
    current_verified = False
    current_backend: str | None = None
    offline_metrics: dict[str, Any] | None = None
    integrity_ok = True
    problems: list[str] = []

    if current_artifact_id is not None:
        try:
            registry.verify_candidate(current_artifact_id)
            current_verified = True
        except RegistryError as exc:
            integrity_ok = False
            problems.append(f"candidate verify failed: {exc}")
        record = _channel_record(registry, "CURRENT") or {}
        for eid in record.get("evidence_ids", []):
            try:
                evidence = registry.load_evidence(eid)
            except RegistryError as exc:
                integrity_ok = False
                problems.append(f"evidence {eid} failed: {exc}")
                continue
            if evidence.get("kind") == "offline_eval":
                current_backend = evidence.get("backend")
                offline_metrics = dict(evidence.get("metrics", {}))

    stages_ok = True
    if resolved_run_id is not None:
        run_json = registry.runs_dir / resolved_run_id / "run.json"
        if run_json.is_file():
            try:
                summary = json.loads(run_json.read_text("utf-8"))
                stages_ok = summary.get("status") in ("promoted", "evaluated", "cached")
            except (OSError, json.JSONDecodeError):
                stages_ok = False

    return {
        "run_id": resolved_run_id,
        "state_root": str(registry.root),
        "current_artifact_id": current_artifact_id,
        "current_verified": current_verified,
        "current_backend": current_backend,
        "offline_metrics": offline_metrics,
        "circuit_open": _circuit_open(registry),
        "integrity_ok": integrity_ok,
        "stages_ok": stages_ok,
        "ok": integrity_ok and (current_artifact_id is None or current_verified),
        "problems": problems,
    }
