"""Unified control-plane CLI — the single operator entry point.

Real-world analog: the ``argo``/``kubectl rollouts``/``mlflow`` command line — one binary
that drives the whole release lifecycle. It owns the guarded subprocess environment (no
CUDA variables ever appear on the operator's command line; the device guard sets them for
the child), and it never imports torch/transformers/scope_bot: every model-touching
action goes through :func:`pipeline.dag.run_pipeline`, which routes real work to the
guarded GPU worker.

Subcommands::

    run       INGEST..PROMOTE (--config --state --backend --promote --actor)
    verify    assert release state (--config --state --expect-current --expect-backend)
    status    show CURRENT/SHADOW/CANARY + circuit + latest run
    review    list | set (--label-id --decision --reviewer)
    rollback  revert CURRENT to a prior artifact (--target --reason --actor)
    dashboard re-render the static HTML dashboard
    serve     start the HTTP serving microservice (--http HOST:PORT)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pipeline import dag, release, serving
from pipeline.adapters import make_mock_gate, policy_wrapped_gate
from pipeline.build_plane import record_review
from pipeline.config import load_config
from pipeline.observability import render_dashboard
from pipeline.registry import FileRegistry, RegistryError
from pipeline.source_fingerprint import load_eval50_cases

__all__ = ["main"]


def _print(obj: Any) -> None:
    """Emit a JSON object to stdout (machine-readable control-plane output)."""
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _load(args: argparse.Namespace):
    state = Path(args.state) if getattr(args, "state", None) else None
    return load_config(Path(args.config), state_override=state)


def _registry(config) -> FileRegistry:
    return FileRegistry(config.state_root, environment=config.environment)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Subcommand handlers.
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    config = _load(args)
    summary = dag.run_pipeline(
        config, backend=args.backend, promote_current=bool(args.promote), actor=args.actor)
    _print(summary)
    ok = summary.get("status") in ("promoted", "evaluated")
    if not ok and summary.get("status") == "circuit_tripped":
        print("circuit tripped: CURRENT rolled back on a mined safety regression",
              file=sys.stderr)
    return 0 if ok else 1


def _cmd_verify(args: argparse.Namespace) -> int:
    config = _load(args)
    report = dag.verify_run(config, run_id=getattr(args, "run_id", None))
    _print(report)
    ok = bool(report.get("ok"))
    if args.expect_current and report.get("current_artifact_id") is None:
        print("verify: expected a CURRENT release, found none", file=sys.stderr)
        ok = False
    if args.expect_backend and report.get("current_backend") != args.expect_backend:
        print(
            f"verify: expected CURRENT backed by {args.expect_backend!r} evidence, "
            f"found {report.get('current_backend')!r}", file=sys.stderr)
        ok = False
    return 0 if ok else 1


def _cmd_status(args: argparse.Namespace) -> int:
    config = _load(args)
    registry = _registry(config)
    latest = dag._latest_run_id(registry)
    run_status = None
    if latest is not None:
        run_json = registry.runs_dir / latest / "run.json"
        if run_json.is_file():
            try:
                run_status = json.loads(run_json.read_text("utf-8")).get("status")
            except (OSError, json.JSONDecodeError):
                run_status = None
    status = {
        "state_root": str(registry.root),
        "environment": config.environment,
        "channels": {
            "CURRENT": release.resolve_channel(registry, "CURRENT"),
            "SHADOW": release.resolve_channel(registry, "SHADOW"),
            "CANARY": release.resolve_channel(registry, "CANARY"),
        },
        "circuit_open": dag._circuit_open(registry),
        "latest_run": latest,
        "latest_run_status": run_status,
    }
    _print(status)
    return 0


def _cmd_review_list(args: argparse.Namespace) -> int:
    config = _load(args)
    registry = _registry(config)
    queued: dict[str, dict[str, Any]] = {}
    decided: set[str] = set()
    path = registry.reviews_path
    if path.is_file():
        for line in path.read_text("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind, label_id = record.get("kind"), record.get("label_id")
            if kind == "queued" and label_id:
                queued[label_id] = record
            elif kind == "decision" and label_id:
                decided.add(label_id)
    pending = [item for lid, item in queued.items() if lid not in decided]
    _print({"pending_count": len(pending), "pending": pending})
    return 0


def _cmd_review_set(args: argparse.Namespace) -> int:
    config = _load(args)
    registry = _registry(config)
    try:
        record = record_review(
            registry.reviews_path, label_id=args.label_id,
            decision=args.decision, reviewer=args.reviewer)
    except ValueError as exc:
        print(f"review set: {exc}", file=sys.stderr)
        return 1
    _print(record)
    return 0


def _cmd_rollback(args: argparse.Namespace) -> int:
    config = _load(args)
    registry = _registry(config)
    try:
        record = release.rollback(
            registry, target_artifact_id=args.target, reason=args.reason, actor=args.actor)
    except (release.ReleaseError, RegistryError) as exc:
        print(f"rollback: {exc}", file=sys.stderr)
        return 1
    _print(record)
    return 0


def _load_current_evidence(registry: FileRegistry) -> list[dict[str, Any]]:
    record = dag._channel_record(registry, "CURRENT") or {}
    evidence: list[dict[str, Any]] = []
    for eid in record.get("evidence_ids", []):
        try:
            evidence.append(dict(registry.load_evidence(eid)))
        except RegistryError:
            continue
    return evidence


def _drift_alerts(config: Any) -> list:
    """Compute conservative drift alerts from the served conversation log (older half vs
    newer half). Returns [] until there are >= drift.min_samples turns in each window —
    which is the honest behavior at demo volume, but it IS a live caller of detect_drift."""
    from feedback_log import read_sessions

    from pipeline.observability import detect_drift
    turns = [t for session in read_sessions().values() for t in session if t.get("disposition")]
    turns.sort(key=lambda t: t.get("ts", ""))
    mid = len(turns) // 2
    return list(detect_drift(
        turns[:mid], turns[mid:],
        min_samples=config.drift.min_samples,
        max_rate_delta=config.drift.max_disposition_rate_delta))


def _cmd_dashboard(args: argparse.Namespace) -> int:
    config = _load(args)
    registry = _registry(config)
    current = dag._channel_record(registry, "CURRENT") or {"channel": "CURRENT", "artifact_id": None}
    release_view = dict(current)
    release_view.setdefault("channel", "CURRENT")
    output = registry.observability_dir / "dashboard.html"
    render_dashboard(output, release=release_view,
                     evidence=_load_current_evidence(registry), alerts=_drift_alerts(config))
    _print({"dashboard": str(output)})
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from pipeline.serving_http import build_http_server, parse_host_port, serve_forever

    config = _load(args)
    registry = _registry(config)
    artifact_id = release.resolve_channel(registry, "CURRENT")
    if artifact_id is None:
        print("serve: no CURRENT release to serve (or circuit is open)", file=sys.stderr)
        return 1
    try:
        registry.verify_candidate(artifact_id)
    except RegistryError as exc:
        print(f"serve: CURRENT candidate failed verification: {exc}", file=sys.stderr)
        return 1

    cards_path = registry.artifact_dir(artifact_id) / "files" / "cards.json"
    cards = json.loads(cards_path.read_text("utf-8"))
    backend = getattr(args, "backend", "mock")

    if backend == "real":
        # A leaf model-serving process (like gpu_worker): guard the A5000 IN THIS process
        # BEFORE any torch import, then load the promoted candidate's real bot and serve it.
        import os
        from pipeline.device_guard import (
            assert_model_devices, child_preflight, pinned_environment,
        )
        pinned = pinned_environment()
        os.environ.update({k: pinned[k] for k in
                           ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")})
        device = child_preflight()  # fail closed unless exactly the A5000 is visible
        from feedback_log import TurnLogger
        from pipeline.adapters import load_real_bot, real_gate, real_respond
        bot = load_real_bot(registry.artifact_dir(artifact_id))
        assert_model_devices(getattr(bot, "m", bot))
        gate = real_gate(bot, cards)
        respond = real_respond(bot)
        # Real serving closes the feedback loop: turns land in logs/conversations.jsonl,
        # which mine_signals/adapter.learn later turn into the next candidate's exemplars.
        turn_logger: object = TurnLogger()
        backend_label = f"real, {device.name}"
    else:
        cases = load_eval50_cases(config.eval_source_path)
        gate = policy_wrapped_gate(make_mock_gate(cases), cards)
        respond = dag._mock_respond(gate)
        turn_logger = dag._MemoryTurnLogger()
        backend_label = "mock"

    event_path = registry.observability_dir / "events.jsonl"
    serve = serving.make_server(
        artifact_id=artifact_id, cards=cards, gate=gate, respond=respond,
        turn_logger=turn_logger, event_path=event_path)

    def _healthz() -> dict[str, Any]:
        return {"ok": True, "artifact_id": artifact_id, "channel": "CURRENT",
                "circuit_open": dag._circuit_open(registry), "backend": backend}

    host, port = parse_host_port(args.http)
    server = build_http_server(host, port, serve=serve, gate=gate,
                               artifact_id=artifact_id, healthz=_healthz)
    print(f"serving CURRENT {artifact_id} on http://{host}:{port} "
          f"({backend_label} ServeFn; Ctrl-C to stop)", file=sys.stderr)
    serve_forever(server)
    return 0


# ---------------------------------------------------------------------------
# Argument parser.
# ---------------------------------------------------------------------------


def _add_state_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="path to config.ci.json / config.demo.json")
    parser.add_argument("--state", default=None, help="registry/state root (default: <root>/.pipeline-<env>)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline", description="Scope-bot release control plane.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the stage DAG")
    _add_state_args(p_run)
    p_run.add_argument("--backend", choices=("mock", "real"), default="mock")
    p_run.add_argument("--promote", action="store_true", help="promote CURRENT on a clean run")
    p_run.add_argument("--actor", default="operator")
    p_run.set_defaults(func=_cmd_run)

    p_verify = sub.add_parser("verify", help="assert release state and integrity")
    _add_state_args(p_verify)
    p_verify.add_argument("--expect-current", action="store_true")
    p_verify.add_argument("--expect-backend", choices=("mock", "real"), default=None)
    p_verify.add_argument("--run-id", default=None)
    p_verify.set_defaults(func=_cmd_verify)

    p_status = sub.add_parser("status", help="show channels + circuit + latest run")
    _add_state_args(p_status)
    p_status.set_defaults(func=_cmd_status)

    p_review = sub.add_parser("review", help="human review queue")
    review_sub = p_review.add_subparsers(dest="review_command", required=True)
    p_review_list = review_sub.add_parser("list", help="list pending review items")
    _add_state_args(p_review_list)
    p_review_list.set_defaults(func=_cmd_review_list)
    p_review_set = review_sub.add_parser("set", help="record a review decision")
    _add_state_args(p_review_set)
    p_review_set.add_argument("--label-id", required=True)
    p_review_set.add_argument("--decision", required=True, choices=("approve", "reject"))
    p_review_set.add_argument("--reviewer", required=True)
    p_review_set.set_defaults(func=_cmd_review_set)

    p_rollback = sub.add_parser("rollback", help="revert CURRENT to a prior artifact")
    _add_state_args(p_rollback)
    p_rollback.add_argument("--target", required=True, help="artifact id to roll back to")
    p_rollback.add_argument("--reason", required=True)
    p_rollback.add_argument("--actor", default="operator")
    p_rollback.set_defaults(func=_cmd_rollback)

    p_dashboard = sub.add_parser("dashboard", help="re-render the static dashboard")
    _add_state_args(p_dashboard)
    p_dashboard.set_defaults(func=_cmd_dashboard)

    p_serve = sub.add_parser("serve", help="start the HTTP serving microservice")
    _add_state_args(p_serve)
    p_serve.add_argument("--http", required=True, help="HOST:PORT to bind (e.g. 127.0.0.1:8080)")
    p_serve.add_argument("--backend", choices=("mock", "real"), default="mock",
                         help="mock (CPU, default/CI) or real (loads Qwen on the A5000, closes the feedback loop)")
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"pipeline: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"pipeline: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
