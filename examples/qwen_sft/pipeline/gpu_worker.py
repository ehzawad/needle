"""The private, GPU-guarded real-backend worker — the ONLY module that loads the model.

Real-world analog: a Kubernetes GPU pod scheduled by the NVIDIA device plugin — an
isolated job that sees exactly the GPUs it was granted and does the heavy, device-bound
work the control plane must never touch. The control plane launches this via
``device_guard.launch_gpu_worker`` under an environment overwritten to expose only the
A5000; this process then RE-verifies the device before importing anything that could
grab a GPU, loads the candidate once, and computes offline + shadow + canary evidence.

Import order is mandatory and enforced by the AST-scan tests: the device guard runs
FIRST, and only after it succeeds may the model-loading adapters be imported. No torch,
transformers, ``scope_bot``, or ``eval50`` import may sit at module top level here.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pipeline import evaluation, serving
from pipeline.source_fingerprint import eval_suite_sha256, load_eval50_cases

__all__ = ["main"]


class _MemoryTurnLogger:
    """A no-side-effects TurnLogger stand-in used for canary probes.

    Matches ``feedback_log.TurnLogger.log`` structurally so the serving wrapper can log
    every probe, but keeps records in memory instead of appending to the production
    conversation log (canary probes are synthetic and must not pollute the label source).
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def log(
        self,
        session_id: str,
        turn: int,
        query: str,
        gate: dict,
        reply: str,
        shortlist: list | None = None,
        extra: dict | None = None,
    ) -> dict:
        record: dict[str, Any] = {
            "session_id": session_id,
            "turn": turn,
            "query": query,
            "disposition": gate.get("disposition"),
            "card_id": gate.get("card_id"),
            "reply": reply,
        }
        if extra:
            record.update(extra)
        self.records.append(record)
        return record


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pipeline.gpu_worker",
        description="A5000-guarded real-backend evidence worker.",
    )
    parser.add_argument("--candidate-dir", required=True, help="registered candidate directory")
    parser.add_argument("--cards", default=None, help="cards.json (default: <candidate>/files/cards.json)")
    parser.add_argument("--eval-source", required=True, help="eval50.py to parse the frozen suite from")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--suite-sha256", default=None, help="suite digest to stamp on evidence")
    parser.add_argument("--shadow-turns", default=None, help="JSONL of prior CURRENT turns to replay")
    parser.add_argument("--current-artifact-id", default=None)
    parser.add_argument("--approved-expansions", default=None, help="JSON list of approved expansion ids")
    parser.add_argument("--event-path", default=None, help="observability events.jsonl path")
    parser.add_argument("--out", default=None, help="write the JSON evidence bundle here (else stdout only)")
    return parser.parse_args(argv)


def _load_jsonl(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    file = Path(path)
    if not file.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_json_list(path: str | None) -> list[str]:
    if not path:
        return []
    file = Path(path)
    if not file.exists():
        return []
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [str(x) for x in data] if isinstance(data, (list, tuple)) else []


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    # --- MANDATORY: guard the device BEFORE importing anything that can grab a GPU. ---
    from pipeline.device_guard import child_preflight

    device = child_preflight()

    # Only after a successful guard may the model-loading adapters be imported.
    from pipeline.adapters import load_real_bot, real_gate, real_respond
    from pipeline.device_guard import assert_model_devices

    try:
        candidate_dir = Path(args.candidate_dir)
        cards_path = (
            Path(args.cards)
            if args.cards
            else candidate_dir / "files" / "cards.json"
        )
        cards = json.loads(Path(cards_path).read_text(encoding="utf-8"))

        cases = load_eval50_cases(Path(args.eval_source))
        suite_sha256 = args.suite_sha256 or eval_suite_sha256(cases)

        # Load the candidate ONCE (the single, guarded model load for this run).
        bot = load_real_bot(candidate_dir)
        assert_model_devices(getattr(bot, "m", bot))
        gate = real_gate(bot, cards)
        respond = real_respond(bot)

        # 1) Offline eval on the exact frozen suite.
        offline = evaluation.evaluate(
            gate,
            cases,
            artifact_id=args.artifact_id,
            backend="real",
            suite_sha256=suite_sha256,
            device=device,
        )

        # 2) Shadow replay of prior CURRENT turns (non-emitting).
        shadow = evaluation.evaluate_shadow(
            gate,
            _load_jsonl(args.shadow_turns),
            artifact_id=args.artifact_id,
            current_artifact_id=args.current_artifact_id,
            approved_expansions=_load_json_list(args.approved_expansions),
        )

        # 3) Canary through the full serving path (gate + respond + cross-check).
        event_path = (
            Path(args.event_path)
            if args.event_path
            else (
                (Path(args.out).resolve().parent if args.out else Path.cwd())
                / "observability"
                / "events.jsonl"
            )
        )
        serve = serving.make_server(
            artifact_id=args.artifact_id,
            cards=cards,
            gate=gate,
            respond=respond,
            turn_logger=_MemoryTurnLogger(),
            event_path=event_path,
        )
        canary = evaluation.evaluate_canary(
            serve, cases, artifact_id=args.artifact_id
        )

        bundle: dict[str, Any] = {
            "schema_version": 1,
            "artifact_id": args.artifact_id,
            "backend": "real",
            "suite_sha256": suite_sha256,
            "device": asdict(device),
            "offline_eval": evaluation.eval_report_to_dict(offline),
            "shadow": dict(shadow),
            "canary": dict(canary),
        }
    except Exception as exc:  # noqa: BLE001 — any real-work failure must exit nonzero.
        print(f"[gpu_worker] real evaluation failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    text = json.dumps(bundle, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")

    # Emit a sentinel-delimited JSON block so the control plane can parse it even amid
    # the model-loading noise transformers writes to stdout/stderr.
    sys.stdout.write("===EVIDENCE_JSON_BEGIN===\n")
    sys.stdout.write(json.dumps(bundle, sort_keys=True) + "\n")
    sys.stdout.write("===EVIDENCE_JSON_END===\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
