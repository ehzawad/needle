"""Release controller — staged channels, gated promotion, rollback, circuit breaker.

Real-world analog: Argo Rollouts / Spinnaker managed release channels plus an Envoy /
Hystrix-style circuit breaker — but file-backed and single-node, with no traffic
routing. Three channel pointers live under ``channels/`` as atomically-replaced JSON:

  * ``SHADOW`` and ``CANARY`` are staging pointers set by :func:`set_channel`;
  * ``CURRENT`` is the served release, changed ONLY through :func:`promote`,
    :func:`rollback`, or :func:`trip_circuit`.

The safety-bearing rules the council fixed live here:

  * Every channel change is serialized by an ``fcntl.flock`` on ``.release.lock`` and
    written durably: temp file -> ``fsync`` -> ``os.replace`` -> ``fsync`` directory, so
    a reader always sees a whole pointer and a crash never leaves a torn one.
  * :func:`promote` requires offline, shadow, AND canary evidence ids for the artifact,
    and REJECTS mock-backed evidence in a demo registry — CI mock evidence can update
    ``CURRENT`` only in a CI registry.
  * :func:`trip_circuit` writes ``circuit.json`` FIRST (so serving fails closed) and
    only then rolls ``CURRENT`` back to the last-known-good artifact; if there is no
    prior good artifact, serving is left disabled rather than pinned to the bad one.
  * Every change is appended to ``releases/history.jsonl`` for an audit trail.

Pure standard library. Never imports torch/transformers/scope_bot.
"""
from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from pipeline.contracts import Channel
from pipeline.registry import (
    FileRegistry,
    RegistryError,
    atomic_write_bytes,
    dumps_bytes,
    utc_now_iso,
    with_sha256_prefix,
    write_all,
)
from pipeline.source_fingerprint import canonical_json, sha256_bytes

_CIRCUIT_ACTOR = "circuit-breaker"


class ReleaseError(RuntimeError):
    """Raised when a release/promotion/rollback precondition is violated (fail-closed)."""


def _norm(value: object | None) -> str | None:
    return with_sha256_prefix(value)


@contextmanager
def _release_lock(registry: FileRegistry) -> Iterator[None]:
    """Serialize all channel changes with an exclusive advisory lock on ``.release.lock``.

    Every mutating entry point holds this for its whole read-modify-write, so two
    concurrent promotions/rollbacks can never interleave a channel pointer.
    """
    fd = os.open(str(registry.lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _read_channel(registry: FileRegistry, channel: str) -> dict[str, Any] | None:
    path = registry.channel_path(channel)
    if not path.is_file():
        return None
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict):
        raise ReleaseError(f"release: channel {channel} is malformed")
    return data


def _write_channel(registry: FileRegistry, channel: str, record: Mapping[str, Any]) -> None:
    atomic_write_bytes(registry.channel_path(channel), dumps_bytes(dict(record)))


def _read_circuit(registry: FileRegistry) -> dict[str, Any]:
    """Return the circuit state, defaulting to closed. An unreadable/malformed file is
    treated as OPEN so serving fails closed rather than trusting a damaged breaker."""
    path = registry.circuit_path
    if not path.is_file():
        return {"open": False}
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"open": True, "reason": "unreadable circuit.json"}
    if not isinstance(data, dict):
        return {"open": True, "reason": "malformed circuit.json"}
    return data


def _circuit_blocks(circuit: Mapping[str, Any], artifact_id: str | None) -> bool:
    """Artifact-scoped circuit decision (fail-closed).

    The circuit breaker is a per-artifact quarantine, not a global kill switch: a CLOSED
    circuit never blocks, and an OPEN circuit blocks ONLY the artifact it names, so a
    server for a different (e.g. rolled-back) artifact keeps serving. But an open circuit
    whose ``bad_artifact_id`` is missing/invalid — including the ``open`` sentinel
    :func:`_read_circuit` returns for an unreadable/malformed ``circuit.json`` — blocks
    fail-closed, because we cannot prove the artifact in hand is the safe one.
    """
    if not circuit.get("open"):
        return False
    bad = circuit.get("bad_artifact_id")
    if not isinstance(bad, str) or not bad:
        return True
    return _norm(bad) == _norm(artifact_id)


def _append_history(registry: FileRegistry, record: Mapping[str, Any]) -> None:
    """Durably append one JSON line to ``releases/history.jsonl`` (called under lock)."""
    path = registry.history_path
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        write_all(fd, line)  # loop over short os.write returns so no audit tail is lost
        os.fsync(fd)
    finally:
        os.close(fd)


def _release_id(
    channel: str,
    artifact_id: str | None,
    evidence_ids: Sequence[str],
    actor: str,
    ts: str,
) -> str:
    """Content id of a release event — unique per channel/artifact/evidence/actor/time."""
    payload = {
        "channel": channel,
        "artifact_id": _norm(artifact_id),
        "evidence_ids": [str(e) for e in evidence_ids],
        "actor": actor,
        "ts": ts,
    }
    return f"sha256:{sha256_bytes(canonical_json(payload))}"


def _require_actor(actor: str, where: str) -> str:
    if not isinstance(actor, str) or not actor.strip():
        raise ReleaseError(f"{where}: a non-empty actor is required")
    return actor


def set_channel(
    registry: FileRegistry,
    channel: Literal["SHADOW", "CANARY"],
    artifact_id: str,
    *,
    evidence_ids: Sequence[str],
    actor: str,
) -> Mapping[str, Any]:
    """Point a staging channel (``SHADOW``/``CANARY``) at a verified artifact.

    Refuses to touch ``CURRENT`` (that is promotion's job), verifies the candidate is
    intact, and checks each cited evidence id exists, is intact, and references this
    artifact. Returns the written channel record.
    """
    if channel not in ("SHADOW", "CANARY"):
        raise ReleaseError(
            "set_channel manages only SHADOW/CANARY; CURRENT changes via "
            "promote/rollback/trip_circuit"
        )
    _require_actor(actor, "set_channel")
    with _release_lock(registry):
        try:
            registry.verify_candidate(artifact_id)
        except RegistryError as exc:
            raise ReleaseError(f"set_channel: candidate {artifact_id} failed verification: {exc}")
        for eid in evidence_ids:
            evidence = registry.load_evidence(eid)
            if _norm(evidence.get("artifact_id")) != _norm(artifact_id):
                raise ReleaseError(
                    f"set_channel: evidence {eid} references "
                    f"{evidence.get('artifact_id')}, not {artifact_id}"
                )
        previous = _read_channel(registry, channel)
        previous_aid = previous.get("artifact_id") if previous else None
        now = utc_now_iso()
        record = {
            "artifact_id": _norm(artifact_id),
            "previous_artifact_id": _norm(previous_aid),
            "release_id": _release_id(channel, artifact_id, evidence_ids, actor, now),
            "evidence_ids": [str(e) for e in evidence_ids],
            "updated_at": now,
            "actor": actor,
        }
        _write_channel(registry, channel, record)
        _append_history(
            registry, {"ts": now, "action": "set_channel", "channel": channel, **record}
        )
        return record


def promote(
    registry: FileRegistry,
    artifact_id: str,
    *,
    offline_evidence_id: str,
    shadow_evidence_id: str,
    canary_evidence_id: str,
    actor: str,
) -> Mapping[str, Any]:
    """Atomically make ``artifact_id`` the served ``CURRENT`` release.

    Preconditions, all enforced under the release lock: the candidate verifies; the
    offline/shadow/canary evidence ids each exist, are intact, are of the right kind,
    reference this artifact, and are not explicitly failed; the circuit is not open
    against this artifact; and — the demo guarantee — none of the evidence is
    mock-backed when the registry ``environment`` is ``demo``. Returns the new
    ``CURRENT`` record.
    """
    _require_actor(actor, "promote")
    with _release_lock(registry):
        try:
            registry.verify_candidate(artifact_id)
        except RegistryError as exc:
            raise ReleaseError(f"promote: candidate {artifact_id} failed verification: {exc}")

        required = {
            "offline_eval": offline_evidence_id,
            "shadow": shadow_evidence_id,
            "canary": canary_evidence_id,
        }
        for kind, eid in required.items():
            if not eid:
                raise ReleaseError(f"promote: a {kind} evidence id is required")
            evidence = registry.load_evidence(eid)
            if evidence.get("kind") != kind:
                raise ReleaseError(
                    f"promote: evidence {eid} is kind {evidence.get('kind')!r}, expected {kind!r}"
                )
            if _norm(evidence.get("artifact_id")) != _norm(artifact_id):
                raise ReleaseError(
                    f"promote: {kind} evidence references "
                    f"{evidence.get('artifact_id')}, not {artifact_id}"
                )
            if evidence.get("passed") is not True:
                raise ReleaseError(
                    f"promote: {kind} evidence did not pass (passed="
                    f"{evidence.get('passed')!r}); refusing promotion"
                )
            if registry.environment == "demo" and evidence.get("backend") != "real":
                raise ReleaseError(
                    f"promote: demo promotion requires real-backed {kind} evidence, got "
                    f"backend={evidence.get('backend')!r}"
                )

        circuit = _read_circuit(registry)
        if _circuit_blocks(circuit, artifact_id):
            raise ReleaseError(f"promote: circuit is open against {artifact_id}")

        previous = _read_channel(registry, "CURRENT")
        previous_aid = previous.get("artifact_id") if previous else None
        now = utc_now_iso()
        evidence_ids = [offline_evidence_id, shadow_evidence_id, canary_evidence_id]
        record = {
            "artifact_id": _norm(artifact_id),
            "previous_artifact_id": _norm(previous_aid),
            "release_id": _release_id("CURRENT", artifact_id, evidence_ids, actor, now),
            "evidence_ids": evidence_ids,
            "updated_at": now,
            "actor": actor,
        }
        _write_channel(registry, "CURRENT", record)
        _append_history(
            registry, {"ts": now, "action": "promote", "channel": "CURRENT", **record}
        )
        return record


def rollback(
    registry: FileRegistry,
    *,
    target_artifact_id: str,
    reason: str,
    actor: str,
) -> Mapping[str, Any]:
    """Atomically roll ``CURRENT`` back to a named, intact prior artifact.

    A clean operational rollback (distinct from a circuit trip): it verifies the target
    candidate, then replaces ``CURRENT`` the same durable way promotion does, recording
    the reason. Returns the new ``CURRENT`` record.
    """
    _require_actor(actor, "rollback")
    if not target_artifact_id:
        raise ReleaseError("rollback: a target artifact id is required")
    with _release_lock(registry):
        try:
            registry.verify_candidate(target_artifact_id)
        except RegistryError as exc:
            raise ReleaseError(
                f"rollback: target {target_artifact_id} failed verification: {exc}"
            )
        previous = _read_channel(registry, "CURRENT")
        previous_aid = previous.get("artifact_id") if previous else None
        now = utc_now_iso()
        record = {
            "artifact_id": _norm(target_artifact_id),
            "previous_artifact_id": _norm(previous_aid),
            "release_id": _release_id("CURRENT", target_artifact_id, [], actor, now),
            "evidence_ids": [],
            "updated_at": now,
            "actor": actor,
            "reason": reason,
        }
        _write_channel(registry, "CURRENT", record)
        _append_history(
            registry, {"ts": now, "action": "rollback", "channel": "CURRENT", **record}
        )
        return record


def trip_circuit(
    registry: FileRegistry,
    *,
    bad_artifact_id: str,
    evidence_id: str,
    reason: str,
) -> Mapping[str, Any]:
    """Open the circuit against a bad artifact, then roll ``CURRENT`` back.

    Ordering is the safety property: ``circuit.json`` is written and fsynced FIRST so a
    concurrent serving read fails closed, and only then is ``CURRENT`` moved to the
    last-known-good artifact (the current ``CURRENT``'s ``previous_artifact_id``, if it
    verifies). With no prior good artifact, ``CURRENT`` is set to ``null`` — serving
    stays disabled rather than pinned to the failed candidate. Returns the circuit
    record.
    """
    if not bad_artifact_id:
        raise ReleaseError("trip_circuit: a bad artifact id is required")
    with _release_lock(registry):
        now = utc_now_iso()
        current = _read_channel(registry, "CURRENT")
        prior_good = current.get("previous_artifact_id") if current else None

        rolled_back_to: str | None = None
        if prior_good:
            try:
                registry.verify_candidate(prior_good)
                rolled_back_to = _norm(prior_good)
            except RegistryError:
                rolled_back_to = None  # prior artifact is gone/corrupt -> disable serving

        # (1) Open the breaker FIRST and make it durable, so serving fails closed.
        circuit = {
            "open": True,
            "bad_artifact_id": _norm(bad_artifact_id),
            "evidence_id": evidence_id,
            "reason": reason,
            "tripped_at": now,
            "rolled_back_to": rolled_back_to,
        }
        atomic_write_bytes(registry.circuit_path, dumps_bytes(circuit))

        # (2) Only now move CURRENT off the bad artifact.
        record = {
            "artifact_id": rolled_back_to,
            "previous_artifact_id": _norm(bad_artifact_id),
            "release_id": _release_id("CURRENT", rolled_back_to, [evidence_id], _CIRCUIT_ACTOR, now),
            "evidence_ids": [evidence_id],
            "updated_at": now,
            "actor": _CIRCUIT_ACTOR,
            "reason": reason,
        }
        _write_channel(registry, "CURRENT", record)
        _append_history(
            registry,
            {"ts": now, "action": "trip_circuit", "channel": "CURRENT", **circuit},
        )
        return circuit


def resolve_channel(registry: FileRegistry, channel: Channel) -> str | None:
    """Resolve a channel to its artifact id, or ``None`` if unset/disabled.

    For ``CURRENT`` this also honors the circuit breaker, artifact-scoped: the breaker
    disables serving only when it is open against the very artifact ``CURRENT`` names (or
    when its state is open-but-unattributable/unreadable, which fails closed). A breaker
    open against some OTHER artifact — e.g. after a rollback moved ``CURRENT`` back to a
    good release — does not disable this one. A ``null`` artifact pointer (serving
    disabled) also resolves to ``None``.
    """
    record = _read_channel(registry, channel)
    if record is None:
        return None
    artifact_id = record.get("artifact_id")
    if artifact_id is None:
        return None
    if channel == "CURRENT":
        circuit = _read_circuit(registry)
        if _circuit_blocks(circuit, artifact_id):
            return None
    return artifact_id
