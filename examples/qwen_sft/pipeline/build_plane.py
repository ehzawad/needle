"""Build plane — mine weak labels, gate them behind human review, register a candidate.

Real-world analog: a training pipeline's dataset-build stage feeding a model registry,
with a Label Studio / Scale review queue holding human authority over any label that
would EXPAND what the bot answers. The mining and the exemplar build are delegated to
the existing ``mine_signals`` / ``adapter.learn`` tools (no filtering is reimplemented);
this module only orchestrates them, records review decisions in an append-only log, and
assembles the immutable candidate identity.

Safety asymmetry (the council invariant): answer-expanding ``OVER_CLARIFY`` /
``OVER_ABSTAIN`` labels enter a candidate ONLY with an explicit, non-CI human approval.
Pending review items are simply excluded (the existing builder's ``approved()`` gate
drops any needs-review label without an approval) and therefore never block an
otherwise-safe build.
"""
from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pipeline.adapters import build_existing, mine_existing
from pipeline.contracts import BlobRef
from pipeline.source_fingerprint import canonical_json, sha256_bytes

if TYPE_CHECKING:  # annotations only — no runtime coupling to the registry lane
    from pipeline.registry import FileRegistry

# Mirrors mine_signals.ANSWER_EXPANDING: the labels that would make the bot answer MORE,
# and so require explicit human approval before they can shape a candidate.
_ANSWER_EXPANDING = frozenset({"OVER_CLARIFY", "OVER_ABSTAIN"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _blobref_dict(ref: BlobRef) -> dict[str, Any]:
    return {"sha256": ref.sha256, "bytes": ref.bytes, "media_type": ref.media_type}


def _as_blobref(value: Any) -> BlobRef:
    """Accept a :class:`BlobRef` or its JSON dict form (a snapshot may round-trip through
    JSON between stages) and return a :class:`BlobRef`."""
    if isinstance(value, BlobRef):
        return value
    if isinstance(value, Mapping):
        return BlobRef(
            sha256=value["sha256"],
            bytes=int(value["bytes"]),
            media_type=value["media_type"],
        )
    raise TypeError(f"expected BlobRef or mapping, got {type(value).__name__}")


def _label_id(record: Mapping[str, Any]) -> str:
    """Content-addressed, stable id for a mined label — the review queue key.

    Computed identically by :func:`write_review_queue` and :func:`build_candidate` from
    the SAME labels blob, so a decision recorded against a queued label id resolves
    back to exactly that label at build time.
    """
    identity = {
        "session_id": record.get("session_id"),
        "turn": record.get("turn"),
        "query": record.get("query"),
        "signal": record.get("signal"),
        "mined_disposition": record.get("mined_disposition"),
        "mined_card": record.get("mined_card"),
    }
    return "sha256:" + sha256_bytes(canonical_json(identity))


def _is_human_reviewer(reviewer: str) -> bool:
    """A non-CI human authority: a non-empty reviewer that is not the CI actor.

    Answer-expanding labels honor an approval only from such a reviewer, so an automated
    CI run can never approve one into a candidate.
    """
    return bool(reviewer) and reviewer.strip().lower() != "ci"


def _read_labels(labels_blob: BlobRef, registry: FileRegistry) -> list[dict[str, Any]]:
    data = registry.read_blob(_as_blobref(labels_blob))
    labels: list[dict[str, Any]] = []
    for line in data.decode("utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            labels.append(json.loads(stripped))
    return labels


def _read_queue(queue_path: Path) -> tuple[set[str], set[str]]:
    """Fold the append-only review log into (queued_label_ids, decided_label_ids)."""
    queued: set[str] = set()
    decided: set[str] = set()
    path = Path(queue_path)
    if not path.exists():
        return queued, decided
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        kind = record.get("kind")
        label_id = record.get("label_id")
        if kind == "queued" and label_id:
            queued.add(label_id)
        elif kind == "decision" and label_id:
            decided.add(label_id)
    return queued, decided


def _decisions(reviews_path: Path) -> dict[str, tuple[str, str]]:
    """Latest (decision, reviewer) per label id from the append-only review log."""
    decisions: dict[str, tuple[str, str]] = {}
    path = Path(reviews_path)
    if not path.exists():
        return decisions
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "decision" and record.get("label_id"):
            decisions[record["label_id"]] = (
                record.get("decision", ""),
                record.get("reviewer", ""),
            )
    return decisions


def mine_labels(snapshot: Mapping[str, Any], registry: FileRegistry) -> BlobRef:
    """Mine weak (query -> correct-disposition) labels from the snapshot and store them.

    Mines from the IMMUTABLE snapshot content (the cards + logs blobs), not the live
    files, so labels are bound to exactly what ingest froze. Delegates to the existing
    miner via :func:`~pipeline.adapters.mine_existing`; returns the labels JSONL blob.
    """
    cards_bytes = registry.read_blob(_as_blobref(snapshot["cards"]))
    logs_bytes = registry.read_blob(_as_blobref(snapshot["logs"]))
    with tempfile.TemporaryDirectory(prefix="mine-") as tmp:
        tmp_dir = Path(tmp)
        cards_path = tmp_dir / "cards.json"
        cards_path.write_bytes(cards_bytes)
        logs_path = tmp_dir / "conversations.jsonl"
        logs_path.write_bytes(logs_bytes)
        labels = mine_existing(logs_path, cards_path)
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in labels)
    return registry.put_blob(payload.encode("utf-8"), media_type="application/jsonl")


def write_review_queue(labels_blob: BlobRef, registry: FileRegistry, queue_path: Path) -> int:
    """Enqueue every needs-review label into the append-only review log (idempotent).

    Only labels the miner flagged ``needs_review`` are queued (answer-expanding, or low
    confidence). A label already queued is not re-appended. Returns the number of items
    still PENDING (queued and not yet decided) — what a reviewer would see with
    ``pipeline review list``.
    """
    labels = _read_labels(labels_blob, registry)
    path = Path(queue_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    queued, decided = _read_queue(path)
    with path.open("a", encoding="utf-8") as handle:
        for record in labels:
            if not record.get("needs_review"):
                continue
            label_id = _label_id(record)
            if label_id in queued:
                continue
            item = {
                "kind": "queued",
                "label_id": label_id,
                "query": record.get("query"),
                "signal": record.get("signal"),
                "mined_disposition": record.get("mined_disposition"),
                "confidence": record.get("confidence"),
                "answer_expanding": record.get("signal") in _ANSWER_EXPANDING,
                "needs_review": True,
                "ts": _utc_now(),
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            queued.add(label_id)
    return len(queued - decided)


def record_review(
    queue_path: Path,
    *,
    label_id: str,
    decision: Literal["approve", "reject"],
    reviewer: str,
) -> Mapping[str, Any]:
    """Append a human approve/reject decision for a queued label (append-only log).

    Raises ``ValueError`` on an invalid decision, an empty reviewer, or a label id that
    was never queued. Returns the recorded decision.
    """
    if decision not in ("approve", "reject"):
        raise ValueError(f"record_review: decision must be 'approve' or 'reject', got {decision!r}")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("record_review: reviewer must be a non-empty string")
    path = Path(queue_path)
    queued, _ = _read_queue(path)
    if label_id not in queued:
        raise ValueError(
            f"record_review: unknown label_id {label_id!r} (not in review queue {path})"
        )
    record = {
        "kind": "decision",
        "label_id": label_id,
        "decision": decision,
        "reviewer": reviewer,
        "ts": _utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _reviews_digest_bytes(reviews_path: Path) -> bytes:
    """Canonical bytes of the resolved decisions — a stable ``reviews_sha256`` even when
    no reviews exist (an empty decision map hashes deterministically)."""
    decisions = _decisions(reviews_path)
    canonical = {
        label_id: {"decision": decision, "reviewer": reviewer}
        for label_id, (decision, reviewer) in sorted(decisions.items())
    }
    return canonical_json(canonical)


def build_candidate(
    snapshot: Mapping[str, Any],
    labels_blob: BlobRef,
    reviews_path: Path,
    registry: FileRegistry,
    *,
    parent_artifact_id: str | None,
) -> Mapping[str, Any]:
    """Assemble and register an immutable candidate from the snapshot + reviewed labels.

    The mined labels are stamped with their human review decisions (latest wins;
    answer-expanding approvals require a non-CI reviewer), then handed to the EXISTING
    exemplar builder — which applies its own confidence / approval / kb-version filtering.
    Pending answer-expanding labels stay unstamped and are dropped by that gate, so they
    never block a safe build. The resulting cards + exemplar bank, the AST-derived runtime
    identity, the evaluation contract, and the lineage are registered as one content-
    addressed artifact.
    """
    labels = _read_labels(labels_blob, registry)
    decisions = _decisions(reviews_path)

    merged: list[dict[str, Any]] = []
    for record in labels:
        stamped = dict(record)
        decided = decisions.get(_label_id(record))
        if decided is not None:
            action, reviewer = decided
            is_expanding = record.get("signal") in _ANSWER_EXPANDING
            if action == "approve":
                # Answer-expanding labels demand explicit non-CI human authority.
                if not is_expanding or _is_human_reviewer(reviewer):
                    stamped["review"] = "approve"
            elif action == "reject":
                stamped["review"] = "reject"
        merged.append(stamped)

    cards_blob = _as_blobref(snapshot["cards"])
    cards_bytes = registry.read_blob(cards_blob)

    with tempfile.TemporaryDirectory(prefix="build-") as tmp:
        tmp_dir = Path(tmp)
        cards_path = tmp_dir / "cards.json"
        cards_path.write_bytes(cards_bytes)
        merged_path = tmp_dir / "merged_labels.jsonl"
        with merged_path.open("w", encoding="utf-8") as handle:
            for record in merged:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        exemplar_bytes = build_existing(cards_path, merged_path, tmp_dir / "staging")

    exemplar_blob = registry.put_blob(exemplar_bytes, media_type="application/json")

    identity = {
        "payload": {
            "cards": _blobref_dict(cards_blob),
            "exemplar_bank": _blobref_dict(exemplar_blob),
        },
        "runtime": dict(snapshot["runtime"]),
        "evaluation_contract": dict(snapshot["evaluation_contract"]),
        "lineage": {
            "snapshot_id": snapshot["snapshot_id"],
            "labels_sha256": _as_blobref(labels_blob).sha256,
            "reviews_sha256": sha256_bytes(_reviews_digest_bytes(reviews_path)),
            "parent_artifact_id": parent_artifact_id,
        },
    }
    return registry.register_candidate(
        identity,
        {"cards.json": cards_bytes, "exemplars.json": exemplar_bytes},
    )
