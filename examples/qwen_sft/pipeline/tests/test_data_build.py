"""Data-plane + build-plane tests — validation, snapshot freeze, and the build flywheel.

Real-world analog: Great Expectations validation checks plus a dataset-build integration
test against a fake model registry. They prove invalid cards/logs fail closed, a frozen
source-hash mismatch aborts ingest, and the mine -> review-queue -> build path registers
a candidate whose exemplar bank came from the EXISTING builder (no reimplemented
filtering) with answer-expanding labels held behind human review. Torch is never
imported: an in-memory ``FakeRegistry`` stands in for the registry lane.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PKG = Path(__file__).resolve().parents[1]

from pipeline.build_plane import (  # noqa: E402
    build_candidate,
    mine_labels,
    record_review,
    write_review_queue,
)
from pipeline.config import load_config  # noqa: E402
from pipeline.contracts import BlobRef  # noqa: E402
from pipeline.data_plane import (  # noqa: E402
    DataValidationError,
    SourceFreezeError,
    ingest,
    validate_cards,
    validate_conversation_jsonl,
)
from pipeline.source_fingerprint import canonical_json, sha256_bytes  # noqa: E402

_REAL_CARDS = json.loads((_ROOT / "seed16" / "cards.json").read_text(encoding="utf-8"))


class FakeRegistry:
    """Minimal in-memory content-addressed store matching the registry lane's contract
    surface used by ingest/mine/build (put_blob / read_blob / register_candidate)."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.candidates: list[dict[str, Any]] = []

    def put_blob(self, data: bytes, *, media_type: str) -> BlobRef:
        digest = hashlib.sha256(data).hexdigest()
        self.blobs[digest] = data
        return BlobRef(sha256=digest, bytes=len(data), media_type=media_type)

    def read_blob(self, ref: BlobRef) -> bytes:
        return self.blobs[ref.sha256]

    def register_candidate(
        self, identity: dict[str, Any], files: dict[str, bytes]
    ) -> dict[str, Any]:
        for data in files.values():
            self.put_blob(data, media_type="application/octet-stream")
        artifact_id = "sha256:" + sha256_bytes(canonical_json(identity))
        manifest = {"artifact_id": artifact_id, "identity": identity, "files": dict(files)}
        self.candidates.append(manifest)
        return manifest


def _config(state: Path):
    return load_config(_PKG / "config.ci.json", state_override=state)


class ValidateCardsTest(unittest.TestCase):
    def test_accepts_real_cards(self) -> None:
        validate_cards(_REAL_CARDS)  # must not raise

    def test_rejects_non_list(self) -> None:
        with self.assertRaises(DataValidationError):
            validate_cards({"intent_id": "x"})  # type: ignore[arg-type]

    def test_rejects_empty(self) -> None:
        with self.assertRaises(DataValidationError):
            validate_cards([])

    def test_rejects_missing_intent_id(self) -> None:
        with self.assertRaises(DataValidationError):
            validate_cards([{"supported_goal": "g"}])

    def test_rejects_duplicate_intent_id(self) -> None:
        card = {"intent_id": "a", "supported_goal": "g"}
        with self.assertRaises(DataValidationError):
            validate_cards([card, dict(card)])

    def test_rejects_bad_included_type(self) -> None:
        with self.assertRaises(DataValidationError):
            validate_cards([{"intent_id": "a", "supported_goal": "g", "included": [1, 2]}])


class ValidateLogsTest(unittest.TestCase):
    def test_accepts_sample_logs(self) -> None:
        turns = validate_conversation_jsonl(_ROOT / "logs" / "conversations.sample.jsonl")
        self.assertGreater(len(turns), 0)
        self.assertIsInstance(turns, tuple)

    def _write(self, tmp: Path, lines: list[str]) -> Path:
        path = tmp / "log.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_rejects_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), ["{not json}"])
            with self.assertRaises(DataValidationError):
                validate_conversation_jsonl(path)

    def test_rejects_missing_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), [json.dumps({"turn": 0, "query": "hi"})])
            with self.assertRaises(DataValidationError):
                validate_conversation_jsonl(path)

    def test_rejects_non_integer_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp), [json.dumps({"session_id": "s", "turn": "0", "query": "hi"})]
            )
            with self.assertRaises(DataValidationError):
                validate_conversation_jsonl(path)

    def test_rejects_invalid_disposition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp),
                [json.dumps({"session_id": "s", "turn": 0, "query": "hi", "disposition": "MAYBE"})],
            )
            with self.assertRaises(DataValidationError):
                validate_conversation_jsonl(path)


class IngestTest(unittest.TestCase):
    def test_ingest_produces_content_addressed_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = ingest(_config(Path(tmp)), FakeRegistry())
        self.assertTrue(snapshot["snapshot_id"].startswith("sha256:"))
        self.assertIsInstance(snapshot["cards"], BlobRef)
        self.assertIsInstance(snapshot["logs"], BlobRef)
        self.assertEqual(snapshot["runtime"]["model_id"], "Qwen/Qwen3-4B-Instruct-2507")
        # Structure-only eval-suite digest (the reconciled council value).
        self.assertEqual(
            snapshot["evaluation_contract"]["eval_suite_sha256"],
            "03a6f1db1b594d7b6fd2b12d23bf08d03ebce0e46e0d13abffa659bdab937461",
        )

    def test_ingest_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = ingest(_config(Path(tmp)), FakeRegistry())
            second = ingest(_config(Path(tmp)), FakeRegistry())
        self.assertEqual(first["snapshot_id"], second["snapshot_id"])

    def test_frozen_source_mismatch_aborts_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            tampered = dataclasses.replace(
                config,
                frozen=dataclasses.replace(config.frozen, scope_bot_sha256="0" * 64),
            )
            with self.assertRaises(SourceFreezeError):
                ingest(tampered, FakeRegistry())

    def test_live_served_record_enters_snapshot_and_mining(self) -> None:
        # A record served into the configured live feedback log is validated, MERGED with
        # the bootstrap sample, and content-addressed into the snapshot's logs blob — which
        # is exactly what MINE consumes. The bootstrap-alone case stays valid (CI's null
        # path); here we point feedback_logs at a temp live file with one served turn.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            live = tmp_dir / "conversations.jsonl"
            probe = "__live_served_probe_query__"
            live.write_text(
                json.dumps({
                    "session_id": "live-session", "turn": 1, "query": probe,
                    "disposition": "CLARIFY", "card_id": None, "candidates": [],
                    "reason": "ambiguous", "reply": "which did you mean?",
                    "kb_version": "1acfd6ce4a70", "ts": "2026-07-16T00:00:00+00:00",
                }) + "\n",
                encoding="utf-8",
            )
            config = dataclasses.replace(
                _config(tmp_dir / "state"), feedback_logs_path=live)
            registry = FakeRegistry()
            snapshot = ingest(config, registry)

            # The live turn is in the merged logs blob (the snapshot's, and MINE's, input).
            merged = registry.read_blob(snapshot["logs"]).decode("utf-8")
            self.assertIn(probe, merged)
            self.assertEqual(snapshot["counts"]["live_turns"], 1)
            self.assertGreater(snapshot["counts"]["bootstrap_turns"], 0)
            self.assertEqual(snapshot["feedback_logs_path"], str(live))

            # MINE reads snapshot["logs"] — prove the exact live record reaches its input.
            labels_blob = mine_labels(snapshot, registry)
            self.assertIsInstance(labels_blob, BlobRef)
            mined_input = registry.read_blob(snapshot["logs"]).decode("utf-8")
            self.assertIn(probe, mined_input)

            # A snapshot without the live file (bootstrap alone) differs -> a new served
            # turn invalidates the snapshot id (and thus INGEST/MINE/BUILD downstream).
            bootstrap_only = ingest(_config(tmp_dir / "state2"), FakeRegistry())
            self.assertNotEqual(snapshot["snapshot_id"], bootstrap_only["snapshot_id"])


class BuildFlowTest(unittest.TestCase):
    def _ingest(self, registry: FakeRegistry, state: Path):
        return ingest(_config(state), registry)

    def test_mine_review_build_flywheel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            registry = FakeRegistry()
            snapshot = self._ingest(registry, tmp_dir)

            labels_blob = mine_labels(snapshot, registry)
            self.assertIsInstance(labels_blob, BlobRef)

            queue_path = tmp_dir / "reviews.jsonl"
            pending = write_review_queue(labels_blob, registry, queue_path)
            # The sample logs surface exactly one answer-expanding (OVER_ABSTAIN) label.
            self.assertEqual(pending, 1)

            candidate = build_candidate(
                snapshot, labels_blob, queue_path, registry, parent_artifact_id=None
            )
            document = json.loads(candidate["files"]["exemplars.json"].decode("utf-8"))
            self.assertEqual(document["kb_version"], "1acfd6ce4a70")
            # The existing builder keeps the four high-confidence, approved exemplars; the
            # pending answer-expanding label is excluded and does NOT block the build.
            self.assertEqual(len(document["exemplars"]), 4)
            self.assertTrue(candidate["artifact_id"].startswith("sha256:"))

    def test_write_review_queue_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            registry = FakeRegistry()
            snapshot = self._ingest(registry, tmp_dir)
            labels_blob = mine_labels(snapshot, registry)
            queue_path = tmp_dir / "reviews.jsonl"
            first = write_review_queue(labels_blob, registry, queue_path)
            second = write_review_queue(labels_blob, registry, queue_path)
            self.assertEqual(first, second)
            queued_lines = [
                json.loads(line)
                for line in queue_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("kind") == "queued"
            ]
            self.assertEqual(len(queued_lines), 1)

    def test_record_review_rejects_unknown_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            registry = FakeRegistry()
            snapshot = self._ingest(registry, tmp_dir)
            labels_blob = mine_labels(snapshot, registry)
            queue_path = tmp_dir / "reviews.jsonl"
            write_review_queue(labels_blob, registry, queue_path)
            with self.assertRaises(ValueError):
                record_review(
                    queue_path, label_id="sha256:deadbeef", decision="approve", reviewer="alice"
                )

    def test_human_approval_of_expanding_label_is_safe(self) -> None:
        # Approving the answer-expanding label with a human reviewer is honored at the
        # review layer, but the existing builder's own confidence/disposition floors keep
        # it out of the bank — the safety layers compose. The build still succeeds.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            registry = FakeRegistry()
            snapshot = self._ingest(registry, tmp_dir)
            labels_blob = mine_labels(snapshot, registry)
            queue_path = tmp_dir / "reviews.jsonl"
            write_review_queue(labels_blob, registry, queue_path)
            queued = next(
                json.loads(line)
                for line in queue_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("kind") == "queued"
            )
            record_review(
                queue_path, label_id=queued["label_id"], decision="approve", reviewer="alice"
            )
            candidate = build_candidate(
                snapshot, labels_blob, queue_path, registry, parent_artifact_id=None
            )
            document = json.loads(candidate["files"]["exemplars.json"].decode("utf-8"))
            self.assertEqual(len(document["exemplars"]), 4)


if __name__ == "__main__":
    unittest.main()
