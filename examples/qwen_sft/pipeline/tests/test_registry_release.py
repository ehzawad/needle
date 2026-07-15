"""Registry + release-controller tests — immutability, integrity, and staged promotion.

Real-world analog: the contract tests you would put around an MLflow model registry and
an Argo Rollouts / circuit-breaker release controller. They pin the council's
non-negotiables without any GPU or model:

  * blobs are content-addressed and immutable;
  * ``verify_candidate`` fails closed on a corrupted blob;
  * ``promote`` flips ``CURRENT`` atomically and only with the required evidence;
  * a demo registry refuses mock-backed evidence (CI mock evidence cannot reach demo);
  * ``rollback`` restores the previous artifact;
  * ``trip_circuit`` opens the breaker first, then reverts ``CURRENT``.

Pure standard library; nothing here imports torch/transformers/scope_bot.
"""
from __future__ import annotations

import stat
import sys
import tempfile
import unittest
from pathlib import Path

# Make the example root importable so ``pipeline.*`` resolves regardless of runner.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.registry import FileRegistry, RegistryError  # noqa: E402
from pipeline.release import (  # noqa: E402
    ReleaseError,
    promote,
    resolve_channel,
    rollback,
    set_channel,
    trip_circuit,
)
from pipeline.source_fingerprint import sha256_bytes  # noqa: E402

_CARDS = b'{"cards":[{"id":"acct.balance"}]}'
_EXEMPLARS = b'[{"q":"balance?","card":"acct.balance"}]'


def _seed_candidate(
    registry: FileRegistry,
    *,
    backend: str = "mock",
    parent: str | None = None,
    salt: str = "",
) -> tuple[dict, dict]:
    """Register a candidate and write its offline/shadow/canary evidence.

    ``salt`` perturbs the lineage so distinct calls yield distinct ``artifact_id``s
    (letting a test hold two candidates at once).
    """
    cards_ref = registry.put_blob(_CARDS, media_type="application/json")
    exemplar_ref = registry.put_blob(_EXEMPLARS, media_type="application/json")
    identity = {
        "payload": {"cards": cards_ref, "exemplar_bank": exemplar_ref},
        "runtime": {
            "model_id": "Qwen/Qwen3-4B-Instruct-2507",
            "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
            "judge_prompt_sha256": "aa" * 32,
            "generation_prompt_sha256": "bb" * 32,
            "scope_bot_source_sha256": "cc" * 32,
            "scope_policy_source_sha256": "dd" * 32,
            "builder_source_sha256": "ee" * 32,
        },
        "evaluation_contract": {"eval_suite_sha256": "ff" * 32, "thresholds_sha256": "11" * 32},
        "lineage": {
            "snapshot_id": "sha256:" + "22" * 32,
            "labels_sha256": "33" * 32,
            "reviews_sha256": "44" * 32 + salt,
            "parent_artifact_id": parent,
        },
    }
    manifest = registry.register_candidate(
        identity, {"cards.json": _CARDS, "exemplars.json": _EXEMPLARS}
    )
    artifact_id = manifest["artifact_id"]
    offline = registry.write_evidence(
        "offline_eval",
        {"artifact_id": artifact_id, "backend": backend, "suite_sha256": "ff" * 32,
         "passed": True, "metrics": {"harmful_answers": 0}},
    )
    shadow = registry.write_evidence(
        "shadow",
        {"artifact_id": artifact_id, "backend": backend, "passed": True,
         "metrics": {"unapproved_expansions": 0}},
    )
    canary = registry.write_evidence(
        "canary",
        {"artifact_id": artifact_id, "backend": backend, "passed": True,
         "metrics": {"harmful_answers": 0}},
    )
    return manifest, {
        "offline": offline["evidence_id"],
        "shadow": shadow["evidence_id"],
        "canary": canary["evidence_id"],
    }


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = FileRegistry(self.root, environment="ci")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_blob_content_address_and_immutability(self) -> None:
        ref = self.registry.put_blob(b"hello world", media_type="text/plain")
        self.assertEqual(ref.sha256, sha256_bytes(b"hello world"))
        self.assertEqual(ref.bytes, len(b"hello world"))
        self.assertEqual(self.registry.read_blob(ref), b"hello world")

        blob_path = self.registry.blobs_dir / ref.sha256
        mode = stat.S_IMODE(blob_path.stat().st_mode)
        self.assertEqual(mode & 0o222, 0, "blob must be stored read-only (immutable)")

        # Re-storing identical bytes is idempotent: same ref, same on-disk digest.
        again = self.registry.put_blob(b"hello world", media_type="text/plain")
        self.assertEqual(again.sha256, ref.sha256)
        self.assertEqual(sha256_bytes(blob_path.read_bytes()), ref.sha256)

    def test_read_blob_detects_corruption(self) -> None:
        ref = self.registry.put_blob(b"payload", media_type="text/plain")
        blob_path = self.registry.blobs_dir / ref.sha256
        blob_path.chmod(0o644)
        blob_path.write_bytes(b"tampered")
        with self.assertRaises(RegistryError):
            self.registry.read_blob(ref)

    def test_register_candidate_is_content_addressed_and_idempotent(self) -> None:
        manifest_a, _ = _seed_candidate(self.registry)
        manifest_b, _ = _seed_candidate(self.registry)  # same identity -> same id
        self.assertEqual(manifest_a["artifact_id"], manifest_b["artifact_id"])
        self.assertTrue(manifest_a["artifact_id"].startswith("sha256:"))
        # A healthy artifact verifies, and its files/manifest are stored read-only.
        self.registry.verify_candidate(manifest_a["artifact_id"])
        adir = self.registry.artifact_dir(manifest_a["artifact_id"])
        self.assertTrue((adir / "manifest.json").is_file())
        self.assertTrue((adir / "files" / "cards.json").is_file())
        mode = stat.S_IMODE((adir / "manifest.json").stat().st_mode)
        self.assertEqual(mode & 0o222, 0, "manifest must be immutable")

    def test_verify_candidate_rejects_corrupted_blob(self) -> None:
        manifest, _ = _seed_candidate(self.registry)
        cards_ref = manifest["payload"]["cards"]
        blob_path = self.registry.blobs_dir / cards_ref["sha256"]
        blob_path.chmod(0o644)
        blob_path.write_bytes(b'{"cards":[]}')  # different bytes -> digest no longer matches
        with self.assertRaises(RegistryError):
            self.registry.verify_candidate(manifest["artifact_id"])

    def test_verify_candidate_rejects_manifest_tamper(self) -> None:
        manifest, _ = _seed_candidate(self.registry)
        manifest_path = self.registry.artifact_dir(manifest["artifact_id"]) / "manifest.json"
        manifest_path.chmod(0o644)
        manifest_path.write_text(
            manifest_path.read_text("utf-8").replace(
                '"model_id": "Qwen/Qwen3-4B-Instruct-2507"',
                '"model_id": "evil/model"',
            ),
            encoding="utf-8",
        )
        with self.assertRaises(RegistryError):
            self.registry.verify_candidate(manifest["artifact_id"])

    def test_load_evidence_detects_tamper(self) -> None:
        manifest, evidence = _seed_candidate(self.registry)
        path = self.registry.evidence_path(evidence["offline"])
        path.chmod(0o644)
        path.write_text(
            path.read_text("utf-8").replace('"backend": "mock"', '"backend": "real"'),
            encoding="utf-8",
        )
        with self.assertRaises(RegistryError):
            self.registry.load_evidence(evidence["offline"])


class ReleaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _registry(self, environment: str = "ci") -> FileRegistry:
        return FileRegistry(self.root, environment=environment)  # type: ignore[arg-type]

    def test_promote_flips_current_atomically(self) -> None:
        registry = self._registry("ci")
        manifest, evidence = _seed_candidate(registry)
        artifact_id = manifest["artifact_id"]
        self.assertIsNone(resolve_channel(registry, "CURRENT"))

        record = promote(
            registry,
            artifact_id,
            offline_evidence_id=evidence["offline"],
            shadow_evidence_id=evidence["shadow"],
            canary_evidence_id=evidence["canary"],
            actor="ci",
        )
        self.assertEqual(record["artifact_id"], artifact_id)
        self.assertIsNone(record["previous_artifact_id"])
        self.assertEqual(resolve_channel(registry, "CURRENT"), artifact_id)
        self.assertTrue(registry.channel_path("CURRENT").is_file())
        # History recorded the promotion.
        history = registry.history_path.read_text("utf-8").strip().splitlines()
        self.assertTrue(any('"action": "promote"' in line for line in history))

    def test_promote_requires_all_three_evidence_ids(self) -> None:
        registry = self._registry("ci")
        manifest, evidence = _seed_candidate(registry)
        with self.assertRaises(ReleaseError):
            promote(
                registry,
                manifest["artifact_id"],
                offline_evidence_id=evidence["offline"],
                shadow_evidence_id="",
                canary_evidence_id=evidence["canary"],
                actor="ci",
            )
        self.assertIsNone(resolve_channel(registry, "CURRENT"))

    def test_demo_registry_rejects_mock_evidence(self) -> None:
        registry = self._registry("demo")
        manifest, evidence = _seed_candidate(registry, backend="mock")
        with self.assertRaises(ReleaseError):
            promote(
                registry,
                manifest["artifact_id"],
                offline_evidence_id=evidence["offline"],
                shadow_evidence_id=evidence["shadow"],
                canary_evidence_id=evidence["canary"],
                actor="demo",
            )
        self.assertIsNone(resolve_channel(registry, "CURRENT"))

    def test_demo_registry_accepts_real_evidence(self) -> None:
        registry = self._registry("demo")
        manifest, evidence = _seed_candidate(registry, backend="real")
        record = promote(
            registry,
            manifest["artifact_id"],
            offline_evidence_id=evidence["offline"],
            shadow_evidence_id=evidence["shadow"],
            canary_evidence_id=evidence["canary"],
            actor="demo",
        )
        self.assertEqual(record["artifact_id"], manifest["artifact_id"])
        self.assertEqual(resolve_channel(registry, "CURRENT"), manifest["artifact_id"])

    def test_ci_registry_accepts_mock_evidence(self) -> None:
        registry = self._registry("ci")
        manifest, evidence = _seed_candidate(registry, backend="mock")
        record = promote(
            registry,
            manifest["artifact_id"],
            offline_evidence_id=evidence["offline"],
            shadow_evidence_id=evidence["shadow"],
            canary_evidence_id=evidence["canary"],
            actor="ci",
        )
        self.assertEqual(resolve_channel(registry, "CURRENT"), manifest["artifact_id"])

    def test_set_channel_rejects_current(self) -> None:
        registry = self._registry("ci")
        manifest, evidence = _seed_candidate(registry)
        with self.assertRaises(ReleaseError):
            set_channel(
                registry,
                "CURRENT",  # type: ignore[arg-type]
                manifest["artifact_id"],
                evidence_ids=[evidence["shadow"]],
                actor="ci",
            )

    def test_set_shadow_channel(self) -> None:
        registry = self._registry("ci")
        manifest, evidence = _seed_candidate(registry)
        record = set_channel(
            registry,
            "SHADOW",
            manifest["artifact_id"],
            evidence_ids=[evidence["shadow"]],
            actor="ci",
        )
        self.assertEqual(record["artifact_id"], manifest["artifact_id"])
        self.assertEqual(resolve_channel(registry, "SHADOW"), manifest["artifact_id"])

    def test_rollback_restores_previous(self) -> None:
        registry = self._registry("ci")
        manifest_a, evidence_a = _seed_candidate(registry, salt="a")
        manifest_b, evidence_b = _seed_candidate(registry, salt="b")
        artifact_a, artifact_b = manifest_a["artifact_id"], manifest_b["artifact_id"]
        self.assertNotEqual(artifact_a, artifact_b)

        promote(
            registry, artifact_a,
            offline_evidence_id=evidence_a["offline"],
            shadow_evidence_id=evidence_a["shadow"],
            canary_evidence_id=evidence_a["canary"], actor="ci",
        )
        promote(
            registry, artifact_b,
            offline_evidence_id=evidence_b["offline"],
            shadow_evidence_id=evidence_b["shadow"],
            canary_evidence_id=evidence_b["canary"], actor="ci",
        )
        self.assertEqual(resolve_channel(registry, "CURRENT"), artifact_b)

        record = rollback(registry, target_artifact_id=artifact_a, reason="regression", actor="ci")
        self.assertEqual(record["artifact_id"], artifact_a)
        self.assertEqual(record["previous_artifact_id"], artifact_b)
        self.assertEqual(resolve_channel(registry, "CURRENT"), artifact_a)

    def test_trip_circuit_opens_and_reverts(self) -> None:
        registry = self._registry("ci")
        manifest_a, evidence_a = _seed_candidate(registry, salt="a")
        manifest_b, evidence_b = _seed_candidate(registry, salt="b")
        artifact_a, artifact_b = manifest_a["artifact_id"], manifest_b["artifact_id"]

        promote(
            registry, artifact_a,
            offline_evidence_id=evidence_a["offline"],
            shadow_evidence_id=evidence_a["shadow"],
            canary_evidence_id=evidence_a["canary"], actor="ci",
        )
        promote(
            registry, artifact_b,
            offline_evidence_id=evidence_b["offline"],
            shadow_evidence_id=evidence_b["shadow"],
            canary_evidence_id=evidence_b["canary"], actor="ci",
        )

        circuit = trip_circuit(
            registry, bad_artifact_id=artifact_b, evidence_id=evidence_b["canary"],
            reason="canary leak",
        )
        self.assertTrue(circuit["open"])
        self.assertEqual(circuit["bad_artifact_id"], artifact_b)
        self.assertEqual(circuit["rolled_back_to"], artifact_a)
        self.assertTrue(registry.circuit_path.is_file())
        # CURRENT reverted to the last good artifact; the bad one is not served.
        self.assertEqual(resolve_channel(registry, "CURRENT"), artifact_a)

    def test_trip_circuit_without_prior_good_disables_serving(self) -> None:
        registry = self._registry("ci")
        manifest, evidence = _seed_candidate(registry)
        artifact_id = manifest["artifact_id"]
        promote(
            registry, artifact_id,
            offline_evidence_id=evidence["offline"],
            shadow_evidence_id=evidence["shadow"],
            canary_evidence_id=evidence["canary"], actor="ci",
        )
        circuit = trip_circuit(
            registry, bad_artifact_id=artifact_id, evidence_id=evidence["canary"],
            reason="canary leak, no prior good",
        )
        self.assertTrue(circuit["open"])
        self.assertIsNone(circuit["rolled_back_to"])
        # No prior good artifact -> serving disabled, not pinned to the bad candidate.
        self.assertIsNone(resolve_channel(registry, "CURRENT"))

    def test_promote_blocked_when_circuit_open_against_artifact(self) -> None:
        registry = self._registry("ci")
        manifest, evidence = _seed_candidate(registry)
        artifact_id = manifest["artifact_id"]
        promote(
            registry, artifact_id,
            offline_evidence_id=evidence["offline"],
            shadow_evidence_id=evidence["shadow"],
            canary_evidence_id=evidence["canary"], actor="ci",
        )
        trip_circuit(
            registry, bad_artifact_id=artifact_id, evidence_id=evidence["canary"],
            reason="canary leak",
        )
        with self.assertRaises(ReleaseError):
            promote(
                registry, artifact_id,
                offline_evidence_id=evidence["offline"],
                shadow_evidence_id=evidence["shadow"],
                canary_evidence_id=evidence["canary"], actor="ci",
            )


if __name__ == "__main__":
    unittest.main()
