"""End-to-end mock-DAG test — the integrator's contract for the whole control plane.

Real-world analog: the CI job that runs the full release pipeline on a model-free stub
before any GPU spend, then asserts the release actually flipped and the evidence is the
council's frozen headline (0/25 harmful, 20/20 right-card, 5/5 ambiguous clarify).

Pure standard library; nothing here imports torch/transformers/scope_bot. Every stage,
gate, and cross-check runs on the deterministic mock gate.
"""
from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import dag, release  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.registry import FileRegistry  # noqa: E402

_CI_CONFIG = _ROOT / "pipeline" / "config.ci.json"
_PKG_DIR = _ROOT / "pipeline"


class MockDagTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.state = Path(self._tmp.name) / "state"
        self.config = load_config(_CI_CONFIG, state_override=self.state)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_mock_dag_promotes_and_reproduces_headline(self) -> None:
        summary = dag.run_pipeline(
            self.config, backend="mock", promote_current=True, actor="ci")

        self.assertEqual(summary["status"], "promoted", summary.get("block_reason"))
        self.assertTrue(summary["promoted"])
        artifact_id = summary["artifact_id"]
        self.assertIsNotNone(artifact_id)
        self.assertTrue(artifact_id.startswith("sha256:"))

        # CURRENT resolves to exactly the evaluated/promoted artifact.
        registry = FileRegistry(self.config.state_root, environment="ci")
        self.assertEqual(release.resolve_channel(registry, "CURRENT"), artifact_id)
        self.assertEqual(summary["current_artifact_id"], artifact_id)

        # Offline metrics are the frozen headline: 0/25, 20/20, 0 wrong, 5/5.
        metrics = summary["offline_metrics"]
        self.assertEqual(metrics["harmful_answers"], 0)
        self.assertEqual(metrics["harmful_total"], 25)
        self.assertEqual(metrics["right_card_answers"], 20)
        self.assertEqual(metrics["in_scope_total"], 20)
        self.assertEqual(metrics["wrong_card_answers"], 0)
        self.assertEqual(metrics["ambiguous_clarifies"], 5)
        self.assertEqual(metrics["ambiguous_total"], 5)
        self.assertEqual(metrics["errors"], 0)

        # All eight pre-promotion stages plus PROMOTE ran, none blocked.
        for stage in ("INGEST", "MINE", "REVIEW_QUEUE", "BUILD_CANDIDATE", "REGISTER",
                      "OFFLINE_EVAL", "SHADOW", "CANARY", "PROMOTE"):
            self.assertIn(stage, summary["stages"])
            self.assertIn(summary["stages"][stage]["status"], ("success", "cached"))

    def test_verify_run_confirms_current_and_backend(self) -> None:
        dag.run_pipeline(self.config, backend="mock", promote_current=True, actor="ci")
        report = dag.verify_run(self.config)
        self.assertTrue(report["ok"])
        self.assertIsNotNone(report["current_artifact_id"])
        self.assertTrue(report["current_verified"])
        self.assertEqual(report["current_backend"], "mock")
        self.assertFalse(report["circuit_open"])

    def test_resumable_stage_cache_reuses_input_stages(self) -> None:
        """The deterministic run id gives a stable per-run stage dir, so re-running reuses
        the input-only stages (INGEST/MINE/REVIEW_QUEUE) from cache. A second full run
        after a promotion legitimately builds the NEXT lineage generation (its candidate's
        parent is the now-CURRENT artifact — part of the content-addressed identity)."""
        first = dag.run_pipeline(self.config, backend="mock", promote_current=True, actor="ci")
        second = dag.run_pipeline(self.config, backend="mock", promote_current=True, actor="ci")
        self.assertEqual(first["status"], "promoted")
        self.assertEqual(second["status"], "promoted")
        for stage in ("INGEST", "MINE", "REVIEW_QUEUE"):
            self.assertEqual(second["stages"][stage]["status"], "cached",
                             f"expected {stage} to be reused from cache on re-run")
        # The second generation records the first as its parent (explicit lineage).
        self.assertEqual(second["current_before"], first["artifact_id"])

    def test_no_top_level_torch_import(self) -> None:
        """AST-scan every pipeline module: NONE may import torch/transformers at module
        scope. The guarded lazy imports inside gpu_worker / device_guard / adapters
        function bodies are allowed and are exactly how model work stays isolated."""
        offenders: list[str] = []
        for path in sorted(_PKG_DIR.glob("*.py")):
            tree = ast.parse(path.read_text("utf-8"), filename=str(path))
            for node in tree.body:  # module-level statements only
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module] if node.module else []
                else:
                    continue
                for name in names:
                    if name and name.split(".", 1)[0] in ("torch", "transformers"):
                        offenders.append(f"{path.name}:{node.lineno} imports {name}")
        self.assertEqual(offenders, [], f"top-level model imports: {offenders}")


if __name__ == "__main__":
    unittest.main()
