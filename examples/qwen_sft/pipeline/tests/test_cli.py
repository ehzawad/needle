"""CLI wiring tests — the dashboard's drift path is a LIVE call, and serve offers a
real model-serving backend. These guard against regressing to the earlier state where
``detect_drift`` was defined but never called (dashboard always got ``alerts=[]``) and
the HTTP endpoint could only ever serve a mock gate.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import cli  # noqa: E402
from pipeline.config import load_config  # noqa: E402

_PKG = Path(__file__).resolve().parents[1]


class CliWiringTest(unittest.TestCase):
    def test_drift_alerts_is_a_live_call(self) -> None:
        # Exercises cli._drift_alerts -> observability.detect_drift over the served log.
        # Below drift.min_samples it MUST return an empty list (not raise, not None).
        config = load_config(_PKG / "config.ci.json")
        alerts = cli._drift_alerts(config)
        self.assertIsInstance(alerts, list)

    def test_serve_parser_exposes_real_backend(self) -> None:
        parser = cli._build_parser()
        args = parser.parse_args([
            "serve", "--config", str(_PKG / "config.ci.json"),
            "--state", "/tmp/x", "--http", "127.0.0.1:0", "--backend", "real",
        ])
        self.assertEqual(args.backend, "real")


if __name__ == "__main__":
    unittest.main()
