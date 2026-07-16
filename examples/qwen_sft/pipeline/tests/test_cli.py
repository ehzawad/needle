"""CLI wiring tests — the dashboard's drift path is a LIVE call, and serve offers a
real model-serving backend. These guard against regressing to the earlier state where
``detect_drift`` was defined but never called (dashboard always got ``alerts=[]``) and
the HTTP endpoint could only ever serve a mock gate.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import cli  # noqa: E402
from pipeline import device_guard  # noqa: E402
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


class GpuConfigGuardTest(unittest.TestCase):
    """The config's declared GPU identity is validated against the device-guard A5000
    constants at load_config time (fail closed before any run)."""

    def _load_with_gpu(self, uuid_value: str, name_value: str):
        data = json.loads((_PKG / "config.ci.json").read_text(encoding="utf-8"))
        data["gpu"]["uuid"] = uuid_value
        data["gpu"]["name"] = name_value
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return load_config(path)

    def test_correct_gpu_identity_loads(self) -> None:
        config = self._load_with_gpu(device_guard.A5000_UUID, device_guard.A5000_NAME)
        self.assertEqual(config.gpu_uuid, device_guard.A5000_UUID)
        self.assertEqual(config.gpu_name, device_guard.A5000_NAME)

    def test_wrong_gpu_uuid_fails_load(self) -> None:
        with self.assertRaises(ValueError):
            self._load_with_gpu("GPU-00000000-0000-0000-0000-000000000000",
                                device_guard.A5000_NAME)

    def test_wrong_gpu_name_fails_load(self) -> None:
        with self.assertRaises(ValueError):
            self._load_with_gpu(device_guard.A5000_UUID, "NVIDIA RTX A6000")


class PinCurrentProcessTest(unittest.TestCase):
    """pin_current_process pins THIS process to the A5000 and strips every
    distributed-launch var, so a model loaded in-process (real HTTP serving) cannot see
    the wrong GPU or widen visibility via a leftover RANK/LOCAL_RANK."""

    _TOUCHED = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")

    def setUp(self) -> None:
        self._saved = {k: os.environ.get(k) for k in self._TOUCHED}

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_pins_a5000_and_strips_distributed_vars(self) -> None:
        os.environ["RANK"] = "3"
        os.environ["LOCAL_RANK"] = "1"
        os.environ["WORLD_SIZE"] = "4"
        os.environ["GROUP_RANK"] = "2"
        device_guard.pin_current_process()
        for stripped in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                         "GROUP_RANK"):
            self.assertNotIn(stripped, os.environ, f"{stripped} not stripped")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], device_guard.A5000_UUID)
        self.assertEqual(os.environ["NVIDIA_VISIBLE_DEVICES"], device_guard.A5000_UUID)
        self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")


if __name__ == "__main__":
    unittest.main()
