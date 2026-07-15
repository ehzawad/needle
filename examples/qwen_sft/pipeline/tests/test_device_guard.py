"""Device-guard tests — the GPU resource-isolation boundary.

Real-world analog: the Kubernetes NVIDIA device plugin enforcing that a pod sees only
the GPUs it was granted. This box has an A5000 AND an A6000, and PyTorch logical
device 0 defaults to the A6000 here, so ``scope_bot``'s ``device_map={"": 0}`` would
grab the wrong card without pinning.

These tests exercise the guard WITHOUT real torch: the wrong/missing
``CUDA_VISIBLE_DEVICES`` branch of ``child_preflight`` fails closed before any torch
import, and ``pinned_environment`` is a pure dict transform. ``host_preflight`` is
asserted only when nvidia-smi actually reports the A5000 (skipped otherwise) so the
suite stays portable.
"""
from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

# Make the example root importable so ``pipeline.*`` resolves regardless of runner.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pipeline.device_guard as device_guard  # noqa: E402
from pipeline.contracts import DeviceReport  # noqa: E402


def _a5000_visible() -> bool:
    """True only if nvidia-smi is present AND reports the required A5000 UUID, so the
    host_preflight assertion runs on the real host and skips everywhere else."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        rows = device_guard._smi_uuids()
    except Exception:  # nvidia-smi present but unusable
        return False
    return any(uuid == device_guard.A5000_UUID for uuid, _ in rows)


class PinnedEnvironmentTest(unittest.TestCase):
    def test_overwrites_visibility_and_strips_rank(self) -> None:
        base = {
            "CUDA_VISIBLE_DEVICES": "0",  # dangerously points at the A6000 here
            "NVIDIA_VISIBLE_DEVICES": "all",
            "RANK": "3",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "4",
            "UNRELATED": "keep-me",
        }
        env = device_guard.pinned_environment(base)

        # Visibility is OVERWRITTEN to exactly the A5000 UUID, not inherited.
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], device_guard.A5000_UUID)
        self.assertEqual(env["NVIDIA_VISIBLE_DEVICES"], device_guard.A5000_UUID)
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

        # Distributed-launch vars are stripped so no torchrun/DDP path can widen it.
        self.assertNotIn("RANK", env)
        self.assertNotIn("LOCAL_RANK", env)
        self.assertNotIn("WORLD_SIZE", env)

        # Unrelated vars survive, and the caller's dict is not mutated.
        self.assertEqual(env["UNRELATED"], "keep-me")
        self.assertEqual(base["RANK"], "3")
        self.assertEqual(base["CUDA_VISIBLE_DEVICES"], "0")


class ChildPreflightTest(unittest.TestCase):
    def test_rejects_wrong_visible_devices(self) -> None:
        # A wrong CUDA_VISIBLE_DEVICES fails closed BEFORE torch is imported.
        with mock.patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False
        ):
            with self.assertRaises(device_guard.DeviceGuardError):
                device_guard.child_preflight()

    def test_rejects_missing_visible_devices(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(device_guard.DeviceGuardError):
                device_guard.child_preflight()

    def test_rejects_a6000_uuid(self) -> None:
        wrong_uuid = "GPU-00000000-0000-0000-0000-000000000000"
        with mock.patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": wrong_uuid}, clear=False
        ):
            with self.assertRaises(device_guard.DeviceGuardError):
                device_guard.child_preflight()


class HostPreflightTest(unittest.TestCase):
    @unittest.skipUnless(_a5000_visible(), "A5000 not visible via nvidia-smi")
    def test_reports_a5000(self) -> None:
        report = device_guard.host_preflight()
        self.assertIsInstance(report, DeviceReport)
        self.assertEqual(report.uuid, device_guard.A5000_UUID)
        self.assertIn(device_guard.A5000_NAME, report.name)
        self.assertGreaterEqual(report.visible_count, 1)


if __name__ == "__main__":
    unittest.main()
