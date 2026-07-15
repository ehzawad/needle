"""A5000-only device guard — the exclusive GPU launch boundary (fail-closed).

Real-world analog: a Kubernetes GPU resource limit + the NVIDIA device plugin —
a pod may see ONLY the GPUs it was granted. This box has an A5000 AND an A6000, and
(dangerously) PyTorch logical device 0 defaults to the A6000 here. `scope_bot` uses
`device_map={"": 0}`, so without pinning it would grab the wrong GPU. This module
guarantees every model-touching child sees EXACTLY the A5000 and nothing else.

The control plane never imports torch. Real model work happens only inside a child
launched via `launch_gpu_worker`, whose environment is OVERWRITTEN (not inherited)
to expose the A5000 by UUID; `child_preflight()` then re-verifies and fails closed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from pipeline.contracts import DeviceReport

A5000_UUID = "GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567"
A5000_NAME = "NVIDIA RTX A5000"
_STRIP = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK")


class DeviceGuardError(RuntimeError):
    """Raised when the visible GPU is anything other than exactly the A5000."""


def pinned_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an env that exposes ONLY the A5000. Overwrites (never inherits) the
    critical CUDA vars and strips any distributed-launch vars so no torchrun/DDP path
    can widen visibility."""
    env = dict(base if base is not None else os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = A5000_UUID
    env["NVIDIA_VISIBLE_DEVICES"] = A5000_UUID
    for k in _STRIP:
        env.pop(k, None)
    return env


def _smi_uuids() -> list[tuple[str, str]]:
    """(uuid, name) for every physical GPU, via nvidia-smi (no torch, control-plane safe)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        uuid, _, name = line.partition(",")
        rows.append((uuid.strip(), name.strip()))
    return rows


def host_preflight() -> DeviceReport:
    """Control-plane check: the A5000 UUID must exist EXACTLY once with the A5000 name.
    Uses nvidia-smi only; never imports torch."""
    rows = _smi_uuids()
    matches = [(u, n) for (u, n) in rows if u == A5000_UUID]
    if len(matches) != 1:
        raise DeviceGuardError(
            f"host-guard: A5000 UUID {A5000_UUID} found {len(matches)} times among {rows}")
    uuid, name = matches[0]
    if A5000_NAME not in name:
        raise DeviceGuardError(f"host-guard: UUID {uuid} has name '{name}', not {A5000_NAME!r}")
    return DeviceReport(uuid=uuid, name=name, logical_index=-1, visible_count=len(rows),
                        torch_version="host")


def child_preflight() -> DeviceReport:
    """Guarded-worker check, run BEFORE importing scope_bot. Fails closed unless CUDA
    exposes exactly one device and it is the A5000. Because CUDA_VISIBLE_DEVICES is the
    A5000 UUID, CUDA exposes only that exact GPU (or zero, if the UUID is absent)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != A5000_UUID:
        raise DeviceGuardError(
            f"child-guard: CUDA_VISIBLE_DEVICES={visible!r} != required {A5000_UUID!r}")
    import torch  # noqa: PLC0415 — deliberately deferred; never a top-level import
    if not torch.cuda.is_available():
        raise DeviceGuardError("child-guard: no CUDA device visible; the A5000 is required")
    n = torch.cuda.device_count()
    if n != 1:
        raise DeviceGuardError(f"child-guard: expected exactly 1 visible GPU (A5000), saw {n}")
    if torch.cuda.current_device() != 0:
        raise DeviceGuardError(f"child-guard: current device is {torch.cuda.current_device()}, not 0")
    name = torch.cuda.get_device_name(0)
    if A5000_NAME not in name:
        raise DeviceGuardError(f"child-guard: logical cuda:0 is '{name}', not the A5000")
    return DeviceReport(uuid=A5000_UUID, name=name, logical_index=0, visible_count=1,
                        torch_version=torch.__version__)


def assert_model_devices(model: object) -> None:
    """Belt-and-suspenders: after the model loads, refuse any CUDA placement other than
    logical cuda:0 (with one visible device, only cuda:0 can exist — this catches an
    accidental explicit .to('cuda:1'). Reads only tensor.device, so needs no torch import."""
    for tensor in list(getattr(model, "parameters", lambda: [])()) + \
            list(getattr(model, "buffers", lambda: [])()):
        dev = tensor.device
        if dev.type == "cuda" and (dev.index or 0) != 0:
            raise DeviceGuardError(f"model placed a tensor on {dev}, not cuda:0")


def launch_gpu_worker(argv: Sequence[str], *, cwd: Path,
                      timeout_s: int | None = None) -> subprocess.CompletedProcess[str]:
    """Launch a model-touching child under the pinned A5000 environment. This is the
    ONLY sanctioned path to real GPU work; the control plane calls nothing torch itself."""
    host_preflight()  # refuse to even launch if the host can't see exactly the A5000
    cmd = [sys.executable, *argv]
    return subprocess.run(cmd, cwd=str(cwd), env=pinned_environment(os.environ),
                          capture_output=True, text=True, timeout=timeout_s)
