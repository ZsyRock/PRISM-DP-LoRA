#!/usr/bin/env python3
"""Validate one Slurm GPU step before replacing this process with a workload.

The guard is intentionally stricter than ``nvidia-smi`` or
``torch.cuda.device_count()`` alone: it initializes the CUDA driver, requires
exactly one GPU to be visible inside the step cgroup, and executes a small
tensor operation followed by an explicit synchronization.  Only after those
checks pass is a create-only marker installed atomically and the command after
``--`` executed.

An outer launcher may safely retry when the marker is absent.  Once the marker
exists, the workload boundary has been crossed and the launcher must not retry
the command automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CUDA_GUARD_FAILURE_EXIT = 86
WORKLOAD_EXEC_FAILURE_EXIT = 127
GPU_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "SLURM_JOB_ID",
    "SLURM_STEP_ID",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_LOCALID",
    "SLURM_PROCID",
    "SLURM_NTASKS",
)
NVIDIA_SMI_ARGUMENTS = (
    "--query-gpu=index,uuid,name,driver_version,memory.total",
    "--format=csv,noheader",
)


class CudaGuardError(RuntimeError):
    """Raised when a step cannot prove that its single GPU is usable."""


def _load_torch() -> Any:
    import torch

    return torch


def collect_nvidia_smi(
    *,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Return bounded ``nvidia-smi`` diagnostics without making it a gate.

    PyTorch CUDA initialization is the authoritative guard.  ``nvidia-smi`` is
    useful provenance, but a missing binary or a diagnostic-command failure is
    recorded rather than treated as proof that CUDA itself is unusable.
    """

    executable = which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "executable_not_found"}
    command = [executable, *NVIDIA_SMI_ARGUMENTS]
    try:
        completed = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except Exception as exc:  # diagnostics must not mask the CUDA probe
        return {
            "available": True,
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def probe_single_cuda_device(torch_module: Any) -> dict[str, Any]:
    """Initialize and exercise the exactly-one-visible-GPU step contract."""

    try:
        torch_module.cuda.init()
    except Exception as exc:
        raise CudaGuardError(
            f"CUDA driver initialization failed: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        available = bool(torch_module.cuda.is_available())
        device_count = int(torch_module.cuda.device_count())
    except Exception as exc:
        raise CudaGuardError(
            f"CUDA availability query failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not available:
        raise CudaGuardError("torch.cuda.is_available() returned false after cuda.init()")
    if device_count != 1:
        raise CudaGuardError(
            f"expected exactly one visible CUDA device inside the step, found {device_count}"
        )

    try:
        torch_module.cuda.set_device(0)
        properties = torch_module.cuda.get_device_properties(0)
        left = torch_module.ones(
            (16, 16), device="cuda:0", dtype=torch_module.float32
        )
        right = torch_module.ones(
            (16, 16), device="cuda:0", dtype=torch_module.float32
        )
        product = torch_module.matmul(left, right)
        checksum = float(product.sum().item())
        torch_module.cuda.synchronize(0)
    except Exception as exc:
        raise CudaGuardError(
            f"CUDA tensor probe failed: {type(exc).__name__}: {exc}"
        ) from exc
    expected_checksum = float(16 * 16 * 16)
    if checksum != expected_checksum:
        raise CudaGuardError(
            "CUDA tensor probe returned an unexpected checksum: "
            f"expected={expected_checksum}, actual={checksum}"
        )

    return {
        "available": available,
        "device_count": device_count,
        "device_index": 0,
        "device_name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "tensor_probe": {
            "operation": "matmul_16x16_ones",
            "checksum": checksum,
            "synchronized": True,
        },
    }


def write_atomic_marker(path: Path, payload: Mapping[str, Any]) -> None:
    """Install a complete mode-0600 marker atomically without overwriting one."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CudaGuardError(
                f"refusing to replace existing guard marker: {path}"
            ) from exc
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _diagnostic_base(
    *,
    environ: Mapping[str, str],
    hostname: str,
    nvidia_smi: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "hostname": hostname,
        "gpu_environment": {
            key: environ.get(key) for key in GPU_ENVIRONMENT_KEYS
        },
        "nvidia_smi": dict(nvidia_smi),
    }


def _emit(stream: Any, payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, allow_nan=False), file=stream, flush=True)


def run_guarded_workload(
    marker: Path,
    workload: Sequence[str],
    *,
    torch_loader: Callable[[], Any] = _load_torch,
    nvidia_smi_collector: Callable[[], Mapping[str, Any]] = collect_nvidia_smi,
    exec_fn: Callable[[str, Sequence[str]], Any] = os.execvp,
    environ: Mapping[str, str] = os.environ,
    hostname_fn: Callable[[], str] = socket.gethostname,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> int:
    """Run the guard, atomically mark success, then exec ``workload``."""

    if not workload:
        raise ValueError("workload must not be empty")

    try:
        nvidia_smi = dict(nvidia_smi_collector())
    except Exception as exc:  # injected/custom diagnostics remain non-authoritative
        nvidia_smi = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    diagnostic = _diagnostic_base(
        environ=environ,
        hostname=hostname_fn(),
        nvidia_smi=nvidia_smi,
    )
    try:
        cuda = probe_single_cuda_device(torch_loader())
        marker_payload = {
            **diagnostic,
            "state": "cuda_guard_passed_workload_starting",
            "created_at": now_fn().astimezone(timezone.utc).isoformat(),
            "cuda": cuda,
            # Record only the executable, not arguments that might contain a
            # credential or other secret.
            "workload_executable": workload[0],
        }
        write_atomic_marker(marker, marker_payload)
    except Exception as exc:
        failure = {
            **diagnostic,
            "state": "cuda_guard_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _emit(sys.stderr, failure)
        return CUDA_GUARD_FAILURE_EXIT

    _emit(sys.stdout, marker_payload)
    try:
        exec_fn(workload[0], list(workload))
    except OSError as exc:
        # The marker deliberately remains present.  The CUDA guard succeeded
        # and the workload boundary was crossed, so an outer launcher must not
        # interpret this as a retry-safe guard failure.
        _emit(
            sys.stderr,
            {
                **diagnostic,
                "state": "workload_exec_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "workload_executable": workload[0],
            },
        )
        return WORKLOAD_EXEC_FAILURE_EXIT
    return 0


def _parse_arguments(argv: Sequence[str]) -> tuple[Path, list[str]]:
    if "--" not in argv:
        raise ValueError("a workload must follow the required -- separator")
    separator = list(argv).index("--")
    guard_arguments = list(argv[:separator])
    workload = list(argv[separator + 1 :])
    parser = argparse.ArgumentParser(
        description="Validate one CUDA GPU, write a marker, then exec a workload."
    )
    parser.add_argument("--marker", type=Path, required=True)
    parsed = parser.parse_args(guard_arguments)
    if not workload:
        parser.error("the workload after -- must not be empty")
    return parsed.marker, workload


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        marker, workload = _parse_arguments(arguments)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return run_guarded_workload(marker, workload)


if __name__ == "__main__":
    raise SystemExit(main())
