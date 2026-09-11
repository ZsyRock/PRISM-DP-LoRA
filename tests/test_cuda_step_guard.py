from __future__ import annotations

import json
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.cuda_step_guard import (
    CUDA_GUARD_FAILURE_EXIT,
    WORKLOAD_EXEC_FAILURE_EXIT,
    CudaGuardError,
    collect_nvidia_smi,
    probe_single_cuda_device,
    run_guarded_workload,
    write_atomic_marker,
)


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeProduct:
    def sum(self) -> _FakeScalar:
        return _FakeScalar(4096.0)


class _FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        device_count: int = 1,
        init_error: Exception | None = None,
    ) -> None:
        self.available = available
        self.visible_device_count = device_count
        self.init_error = init_error
        self.calls: list[object] = []

    def init(self) -> None:
        self.calls.append("init")
        if self.init_error is not None:
            raise self.init_error

    def is_available(self) -> bool:
        self.calls.append("is_available")
        return self.available

    def device_count(self) -> int:
        self.calls.append("device_count")
        return self.visible_device_count

    def set_device(self, index: int) -> None:
        self.calls.append(("set_device", index))

    def get_device_properties(self, index: int) -> SimpleNamespace:
        self.calls.append(("get_device_properties", index))
        return SimpleNamespace(
            name="Fake H200",
            total_memory=143_771 * 1024 * 1024,
            major=9,
            minor=0,
        )

    def synchronize(self, index: int) -> None:
        self.calls.append(("synchronize", index))


class _FakeTorch:
    float32 = object()

    def __init__(self, cuda: _FakeCuda) -> None:
        self.cuda = cuda
        self.ones_calls: list[tuple[object, str, object]] = []
        self.matmul_calls = 0

    def ones(self, shape, *, device, dtype):
        self.ones_calls.append((shape, device, dtype))
        return object()

    def matmul(self, left, right) -> _FakeProduct:
        self.matmul_calls += 1
        return _FakeProduct()


def _fake_torch(**cuda_options) -> _FakeTorch:
    return _FakeTorch(_FakeCuda(**cuda_options))


def _fixed_nvidia_smi() -> dict[str, object]:
    return {
        "available": True,
        "returncode": 0,
        "stdout": "0, GPU-fake, NVIDIA H200 NVL, 595.71.05, 143771 MiB",
        "stderr": "",
    }


def test_probe_initializes_exactly_one_gpu_and_synchronizes_tensor_work() -> None:
    torch_module = _fake_torch()

    result = probe_single_cuda_device(torch_module)

    assert result["available"] is True
    assert result["device_count"] == 1
    assert result["device_name"] == "Fake H200"
    assert result["compute_capability"] == [9, 0]
    assert result["tensor_probe"] == {
        "operation": "matmul_16x16_ones",
        "checksum": 4096.0,
        "synchronized": True,
    }
    assert torch_module.cuda.calls == [
        "init",
        "is_available",
        "device_count",
        ("set_device", 0),
        ("get_device_properties", 0),
        ("synchronize", 0),
    ]
    assert len(torch_module.ones_calls) == 2
    assert torch_module.matmul_calls == 1


@pytest.mark.parametrize(
    ("torch_module", "message"),
    [
        (_fake_torch(init_error=RuntimeError("cuInit failed")), "driver initialization"),
        (_fake_torch(available=False), "is_available"),
        (_fake_torch(device_count=0), "exactly one visible"),
        (_fake_torch(device_count=2), "exactly one visible"),
    ],
)
def test_probe_rejects_driver_or_visibility_failures(
    torch_module: _FakeTorch, message: str
) -> None:
    with pytest.raises(CudaGuardError, match=message):
        probe_single_cuda_device(torch_module)


def test_guard_failure_never_creates_marker_or_executes_workload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = tmp_path / "guard-passed.json"
    exec_calls: list[tuple[str, list[str]]] = []

    def fake_exec(executable: str, arguments) -> None:
        exec_calls.append((executable, list(arguments)))

    return_code = run_guarded_workload(
        marker,
        ["python", "train.py"],
        torch_loader=lambda: _fake_torch(init_error=RuntimeError("driver down")),
        nvidia_smi_collector=_fixed_nvidia_smi,
        exec_fn=fake_exec,
        environ={"CUDA_VISIBLE_DEVICES": "3", "SLURM_STEP_GPUS": "3"},
        hostname_fn=lambda: "bad-node",
    )

    assert return_code == CUDA_GUARD_FAILURE_EXIT
    assert not marker.exists()
    assert exec_calls == []
    failure = json.loads(capsys.readouterr().err)
    assert failure["state"] == "cuda_guard_failed"
    assert failure["hostname"] == "bad-node"
    assert failure["gpu_environment"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert failure["gpu_environment"]["SLURM_STEP_GPUS"] == "3"
    assert "driver down" in failure["error"]


def test_success_installs_private_marker_before_exec_and_records_provenance(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "nested" / "guard-passed.json"
    observed: dict[str, object] = {}

    def fake_exec(executable: str, arguments) -> None:
        observed["executable"] = executable
        observed["arguments"] = list(arguments)
        observed["marker_exists_during_exec"] = marker.is_file()
        observed["marker_mode_during_exec"] = stat.S_IMODE(marker.stat().st_mode)
        observed["payload_during_exec"] = json.loads(marker.read_text(encoding="utf-8"))

    return_code = run_guarded_workload(
        marker,
        ["/environment/bin/python", "-u", "train_eval.py", "--token", "secret"],
        torch_loader=lambda: _fake_torch(),
        nvidia_smi_collector=_fixed_nvidia_smi,
        exec_fn=fake_exec,
        environ={
            "CUDA_VISIBLE_DEVICES": "0",
            "SLURM_JOB_ID": "900001",
            "SLURM_STEP_ID": "7",
            "SLURM_JOB_GPUS": "1,2",
            "SLURM_STEP_GPUS": "1",
            "SLURM_LOCALID": "0",
            "SLURM_PROCID": "0",
            "SLURM_NTASKS": "1",
        },
        hostname_fn=lambda: "good-node",
        now_fn=lambda: datetime(2026, 8, 5, 8, 0, tzinfo=timezone.utc),
    )

    assert return_code == 0
    assert observed["executable"] == "/environment/bin/python"
    assert observed["arguments"] == [
        "/environment/bin/python",
        "-u",
        "train_eval.py",
        "--token",
        "secret",
    ]
    assert observed["marker_exists_during_exec"] is True
    assert observed["marker_mode_during_exec"] == 0o600
    payload = observed["payload_during_exec"]
    assert payload["state"] == "cuda_guard_passed_workload_starting"
    assert payload["created_at"] == "2026-08-05T08:00:00+00:00"
    assert payload["hostname"] == "good-node"
    assert payload["gpu_environment"]["SLURM_JOB_GPUS"] == "1,2"
    assert payload["gpu_environment"]["SLURM_STEP_GPUS"] == "1"
    assert payload["cuda"]["device_count"] == 1
    assert payload["nvidia_smi"]["returncode"] == 0
    assert payload["workload_executable"] == "/environment/bin/python"
    assert "secret" not in marker.read_text(encoding="utf-8")


def test_workload_exec_failure_keeps_marker_and_is_not_a_guard_failure(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "guard-passed.json"

    def failing_exec(executable: str, arguments) -> None:
        assert marker.exists()
        raise FileNotFoundError(executable)

    return_code = run_guarded_workload(
        marker,
        ["missing-workload"],
        torch_loader=lambda: _fake_torch(),
        nvidia_smi_collector=_fixed_nvidia_smi,
        exec_fn=failing_exec,
    )

    assert return_code == WORKLOAD_EXEC_FAILURE_EXIT
    assert marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8"))["state"] == (
        "cuda_guard_passed_workload_starting"
    )


def test_atomic_marker_is_create_only(tmp_path: Path) -> None:
    marker = tmp_path / "guard-passed.json"
    write_atomic_marker(marker, {"schema_version": 1, "value": "first"})

    with pytest.raises(CudaGuardError, match="refusing to replace"):
        write_atomic_marker(marker, {"schema_version": 1, "value": "second"})

    assert json.loads(marker.read_text(encoding="utf-8"))["value"] == "first"
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600


def test_nvidia_smi_is_optional_and_bounded() -> None:
    assert collect_nvidia_smi(which=lambda _: None) == {
        "available": False,
        "reason": "executable_not_found",
    }

    observed: dict[str, object] = {}

    def fake_runner(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="one gpu\n", stderr="")

    result = collect_nvidia_smi(
        which=lambda _: "/usr/bin/nvidia-smi", runner=fake_runner
    )

    assert result["returncode"] == 0
    assert result["stdout"] == "one gpu"
    assert observed["command"][0] == "/usr/bin/nvidia-smi"
    assert observed["kwargs"]["timeout"] == 20
    assert observed["kwargs"]["check"] is False
