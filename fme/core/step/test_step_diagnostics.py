import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.device import get_device
from fme.core.step.step_diagnostics import CORRECTION_DELTAS, StepDiagnostics


def _get_diagnostics(n_samples: int = 2, n_times: int = 3) -> StepDiagnostics:
    return StepDiagnostics(
        delta={
            "a": torch.randn(n_samples, n_times, 4, 5),
            "b": torch.randn(n_samples, n_times, 4, 5),
        }
    )


def test_to_device_to_cpu_preserve_keys_and_values():
    diagnostics = _get_diagnostics()
    on_device = diagnostics.to_device()
    assert set(on_device.delta) == {"a", "b"}
    for tensor in on_device.delta.values():
        assert tensor.device.type == get_device().type
    on_cpu = on_device.to_cpu()
    assert set(on_cpu.delta) == {"a", "b"}
    for name, tensor in on_cpu.delta.items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, diagnostics.delta[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
def test_pin_memory_preserves_keys():
    diagnostics = _get_diagnostics()
    pinned = diagnostics.pin_memory()
    assert set(pinned.delta) == {"a", "b"}
    for tensor in pinned.delta.values():
        assert tensor.is_pinned()


def test_broadcast_ensemble_repeat_interleaves_sample_dim():
    n_samples, n_ensemble = 2, 3
    diagnostics = _get_diagnostics(n_samples=n_samples)
    broadcast = diagnostics.broadcast_ensemble(n_ensemble)
    for name, tensor in broadcast.delta.items():
        assert tensor.shape[0] == n_samples * n_ensemble
        expected = torch.repeat_interleave(diagnostics.delta[name], n_ensemble, dim=0)
        torch.testing.assert_close(tensor, expected)


def test_empty_delta_is_valid():
    diagnostics = StepDiagnostics(delta={})
    assert diagnostics.to_device().delta == {}
    assert diagnostics.to_cpu().delta == {}
    assert diagnostics.pin_memory().delta == {}
    assert diagnostics.broadcast_ensemble(3).delta == {}
    assert diagnostics.sample_dim_size() is None
    datasets = diagnostics.to_datasets(
        xr.DataArray(np.zeros((2, 3)), dims=["sample", "time"])
    )
    assert set(datasets) == {CORRECTION_DELTAS}
    assert len(datasets[CORRECTION_DELTAS].data_vars) == 0


def test_sample_dim_size():
    assert _get_diagnostics(n_samples=4).sample_dim_size() == 4


def test_to_datasets_round_trips_values_with_time():
    n_samples, n_times = 2, 3
    diagnostics = _get_diagnostics(n_samples=n_samples, n_times=n_times)
    time = xr.DataArray(
        np.arange(n_samples * n_times).reshape(n_samples, n_times),
        dims=["sample", "time"],
    )
    datasets = diagnostics.to_datasets(time)
    assert set(datasets) == {CORRECTION_DELTAS}
    ds = datasets[CORRECTION_DELTAS]
    assert set(ds.data_vars) == {"a", "b"}
    for name in ("a", "b"):
        assert ds[name].dims[:2] == ("sample", "time")
        np.testing.assert_allclose(
            ds[name].values, diagnostics.delta[name].cpu().numpy()
        )
    np.testing.assert_array_equal(ds["valid_time"].values, time.values)
