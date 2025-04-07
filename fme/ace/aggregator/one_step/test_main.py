import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.aggregator.one_step import OneStepAggregator
from fme.ace.stepper import TrainOutput
from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.typing_ import EnsembleTensorDict


def test_labels_exist():
    batch_size = 10
    n_ensemble = 2
    n_time = 3
    nx, ny = 2, 2
    loss = 1.0
    lat_lon_coordinates = LatLonCoordinates(torch.arange(nx), torch.arange(ny))
    # keep area weights ones for simplicity
    lat_lon_coordinates._area_weights = torch.ones(nx, ny)
    agg = OneStepAggregator(
        lat_lon_coordinates.to(device=get_device()), save_diagnostics=False
    )
    target_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, 1, n_time, nx, ny, device=get_device())},
    )
    gen_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, n_ensemble, n_time, nx, ny, device=get_device())},
    )
    agg.record_batch(
        batch=TrainOutput(
            metrics={"loss": loss},
            target_data=target_data,
            gen_data=gen_data,
            time=xr.DataArray(np.zeros((batch_size, n_time)), dims=["sample", "time"]),
            normalize=lambda x: x,
        ),
    )
    logs = agg.get_logs(label="test")
    expected_keys = [
        "test/mean/loss",
        "test/mean/weighted_rmse/a",
        "test/mean/weighted_bias/a",
        "test/mean/weighted_grad_mag_percent_diff/a",
        "test/snapshot/image-full-field/a",
        "test/snapshot/image-residual/a",
        "test/snapshot/image-error/a",
        "test/mean_map/image-full-field/a",
        "test/mean_map/image-error/a",
        "test/power_spectrum/positive_norm_bias/a",
        "test/power_spectrum/negative_norm_bias/a",
        "test/power_spectrum/mean_abs_norm_bias/a",
        "test/power_spectrum/smallest_scale_norm_bias/a",
        "test/crps/a",
        "test/crps/mean_map/a",
        "test/ssr_bias/a",
        "test/ssr_bias/mean_map/a",
    ]
    assert set(logs.keys()) == set(expected_keys)


def test_aggregator_raises_on_no_data():
    """
    Basic test the aggregator combines loss correctly
    with multiple batches and no distributed training.
    """
    nx, ny = 2, 2
    lat_lon_coordinates = LatLonCoordinates(torch.arange(nx), torch.arange(ny))
    # keep area weights ones for simplicity
    lat_lon_coordinates._area_weights = torch.ones(nx, ny)
    agg = OneStepAggregator(
        lat_lon_coordinates.to(device=get_device()), save_diagnostics=False
    )
    with pytest.raises(ValueError) as excinfo:
        agg.record_batch(
            batch=TrainOutput(
                metrics={"loss": 1.0},
                target_data=EnsembleTensorDict({}),
                gen_data=EnsembleTensorDict({}),
                time=xr.DataArray(np.zeros((0, 0)), dims=["sample", "time"]),
                normalize=lambda x: x,
            ),
        )
        # check that the raised exception contains the right substring
        assert "No data" in str(excinfo.value)


@pytest.mark.parametrize(
    "epoch", [pytest.param(None, id="no epoch"), pytest.param(2, id="epoch 2")]
)
def test_flush_diagnostics(tmpdir, epoch):
    nx, ny, batch_size, n_ensemble, n_time = 3, 3, 10, 2, 3
    lat_lon_coordinates = LatLonCoordinates(torch.arange(nx), torch.arange(ny))
    agg = OneStepAggregator(
        lat_lon_coordinates.to(device=get_device()), output_dir=(tmpdir / "val")
    )
    target_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, 1, n_time, nx, ny, device=get_device())}
    )
    gen_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, n_ensemble, n_time, nx, ny, device=get_device())}
    )
    time = xr.DataArray(np.zeros((batch_size, n_time)), dims=["sample", "time"])
    agg.record_batch(
        batch=TrainOutput(
            metrics={"loss": 1.0},
            target_data=target_data,
            gen_data=gen_data,
            time=time,
            normalize=lambda x: x,
        ),
    )
    if epoch is not None:
        agg.flush_diagnostics(subdir=f"epoch_{epoch:04d}")
        output_dir = tmpdir / "val" / f"epoch_{epoch:04d}"
    else:
        agg.flush_diagnostics()
        output_dir = tmpdir / "val"
    expected_files = [
        "mean",
        "snapshot",
        "mean_map",
    ]
    for file in expected_files:
        assert (output_dir / f"{file}_diagnostics.nc").exists()


def test_agg_raises_without_output_dir():
    lat_lon_coordinates = LatLonCoordinates(torch.arange(2), torch.arange(2))
    with pytest.raises(
        ValueError, match="Output directory must be set to save diagnostics"
    ):
        OneStepAggregator(lat_lon_coordinates, save_diagnostics=True, output_dir=None)
