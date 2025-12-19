import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.aggregator.train import TrainAggregator, TrainAggregatorConfig
from fme.ace.stepper.single_module import TrainOutput
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.typing_ import EnsembleTensorDict


@pytest.mark.parametrize(
    "config, expected_keys",
    [
        (
            TrainAggregatorConfig(spherical_power_spectrum=False, weighted_rmse=False),
            ["test/mean/loss"],
        ),
        (
            TrainAggregatorConfig(),
            [
                "test/power_spectrum/positive_norm_bias/a",
                "test/power_spectrum/negative_norm_bias/a",
                "test/power_spectrum/mean_abs_norm_bias/a",
                "test/power_spectrum/smallest_scale_norm_bias/a",
                "test/mean/weighted_rmse/a",
                "test/mean/loss",
            ],
        ),
    ],
)
def test_labels_exist(config: TrainAggregatorConfig, expected_keys: list[str]):
    batch_size = 10
    n_ensemble = 2
    n_time = 3
    nx, ny = 2, 2
    loss = 1.0
    device = get_device()
    gridded_operations = LatLonOperations(
        area_weights=torch.ones(nx, ny, device=device)
    )
    agg = TrainAggregator(config=config, operations=gridded_operations)
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
    assert set(logs.keys()) == set(expected_keys)
    assert not np.isnan(float(logs["test/mean/loss"]))


@pytest.mark.parametrize(
    "config",
    [
        TrainAggregatorConfig(spherical_power_spectrum=False, weighted_rmse=False),
        TrainAggregatorConfig(),
    ],
)
def test_aggregator_gets_logs_with_no_batches(config: TrainAggregatorConfig):
    ny, nx = 4, 8
    device = get_device()
    gridded_operations = LatLonOperations(
        area_weights=torch.ones(ny, nx, device=device)
    )
    agg = TrainAggregator(config=config, operations=gridded_operations)
    logs = agg.get_logs(label="test")
    assert np.isnan(logs.pop("test/mean/loss"))
    assert logs == {}
