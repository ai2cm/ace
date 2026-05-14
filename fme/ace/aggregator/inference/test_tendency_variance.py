import torch

from fme.ace.aggregator.inference.data import InferenceBatchData, make_dummy_time
from fme.ace.aggregator.inference.tendency_variance import (
    TendencyVarianceRatioAggregator,
    TendencyVarianceRatioMetricConfig,
)
from fme.core.device import get_device


def test_aggregator_produces_ratio():
    torch.manual_seed(0)
    agg = TendencyVarianceRatioAggregator()
    n_sample, n_time, h, w = 4, 5, 6, 6
    device = get_device()

    target = {"a": torch.randn(n_sample, n_time, h, w, device=device)}
    gen = {"a": 2.0 * target["a"]}

    batch = InferenceBatchData(
        prediction=gen,
        target=target,
        time=make_dummy_time(n_sample, n_time),
        i_time_start=0,
    )
    agg.record_batch(batch)
    logs = agg.get_logs("test")
    assert "test/tendency_variance_ratio/a" in logs
    assert abs(logs["test/tendency_variance_ratio/a"] - 4.0) < 0.2


def test_aggregator_skips_without_target():
    agg = TendencyVarianceRatioAggregator()
    batch = InferenceBatchData(
        prediction={"a": torch.randn(2, 3, 4, 4)},
        time=make_dummy_time(2, 3),
        i_time_start=0,
    )
    agg.record_batch(batch)
    assert agg.get_logs("test") == {}


def test_metric_config_build():
    import datetime

    import numpy as np
    import xarray as xr

    from fme.ace.aggregator.inference.build_context import MetricBuildContext
    from fme.core.coordinates import LatLonCoordinates
    from fme.core.gridded_ops import LatLonOperations

    config = TendencyVarianceRatioMetricConfig()
    assert config.get_name() == "tendency_variance"

    ops = LatLonOperations(torch.ones(4, 8))
    coords = LatLonCoordinates(
        lat=xr.DataArray(np.linspace(-90, 90, 4), dims=["lat"]),
        lon=xr.DataArray(np.linspace(0, 360, 8, endpoint=False), dims=["lon"]),
    )
    ctx = MetricBuildContext(
        ops=ops,
        horizontal_coordinates=coords,
        n_timesteps=10,
        n_ic_steps=1,
        timestep=datetime.timedelta(hours=6),
        variable_metadata=None,
        channel_mean_names=None,
        monthly_reference_data=None,
        time_mean_reference_data=None,
        initial_time=xr.DataArray([0]),
    )
    result = config.build(ctx)
    assert result.aggregator is not None
