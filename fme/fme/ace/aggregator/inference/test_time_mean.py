import numpy as np
import torch

from fme.ace.aggregator.inference.time_mean import (
    TimeMeanAggregator,
    TimeMeanEvaluatorAggregator,
)
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations


def test_rmse_of_time_mean_all_channels():
    torch.manual_seed(0)
    area_weights = torch.ones(1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="norm",
    )
    target_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
    }
    gen_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
    }
    agg.record_batch(
        target_data=target_data_norm,
        gen_data=gen_data_norm,
        target_data_norm=target_data_norm,
        gen_data_norm=gen_data_norm,
    )
    logs = agg.get_logs(label="time_mean_norm")
    assert logs["time_mean_norm/rmse/a"] == 1.0
    assert logs["time_mean_norm/rmse/b"] == 2.0
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.5


def test_custom_channel_mean_names():
    torch.manual_seed(0)
    area_weights = torch.ones(1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="norm",
        channel_mean_names=["a"],
    )
    target_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
    }
    gen_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
    }
    agg.record_batch(
        target_data=target_data_norm,
        gen_data=gen_data_norm,
        target_data_norm=target_data_norm,
        gen_data_norm=gen_data_norm,
    )
    logs = agg.get_logs(label="time_mean_norm")
    assert logs["time_mean_norm/rmse/a"] == 1.0
    assert logs["time_mean_norm/rmse/b"] == 2.0
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.0


def test_mean_all_channels_not_in_denorm():
    area_weights = torch.ones(1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="denorm",
    )
    target_data = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
    }
    gen_data = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
    }
    agg.record_batch(
        target_data=target_data,
        gen_data=gen_data,
        target_data_norm=target_data,
        gen_data_norm=gen_data,
    )
    logs = agg.get_logs(label="time_mean")
    assert "time_mean/rmse/channel_mean" not in list(logs.keys())
    ds = agg.get_dataset()
    assert "bias_map-a" in ds
    assert np.all(ds["bias_map-a"].values == 1.0)


def test_bias_values():
    area_weights = torch.ones(1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="denorm",
    )
    # use constant values so area-weighting doesn't matter
    target_data = {
        "a": (torch.rand(1) * torch.ones(size=[2, 3, 4, 5])).to(device=get_device()),
    }
    gen_data = {
        "a": (torch.rand(1) * torch.ones(size=[2, 3, 4, 5])).to(device=get_device()),
    }
    agg.record_batch(
        target_data=target_data,
        gen_data=gen_data,
        target_data_norm=target_data,
        gen_data_norm=gen_data,
    )
    ds = agg.get_dataset()
    assert "bias_map-a" in ds
    np.testing.assert_array_equal(
        ds["bias_map-a"].values,
        (
            gen_data["a"].cpu().numpy().mean(axis=(0, 1))
            - target_data["a"].cpu().numpy().mean(axis=(0, 1))
        ),
    )
    assert "gen_map-a" in ds
    np.testing.assert_array_equal(
        ds["gen_map-a"].values,
        (gen_data["a"].cpu().numpy().mean(axis=(0, 1))),
    )


def test_aggregator_mean_values():
    area_weights = torch.ones(1).to(get_device())
    agg = TimeMeanAggregator(LatLonOperations(area_weights))
    # use constant values so area-weighting doesn't matter
    data = {
        "a": (torch.rand(1) * torch.ones(size=[2, 3, 4, 5])).to(device=get_device()),
    }
    agg.record_batch(
        data=data,
        i_time_start=0,
    )
    ds = agg.get_dataset()
    assert "gen_map-a" in ds
    np.testing.assert_array_equal(
        ds["gen_map-a"].values,
        (data["a"].cpu().numpy().mean(axis=(0, 1))),
    )
