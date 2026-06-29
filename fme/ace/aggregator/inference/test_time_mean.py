import numpy as np
import torch

from fme.ace.aggregator.inference.data import InferenceBatchData, make_dummy_time
from fme.ace.aggregator.inference.spectrum import get_spectrum_bias_metrics
from fme.ace.aggregator.inference.time_mean import (
    TimeMeanAggregator,
    TimeMeanEvaluatorAggregator,
)
from fme.core.device import get_device
from fme.core.fill import SmoothFloodFill
from fme.core.gridded_ops import LatLonOperations
from fme.core.metrics import spherical_power_spectrum


def get_gridded_operations(nlat: int, nlon: int):
    return LatLonOperations(torch.ones(nlat, nlon, device=get_device()))


SPECTRUM_BIAS_SUFFIXES = [
    "smallest_scale_norm_bias",
    "positive_norm_bias",
    "negative_norm_bias",
    "mean_abs_norm_bias",
]


def test_rmse_of_time_mean_all_channels():
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
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
        InferenceBatchData(
            prediction=gen_data_norm,
            prediction_norm=gen_data_norm,
            target=target_data_norm,
            target_norm=target_data_norm,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean_norm")
    assert logs["time_mean_norm/rmse/a"] == 1.0
    assert logs["time_mean_norm/rmse/b"] == 2.0
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.5


def test_channel_mean_excludes_all_nan_target_channels():
    """A variable whose target is entirely NaN (e.g. filled by
    allow_missing_variables) has a NaN RMSE and is excluded from the channel
    mean rather than poisoning it."""
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="norm",
    )
    target_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
        # "c" is missing from the data: entirely-NaN target.
        "c": torch.full([2, 3, 4, 4], torch.nan, device=get_device()),
    }
    gen_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
        "c": torch.ones([2, 3, 4, 4], device=get_device()),
    }
    agg.record_batch(
        InferenceBatchData(
            prediction=gen_data_norm,
            prediction_norm=gen_data_norm,
            target=target_data_norm,
            target_norm=target_data_norm,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean_norm")
    # "c" is still recorded per-variable as NaN...
    assert np.isnan(logs["time_mean_norm/rmse/c"])
    # ...but excluded from channel_mean: mean of "a" (1) and "b" (2).
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.5


def test_custom_channel_mean_names():
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
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
        InferenceBatchData(
            prediction=gen_data_norm,
            prediction_norm=gen_data_norm,
            target=target_data_norm,
            target_norm=target_data_norm,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean_norm")
    assert logs["time_mean_norm/rmse/a"] == 1.0
    assert logs["time_mean_norm/rmse/b"] == 2.0
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.0


def test_mean_all_channels_not_in_denorm():
    area_weights = torch.ones(1, 1).to(get_device())
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
        InferenceBatchData(
            prediction=gen_data,
            prediction_norm=gen_data,
            target=target_data,
            target_norm=target_data,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean")
    assert "time_mean/rmse/channel_mean" not in list(logs.keys())
    ds = agg.get_dataset()
    assert "bias_map-a" in ds
    assert np.all(ds["bias_map-a"].values == 1.0)


def test_bias_values():
    area_weights = torch.ones(1, 1).to(get_device())
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
        InferenceBatchData(
            prediction=gen_data,
            prediction_norm=gen_data,
            target=target_data,
            target_norm=target_data,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
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


def test_log_variables_does_not_affect_channel_mean():
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
    target_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
    }
    gen_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
    }
    batch = InferenceBatchData(
        prediction=gen_data_norm,
        prediction_norm=gen_data_norm,
        target=target_data_norm,
        target_norm=target_data_norm,
        time=make_dummy_time(2, 3),
        i_time_start=0,
    )
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="norm",
        log_variables=frozenset(["a"]),
    )
    agg.record_batch(batch)
    logs = agg.get_logs(label="time_mean_norm")
    assert logs["time_mean_norm/rmse/a"] == 1.0
    assert "time_mean_norm/rmse/b" not in logs
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.5


def test_empty_log_variables_still_computes_channel_mean():
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="norm",
        log_variables=frozenset(),
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
        InferenceBatchData(
            prediction=gen_data_norm,
            prediction_norm=gen_data_norm,
            target=target_data_norm,
            target_norm=target_data_norm,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean_norm")
    assert "time_mean_norm/rmse/a" not in logs
    assert "time_mean_norm/rmse/b" not in logs
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.5


def test_log_variables_with_channel_mean_names():
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="norm",
        channel_mean_names=["a"],
        log_variables=frozenset(["b"]),
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
        InferenceBatchData(
            prediction=gen_data_norm,
            prediction_norm=gen_data_norm,
            target=target_data_norm,
            target_norm=target_data_norm,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean_norm")
    assert "time_mean_norm/rmse/a" not in logs
    assert logs["time_mean_norm/rmse/b"] == 2.0
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.0


def test_log_variables_filters_dataset():
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="denorm",
        log_variables=frozenset(["a"]),
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
        InferenceBatchData(
            prediction=gen_data,
            prediction_norm=gen_data,
            target=target_data,
            target_norm=target_data,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    ds = agg.get_dataset()
    assert "bias_map-a" in ds
    assert "gen_map-a" in ds
    assert "bias_map-b" not in ds
    assert "gen_map-b" not in ds


def test_aggregator_mean_values():
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanAggregator(LatLonOperations(area_weights))
    # use constant values so area-weighting doesn't matter
    data = {
        "a": (torch.rand(1) * torch.ones(size=[2, 3, 4, 5])).to(device=get_device()),
    }
    agg.record_batch(
        InferenceBatchData(
            prediction=data,
            prediction_norm=data,
            target=None,
            target_norm=None,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    ds = agg.get_dataset()
    assert "gen_map-a" in ds
    np.testing.assert_allclose(
        ds["gen_map-a"].values,
        (data["a"].cpu().numpy().mean(axis=(0, 1))),
    )


def test_time_mean_spectrum_bias_metrics():
    torch.manual_seed(0)
    nlat, nlon = 8, 16
    ops = get_gridded_operations(nlat, nlon)
    agg = TimeMeanEvaluatorAggregator(
        ops,
        horizontal_dims=["lat", "lon"],
        target="denorm",
    )
    gen_data = {"a": torch.randn(2, 3, nlat, nlon, device=get_device())}
    target_data = {"a": torch.randn(2, 3, nlat, nlon, device=get_device())}
    agg.record_batch(
        InferenceBatchData(
            prediction=gen_data,
            prediction_norm=gen_data,
            target=target_data,
            target_norm=target_data,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean")

    for suffix in SPECTRUM_BIAS_SUFFIXES:
        assert f"time_mean/{suffix}/a" in logs

    sht = ops.get_real_sht()
    nan_fill = SmoothFloodFill(num_steps=4)
    gen_mean = agg._gen_agg.get_data()["a"]
    target_mean = agg._target_agg.get_data()["a"]
    expected = get_spectrum_bias_metrics(
        spherical_power_spectrum(nan_fill(gen_mean, "a"), sht),
        spherical_power_spectrum(nan_fill(target_mean, "a"), sht),
    )
    for suffix, value in expected.items():
        torch.testing.assert_close(logs[f"time_mean/{suffix}/a"], value)


def test_time_mean_spectrum_bias_not_emitted_for_norm():
    torch.manual_seed(0)
    nlat, nlon = 8, 16
    agg = TimeMeanEvaluatorAggregator(
        get_gridded_operations(nlat, nlon),
        horizontal_dims=["lat", "lon"],
        target="norm",
    )
    gen_data = {"a": torch.randn(2, 3, nlat, nlon, device=get_device())}
    target_data = {"a": torch.randn(2, 3, nlat, nlon, device=get_device())}
    agg.record_batch(
        InferenceBatchData(
            prediction=gen_data,
            prediction_norm=gen_data,
            target=target_data,
            target_norm=target_data,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean_norm")
    assert not any(suffix in key for key in logs for suffix in SPECTRUM_BIAS_SUFFIXES)


def test_time_mean_spectrum_bias_report_spectrum_bias_false():
    torch.manual_seed(0)
    nlat, nlon = 8, 16
    agg = TimeMeanEvaluatorAggregator(
        get_gridded_operations(nlat, nlon),
        horizontal_dims=["lat", "lon"],
        target="denorm",
        report_spectrum_bias=False,
    )
    gen_data = {"a": torch.randn(2, 3, nlat, nlon, device=get_device())}
    target_data = {"a": torch.randn(2, 3, nlat, nlon, device=get_device())}
    agg.record_batch(
        InferenceBatchData(
            prediction=gen_data,
            prediction_norm=gen_data,
            target=target_data,
            target_norm=target_data,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean")
    assert not any(suffix in key for key in logs for suffix in SPECTRUM_BIAS_SUFFIXES)
