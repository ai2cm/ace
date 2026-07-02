import numpy as np
import torch

from fme.ace.aggregator.inference.data import InferenceBatchData, make_dummy_time
from fme.ace.aggregator.inference.time_mean import (
    TimeMeanAggregator,
    TimeMeanEvaluatorAggregator,
)
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations


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


def test_time_mean_evaluator_per_cell_masking():
    """Per-cell valid-only time-mean + never-valid cell dropping (A-eval).

    target_mask marks cell (0,0) never-valid, (0,1) valid only at t=0, and the
    bottom row valid at both steps. The masked time-mean must use only valid
    (sample, time) entries per cell, and the never-valid cell must drop from the
    area-weighted rmse/bias.
    """
    area_weights = torch.ones(2, 2).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights), horizontal_dims=["lat", "lon"]
    )
    # (sample=1, time=2, lat=2, lon=2)
    target = torch.tensor(
        [[[[10.0, 20.0], [30.0, 40.0]], [[11.0, 777.0], [31.0, 41.0]]]]
    ).to(get_device())  # 777 sits under an invalid mask, must not leak
    gen = target + 2.0
    mask = torch.tensor([[[[0.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]]]).to(
        get_device()
    )
    agg.record_batch(
        InferenceBatchData(
            prediction={"ta1000": gen},
            prediction_norm={"ta1000": gen},
            target={"ta1000": target},
            target_norm={"ta1000": target},
            target_mask={"ta1000": mask},
            time=make_dummy_time(1, 2),
            i_time_start=1,  # count both timesteps (no initial-condition drop)
        )
    )
    pair = agg._get_target_gen_pairs()[0]
    tmean = pair.target.cpu().numpy()
    # (0,1) uses only t=0 (20, not blended with the masked 777); bottom row avgs
    np.testing.assert_allclose(tmean[0, 1], 20.0)
    np.testing.assert_allclose(tmean[1, 0], 30.5)
    np.testing.assert_allclose(tmean[1, 1], 40.5)
    # gen is target+2 at every valid cell -> masked rmse over the 3 valid cells
    # is exactly 2.0; the never-valid (0,0) is dropped (else it would be
    # sqrt(3) by inflating the denominator).
    np.testing.assert_allclose(pair.rmse(), 2.0, rtol=1e-6)
    np.testing.assert_allclose(pair.weighted_mean_bias(), 2.0, rtol=1e-6)
