import numpy as np
import pytest
import torch

from fme.ace.aggregator.one_step.reduced import MeanAggregator
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.testing import mock_distributed


def test_mean_metrics_call_distributed():
    """
    All reduced metrics should be reduced across processes using Distributed.

    This tests that functionality by modifying the Distributed singleton.
    """
    with mock_distributed(-1.0) as mock:
        area_weights = torch.ones([4]).to(get_device())
        agg = MeanAggregator(LatLonOperations(area_weights))
        sample_data = {"a": torch.ones([2, 3, 4, 4], device=get_device())}
        agg.record_batch(
            loss=1.0,
            target_data=sample_data,
            gen_data=sample_data,
            target_data_norm=sample_data,
            gen_data_norm=sample_data,
        )
        logs = agg.get_logs(label="metrics")
        assert logs["metrics/loss"] == -1.0
        assert logs["metrics/weighted_rmse/a"] == -1.0
        assert logs["metrics/weighted_grad_mag_percent_diff/a"] == -1.0
        assert mock.reduce_called


def test_i_time_start_gets_correct_time_one_step_windows():
    # while this directly tests the "mean" result, this is really a test that
    # the data from the correct timestep is piped into the aggregator.
    target_time = 3
    area_weights = torch.ones([4]).to(get_device())
    agg = MeanAggregator(LatLonOperations(area_weights), target_time=target_time)
    target_data = {"a": torch.zeros([2, 1, 4, 4], device=get_device())}
    for i in range(5):
        sample_data = {
            "a": torch.full([2, 1, 4, 4], fill_value=float(i), device=get_device())
        }
        agg.record_batch(
            loss=1.0,
            target_data=target_data,
            gen_data=sample_data,
            target_data_norm=target_data,
            gen_data_norm=sample_data,
            i_time_start=i,
        )
    logs = agg.get_logs(label="metrics")
    np.testing.assert_allclose(logs["metrics/weighted_bias/a"], target_time)


@pytest.mark.parametrize(
    "window_len, n_windows, target_time",
    [
        pytest.param(3, 1, 1, id="single_window"),
        pytest.param(3, 2, 1, id="first_of_two_windows"),
        pytest.param(3, 2, 4, id="second_of_two_windows"),
    ],
)
def test_i_time_start_gets_correct_time_longer_windows(
    window_len: int, n_windows: int, target_time: int
):
    # while this directly tests the "mean" result, this is really a test that
    # the data from the correct timestep is piped into the aggregator.
    area_weights = torch.ones([4]).to(get_device())
    agg = MeanAggregator(LatLonOperations(area_weights), target_time=target_time)
    target_data = {"a": torch.zeros([2, window_len, 4, 4], device=get_device())}
    i_start = 0
    for i in range(n_windows):
        sample_data = {"a": torch.zeros([2, window_len, 4, 4], device=get_device())}
        for i in range(window_len):
            sample_data["a"][..., i, :, :] = float(i_start + i)
        agg.record_batch(
            loss=1.0,
            target_data=target_data,
            gen_data=sample_data,
            target_data_norm=target_data,
            gen_data_norm=sample_data,
            i_time_start=i_start,
        )
        i_start += window_len
    logs = agg.get_logs(label="metrics")
    np.testing.assert_allclose(
        float(logs["metrics/weighted_bias/a"]), float(target_time), rtol=1e-5
    )


def test_loss():
    """
    Basic test the aggregator combines loss correctly
    with multiple batches and no distributed training.
    """
    torch.manual_seed(0)
    example_data = {
        "a": torch.randn(1, 2, 5, 5, device=get_device()),
    }
    area_weights = torch.ones(1).to(get_device())
    aggregator = MeanAggregator(LatLonOperations(area_weights))
    aggregator.record_batch(
        loss=1.0,
        target_data=example_data,
        gen_data=example_data,
        target_data_norm=example_data,
        gen_data_norm=example_data,
    )
    aggregator.record_batch(
        loss=2.0,
        target_data=example_data,
        gen_data=example_data,
        target_data_norm=example_data,
        gen_data_norm=example_data,
    )
    logs = aggregator.get_logs(label="metrics")
    assert logs["metrics/loss"] == 1.5
    aggregator.record_batch(
        loss=3.0,
        target_data=example_data,
        gen_data=example_data,
        target_data_norm=example_data,
        gen_data_norm=example_data,
    )
    logs = aggregator.get_logs(label="metrics")
    assert logs["metrics/loss"] == 2.0
