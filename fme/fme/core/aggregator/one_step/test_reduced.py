from fme.core.aggregator.one_step.reduced import MeanAggregator
from fme.core.testing import mock_distributed
from fme.core.device import get_device
import torch


def test_mean_metrics_call_distributed():
    """
    All reduced metrics should be reduced across processes using Distributed.

    This tests that functionality by modifying the Distributed singleton.
    """
    with mock_distributed(-1.0) as mock:
        agg = MeanAggregator()
        sample_data = {"a": torch.ones([2, 3, 4, 4], device=get_device())}
        agg.record_batch(1.0, sample_data, sample_data, sample_data, sample_data)
        logs = agg.get_logs(label="metrics")
        assert logs["metrics/loss"] == -1.0
        assert logs["metrics/l1/a"] == -1.0
        assert logs["metrics/weighted_rmse/a"] == -1.0
        assert logs["metrics/weighted_grad_mag_percent_diff/a"] == -1.0
        assert mock.reduce_called
