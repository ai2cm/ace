import torch

from fme.ace.aggregator.inference.reduced import MeanAggregator
from fme.ace.aggregator.inference.time_mean import TimeMeanEvaluatorAggregator
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.testing import mock_distributed


def test_mean_metrics_call_distributed():
    """
    All mean metrics should be reduced across processes using Distributed.

    This tests that functionality by modifying the Distributed singleton.
    """
    with mock_distributed(-1.0):
        data_a = torch.ones([2, 3, 4, 4], device=get_device())
        area_weights = torch.ones(1).to(get_device())
        agg = MeanAggregator(
            LatLonOperations(area_weights), target="denorm", n_timesteps=3
        )
        sample_data = {"a": data_a}
        agg.record_batch(sample_data, sample_data, sample_data, sample_data)
        logs = agg.get_logs(label="metrics")
        table = logs["metrics/series"]
        # assert all data past the first column in the WandB table is -1
        assert all([all(item == -1 for item in row[1][1:]) for row in table.iterrows()])


def test_time_mean_metrics_call_distributed():
    """
    All time-mean metrics should be reduced across processes using Distributed.

    This tests that functionality by modifying the Distributed singleton.
    """
    torch.manual_seed(0)
    with mock_distributed(0.0) as mock:
        area_weights = torch.ones(1).to(get_device())
        agg = TimeMeanEvaluatorAggregator(
            LatLonOperations(area_weights), horizontal_dims=["lat", "lon"]
        )
        target_data = {"a": torch.ones([2, 3, 4, 4], device=get_device())}
        gen_data = {"a": torch.randn([2, 3, 4, 4], device=get_device())}
        agg.record_batch(
            target_data=target_data,
            gen_data=gen_data,
            target_data_norm=target_data,
            gen_data_norm=gen_data,
        )
        logs = agg.get_logs(label="metrics")
        # the reduction happens on the time-means, so the gen and target data should
        # be filled identically and all errors will be zero, even though we gave them
        # different data above.
        assert logs["metrics/rmse/a"] == 0.0
        assert logs["metrics/bias/a"] == 0.0
        assert mock.reduce_called
