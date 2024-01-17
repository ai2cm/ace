import torch

from fme.core.aggregator.inference.time_mean import TimeMeanAggregator
from fme.core.device import get_device


def test_rmse_of_time_mean_all_channels():
    torch.manual_seed(0)
    area_weights = torch.ones(1).to(get_device())
    agg = TimeMeanAggregator(area_weights, target="norm")
    target_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
    }
    gen_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
    }
    agg.record_batch(
        loss=1.0,
        target_data=target_data_norm,
        gen_data=gen_data_norm,
        target_data_norm=target_data_norm,
        gen_data_norm=gen_data_norm,
    )
    logs = agg.get_logs(label="metrics")
    assert logs["metrics/rmse/all_channels"] == 1.5
