import numpy as np
import torch

from fme.ace.aggregator.one_step.map import MapAggregator
from fme.core.device import get_device
from fme.core.distributed import Distributed

DIMS = ["lat", "lon"]


def _record(agg: MapAggregator, target: torch.Tensor, gen: torch.Tensor):
    agg.record_batch(
        loss=1.0,
        target_data={"a": target},
        gen_data={"a": gen},
        target_data_norm={"a": target},
        gen_data_norm={"a": gen},
    )


def test_getter_idempotent_under_inplace_reduce(monkeypatch):
    # Under real distributed training, reduce_mean's all_reduce mutates its argument
    # in place; single-process reduce is a no-op, so simulate the in-place behavior
    # here. _get_data is called once for the netCDF dataset and once for the wandb
    # logs, so it must be idempotent. Without cloning, the 2nd call re-reduces the
    # accumulators and inflates the maps by `total_ranks`.
    world_size = 4

    def mutating_reduce_mean(tensor):
        tensor.mul_(world_size)  # emulate in-place all_reduce over identical ranks
        return tensor / world_size

    monkeypatch.setattr(Distributed.get_instance(), "reduce_mean", mutating_reduce_mean)

    nx, ny = 3, 4
    agg = MapAggregator(DIMS)
    target = 250.0 + torch.randn(8, 3, nx, ny, device=get_device())
    gen = target + 0.1 * torch.randn(8, 3, nx, ny, device=get_device())
    _record(agg, target=target, gen=gen)

    first = agg.get_dataset()
    second = agg.get_dataset()
    for var in ("gen_map-a", "bias_map-a"):
        np.testing.assert_array_equal(first[var].values, second[var].values)
    # gen_map is a mean field (~250), not inflated by world_size (~1000)
    assert 200.0 < float(np.nanmean(first["gen_map-a"].values)) < 300.0
