import pytest
import torch

from fme.ace.aggregator.loss_metrics import PerStepLossAggregator
from fme.core.device import get_device
from fme.core.distributed import Distributed


def test_per_step_loss_aggregator_means_over_uneven_key_sets():
    aggregator = PerStepLossAggregator()
    aggregator.record(
        {
            "loss_step_0": torch.tensor(1.0, device=get_device()),
            "loss_step_1": torch.tensor(3.0, device=get_device()),
        }
    )
    aggregator.record({"loss_step_0": torch.tensor(2.0, device=get_device())})
    logs = aggregator.get_logs("val")
    assert logs == {
        "val/mean/loss_step_0": pytest.approx(1.5),
        "val/mean/loss_step_1": pytest.approx(3.0),
    }


def test_per_step_loss_aggregator_no_records_gives_no_logs():
    assert PerStepLossAggregator().get_logs("val") == {}


@pytest.mark.parallel
@pytest.mark.parametrize("empty_rank", [None, 0])
def test_per_step_loss_aggregator_mismatched_ranks(empty_rank):
    """Ranks recording different key sets reduce without hanging.

    Data-parallel rank r records r + 1 batches, each covering steps 0..r with
    value r + s for step s, so ranks disagree on both the per-step key sets
    and the per-key record counts; with ``empty_rank`` set, that rank records
    nothing at all. Means must be count-weighted over all recording ranks.
    Reductions run over data-parallel replicas (spatial ranks shard a single
    replica and record identical metrics in practice), so expectations are in
    terms of data-parallel ranks.
    """
    dist = Distributed.get_instance()
    rank = dist.data_parallel_rank
    world_size = dist.total_data_parallel_ranks
    aggregator = PerStepLossAggregator()
    if rank != empty_rank:
        for _ in range(rank + 1):
            aggregator.record(
                {
                    f"loss_step_{step}": torch.tensor(
                        float(rank + step), device=get_device()
                    )
                    for step in range(rank + 1)
                }
            )
    logs = aggregator.get_logs("val")

    recording_ranks = [r for r in range(world_size) if r != empty_rank]
    expected: dict[str, float] = {}
    max_step = max(recording_ranks) if recording_ranks else -1
    for step in range(max_step + 1):
        contributors = [r for r in recording_ranks if r >= step]
        total = sum((r + 1) * (r + step) for r in contributors)
        count = sum(r + 1 for r in contributors)
        expected[f"val/mean/loss_step_{step}"] = total / count
    assert logs == pytest.approx(expected)
