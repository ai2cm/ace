import pytest
import torch
from torch import nn

import fme
from fme.core.distributed import Distributed
from fme.core.optimization import OptimizationConfig
from fme.core.scheduler import SchedulerConfig


@pytest.mark.parallel
def test_non_finite_loss_on_one_data_rank_skips_step_on_all_ranks():
    """A non-finite loss on data-parallel rank 0 only must make every rank skip
    the optimizer step, so the ranks never disagree about the weights.

    Spatial (model-parallel) ranks share a reduced loss, so the NaN is placed on
    every rank of data-parallel group 0.
    """
    dist = Distributed.get_instance()
    torch.manual_seed(0)
    model = nn.Linear(2, 1).to(fme.get_device())
    optimization = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.1,
        scheduler=SchedulerConfig(),
        max_consecutive_non_finite_losses=1,
    ).build(nn.ModuleList([model]), max_epochs=1)
    x = torch.ones(1, 2).to(fme.get_device())
    before = [p.detach().clone() for p in model.parameters()]
    scale = float("nan") if dist.data_parallel_rank == 0 else 1.0
    optimization.accumulate_loss(model(x).sum() * scale)
    optimization.step_weights()
    assert optimization.skipped_non_finite_batches == 1
    for p, p_before in zip(model.parameters(), before):
        torch.testing.assert_close(p.detach(), p_before)
