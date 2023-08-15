import torch.optim.lr_scheduler

from fme.core.scheduler import SchedulerConfig


def test_default_gives_none():
    optimizer = torch.optim.Adam(params=[torch.nn.Parameter()])
    assert SchedulerConfig().build(optimizer) is None


def test_build():
    config = SchedulerConfig(type="StepLR", kwargs={"step_size": 1})
    # define dummy parameters for optimizer
    optimizer = torch.optim.Adam(params=[torch.nn.Parameter()])
    scheduler = config.build(optimizer)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 1
    assert scheduler.optimizer is optimizer
