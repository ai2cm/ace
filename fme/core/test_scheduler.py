import torch.optim.lr_scheduler

from fme.core.scheduler import SchedulerConfig


def test_default_gives_none():
    optimizer = torch.optim.Adam(params=[torch.nn.Parameter()])
    max_epochs = 42
    assert SchedulerConfig().build(optimizer, max_epochs) is None


def test_build():
    config = SchedulerConfig(type="StepLR", kwargs={"step_size": 1})
    # define dummy parameters for optimizer
    optimizer = torch.optim.Adam(params=[torch.nn.Parameter()])
    max_epochs = 42
    scheduler = config.build(optimizer, max_epochs)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 1
    assert scheduler.optimizer is optimizer
