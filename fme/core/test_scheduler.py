import copy
from typing import Literal

import dacite
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
import yaml

import fme
from fme.core.optimization import Optimization, OptimizationConfig
from fme.core.scheduler import (
    HelicalSchedulerConfig,
    SchedulerConfig,
    SequentialSchedulerConfig,
)


def test_default_gives_none():
    optimizer = torch.optim.Adam(params=[torch.nn.Parameter()])
    max_epochs = 42
    assert SchedulerConfig().build(optimizer, max_epochs).scheduler_obj is None


def test_build():
    config = SchedulerConfig(type="StepLR", kwargs={"step_size": 1})
    # define dummy parameters for optimizer
    optimizer = torch.optim.Adam(params=[torch.nn.Parameter()])
    max_epochs = 42
    scheduler = config.build(optimizer, max_epochs).scheduler_obj
    assert scheduler is not None
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 1
    assert scheduler.optimizer is optimizer


def _helical_reference_lr(
    component: Literal["ocean", "atmos"],
    variant: Literal["repeat", "decay"],
    step: np.ndarray,
    peak: float,
    minimum: float,
    warmup_start_factor: float,
    warmup: float,
    period: float,
    total: float,
) -> np.ndarray:
    """Coupled helical LR (ocean: linear warmup then cosine from its peak;
    atmos: cosine rising from its minimum), optionally times a linear envelope
    falling from the component's first peak to zero at ``total``."""
    s = step.astype(float)
    if component == "ocean":
        lr = np.where(
            s < warmup,
            peak * (warmup_start_factor + (1 - warmup_start_factor) * s / warmup),
            minimum
            + (peak - minimum) * (1 + np.cos(2 * np.pi * (s - warmup) / period)) / 2,
        )
        first_peak = warmup
    else:
        lr = minimum + (peak - minimum) * (1 - np.cos(2 * np.pi * s / period)) / 2
        first_peak = period / 2
    if variant == "decay":
        lr = lr * np.where(s <= first_peak, 1.0, (total - s) / (total - first_peak))
    return lr


_HELICAL_TOTAL = 11680
_HELICAL_CASES = {
    # (variant, component): (reference kwargs, scheduler yaml)
    ("repeat", "ocean"): (
        dict(peak=7e-5, minimum=7e-8, warmup_start_factor=0.01, warmup=1668,
             period=3336),
        dict(period=3336, min_factor=1e-3, cosine_start="peak", warmup_iters=1668,
             warmup_start_factor=0.01),
    ),
    ("repeat", "atmos"): (
        dict(peak=7e-7, minimum=0.0, warmup_start_factor=0.0, warmup=0,
             period=6672),
        dict(period=6672, cosine_start="trough"),
    ),
    ("decay", "ocean"): (
        dict(peak=7e-5, minimum=7e-8, warmup_start_factor=0.01, warmup=1460,
             period=2920),
        dict(period=2920, min_factor=1e-3, cosine_start="peak", warmup_iters=1460,
             warmup_start_factor=0.01, decay_end=_HELICAL_TOTAL),
    ),
    ("decay", "atmos"): (
        dict(peak=7e-7, minimum=0.0, warmup_start_factor=0.0, warmup=0,
             period=5840),
        dict(period=5840, cosine_start="trough", decay_end=_HELICAL_TOTAL),
    ),
}  # fmt: skip


def _muon_optimization_from_yaml(lr: float, scheduler: dict) -> Optimization:
    config = dacite.from_dict(
        OptimizationConfig,
        {
            "optimizer_type": "Muon",
            "lr": lr,
            "kwargs": {"adamw_lr": 3 * lr},
            "scheduler": scheduler,
        },
        config=dacite.Config(strict=True),
    )
    assert isinstance(config.scheduler, HelicalSchedulerConfig)
    assert config.has_lr_schedule
    model = nn.Linear(3, 3).to(fme.get_device())
    return config.build(nn.ModuleList([model]), max_epochs=1)


def _group_lrs(optimization: Optimization) -> list[float]:
    return [group["lr"] for group in optimization.optimizer.param_groups]


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step\\(\\)`")
@pytest.mark.parametrize("variant, component", list(_HELICAL_CASES))
def test_helical_matches_coupled_helical_closed_form(variant, component):
    reference_kwargs, scheduler = _HELICAL_CASES[(variant, component)]
    peak = reference_kwargs["peak"]
    optimization = _muon_optimization_from_yaml(peak, scheduler)
    lrs = []
    for _ in range(_HELICAL_TOTAL + 1):
        lrs.append(_group_lrs(optimization))
        optimization.step_scheduler(is_iteration=True)
    expected = _helical_reference_lr(
        component,
        variant,
        np.arange(_HELICAL_TOTAL + 1),
        total=_HELICAL_TOTAL,
        **reference_kwargs,
    )
    lrs_by_group = np.array(lrs).T
    np.testing.assert_allclose(lrs_by_group[0], expected, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(lrs_by_group[1], 3 * expected, rtol=1e-9, atol=1e-15)
    if variant == "decay":
        assert lrs_by_group[:, -1] == pytest.approx([0.0, 0.0])


def _train_and_record_lrs(
    optimization: Optimization, model: nn.Module, x: torch.Tensor, n_steps: int
) -> list[list[float]]:
    lrs = []
    for _ in range(n_steps):
        optimization.accumulate_loss(model(x).pow(2).mean())
        optimization.step_weights()
        optimization.step_scheduler(is_iteration=True)
        lrs.append(_group_lrs(optimization))
    return lrs


def test_helical_resume_mid_cycle_continues_curve():
    torch.manual_seed(0)
    scheduler = HelicalSchedulerConfig(
        period=8, min_factor=0.1, warmup_iters=4, warmup_start_factor=0.2, decay_end=30
    )
    config = OptimizationConfig(
        optimizer_type="Muon", lr=0.02, kwargs={"adamw_lr": 1e-3}, scheduler=scheduler
    )
    n_before, n_after = 11, 15  # resume 3 steps into the second cycle
    x = torch.randn(4, 3, device=fme.get_device())
    model = nn.Linear(3, 3).to(fme.get_device())
    optimization = config.build(nn.ModuleList([model]), max_epochs=1)
    _train_and_record_lrs(optimization, model, x, n_before)
    saved_state = yaml.dump(optimization.get_state())
    saved_model = copy.deepcopy(model.state_dict())
    uninterrupted = _train_and_record_lrs(optimization, model, x, n_after)

    resumed_model = nn.Linear(3, 3).to(fme.get_device())
    resumed_model.load_state_dict(saved_model)
    resumed = config.build(nn.ModuleList([resumed_model]), max_epochs=1)
    resumed.load_state(yaml.load(saved_state, Loader=yaml.CLoader))
    assert _train_and_record_lrs(resumed, resumed_model, x, n_after) == uninterrupted
    for name, param in model.state_dict().items():
        torch.testing.assert_close(resumed_model.state_dict()[name], param)
    assert uninterrupted[-1] == [
        0.02 * scheduler.factor(n_before + n_after),
        1e-3 * scheduler.factor(n_before + n_after),
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(period=0),
        dict(period=4, warmup_iters=-1),
        dict(period=4, min_factor=1.5),
        dict(period=4, warmup_start_factor=-0.1),
        dict(period=4, cosine_start="middle"),
        dict(period=4, warmup_iters=2, decay_end=2),
        dict(period=4, cosine_start="trough", decay_end=2),
    ],
)
def test_helical_config_validation(kwargs):
    with pytest.raises(ValueError):
        HelicalSchedulerConfig(**kwargs)


def test_existing_scheduler_yaml_still_parses_to_its_config():
    def parse(scheduler: dict):
        return dacite.from_dict(
            OptimizationConfig,
            {"scheduler": scheduler},
            config=dacite.Config(strict=True),
        ).scheduler

    assert isinstance(parse({}), SchedulerConfig)
    assert isinstance(parse({"type": "StepLR"}), SchedulerConfig)
    linear = {"type": "LinearLR", "step_each_iteration": True}
    sequential = parse({"schedulers": [linear, linear], "milestones": [2]})
    assert isinstance(sequential, SequentialSchedulerConfig)
