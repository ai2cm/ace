import pathlib
from typing import Any, Protocol

import torch
from torch import nn

from fme.core.ema import EMA_CHECKPOINT_KEY, EMATracker


class ScaledIdentity(nn.Module):
    """A module with a parameter, defined at module scope so torch.save can
    pickle it.
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.scale


class _HasModulesAndState(Protocol):
    @property
    def modules(self) -> nn.ModuleList: ...

    def get_state(self) -> dict[str, Any]: ...


def _named_parameters(modules: nn.ModuleList) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in modules.named_parameters()}


def save_checkpoint_with_ema(
    stepper: _HasModulesAndState, path: str | pathlib.Path
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Save a checkpoint shaped like a Trainer's ``ckpt.tar``, whose stepper
    weights differ from the EMA weights saved beside them.

    The stepper's parameters are modified in place before saving.

    Returns:
        The saved stepper weights and EMA weights, keyed by the parameter
        name in ``stepper.modules``.
    """
    modules = stepper.modules
    ema = EMATracker(modules, decay=0.5, faster_decay_at_start=False)
    with torch.no_grad():
        for param in modules.parameters():
            param.add_(1.0)
    ema(modules)
    with ema.applied_params(modules):
        ema_weights = _named_parameters(modules)
    stepper_weights = _named_parameters(modules)
    torch.save(
        {"stepper": stepper.get_state(), EMA_CHECKPOINT_KEY: ema.get_state()}, path
    )
    return stepper_weights, ema_weights


def assert_parameters_equal(
    modules: nn.ModuleList, expected: dict[str, torch.Tensor]
) -> None:
    actual = _named_parameters(modules)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value.to(actual[name].device))
