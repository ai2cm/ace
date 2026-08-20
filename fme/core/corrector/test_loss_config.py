from collections.abc import Iterable

import pytest
import torch

from fme.core.corrector.loss_config import (
    CorrectorLossConfig,
    CorrectorRegularizationConfig,
)
from fme.core.device import get_device
from fme.core.loss import CorrectorLoss
from fme.core.normalizer import StandardNormalizer
from fme.core.typing_ import TensorDict

# Deterministic setup: fixed delta dicts and a normalizer with known means/stds.
_SHAPE = (2, 2, 2)  # (batch, lat, lon)
_NAMES = ["a", "b_0", "b_1"]
_MEANS = {"a": 1.0, "b_0": -2.0, "b_1": 0.5}
_STDS = {"a": 2.0, "b_0": 0.5, "b_1": 4.0}


def _fixed(offset: float, scale: float) -> torch.Tensor:
    n = _SHAPE[0] * _SHAPE[1] * _SHAPE[2]
    values = offset + scale * torch.arange(n, dtype=torch.float32)
    return values.reshape(_SHAPE).to(get_device())


def _delta_dict() -> TensorDict:
    return {
        "a": _fixed(0.1, 0.05),
        "b_0": _fixed(-0.2, 0.1),
        "b_1": _fixed(0.3, -0.15),
    }


def _normalizer(names: Iterable[str] = _NAMES) -> StandardNormalizer:
    return StandardNormalizer(
        means={name: torch.as_tensor(_MEANS[name]) for name in names},
        stds={name: torch.as_tensor(_STDS[name]) for name in names},
    )


def _build(config: CorrectorLossConfig) -> CorrectorLoss:
    return config.build(
        normalizer=_normalizer(),
        channel_dim=-3,
    )


def test_config_post_init_errors():
    # both features off; weight <= 0 — each raises in __post_init__.
    with pytest.raises(ValueError, match="at least one"):
        CorrectorLossConfig()
    for weight in (0.0, -1.0):
        with pytest.raises(ValueError, match="weight"):
            CorrectorRegularizationConfig(norm="L2", weight=weight)


def test_penalty_covers_every_delta():
    # the penalty needs no name configuration: it covers exactly the delta
    # keys of each call.
    config = CorrectorLossConfig(
        regularization=CorrectorRegularizationConfig(norm="L2", weight=2.0)
    )
    corrector_loss = _build(config)
    assert corrector_loss.penalty_weight == 2.0
    penalty = corrector_loss.penalty(_delta_dict())
    assert penalty is not None
    assert list(penalty.get_channel_losses()) == ["a", "b_0", "b_1"]


@pytest.mark.parametrize(
    "norm,reduce",
    [
        ("L1", lambda x: x.abs().mean()),
        ("L2", lambda x: (x**2).mean()),
    ],
)
def test_penalty_matches_the_configured_norm(norm, reduce):
    # the penalty is the mean of the normalized delta under the configured
    # norm; "L2" is the mean square, not its root.
    config = CorrectorLossConfig(
        regularization=CorrectorRegularizationConfig(norm=norm)
    )
    penalty = _build(config).penalty(_delta_dict())
    assert penalty is not None
    normalized = _delta_dict()["a"] / _STDS["a"]
    torch.testing.assert_close(
        penalty.get_channel_losses()["a"].loss, reduce(normalized)
    )
