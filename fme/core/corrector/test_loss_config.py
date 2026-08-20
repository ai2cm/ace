from collections.abc import Iterable

import dacite
import pytest
import torch

from fme.core.corrector.loss_config import (
    CorrectorLossConfig,
    CorrectorRegularizationConfig,
    PreCorrectorOptimizationConfig,
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
_LOSS_NAMES = list(_NAMES)


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


def _build(
    config: CorrectorLossConfig,
    loss_names: Iterable[str] = _LOSS_NAMES,
) -> CorrectorLoss:
    return config.build(
        list(loss_names),
        normalizer=_normalizer(),
        channel_dim=-3,
    )


def _resolved(config: CorrectorLossConfig) -> CorrectorLoss:
    """A built corrector loss whose names are resolved against ``_delta_dict``."""
    corrector_loss = _build(config)
    corrector_loss.resolve_names(_delta_dict().keys())
    return corrector_loss


def test_config_post_init_errors():
    # both features None; a present feature selecting nothing; weight <= 0 —
    # each raises in __post_init__.
    with pytest.raises(ValueError, match="at least one"):
        CorrectorLossConfig()
    with pytest.raises(ValueError, match="names_and_prefixes"):
        PreCorrectorOptimizationConfig(names_and_prefixes=[])
    with pytest.raises(ValueError, match="names_and_prefixes"):
        CorrectorRegularizationConfig(names_and_prefixes=[], norm="L2")
    for weight in (0.0, -1.0):
        with pytest.raises(ValueError, match="weight"):
            CorrectorRegularizationConfig(
                names_and_prefixes=["a"], norm="L2", weight=weight
            )


@pytest.mark.parametrize(
    "feature_config",
    [PreCorrectorOptimizationConfig, CorrectorRegularizationConfig],
)
def test_names_and_prefixes_is_required_by_dacite(feature_config):
    # the YAML path, not the constructor: omitting the selection is a
    # missing-value error from dacite, not a config that parses and no-ops.
    with pytest.raises(dacite.exceptions.MissingValueError):
        dacite.from_dict(feature_config, {}, config=dacite.Config(strict=True))


@pytest.mark.parametrize("entry", ["missing_var", "missing_"])
@pytest.mark.parametrize("feature", ["precorrector_optimization", "regularization"])
def test_build_errors_on_entry_matching_no_loss_name(entry, feature):
    # an entry matching nothing the loss covers raises at build.
    # PARAMETERIZE: entry in {exact name, trailing-underscore prefix}.
    if feature == "precorrector_optimization":
        config = CorrectorLossConfig(
            precorrector_optimization=PreCorrectorOptimizationConfig(
                names_and_prefixes=[entry]
            )
        )
    else:
        config = CorrectorLossConfig(
            regularization=CorrectorRegularizationConfig(
                names_and_prefixes=[entry], norm="L2"
            )
        )
    with pytest.raises(ValueError, match="match none of the variables the loss"):
        _build(config)


def test_build_defers_regularizer_channels():
    # a config whose entries all match loss_names builds, and the regularizer
    # reports no names until the corrector loss has seen a delta.
    config = CorrectorLossConfig(
        regularization=CorrectorRegularizationConfig(
            names_and_prefixes=["b_"], norm="L2", weight=2.0
        )
    )
    corrector_loss = _build(config)
    assert corrector_loss.penalty_weight == 2.0
    with pytest.raises(RuntimeError, match="channels were resolved"):
        corrector_loss.penalty(_delta_dict())
    corrector_loss.resolve_names(_delta_dict().keys())
    penalty = corrector_loss.penalty(_delta_dict())
    assert penalty is not None
    assert list(penalty.get_channel_losses()) == ["b_0", "b_1"]


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
        regularization=CorrectorRegularizationConfig(
            names_and_prefixes=["a"], norm=norm
        )
    )
    penalty = _resolved(config).penalty(_delta_dict())
    assert penalty is not None
    normalized = _delta_dict()["a"] / _STDS["a"]
    torch.testing.assert_close(
        penalty.get_channel_losses()["a"].loss, reduce(normalized)
    )
