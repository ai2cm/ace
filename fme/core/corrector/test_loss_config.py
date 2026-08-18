from collections.abc import Iterable

import pytest
import torch

from fme.core.corrector.loss_config import (
    CorrectorLossConfig,
    CorrectorRegularizationConfig,
    PreCorrectorOptimizationConfig,
)
from fme.core.device import get_device
from fme.core.loss import CorrectorLoss, LossConfig
from fme.core.normalizer import StandardNormalizer
from fme.core.typing_ import TensorDict

# Deterministic setup: fixed delta dicts and a normalizer with known means/stds.
_SHAPE = (2, 2, 2)  # (batch, lat, lon)
_NAMES = ["a", "b_0", "b_1"]
_MEANS = {"a": 1.0, "b_0": -2.0, "b_1": 0.5}
_STDS = {"a": 2.0, "b_0": 0.5, "b_1": 4.0}
_MODIFIED = frozenset(_NAMES)


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
    corrector_modified_names: frozenset[str] = _MODIFIED,
) -> CorrectorLoss:
    return config.build(
        corrector_modified_names,
        normalizer=_normalizer(sorted(corrector_modified_names | frozenset(_NAMES))),
        gridded_operations=None,
    )


def test_config_post_init_errors():
    # both features None; a present feature with names_and_prefixes None
    # or empty; weight <= 0; EnsembleLoss / NaN / global_mean_type — each
    # raises in __post_init__.
    with pytest.raises(ValueError, match="at least one"):
        CorrectorLossConfig()
    empty_selections: list[list[str] | None] = [None, []]
    for names in empty_selections:
        with pytest.raises(ValueError, match="names_and_prefixes"):
            PreCorrectorOptimizationConfig(names_and_prefixes=names)
        with pytest.raises(ValueError, match="names_and_prefixes"):
            CorrectorRegularizationConfig(names_and_prefixes=names)
    for weight in (0.0, -1.0):
        with pytest.raises(ValueError, match="weight"):
            CorrectorRegularizationConfig(names_and_prefixes=["a"], weight=weight)
    for loss_type in ("EnsembleLoss", "NaN"):
        with pytest.raises(ValueError, match=loss_type):
            CorrectorRegularizationConfig(
                names_and_prefixes=["a"], loss=LossConfig(type=loss_type)
            )
    with pytest.raises(ValueError, match="global_mean_type"):
        CorrectorRegularizationConfig(
            names_and_prefixes=["a"], loss=LossConfig(global_mean_type="LpLoss")
        )


@pytest.mark.parametrize("entry", ["missing_var", "missing_"])
@pytest.mark.parametrize("feature", ["precorrector_optimization", "regularization"])
def test_build_errors_on_entry_matching_no_modified_name(entry, feature):
    # an entry matching no corrector-modified name raises at build.
    # PARAMETERIZE: entry in {exact name, trailing-underscore prefix}.
    if feature == "precorrector_optimization":
        config = CorrectorLossConfig(
            precorrector_optimization=PreCorrectorOptimizationConfig(
                names_and_prefixes=[entry]
            )
        )
    else:
        config = CorrectorLossConfig(
            regularization=CorrectorRegularizationConfig(names_and_prefixes=[entry])
        )
    with pytest.raises(ValueError, match="selects no variable the corrector modifies"):
        _build(config)


def test_build_errors_without_modified_names():
    # empty corrector_modified_names raises.
    config = CorrectorLossConfig(
        precorrector_optimization=PreCorrectorOptimizationConfig(
            names_and_prefixes=["a"]
        )
    )
    with pytest.raises(ValueError, match="modifies no variables"):
        _build(config, corrector_modified_names=frozenset())


def test_build_regularizer_packs_matched_names():
    # the built WeightedMappingLoss packs exactly
    # selection.matched(corrector_modified_names); a prefix entry matches all its
    # level names.
    config = CorrectorLossConfig(
        regularization=CorrectorRegularizationConfig(
            names_and_prefixes=["b_"], weight=2.0
        )
    )
    corrector_loss = _build(config)
    assert corrector_loss.penalty_weight == 2.0
    penalty = corrector_loss.penalty(_delta_dict())
    assert penalty is not None
    assert list(penalty.get_channel_losses()) == ["b_0", "b_1"]
