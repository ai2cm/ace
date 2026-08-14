from collections.abc import Collection, Iterable

import pytest
import torch

from fme.core.corrector.loss import (
    CorrectorLoss,
    CorrectorLossConfig,
    CorrectorRegularizationConfig,
    PreCorrectorOptimizationConfig,
    StepOutputLoss,
)
from fme.core.device import get_device
from fme.core.loss import LossConfig, StepLoss, StepLossConfig
from fme.core.normalizer import StandardNormalizer
from fme.core.typing_ import TensorDict, TensorMapping

# Deterministic setup: fixed prediction/target/delta dicts, a normalizer with
# known means/stds and an MSE step loss, so every penalty below is computable
# by hand.
_SHAPE = (2, 2, 2)  # (batch, lat, lon)
_NAMES = ["a", "b_0", "b_1"]
_MEANS = {"a": 1.0, "b_0": -2.0, "b_1": 0.5}
_STDS = {"a": 2.0, "b_0": 0.5, "b_1": 4.0}
_MODIFIED = frozenset(_NAMES)


def _fixed(offset: float, scale: float) -> torch.Tensor:
    n = _SHAPE[0] * _SHAPE[1] * _SHAPE[2]
    values = offset + scale * torch.arange(n, dtype=torch.float32)
    return values.reshape(_SHAPE).to(get_device())


def _predict_dict() -> TensorDict:
    return {
        "a": _fixed(0.0, 0.25),
        "b_0": _fixed(1.0, -0.5),
        "b_1": _fixed(-3.0, 0.75),
    }


def _target_dict() -> TensorDict:
    return {
        "a": _fixed(0.5, 0.2),
        "b_0": _fixed(-1.0, 0.4),
        "b_1": _fixed(2.0, -0.3),
    }


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


def _step_loss(sqrt_loss_step_decay_constant: float = 0.0) -> StepLoss:
    return StepLossConfig(
        type="MSE", sqrt_loss_step_decay_constant=sqrt_loss_step_decay_constant
    ).build(gridded_ops=None, out_names=list(_NAMES), normalizer=_normalizer())


def _build(
    config: CorrectorLossConfig,
    corrector_modified_names: frozenset[str] = _MODIFIED,
    prescribed_prognostic_names: Collection[str] = (),
) -> CorrectorLoss:
    return config.build(
        corrector_modified_names,
        prescribed_prognostic_names=prescribed_prognostic_names,
        normalizer=_normalizer(sorted(corrector_modified_names | frozenset(_NAMES))),
        gridded_operations=None,
    )


def _expected_penalty(deltas: TensorMapping, names: Iterable[str]) -> torch.Tensor:
    per_channel = []
    for name in sorted(names):
        normalized = torch.nan_to_num(deltas[name], nan=0.0) / _STDS[name]
        per_channel.append((normalized**2).mean())
    return torch.stack(per_channel).mean()


def test_config_post_init_errors():
    # GOAL: both features None; a present feature with names_and_prefixes None
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
    # GOAL: an entry matching no corrector-modified name raises at build.
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
    # GOAL: empty corrector_modified_names raises.
    config = CorrectorLossConfig(
        precorrector_optimization=PreCorrectorOptimizationConfig(
            names_and_prefixes=["a"]
        )
    )
    with pytest.raises(ValueError, match="modifies no variables"):
        _build(config, corrector_modified_names=frozenset())


@pytest.mark.parametrize("feature", ["precorrector_optimization", "regularization"])
def test_build_errors_on_prescribed_prognostic_entry(feature):
    # GOAL: an entry matching only a prescribed prognostic name raises, and the
    # message says the delta is dropped after the prescribed overwrite.
    if feature == "precorrector_optimization":
        config = CorrectorLossConfig(
            precorrector_optimization=PreCorrectorOptimizationConfig(
                names_and_prefixes=["a"]
            )
        )
    else:
        config = CorrectorLossConfig(
            regularization=CorrectorRegularizationConfig(names_and_prefixes=["a"])
        )
    with pytest.raises(ValueError, match="prescribed overwrite"):
        _build(config, prescribed_prognostic_names={"a"})
    # the same entry builds when the name is loss-visible
    _build(config)


def test_build_regularizer_packs_matched_names():
    # GOAL: the built WeightedMappingLoss packs exactly
    # selection.matched(loss_visible_names); a prefix entry matches all its
    # level names.
    config = CorrectorLossConfig(
        regularization=CorrectorRegularizationConfig(
            names_and_prefixes=["b_"], weight=2.0
        )
    )
    corrector_loss = _build(config)
    assert corrector_loss.regularization_weight == 2.0
    penalty = corrector_loss.regularization(_delta_dict())
    assert penalty is not None
    assert list(penalty.get_channel_losses()) == ["b_0", "b_1"]


def test_step_output_loss_without_corrector_loss_unchanged():
    # GOAL: with corrector_loss None, total() and get_channel_losses() match a
    # bare StepLoss, deltas ignored.
    predict, target, deltas = _predict_dict(), _target_dict(), _delta_dict()
    bare = _step_loss()(predict, target, 0)
    result = StepOutputLoss(_step_loss(), None)(predict, target, 0, deltas=deltas)
    assert result.corrector_regularization is None
    torch.testing.assert_close(result.total(), bare.total())
    assert result.get_corrector_channel_losses() == {}
    expected = bare.get_channel_losses()
    actual = result.get_channel_losses()
    assert set(actual) == set(expected)
    for name in expected:
        torch.testing.assert_close(actual[name].loss, expected[name].loss)


def test_pre_corrector_outputs_selected_only():
    # GOAL: the main loss sees output − delta for the configured names only;
    # other keys use the network output as-is; targets untouched.
    predict, target, deltas = _predict_dict(), _target_dict(), _delta_dict()
    config = CorrectorLossConfig(
        precorrector_optimization=PreCorrectorOptimizationConfig(
            names_and_prefixes=["a"]
        )
    )
    corrector_loss = _build(config)
    net_output = corrector_loss.pre_corrector_outputs(predict, deltas)
    torch.testing.assert_close(net_output["a"], predict["a"] - deltas["a"])
    for name in ("b_0", "b_1"):
        torch.testing.assert_close(net_output[name], predict[name])
    result = StepOutputLoss(_step_loss(), corrector_loss)(
        predict, target, 0, deltas=deltas
    )
    expected = _step_loss()(net_output, target, 0)
    torch.testing.assert_close(result.main.total(), expected.total())
    assert result.corrector_regularization is None
    # the target dict is never modified
    for name, value in _target_dict().items():
        torch.testing.assert_close(target[name], value)


def test_regularization_analytic_penalty():
    # GOAL: penalty equals the hand-computed mean of (delta/std)^2 — normalizer
    # means cancel against the zeros target.
    deltas = _delta_dict()
    config = CorrectorLossConfig(
        regularization=CorrectorRegularizationConfig(names_and_prefixes=["a", "b_"])
    )
    penalty = _build(config).regularization(deltas)
    assert penalty is not None
    torch.testing.assert_close(penalty.total(), _expected_penalty(deltas, _NAMES))


def test_regularization_masked_points_drop():
    # GOAL: NaN-filled delta points contribute nothing; penalty and gradients
    # stay finite.
    deltas = _delta_dict()
    masked = deltas["a"].clone()
    masked[0] = torch.nan
    masked.requires_grad_(True)
    deltas["a"] = masked
    config = CorrectorLossConfig(
        regularization=CorrectorRegularizationConfig(names_and_prefixes=["a", "b_"])
    )
    penalty = _build(config).regularization(deltas)
    assert penalty is not None
    total = penalty.total()
    assert torch.isfinite(total)
    torch.testing.assert_close(total, _expected_penalty(deltas, _NAMES))
    total.backward()
    assert masked.grad is not None
    assert torch.isfinite(masked.grad).all()
    assert (masked.grad[0] == 0.0).all()


def test_total_and_channel_decomposition():
    # GOAL: total() == main.total() + weight * penalty.total();
    # get_channel_losses() is main-only; get_corrector_channel_losses() covers
    # exactly the selected names.
    predict, target, deltas = _predict_dict(), _target_dict(), _delta_dict()
    config = CorrectorLossConfig(
        precorrector_optimization=PreCorrectorOptimizationConfig(
            names_and_prefixes=["a"]
        ),
        regularization=CorrectorRegularizationConfig(
            names_and_prefixes=["b_"], weight=3.0
        ),
    )
    result = StepOutputLoss(_step_loss(), _build(config))(
        predict, target, 0, deltas=deltas
    )
    assert result.corrector_regularization is not None
    torch.testing.assert_close(
        result.total(),
        result.main.total() + 3.0 * result.corrector_regularization.total(),
    )
    assert list(result.get_channel_losses()) == _NAMES
    corrector_channels = result.get_corrector_channel_losses()
    assert list(corrector_channels) == ["b_0", "b_1"]
    # per-channel penalties are unweighted
    torch.testing.assert_close(
        torch.stack([info.loss for info in corrector_channels.values()]).mean(),
        _expected_penalty(deltas, ["b_0", "b_1"]),
    )


@pytest.mark.parametrize("feature", ["precorrector_optimization", "regularization"])
def test_missing_selected_delta_raises(feature):
    # GOAL: a non-empty delta dict lacking a selected name raises.
    predict, target = _predict_dict(), _target_dict()
    deltas = {k: v for k, v in _delta_dict().items() if k != "a"}
    if feature == "precorrector_optimization":
        config = CorrectorLossConfig(
            precorrector_optimization=PreCorrectorOptimizationConfig(
                names_and_prefixes=["a"]
            )
        )
    else:
        config = CorrectorLossConfig(
            regularization=CorrectorRegularizationConfig(names_and_prefixes=["a"])
        )
    loss = StepOutputLoss(_step_loss(), _build(config))
    with pytest.raises(ValueError, match="produced no delta"):
        loss(predict, target, 0, deltas=deltas)


@pytest.mark.parametrize("deltas", [None, {}])
def test_empty_deltas_inert(deltas):
    # GOAL: empty deltas ⇒ no pre-corrector swap, no penalty; result matches
    # the unconfigured case.
    predict, target = _predict_dict(), _target_dict()
    config = CorrectorLossConfig(
        precorrector_optimization=PreCorrectorOptimizationConfig(
            names_and_prefixes=["a"]
        ),
        regularization=CorrectorRegularizationConfig(names_and_prefixes=["b_"]),
    )
    result = StepOutputLoss(_step_loss(), _build(config))(
        predict, target, 0, deltas=deltas
    )
    unconfigured = StepOutputLoss(_step_loss(), None)(predict, target, 0)
    assert result.corrector_regularization is None
    assert result.get_corrector_channel_losses() == {}
    torch.testing.assert_close(result.total(), unconfigured.total())
