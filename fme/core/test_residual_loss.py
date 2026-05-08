import pytest
import torch

from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.loss import StepLossConfig
from fme.core.normalizer import StandardNormalizer
from fme.core.residual_loss import (
    ResidualPair,
    SnapshotResidualLoss,
    SnapshotResidualLossConfig,
)


def _build_loss(
    out_names: list[str],
    pairs: list[ResidualPair],
    weights: dict[str, float] | None = None,
    stds: dict[str, float] | None = None,
    weight: float = 1.0,
) -> SnapshotResidualLoss:
    means = {name: torch.as_tensor(0.0) for name in out_names}
    if stds is None:
        stds = {name: 1.0 for name in out_names}
    std_tensors = {name: torch.as_tensor(stds[name]) for name in out_names}
    normalizer = StandardNormalizer(means=means, stds=std_tensors)
    config = SnapshotResidualLossConfig(
        pairs=pairs,
        loss=StepLossConfig(type="MSE", weights={} if weights is None else weights),
        weight=weight,
    )
    gridded_ops = LatLonOperations(torch.ones(1, 1))
    return config.build(
        gridded_ops=gridded_ops,
        out_names=out_names,
        normalizer=normalizer,
    )


def test_residual_pair_validation_negative():
    with pytest.raises(ValueError, match="non-negative"):
        ResidualPair(step_a=-1, step_b=0)


def test_residual_pair_validation_equal():
    with pytest.raises(ValueError, match="must differ"):
        ResidualPair(step_a=2, step_b=2)


def test_residual_pair_label_and_max_step():
    pair = ResidualPair(step_a=3, step_b=1)
    assert pair.max_step == 3
    assert pair.label == "step_3_minus_1"


def test_config_requires_pairs():
    with pytest.raises(ValueError, match="at least one"):
        SnapshotResidualLossConfig(pairs=[])


def test_config_max_pair_step():
    config = SnapshotResidualLossConfig(
        pairs=[ResidualPair(1, 0), ResidualPair(40, 30)]
    )
    assert config.max_pair_step == 40


def test_loss_matches_manual_mse():
    """Residual MSE on standardized residuals matches manual computation."""
    torch.manual_seed(0)
    out_names = ["a", "b"]
    std_a, std_b = 2.0, 0.5
    loss = _build_loss(
        out_names,
        pairs=[ResidualPair(1, 0)],
        stds={"a": std_a, "b": std_b},
    )

    n_sample, n_ensemble, h, w = 3, 2, 4, 5
    pred1 = {
        "a": torch.randn(n_sample, n_ensemble, h, w, device=get_device()),
        "b": torch.randn(n_sample, n_ensemble, h, w, device=get_device()),
    }
    ic = {
        "a": torch.randn(n_sample, 1, h, w, device=get_device()),
        "b": torch.randn(n_sample, 1, h, w, device=get_device()),
    }
    target1 = {
        "a": torch.randn(n_sample, 1, h, w, device=get_device()),
        "b": torch.randn(n_sample, 1, h, w, device=get_device()),
    }

    total, per_pair = loss({0: ic, 1: pred1}, {0: ic, 1: target1})

    gen_residual_a = (pred1["a"] - ic["a"]) / std_a
    gen_residual_b = (pred1["b"] - ic["b"]) / std_b
    target_residual_a = (target1["a"] - ic["a"]) / std_a
    target_residual_b = (target1["b"] - ic["b"]) / std_b
    expected = 0.5 * (
        ((gen_residual_a - target_residual_a) ** 2).mean()
        + ((gen_residual_b - target_residual_b) ** 2).mean()
    )
    torch.testing.assert_close(total, expected)
    assert set(per_pair.keys()) == {"residual_loss_step_1_minus_0"}
    torch.testing.assert_close(per_pair["residual_loss_step_1_minus_0"], expected)


def test_loss_matches_manual_mse_no_ic():
    """Pair where both endpoints are forward predictions with ensemble dims."""
    torch.manual_seed(0)
    out_names = ["a"]
    loss = _build_loss(out_names, pairs=[ResidualPair(3, 1)])

    n_sample, n_ensemble, h, w = 2, 4, 3, 3
    pred_shape = (n_sample, n_ensemble, h, w)
    target_shape = (n_sample, 1, h, w)
    pred1 = {"a": torch.randn(*pred_shape, device=get_device())}
    pred3 = {"a": torch.randn(*pred_shape, device=get_device())}
    target1 = {"a": torch.randn(*target_shape, device=get_device())}
    target3 = {"a": torch.randn(*target_shape, device=get_device())}

    total, _ = loss({1: pred1, 3: pred3}, {1: target1, 3: target3})
    expected = (((pred3["a"] - pred1["a"]) - (target3["a"] - target1["a"])) ** 2).mean()
    torch.testing.assert_close(total, expected)


def test_variable_weights_applied():
    out_names = ["a", "b"]
    weights = {"a": 4.0, "b": 1.0}
    loss = _build_loss(out_names, pairs=[ResidualPair(1, 0)], weights=weights)

    n_sample, h, w = 2, 3, 3
    ic = {
        "a": torch.zeros(n_sample, 1, h, w, device=get_device()),
        "b": torch.zeros(n_sample, 1, h, w, device=get_device()),
    }
    pred1 = {
        "a": torch.ones(n_sample, 1, h, w, device=get_device()),
        "b": 2 * torch.ones(n_sample, 1, h, w, device=get_device()),
    }
    target1 = {
        "a": 2 * torch.ones(n_sample, 1, h, w, device=get_device()),
        "b": 2.5 * torch.ones(n_sample, 1, h, w, device=get_device()),
    }

    total, _ = loss({0: ic, 1: pred1}, {0: ic, 1: target1})
    # |a|^2 weight=4 -> 16; |b|^2 weight=1 -> 0.25
    expected = torch.tensor((16.0 + 0.25) / 2.0, device=get_device())
    torch.testing.assert_close(total, expected)


def test_update_max_n_forward_steps_filters_pairs():
    out_names = ["a"]
    loss = _build_loss(
        out_names,
        pairs=[
            ResidualPair(1, 0),
            ResidualPair(3, 1),
            ResidualPair(40, 30),
        ],
    )
    assert {p.max_step for p in loss.active_pairs} == {1, 3, 40}

    loss.update_max_n_forward_steps(2)
    assert [p.max_step for p in loss.active_pairs] == [1]
    assert loss.needed_steps() == {0, 1}

    loss.update_max_n_forward_steps(0)
    assert loss.active_pairs == []
    assert loss.needed_steps() == set()

    loss.update_max_n_forward_steps(40)
    assert {p.max_step for p in loss.active_pairs} == {1, 3, 40}


def test_call_with_no_active_pairs_returns_zero():
    out_names = ["a"]
    loss = _build_loss(out_names, pairs=[ResidualPair(5, 0)])
    loss.update_max_n_forward_steps(1)
    assert loss.active_pairs == []

    total, per_pair = loss({}, {})
    assert per_pair == {}
    torch.testing.assert_close(total, torch.zeros((), device=total.device))


def test_call_missing_step_raises():
    out_names = ["a"]
    loss = _build_loss(out_names, pairs=[ResidualPair(1, 0)])
    n_sample, h, w = 2, 3, 3
    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    with pytest.raises(KeyError, match="prediction"):
        loss({0: ic}, {0: ic, 1: ic})
    with pytest.raises(KeyError, match="target"):
        loss({0: ic, 1: ic}, {0: ic})


def test_total_sums_over_active_pairs():
    """Total returned is the unweighted sum over active pairs."""
    torch.manual_seed(0)
    out_names = ["a"]
    loss = _build_loss(
        out_names,
        pairs=[ResidualPair(1, 0), ResidualPair(2, 0)],
    )

    n_sample, h, w = 2, 3, 3
    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    pred1 = {"a": torch.ones(n_sample, 1, h, w, device=get_device())}
    pred2 = {"a": 2 * torch.ones(n_sample, 1, h, w, device=get_device())}
    tgt1 = {"a": 0.5 * torch.ones(n_sample, 1, h, w, device=get_device())}
    tgt2 = {"a": torch.ones(n_sample, 1, h, w, device=get_device())}

    total, per_pair = loss(
        {0: ic, 1: pred1, 2: pred2},
        {0: ic, 1: tgt1, 2: tgt2},
    )
    expected = ((1 - 0.5) ** 2) + ((2 - 1) ** 2)
    torch.testing.assert_close(
        total, torch.tensor(float(expected), device=total.device)
    )
    assert sum(per_pair.values()).item() == pytest.approx(total.item())


def test_loss_propagates_gradients():
    out_names = ["a"]
    loss = _build_loss(out_names, pairs=[ResidualPair(1, 0)])
    n_sample, h, w = 2, 3, 3
    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    pred1 = {
        "a": torch.ones(n_sample, 1, h, w, device=get_device(), requires_grad=True)
    }
    target1 = {"a": 2 * torch.ones(n_sample, 1, h, w, device=get_device())}

    total, _ = loss({0: ic, 1: pred1}, {0: ic, 1: target1})
    total.backward()
    grad = pred1["a"].grad
    assert grad is not None
    assert torch.all(grad != 0)


def test_loss_detaches_earlier_endpoint():
    """Earlier endpoint never receives a gradient, even with requires_grad=True.

    The residual term constrains the later state's deviation from the earlier
    endpoint treated as a fixed reference, so backward should propagate
    gradients only into the larger-step prediction.
    """
    out_names = ["a"]
    loss = _build_loss(out_names, pairs=[ResidualPair(step_a=1, step_b=0)])
    n_sample, h, w = 2, 3, 3
    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device(), requires_grad=True)}
    pred1 = {
        "a": torch.ones(n_sample, 1, h, w, device=get_device(), requires_grad=True)
    }
    target1 = {"a": 2 * torch.ones(n_sample, 1, h, w, device=get_device())}

    total, _ = loss({0: ic, 1: pred1}, {0: ic, 1: target1})
    total.backward()
    assert pred1["a"].grad is not None
    assert torch.all(pred1["a"].grad != 0)
    assert ic["a"].grad is None


def test_loss_detaches_earlier_endpoint_when_step_a_lt_step_b():
    """When step_a < step_b, step_a is the earlier endpoint and is detached."""
    out_names = ["a"]
    loss = _build_loss(out_names, pairs=[ResidualPair(step_a=0, step_b=1)])
    n_sample, h, w = 2, 3, 3
    pred0 = {
        "a": torch.zeros(n_sample, 1, h, w, device=get_device(), requires_grad=True)
    }
    pred1 = {
        "a": torch.ones(n_sample, 1, h, w, device=get_device(), requires_grad=True)
    }
    tgt0 = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    tgt1 = {"a": 2 * torch.ones(n_sample, 1, h, w, device=get_device())}

    total, _ = loss({0: pred0, 1: pred1}, {0: tgt0, 1: tgt1})
    total.backward()
    assert pred1["a"].grad is not None
    assert torch.all(pred1["a"].grad != 0)
    assert pred0["a"].grad is None


def test_pairs_completing_at_filters_by_max_step():
    out_names = ["a"]
    loss = _build_loss(
        out_names,
        pairs=[ResidualPair(1, 0), ResidualPair(3, 1), ResidualPair(0, 3)],
    )
    assert [p.label for p in loss.pairs_completing_at(1)] == ["step_1_minus_0"]
    assert {p.label for p in loss.pairs_completing_at(3)} == {
        "step_3_minus_1",
        "step_0_minus_3",
    }
    assert loss.pairs_completing_at(2) == []


def test_pairs_completing_at_respects_active_filtering():
    """Pairs filtered out by update_max_n_forward_steps must not appear."""
    out_names = ["a"]
    loss = _build_loss(
        out_names,
        pairs=[ResidualPair(1, 0), ResidualPair(3, 1)],
    )
    loss.update_max_n_forward_steps(1)
    assert [p.label for p in loss.pairs_completing_at(1)] == ["step_1_minus_0"]
    assert loss.pairs_completing_at(3) == []


def test_compute_pair_loss_matches_call_for_single_pair():
    """compute_pair_loss is the per-pair primitive used inside __call__."""
    torch.manual_seed(0)
    out_names = ["a"]
    pair = ResidualPair(step_a=1, step_b=0)
    loss = _build_loss(out_names, pairs=[pair])

    n_sample, h, w = 2, 3, 3
    ic = {"a": torch.randn(n_sample, 1, h, w, device=get_device())}
    pred1 = {"a": torch.randn(n_sample, 1, h, w, device=get_device())}
    target1 = {"a": torch.randn(n_sample, 1, h, w, device=get_device())}

    via_call, _ = loss({0: ic, 1: pred1}, {0: ic, 1: target1})
    via_pair = loss.compute_pair_loss(pair, {0: ic, 1: pred1}, {0: ic, 1: target1})
    torch.testing.assert_close(via_call, via_pair)
