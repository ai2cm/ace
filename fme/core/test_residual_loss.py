import pytest
import torch

from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.loss import StepLossConfig
from fme.core.normalizer import StandardNormalizer
from fme.core.residual_loss import (
    SnapshotResidualLoss,
    SnapshotResidualLossConfig,
    step_label,
)


def _build_loss(
    out_names: list[str],
    steps: list[int],
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
        steps=steps,
        loss=StepLossConfig(type="MSE", weights={} if weights is None else weights),
        weight=weight,
    )
    gridded_ops = LatLonOperations(torch.ones(1, 1))
    return config.build(
        gridded_ops=gridded_ops,
        out_names=out_names,
        normalizer=normalizer,
    )


def test_config_requires_steps():
    with pytest.raises(ValueError, match="at least one"):
        SnapshotResidualLossConfig(steps=[])


def test_config_rejects_step_below_one():
    with pytest.raises(ValueError, match="must be >= 1"):
        SnapshotResidualLossConfig(steps=[0])


def test_config_max_step():
    config = SnapshotResidualLossConfig(steps=[1, 3, 2])
    assert config.max_step == 3


def test_step_label():
    assert step_label(1) == "step_1_minus_0"
    assert step_label(3) == "step_3_minus_2"


def test_loss_matches_manual_mse():
    """Residual MSE on standardized absolute residuals matches manual computation."""
    torch.manual_seed(0)
    out_names = ["a", "b"]
    std_a, std_b = 2.0, 0.5
    loss = _build_loss(
        out_names,
        steps=[1],
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

    total, per_step = loss({0: ic, 1: pred1}, {0: ic, 1: target1})

    gen_residual_a = (pred1["a"] - ic["a"]).abs() / std_a
    gen_residual_b = (pred1["b"] - ic["b"]).abs() / std_b
    target_residual_a = (target1["a"] - ic["a"]).abs() / std_a
    target_residual_b = (target1["b"] - ic["b"]).abs() / std_b
    expected = 0.5 * (
        ((gen_residual_a - target_residual_a) ** 2).mean()
        + ((gen_residual_b - target_residual_b) ** 2).mean()
    )
    torch.testing.assert_close(total, expected)
    assert set(per_step.keys()) == {"residual_loss_step_1_minus_0"}
    torch.testing.assert_close(per_step["residual_loss_step_1_minus_0"], expected)


def test_loss_consecutive_steps():
    """Step 2 computes |gen[2]-gen[1]| vs |target[2]-target[1]|."""
    torch.manual_seed(0)
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[2])

    n_sample, n_ensemble, h, w = 2, 4, 3, 3
    pred_shape = (n_sample, n_ensemble, h, w)
    target_shape = (n_sample, 1, h, w)
    pred1 = {"a": torch.randn(*pred_shape, device=get_device())}
    pred2 = {"a": torch.randn(*pred_shape, device=get_device())}
    target1 = {"a": torch.randn(*target_shape, device=get_device())}
    target2 = {"a": torch.randn(*target_shape, device=get_device())}

    total, _ = loss({1: pred1, 2: pred2}, {1: target1, 2: target2})
    gen_abs = (pred2["a"] - pred1["a"]).abs()
    tgt_abs = (target2["a"] - target1["a"]).abs()
    expected = ((gen_abs - tgt_abs) ** 2).mean()
    torch.testing.assert_close(total, expected)


def test_variable_weights_applied():
    out_names = ["a", "b"]
    weights = {"a": 4.0, "b": 1.0}
    loss = _build_loss(out_names, steps=[1], weights=weights)

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


def test_update_max_n_forward_steps_filters_steps():
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1, 3, 40])
    assert set(loss.active_steps) == {1, 3, 40}

    loss.update_max_n_forward_steps(2)
    assert loss.active_steps == [1]
    assert loss.needed_steps() == {0, 1}

    loss.update_max_n_forward_steps(0)
    assert loss.active_steps == []
    assert loss.needed_steps() == set()

    loss.update_max_n_forward_steps(40)
    assert set(loss.active_steps) == {1, 3, 40}


def test_call_with_no_active_steps_returns_zero():
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[5])
    loss.update_max_n_forward_steps(1)
    assert loss.active_steps == []

    total, per_step = loss({}, {})
    assert per_step == {}
    torch.testing.assert_close(total, torch.zeros((), device=total.device))


def test_call_missing_step_raises():
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1])
    n_sample, h, w = 2, 3, 3
    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    with pytest.raises(KeyError, match="prediction"):
        loss({0: ic}, {0: ic, 1: ic})
    with pytest.raises(KeyError, match="target"):
        loss({0: ic, 1: ic}, {0: ic})


def test_total_sums_over_active_steps():
    """Total returned is the unweighted sum over active steps."""
    torch.manual_seed(0)
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1, 2])

    n_sample, h, w = 2, 3, 3
    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    pred1 = {"a": torch.ones(n_sample, 1, h, w, device=get_device())}
    pred2 = {"a": 2 * torch.ones(n_sample, 1, h, w, device=get_device())}
    tgt1 = {"a": 0.5 * torch.ones(n_sample, 1, h, w, device=get_device())}
    tgt2 = {"a": torch.ones(n_sample, 1, h, w, device=get_device())}

    total, per_step = loss(
        {0: ic, 1: pred1, 2: pred2},
        {0: ic, 1: tgt1, 2: tgt2},
    )
    # step 1: (1 - 0.5)^2 = 0.25  (gen[1]-gen[0] vs tgt[1]-tgt[0])
    # step 2: ((2-1) - (1-0.5))^2 = 0.25  (gen[2]-gen[1] vs tgt[2]-tgt[1])
    expected = 0.25 + 0.25
    torch.testing.assert_close(
        total, torch.tensor(float(expected), device=total.device)
    )
    assert sum(per_step.values()).item() == pytest.approx(total.item())


def test_loss_propagates_gradients():
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1])
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


def test_loss_detaches_reference_endpoint():
    """Reference endpoint (step-1) never receives a gradient."""
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1])
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


def test_loss_detaches_reference_step2():
    """For step=2, gen[1] is the detached reference."""
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[2])
    n_sample, h, w = 2, 3, 3
    pred1 = {
        "a": torch.zeros(n_sample, 1, h, w, device=get_device(), requires_grad=True)
    }
    pred2 = {
        "a": torch.ones(n_sample, 1, h, w, device=get_device(), requires_grad=True)
    }
    tgt1 = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    tgt2 = {"a": 2 * torch.ones(n_sample, 1, h, w, device=get_device())}

    total, _ = loss({1: pred1, 2: pred2}, {1: tgt1, 2: tgt2})
    total.backward()
    assert pred2["a"].grad is not None
    assert torch.all(pred2["a"].grad != 0)
    assert pred1["a"].grad is None


def test_steps_completing_at():
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1, 2, 3])
    assert loss.steps_completing_at(1) == [1]
    assert loss.steps_completing_at(2) == [2]
    assert loss.steps_completing_at(3) == [3]
    assert loss.steps_completing_at(4) == []


def test_steps_completing_at_respects_active_filtering():
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1, 3])
    loss.update_max_n_forward_steps(1)
    assert loss.steps_completing_at(1) == [1]
    assert loss.steps_completing_at(3) == []


def test_compute_step_loss_matches_call():
    """compute_step_loss is the per-step primitive used inside __call__."""
    torch.manual_seed(0)
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1])

    n_sample, h, w = 2, 3, 3
    ic = {"a": torch.randn(n_sample, 1, h, w, device=get_device())}
    pred1 = {"a": torch.randn(n_sample, 1, h, w, device=get_device())}
    target1 = {"a": torch.randn(n_sample, 1, h, w, device=get_device())}

    via_call, _ = loss({0: ic, 1: pred1}, {0: ic, 1: target1})
    via_step = loss.compute_step_loss(1, {0: ic, 1: pred1}, {0: ic, 1: target1})
    torch.testing.assert_close(via_call, via_step)


def test_compute_residuals_returns_abs_diffs():
    """compute_residuals returns absolute temporal differences."""
    out_names = ["a", "b"]
    loss = _build_loss(out_names, steps=[1])

    n_sample, h, w = 2, 3, 3
    ic = {
        "a": torch.ones(n_sample, 1, h, w, device=get_device()),
        "b": 3 * torch.ones(n_sample, 1, h, w, device=get_device()),
    }
    pred1 = {
        "a": torch.zeros(n_sample, 1, h, w, device=get_device()),
        "b": torch.ones(n_sample, 1, h, w, device=get_device()),
    }
    target1 = {
        "a": 2 * torch.ones(n_sample, 1, h, w, device=get_device()),
        "b": torch.zeros(n_sample, 1, h, w, device=get_device()),
    }

    gen_res, tgt_res = loss.compute_residuals(1, {0: ic, 1: pred1}, {0: ic, 1: target1})

    assert set(gen_res.keys()) == {"a", "b"}
    assert set(tgt_res.keys()) == {"a", "b"}
    # pred - ic for "a": 0 - 1 = -1, abs -> 1
    torch.testing.assert_close(
        gen_res["a"], torch.ones(n_sample, 1, h, w, device=get_device())
    )
    # pred - ic for "b": 1 - 3 = -2, abs -> 2
    torch.testing.assert_close(
        gen_res["b"], 2 * torch.ones(n_sample, 1, h, w, device=get_device())
    )
    # target - ic for "a": 2 - 1 = 1, abs -> 1
    torch.testing.assert_close(
        tgt_res["a"], torch.ones(n_sample, 1, h, w, device=get_device())
    )
    # target - ic for "b": 0 - 3 = -3, abs -> 3
    torch.testing.assert_close(
        tgt_res["b"], 3 * torch.ones(n_sample, 1, h, w, device=get_device())
    )


def test_compute_residuals_detaches_reference():
    """compute_residuals detaches the reference endpoint."""
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1])

    n_sample, h, w = 2, 3, 3
    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device(), requires_grad=True)}
    pred1 = {
        "a": torch.ones(n_sample, 1, h, w, device=get_device(), requires_grad=True)
    }
    target1 = {"a": 2 * torch.ones(n_sample, 1, h, w, device=get_device())}

    gen_res, _ = loss.compute_residuals(1, {0: ic, 1: pred1}, {0: ic, 1: target1})
    gen_res["a"].sum().backward()
    assert pred1["a"].grad is not None
    assert ic["a"].grad is None
