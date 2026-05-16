import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.loss import StepLossConfig, WeightedMappingLoss
from fme.core.normalizer import StandardNormalizer
from fme.core.residual_loss import (
    SnapshotResidualLoss,
    SnapshotResidualLossConfig,
    load_std_maps,
    step_label,
)


def _write_scalar_stds(
    tmp_path, out_names: list[str], stds: dict[str, float] | None = None
) -> str:
    if stds is None:
        stds = {name: 1.0 for name in out_names}
    ds = xr.Dataset({name: xr.DataArray(stds[name]) for name in out_names})
    path = tmp_path / "residual_stds.nc"
    ds.to_netcdf(path)
    return str(path)


def _build_loss(
    out_names: list[str],
    steps: list[int],
    tmp_path,
    weights: dict[str, float] | None = None,
    stds: dict[str, float] | None = None,
    weight: float = 1.0,
) -> SnapshotResidualLoss:
    stds_path = _write_scalar_stds(tmp_path, out_names, stds)
    config = SnapshotResidualLossConfig(
        steps=steps,
        residual_stds_path=stds_path,
        loss=StepLossConfig(type="MSE", weights={} if weights is None else weights),
        weight=weight,
    )
    gridded_ops = LatLonOperations(torch.ones(1, 1))
    return config.build(gridded_ops=gridded_ops, out_names=out_names)


def test_config_requires_steps(tmp_path):
    stds_path = _write_scalar_stds(tmp_path, ["a"])
    with pytest.raises(ValueError, match="at least one"):
        SnapshotResidualLossConfig(steps=[], residual_stds_path=stds_path)


def test_config_rejects_step_below_one(tmp_path):
    stds_path = _write_scalar_stds(tmp_path, ["a"])
    with pytest.raises(ValueError, match="must be >= 1"):
        SnapshotResidualLossConfig(steps=[0], residual_stds_path=stds_path)


def test_config_max_step(tmp_path):
    stds_path = _write_scalar_stds(tmp_path, ["a"])
    config = SnapshotResidualLossConfig(steps=[1, 3, 2], residual_stds_path=stds_path)
    assert config.max_step == 3


def test_step_label():
    assert step_label(1) == "step_1_minus_0"
    assert step_label(3) == "step_3_minus_2"


def test_loss_matches_manual_mse(tmp_path):
    """Residual MSE on standardized absolute residuals matches manual computation."""
    torch.manual_seed(0)
    out_names = ["a", "b"]
    std_a, std_b = 2.0, 0.5
    loss = _build_loss(
        out_names,
        steps=[1],
        tmp_path=tmp_path,
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


def test_loss_consecutive_steps(tmp_path):
    """Step 2 computes |gen[2]-gen[1]| vs |target[2]-target[1]|."""
    torch.manual_seed(0)
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[2], tmp_path=tmp_path)

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


def test_variable_weights_applied(tmp_path):
    out_names = ["a", "b"]
    weights = {"a": 4.0, "b": 1.0}
    loss = _build_loss(out_names, steps=[1], tmp_path=tmp_path, weights=weights)

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


def test_update_max_n_forward_steps_filters_steps(tmp_path):
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1, 3, 40], tmp_path=tmp_path)
    assert set(loss.active_steps) == {1, 3, 40}

    loss.update_max_n_forward_steps(2)
    assert loss.active_steps == [1]
    assert loss.needed_steps() == {0, 1}

    loss.update_max_n_forward_steps(0)
    assert loss.active_steps == []
    assert loss.needed_steps() == set()

    loss.update_max_n_forward_steps(40)
    assert set(loss.active_steps) == {1, 3, 40}


def test_call_with_no_active_steps_returns_zero(tmp_path):
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[5], tmp_path=tmp_path)
    loss.update_max_n_forward_steps(1)
    assert loss.active_steps == []

    total, per_step = loss({}, {})
    assert per_step == {}
    torch.testing.assert_close(total, torch.zeros((), device=total.device))


def test_call_missing_step_raises(tmp_path):
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1], tmp_path=tmp_path)
    n_sample, h, w = 2, 3, 3
    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    with pytest.raises(KeyError, match="prediction"):
        loss({0: ic}, {0: ic, 1: ic})
    with pytest.raises(KeyError, match="target"):
        loss({0: ic, 1: ic}, {0: ic})


def test_total_sums_over_active_steps(tmp_path):
    """Total returned is the unweighted sum over active steps."""
    torch.manual_seed(0)
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1, 2], tmp_path=tmp_path)

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


def test_loss_propagates_gradients(tmp_path):
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1], tmp_path=tmp_path)
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


def test_loss_detaches_reference_endpoint(tmp_path):
    """Reference endpoint (step-1) never receives a gradient."""
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1], tmp_path=tmp_path)
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


def test_loss_detaches_reference_step2(tmp_path):
    """For step=2, gen[1] is the detached reference."""
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[2], tmp_path=tmp_path)
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


def test_steps_completing_at(tmp_path):
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1, 2, 3], tmp_path=tmp_path)
    assert loss.steps_completing_at(1) == [1]
    assert loss.steps_completing_at(2) == [2]
    assert loss.steps_completing_at(3) == [3]
    assert loss.steps_completing_at(4) == []


def test_steps_completing_at_respects_active_filtering(tmp_path):
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1, 3], tmp_path=tmp_path)
    loss.update_max_n_forward_steps(1)
    assert loss.steps_completing_at(1) == [1]
    assert loss.steps_completing_at(3) == []


def test_compute_step_loss_matches_call(tmp_path):
    """compute_step_loss is the per-step primitive used inside __call__."""
    torch.manual_seed(0)
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1], tmp_path=tmp_path)

    n_sample, h, w = 2, 3, 3
    ic = {"a": torch.randn(n_sample, 1, h, w, device=get_device())}
    pred1 = {"a": torch.randn(n_sample, 1, h, w, device=get_device())}
    target1 = {"a": torch.randn(n_sample, 1, h, w, device=get_device())}

    via_call, _ = loss({0: ic, 1: pred1}, {0: ic, 1: target1})
    via_step = loss.compute_step_loss(1, {0: ic, 1: pred1}, {0: ic, 1: target1})
    torch.testing.assert_close(via_call, via_step)


def test_compute_residuals_returns_abs_diffs(tmp_path):
    """compute_residuals returns absolute temporal differences."""
    out_names = ["a", "b"]
    loss = _build_loss(out_names, steps=[1], tmp_path=tmp_path)

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


def test_compute_residuals_detaches_reference(tmp_path):
    """compute_residuals detaches the reference endpoint."""
    out_names = ["a"]
    loss = _build_loss(out_names, steps=[1], tmp_path=tmp_path)

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


def _build_loss_with_std_maps(
    out_names: list[str],
    steps: list[int],
    residual_std_maps: dict[str, torch.Tensor],
    stds: dict[str, float] | None = None,
    weight: float = 1.0,
    eps: float = 1e-2,
) -> SnapshotResidualLoss:
    if stds is None:
        stds = {name: 1.0 for name in out_names}
    identity_normalizer = StandardNormalizer(
        means={name: torch.zeros(()) for name in out_names},
        stds={name: torch.ones(()) for name in out_names},
    )
    inner_loss = torch.nn.MSELoss(reduction="none").to(get_device())
    weighted_mapping_loss = WeightedMappingLoss(
        loss=inner_loss,
        weights={},
        out_names=out_names,
        channel_dim=-3,
        normalizer=identity_normalizer,
    )
    return SnapshotResidualLoss(
        steps=steps,
        loss=weighted_mapping_loss,
        residual_stds=stds,
        out_names=out_names,
        weight=weight,
        residual_std_maps=residual_std_maps,
        eps=eps,
    )


def test_std_maps_normalization_scales_residuals():
    """Local std normalization divides residuals by (sigma_l + eps * sigma_g)."""
    h, w = 4, 5
    n_sample = 2
    sigma_l = 2.0 * torch.ones(1, 1, h, w, device=get_device())
    sigma_g = 1.0
    eps = 1e-2
    residual_std_maps = {"a": sigma_l}
    loss = _build_loss_with_std_maps(
        ["a"],
        steps=[1],
        residual_std_maps=residual_std_maps,
        stds={"a": sigma_g},
        eps=eps,
    )

    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    pred1 = {"a": torch.ones(n_sample, 1, h, w, device=get_device())}
    target1 = {"a": 2 * torch.ones(n_sample, 1, h, w, device=get_device())}

    gen_res, tgt_res = loss.compute_residuals(1, {0: ic, 1: pred1}, {0: ic, 1: target1})
    denom = sigma_l + eps * sigma_g
    torch.testing.assert_close(
        gen_res["a"],
        (1.0 / denom) * torch.ones(n_sample, 1, h, w, device=get_device()),
    )
    torch.testing.assert_close(
        tgt_res["a"],
        (2.0 / denom) * torch.ones(n_sample, 1, h, w, device=get_device()),
    )


def test_std_maps_normalization_spatially_varying():
    """Different grid cells get different normalization."""
    h, w = 2, 2
    n_sample = 1
    sigma_l = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=get_device())
    residual_std_maps = {"a": sigma_l.unsqueeze(0).unsqueeze(0)}
    sigma_g = 1.0
    eps = 1e-2
    loss = _build_loss_with_std_maps(
        ["a"],
        steps=[1],
        residual_std_maps=residual_std_maps,
        stds={"a": sigma_g},
        eps=eps,
    )

    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    pred1 = {"a": torch.ones(n_sample, 1, h, w, device=get_device())}
    target1 = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}

    gen_res, _ = loss.compute_residuals(1, {0: ic, 1: pred1}, {0: ic, 1: target1})
    expected = (1.0 / (sigma_l + eps * sigma_g)).unsqueeze(0).unsqueeze(0)
    torch.testing.assert_close(gen_res["a"], expected)


def test_std_maps_normalization_affects_loss(tmp_path):
    """Loss with std maps normalization differs from loss without it."""
    h, w = 3, 3
    n_sample = 2
    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    pred1 = {"a": torch.ones(n_sample, 1, h, w, device=get_device())}
    target1 = {"a": 2 * torch.ones(n_sample, 1, h, w, device=get_device())}

    loss_no_maps = _build_loss(["a"], steps=[1], tmp_path=tmp_path)
    total_no_maps, _ = loss_no_maps({0: ic, 1: pred1}, {0: ic, 1: target1})

    residual_std_maps = {"a": 4.0 * torch.ones(1, 1, h, w, device=get_device())}
    loss_with_maps = _build_loss_with_std_maps(
        ["a"], steps=[1], residual_std_maps=residual_std_maps
    )
    total_with_maps, _ = loss_with_maps({0: ic, 1: pred1}, {0: ic, 1: target1})

    assert not torch.allclose(total_no_maps, total_with_maps)


def test_std_maps_normalization_gradient_flow():
    """Gradients flow through the std-maps-normalized loss."""
    h, w = 3, 3
    n_sample = 2
    residual_std_maps = {"a": 4.0 * torch.ones(1, 1, h, w, device=get_device())}
    loss = _build_loss_with_std_maps(
        ["a"], steps=[1], residual_std_maps=residual_std_maps
    )

    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    pred1 = {
        "a": torch.ones(n_sample, 1, h, w, device=get_device(), requires_grad=True)
    }
    target1 = {"a": 2 * torch.ones(n_sample, 1, h, w, device=get_device())}

    total, _ = loss({0: ic, 1: pred1}, {0: ic, 1: target1})
    total.backward()
    assert pred1["a"].grad is not None
    assert torch.all(pred1["a"].grad != 0)


def test_std_maps_with_nontrivial_scalar_stds():
    """Combined scalar stds and std maps: |r| / (sigma_l + eps * sigma_g)."""
    h, w = 3, 3
    n_sample = 1
    sigma_g = 2.0
    sigma_l_val = 3.0
    eps = 1e-2
    sigma_l = sigma_l_val * torch.ones(1, 1, h, w, device=get_device())
    residual_std_maps = {"a": sigma_l}
    loss = _build_loss_with_std_maps(
        ["a"],
        steps=[1],
        residual_std_maps=residual_std_maps,
        stds={"a": sigma_g},
        eps=eps,
    )

    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    pred1 = {"a": 6.0 * torch.ones(n_sample, 1, h, w, device=get_device())}
    target1 = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}

    gen_res, _ = loss.compute_residuals(1, {0: ic, 1: pred1}, {0: ic, 1: target1})
    # |r| = 6, denom = 3 + 0.01 * 2 = 3.02
    expected_val = 6.0 / (sigma_l_val + eps * sigma_g)
    torch.testing.assert_close(
        gen_res["a"],
        expected_val * torch.ones(n_sample, 1, h, w, device=get_device()),
    )


def test_residual_scale_scales_loss(tmp_path):
    """full_state_stds_path scales per-variable loss by sigma_r / sigma_f."""
    torch.manual_seed(0)
    out_names = ["a", "b"]
    residual_stds = {"a": 2.0, "b": 0.5}
    full_state_stds = {"a": 4.0, "b": 1.0}

    residual_stds_path = _write_scalar_stds(tmp_path, out_names, residual_stds)
    full_state_ds = xr.Dataset(
        {name: xr.DataArray(full_state_stds[name]) for name in out_names}
    )
    full_state_path = tmp_path / "full_state_stds.nc"
    full_state_ds.to_netcdf(full_state_path)

    config_with = SnapshotResidualLossConfig(
        steps=[1],
        residual_stds_path=residual_stds_path,
        full_state_stds_path=str(full_state_path),
    )
    config_without = SnapshotResidualLossConfig(
        steps=[1],
        residual_stds_path=residual_stds_path,
    )
    gridded_ops = LatLonOperations(torch.ones(1, 1))
    loss_with = config_with.build(gridded_ops=gridded_ops, out_names=out_names)
    loss_without = config_without.build(gridded_ops=gridded_ops, out_names=out_names)

    n_sample, h, w = 2, 3, 3
    ic = {
        "a": torch.zeros(n_sample, 1, h, w, device=get_device()),
        "b": torch.zeros(n_sample, 1, h, w, device=get_device()),
    }
    pred1 = {
        "a": torch.randn(n_sample, 1, h, w, device=get_device()),
        "b": torch.randn(n_sample, 1, h, w, device=get_device()),
    }
    target1 = {
        "a": torch.randn(n_sample, 1, h, w, device=get_device()),
        "b": torch.randn(n_sample, 1, h, w, device=get_device()),
    }

    gen_with, tgt_with = loss_with.compute_residuals(
        1, {0: ic, 1: pred1}, {0: ic, 1: target1}
    )
    gen_without, tgt_without = loss_without.compute_residuals(
        1, {0: ic, 1: pred1}, {0: ic, 1: target1}
    )

    import math

    for name in out_names:
        scale = math.sqrt(residual_stds[name] / full_state_stds[name])
        torch.testing.assert_close(gen_with[name], gen_without[name] * scale)
        torch.testing.assert_close(tgt_with[name], tgt_without[name] * scale)


def test_residual_scale_single_variable_loss_value(tmp_path):
    """Verify that the MSE loss is scaled by sigma_r / sigma_f."""
    out_names = ["a"]
    sigma_r = 2.0
    sigma_f = 8.0

    residual_stds_path = _write_scalar_stds(tmp_path, out_names, {"a": sigma_r})
    full_state_ds = xr.Dataset({"a": xr.DataArray(sigma_f)})
    full_state_path = tmp_path / "full_state_stds.nc"
    full_state_ds.to_netcdf(full_state_path)

    config = SnapshotResidualLossConfig(
        steps=[1],
        residual_stds_path=residual_stds_path,
        full_state_stds_path=str(full_state_path),
    )
    gridded_ops = LatLonOperations(torch.ones(1, 1))
    loss = config.build(gridded_ops=gridded_ops, out_names=out_names)

    n_sample, h, w = 2, 3, 3
    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    pred1 = {"a": torch.ones(n_sample, 1, h, w, device=get_device())}
    target1 = {"a": 2 * torch.ones(n_sample, 1, h, w, device=get_device())}

    total, _ = loss({0: ic, 1: pred1}, {0: ic, 1: target1})
    # Without scaling: MSE((1/sigma_r), (2/sigma_r)) = (1/sigma_r)^2 = 0.25
    # With scaling: 0.25 * (sigma_r / sigma_f) = 0.25 * 0.25 = 0.0625
    expected = torch.tensor(
        (1.0 / sigma_r) ** 2 * (sigma_r / sigma_f), device=get_device()
    )
    torch.testing.assert_close(total, expected)


def test_load_std_maps(tmp_path):
    """load_std_maps reads 2D variables from a netCDF file."""
    h, w = 4, 8
    ds = xr.Dataset(
        {
            "a": xr.DataArray(
                np.random.rand(h, w).astype(np.float32), dims=["lat", "lon"]
            ),
            "b": xr.DataArray(
                np.random.rand(h, w).astype(np.float32), dims=["lat", "lon"]
            ),
        }
    )
    path = tmp_path / "std_maps.nc"
    ds.to_netcdf(path)

    result = load_std_maps(str(path), ["a", "b"])
    assert set(result.keys()) == {"a", "b"}
    assert result["a"].shape == (1, 1, h, w)
    assert result["b"].shape == (1, 1, h, w)
    np.testing.assert_allclose(
        result["a"].cpu().numpy().squeeze(), ds["a"].values, rtol=1e-6
    )


def test_load_std_maps_missing_variable(tmp_path):
    """load_std_maps raises if a requested variable is absent."""
    ds = xr.Dataset(
        {"a": xr.DataArray(np.ones((3, 3), dtype=np.float32), dims=["lat", "lon"])}
    )
    path = tmp_path / "std_maps.nc"
    ds.to_netcdf(path)

    with pytest.raises(ValueError, match="not found"):
        load_std_maps(str(path), ["a", "b"])


def test_load_std_maps_wrong_ndim(tmp_path):
    """load_std_maps rejects non-2D variables."""
    ds = xr.Dataset(
        {"a": xr.DataArray(np.ones((2, 3, 4), dtype=np.float32), dims=["t", "y", "x"])}
    )
    path = tmp_path / "std_maps.nc"
    ds.to_netcdf(path)

    with pytest.raises(ValueError, match="2D"):
        load_std_maps(str(path), ["a"])


def test_config_build_with_std_maps_path(tmp_path):
    """SnapshotResidualLossConfig.build loads std maps from a file."""
    h, w = 4, 5
    ds = xr.Dataset(
        {"a": xr.DataArray(np.full((h, w), 2.0, dtype=np.float32), dims=["lat", "lon"])}
    )
    maps_path = tmp_path / "std_maps.nc"
    ds.to_netcdf(maps_path)

    stds_path = _write_scalar_stds(tmp_path, ["a"], stds={"a": 1.0})
    config = SnapshotResidualLossConfig(
        steps=[1], residual_stds_path=stds_path, std_maps_path=str(maps_path)
    )
    gridded_ops = LatLonOperations(torch.ones(1, 1))
    loss = config.build(gridded_ops=gridded_ops, out_names=["a"])

    n_sample = 2
    ic = {"a": torch.zeros(n_sample, 1, h, w, device=get_device())}
    pred1 = {"a": 3.0 * torch.ones(n_sample, 1, h, w, device=get_device())}
    target1 = {"a": 6.0 * torch.ones(n_sample, 1, h, w, device=get_device())}

    gen_res, tgt_res = loss.compute_residuals(1, {0: ic, 1: pred1}, {0: ic, 1: target1})
    # sigma_l = 2.0, sigma_g = 1.0, eps = 1e-2 (default)
    denom = 2.0 + loss._eps * 1.0
    torch.testing.assert_close(
        gen_res["a"],
        (3.0 / denom) * torch.ones(n_sample, 1, h, w, device=get_device()),
    )
    torch.testing.assert_close(
        tgt_res["a"],
        (6.0 / denom) * torch.ones(n_sample, 1, h, w, device=get_device()),
    )
