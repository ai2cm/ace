import pytest
import torch

from fme.core.coordinates import DepthCoordinate, HybridSigmaPressureCoordinate
from fme.core.device import get_device
from fme.core.loss import StepLossConfig
from fme.core.normalizer import StandardNormalizer
from fme.core.ocean_eos import G_EARTH, RHO_0, wright97_anomaly
from fme.core.optimized_derived import (
    OptimizedDerivedVariableConfig,
    build_optimized_derived_variables,
)

N_LAT, N_LON, N_LEVELS = 4, 6, 2
IDEPTH = torch.tensor([0.0, 10.0, 1000.0])
NAMES = [f"{v}_{k}" for v in ("so", "thetao") for k in range(N_LEVELS)]


def _mask() -> torch.Tensor:
    mask = torch.ones(N_LAT, N_LON, N_LEVELS)
    mask[0, :, :] = 0.0  # land
    mask[1, :, 1] = 0.0  # below the sea floor at level 1
    return mask


def _normalizer(mean_S=35.0, mean_T=10.0, std_S=0.5, std_T=3.0):
    means = {f"so_{k}": mean_S for k in range(N_LEVELS)}
    means.update({f"thetao_{k}": mean_T for k in range(N_LEVELS)})
    stds = {f"so_{k}": std_S for k in range(N_LEVELS)}
    stds.update({f"thetao_{k}": std_T for k in range(N_LEVELS)})
    return StandardNormalizer(
        means={k: torch.tensor(v) for k, v in means.items()},
        stds={k: torch.tensor(v) for k, v in stds.items()},
    )


def _build(configs=None, vertical_coordinate=None, loss_names=NAMES):
    return build_optimized_derived_variables(
        configs if configs is not None else [OptimizedDerivedVariableConfig()],
        vertical_coordinate=vertical_coordinate
        if vertical_coordinate is not None
        else DepthCoordinate(IDEPTH, _mask()),
        network_normalizer=_normalizer(),
        loss_normalizer=_normalizer(),
        loss_names=loss_names,
    )


def _data(shape=(2, 1, N_LAT, N_LON), seed=0):
    g = torch.Generator().manual_seed(seed)
    data = {}
    for k in range(N_LEVELS):
        data[f"so_{k}"] = 34.0 + 2.0 * torch.rand(shape, generator=g)
        data[f"thetao_{k}"] = 2.0 + 20.0 * torch.rand(shape, generator=g)
    return {k: v.to(get_device()) for k, v in data.items()}


def test_derive_values_and_mask():
    derived = _build()
    assert derived.names == ["rho_wright97_0", "rho_wright97_1"]
    data = _data()
    data["thetao_0"][0, 0, 2, 3] = torch.nan
    out = derived(data)
    mask = _mask().to(get_device()) > 0
    for k in range(N_LEVELS):
        p = RHO_0 * G_EARTH * 0.5 * float(IDEPTH[k] + IDEPTH[k + 1])
        expected = wright97_anomaly(
            data[f"so_{k}"], data[f"thetao_{k}"], torch.tensor(p), RHO_0
        )
        valid = mask[..., k] & expected.isfinite()
        torch.testing.assert_close(out[f"rho_wright97_{k}"][valid], expected[valid])
        assert out[f"rho_wright97_{k}"][~valid.expand_as(expected)].isnan().all()
    assert out["rho_wright97_0"][0, 0, 2, 3].isnan()
    assert out["rho_wright97_0"][0, 0, 1].isfinite().all()
    assert out["rho_wright97_1"][0, 0, 1].isnan().all()


def test_levels_weight_and_stds_override():
    derived = _build(
        [
            OptimizedDerivedVariableConfig(
                levels=[1], weight=0.25, stds={"rho_wright97_1": 2.0}
            )
        ]
    )
    assert derived.names == ["rho_wright97_1"]
    assert derived.weights == {"rho_wright97_1": 0.25}
    assert derived.stds == {"rho_wright97_1": 2.0}
    assert set(derived(_data())) == {"rho_wright97_1"}


def test_linearized_std():
    derived = _build()
    for k in range(N_LEVELS):
        p = torch.tensor(RHO_0 * G_EARTH * 0.5 * float(IDEPTH[k] + IDEPTH[k + 1]))
        S = torch.tensor(35.0, dtype=torch.float64, requires_grad=True)
        T = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
        dS, dT = torch.autograd.grad(wright97_anomaly(S, T, p.double()), (S, T))
        expected = float(((dT * 3.0) ** 2 + (dS * 0.5) ** 2).sqrt())
        assert derived.stds[f"rho_wright97_{k}"] == pytest.approx(expected, rel=1e-6)
    # thermal term dominates here: roughly alpha * rho * std_T
    assert 0.1 < derived.stds["rho_wright97_0"] < 1.5


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"configs": [OptimizedDerivedVariableConfig(levels=[2])]}, "outside"),
        ({"configs": [OptimizedDerivedVariableConfig()] * 2}, "not unique"),
        ({"loss_names": ["so_0", "so_1", "thetao_0"]}, "thetao_1"),
        (
            {"configs": [OptimizedDerivedVariableConfig(stds={"rho_wright97_9": 1.0})]},
            "rho_wright97_9",
        ),
        (
            {"loss_names": NAMES + ["rho_wright97_0"]},
            "not unique",
        ),
        (
            {
                "vertical_coordinate": HybridSigmaPressureCoordinate(
                    ak=torch.tensor([0.0, 1.0]), bk=torch.tensor([0.0, 1.0])
                )
            },
            "DepthCoordinate",
        ),
    ],
)
def test_build_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _build(**kwargs)


def test_step_loss_gradient_reaches_thetao_and_so():
    """With the output-name weights at zero, the whole loss is the rho_wright97 term,
    so nonzero gradients on thetao and so come through the EOS."""
    derived = _build()
    step_loss = StepLossConfig(type="MSE", weights={n: 0.0 for n in NAMES}).build(
        gridded_ops=None,
        out_names=NAMES,
        normalizer=_normalizer(),
        channel_dim=-3,
        derived=derived,
    )
    mask = _mask().to(get_device()) > 0
    target = _data(seed=0)
    for k in range(N_LEVELS):  # NaN off-mask, as in the ocean datasets
        for v in ("so", "thetao"):
            target[f"{v}_{k}"] = torch.where(
                mask[..., k], target[f"{v}_{k}"], torch.nan
            )
    target["so_1"][1, 0, 3, 4] = torch.nan  # NaN target point on-mask
    predict = {k: v.clone().requires_grad_() for k, v in _data(seed=1).items()}
    # A NaN prediction at a masked point must not poison the gradient.
    with torch.no_grad():
        predict["thetao_0"][0, 0, 0, 0] = torch.nan
    output = step_loss(predict, target, step=0)
    assert set(output.get_channel_losses()) == set(
        NAMES + ["rho_wright97_0", "rho_wright97_1"]
    )
    total = output.total()
    assert total.isfinite() and total > 0
    total.backward()
    for k in range(N_LEVELS):
        for v in ("so", "thetao"):
            grad = predict[f"{v}_{k}"].grad
            assert grad is not None and grad.isfinite().all()
            valid = mask[..., k].expand_as(grad)
            assert (grad[valid] != 0).any()
            assert (grad[~valid] == 0).all()
    assert predict["so_1"].grad[1, 0, 3, 4] == 0


def test_step_loss_without_derived_is_unchanged():
    config = StepLossConfig(type="MSE")
    kwargs = dict(
        gridded_ops=None, out_names=NAMES, normalizer=_normalizer(), channel_dim=-3
    )
    plain = config.build(**kwargs)
    with_none = config.build(**kwargs, derived=None)
    predict, target = _data(seed=1), _data(seed=0)
    a = plain(predict, target, step=0)
    b = with_none(predict, target, step=0)
    assert set(a.get_channel_losses()) == set(NAMES)
    torch.testing.assert_close(a.total(), b.total())


def test_loss_weights_on_derived_names_rejected():
    with pytest.raises(ValueError, match="rho_wright97_0"):
        StepLossConfig(weights={"rho_wright97_0": 2.0}).build(
            gridded_ops=None,
            out_names=NAMES,
            normalizer=_normalizer(),
            derived=_build(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight": -1.0},
        {"levels": []},
        {"levels": [0, 0]},
        {"stds": {"rho_wright97_0": 0.0}},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        OptimizedDerivedVariableConfig(**kwargs)
