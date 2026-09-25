import pytest
import torch

from fme.core.coordinates import DepthCoordinate, HybridSigmaPressureCoordinate
from fme.core.device import get_device
from fme.core.loss import CorrectorLoss, StepLossConfig, StepOutputLoss
from fme.core.name_and_prefix_matcher import NameAndPrefixSelection
from fme.core.normalizer import StandardNormalizer
from fme.core.ocean_eos import G_EARTH, RHO_0, wright97_anomaly
from fme.core.optimized_derived import (
    SO_CLAMP_RANGE,
    THETAO_CLAMP_RANGE,
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
        {"thetao_clamp": [1.0]},
        {"thetao_clamp": [5.0, 5.0]},
        {"so_clamp": [45.0, 0.0]},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        OptimizedDerivedVariableConfig(**kwargs)


# A corrector that changes thetao: thetao_c = A * thetao_net + B on every level.
_A, _B = 0.5, 1.0
THETAO_NAMES = [f"thetao_{k}" for k in range(N_LEVELS)]


def _corrected(net):
    corrected = dict(net)
    for name in THETAO_NAMES:
        corrected[name] = _A * net[name] + _B
    deltas = {name: corrected[name] - net[name] for name in THETAO_NAMES}
    return corrected, deltas


def _step_output_loss(weights=None):
    step_loss = StepLossConfig(type="MSE", weights=weights or {}).build(
        gridded_ops=None,
        out_names=NAMES,
        normalizer=_normalizer(),
        channel_dim=-3,
        derived=_build(),
    )
    corrector_loss = CorrectorLoss(
        precorrector_selection=NameAndPrefixSelection(("thetao_",)),
        regularizer=None,
    )
    return step_loss, StepOutputLoss(step_loss, corrector_loss)


def test_precorrector_rho_from_corrected_thetao():
    """precorrector_optimization on thetao_: the thetao channels see the network
    output, the rho channels see W97 of the corrected thetao."""
    step_loss, loss = _step_output_loss()
    net, target = _data(seed=1), _data(seed=0)
    corrected, deltas = _corrected(net)
    derived_inputs = []
    derive = step_loss._derive
    assert derive is not None

    def recording_derive(data):
        derived_inputs.append(data)
        return derive(data)

    step_loss._derive = recording_derive
    channels = loss(corrected, target, step=0, deltas=deltas).get_channel_losses()
    step_loss._derive = derive
    # the prediction's derived fields are W97(so, thetao_c), computed from corrected
    assert derived_inputs[0] is corrected
    on_corrected = step_loss(corrected, target, step=0).get_channel_losses()
    on_net = step_loss(net, target, step=0).get_channel_losses()
    for name in _build().names:
        torch.testing.assert_close(channels[name].loss, on_corrected[name].loss)
        assert not torch.allclose(channels[name].loss, on_net[name].loss)
    for name in NAMES:
        torch.testing.assert_close(channels[name].loss, on_net[name].loss)


def test_precorrector_rho_gradient_through_corrector():
    """Only rho weighted: dL/d thetao_net = A * dL/d thetao_c, and
    dL/d so_net = dL/d so_c, i.e. the rho gradient passes through the corrector."""
    _, loss = _step_output_loss(weights={n: 0.0 for n in NAMES})
    step_loss, _ = _step_output_loss(weights={n: 0.0 for n in NAMES})
    target = _data(seed=0)
    net = {k: v.clone().requires_grad_() for k, v in _data(seed=1).items()}
    corrected, deltas = _corrected(net)
    loss(corrected, target, step=0, deltas=deltas).total().backward()
    # reference: the same rho-only loss taken directly on the corrected fields
    corrected_leaf = {
        k: v.detach().clone().requires_grad_() for k, v in corrected.items()
    }
    step_loss(corrected_leaf, target, step=0).total().backward()
    # and on the network output with no corrector
    net_leaf = {k: v.detach().clone().requires_grad_() for k, v in net.items()}
    step_loss(net_leaf, target, step=0).total().backward()
    for k in range(N_LEVELS):
        g_net = net[f"thetao_{k}"].grad
        g_c = corrected_leaf[f"thetao_{k}"].grad
        assert g_net is not None and g_c is not None
        assert (g_c != 0).any()
        torch.testing.assert_close(g_net, _A * g_c)
        assert not torch.allclose(g_net, net_leaf[f"thetao_{k}"].grad)
        torch.testing.assert_close(net[f"so_{k}"].grad, corrected_leaf[f"so_{k}"].grad)


def test_config_defaults():
    config = OptimizedDerivedVariableConfig()
    assert config.thetao_clamp == list(THETAO_CLAMP_RANGE)
    assert config.so_clamp == list(SO_CLAMP_RANGE)


# T* = W97 denominator zero at level 0, S = 35, near -81.5 degC
_T_SCAN = torch.linspace(-120.0, 20.0, 1401, dtype=torch.float32)


def test_unclamped_eos_gradient_blows_up_near_pole():
    """Reference for the clamp test: without the clamp the scan crosses T*."""
    p = torch.tensor(RHO_0 * G_EARTH * 5.0)
    T = _T_SCAN.clone().requires_grad_()
    rho = wright97_anomaly(torch.full_like(T, 35.0), T, p)
    (grad,) = torch.autograd.grad(rho.sum(), T)
    assert not grad.isfinite().all() or grad.abs().max() > 1e6


def test_clamp_gradient_finite_at_pole_adjacent_inputs():
    """thetao scanned across T*: loss and gradients finite, and zero rho
    gradient outside the clamp box; so above the box likewise."""
    shape = (1, 1, 1, _T_SCAN.numel())
    mask = torch.ones(1, _T_SCAN.numel(), N_LEVELS)
    derived = build_optimized_derived_variables(
        [OptimizedDerivedVariableConfig()],
        vertical_coordinate=DepthCoordinate(IDEPTH, mask),
        network_normalizer=_normalizer(),
        loss_normalizer=_normalizer(),
        loss_names=NAMES,
    )
    step_loss = StepLossConfig(type="MSE", weights={n: 0.0 for n in NAMES}).build(
        gridded_ops=None,
        out_names=NAMES,
        normalizer=_normalizer(),
        channel_dim=-3,
        derived=derived,
    )
    device = get_device()
    target = {
        **{f"so_{k}": torch.full(shape, 35.0) for k in range(N_LEVELS)},
        **{f"thetao_{k}": torch.full(shape, 10.0) for k in range(N_LEVELS)},
    }
    target = {k: v.to(device) for k, v in target.items()}
    predict = {k: v.clone() for k, v in target.items()}
    predict["thetao_0"] = _T_SCAN.reshape(shape).to(device)
    predict["so_1"] = torch.linspace(20.0, 200.0, _T_SCAN.numel()).reshape(shape)
    predict["so_1"] = predict["so_1"].to(device)
    predict = {k: v.clone().requires_grad_() for k, v in predict.items()}
    total = step_loss(predict, target, step=0).total()
    assert total.isfinite()
    total.backward()
    for name, lo, hi in (
        ("thetao_0", *THETAO_CLAMP_RANGE),
        ("so_1", *SO_CLAMP_RANGE),
    ):
        x = predict[name].detach()
        grad = predict[name].grad
        assert grad is not None and grad.isfinite().all()
        outside = (x < lo) | (x > hi)
        assert outside.any() and (grad[outside] == 0).all()
        assert (grad[~outside] != 0).any()
