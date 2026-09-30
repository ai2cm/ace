import pytest
import torch

from fme.core.constants import DENSITY_OF_SEA_WATER_CM4, SPECIFIC_HEAT_OF_SEA_WATER_CM4
from fme.core.coordinates import DepthCoordinate, HybridSigmaPressureCoordinate
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.loss import CorrectorLoss, StepLossConfig, StepOutputLoss
from fme.core.name_and_prefix_matcher import NameAndPrefixSelection
from fme.core.normalizer import StandardNormalizer
from fme.core.ocean_data import OceanData
from fme.core.ocean_eos import G_EARTH, RHO_0, wright97_anomaly
from fme.core.optimized_derived import (
    LAYER_OHC_DEFAULT_BANDS,
    LAYER_OHC_DEFAULT_STDS,
    PBO_WRIGHT97_STD,
    SO_CLAMP_RANGE,
    STERIC_HEIGHT_WRIGHT97_STD,
    THETAO_CLAMP_RANGE,
    OptimizedDerivedVariableConfig,
    build_optimized_derived_variables,
    layer_ohc_name,
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


# ---------------------------------------------------------------- column variables
# pbo_wright97 = P - <P>,  P = RHO_0 g zos + g sum_k rho_k dz_k
# steric_height_wright97 = -(C - <C>) / RHO_0

DEPTHO = torch.full((N_LAT, N_LON), 1000.0)
DEPTHO[2, :] = 400.0  # partial bottom cell at level 1
DEPTHO[1, :] = 10.0  # level 1 dry (mask[1, :, 1] == 0)
COLUMN_NAMES = NAMES + ["zos"]


def _ops() -> LatLonOperations:
    # non-uniform in latitude, uniform in longitude
    area = torch.linspace(1.0, 2.0, N_LAT)[:, None].expand(N_LAT, N_LON).clone()
    return LatLonOperations(area)


def _build_column(names=("pbo_wright97",), loss_names=COLUMN_NAMES, ops="default"):
    return build_optimized_derived_variables(
        [OptimizedDerivedVariableConfig(name=n, stds={n: 1.0}) for n in names],
        vertical_coordinate=DepthCoordinate(IDEPTH, _mask(), DEPTHO),
        network_normalizer=_normalizer(),
        loss_normalizer=_normalizer(),
        loss_names=list(loss_names),
        gridded_operations=_ops() if ops == "default" else ops,
    )


def _column_data(seed=0):
    data = _data(seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    data["zos"] = (0.5 * torch.randn((2, 1, N_LAT, N_LON), generator=g)).to(
        get_device()
    )
    return data


def _expected_column(data):
    """Hand computation: dz from idepth and deptho, uniform-in-lon area mean
    over mask_0."""
    mask = _mask().to(get_device())
    C = torch.zeros_like(data["zos"])
    for k in range(N_LEVELS):
        z_top, z_bot = float(IDEPTH[k]), float(IDEPTH[k + 1])
        dz = (DEPTHO.clamp(z_top, z_bot) - z_top).to(get_device()) * mask[..., k]
        p = torch.tensor(RHO_0 * G_EARTH * 0.5 * (z_top + z_bot))
        rho = wright97_anomaly(data[f"so_{k}"], data[f"thetao_{k}"], p, RHO_0)
        C = C + torch.where(mask[..., k] > 0, rho * dz, 0.0)
    wet = (mask[..., 0] > 0).expand_as(C)
    w = (_ops()._cpu_area.to(get_device()) * wet).to(C.dtype)

    def demean(x):
        return x - (x * w).sum((-2, -1), keepdim=True) / w.sum((-2, -1), keepdim=True)

    P = RHO_0 * G_EARTH * data["zos"] + G_EARTH * C
    return (
        torch.where(wet, demean(P), torch.nan),
        torch.where(wet, -demean(C) / RHO_0, torch.nan),
    )


def test_column_values_and_mask():
    derived = _build_column(("pbo_wright97", "steric_height_wright97"))
    assert derived.names == ["pbo_wright97", "steric_height_wright97"]
    data = _column_data()
    out = derived(data)
    pbo, steric = _expected_column(data)
    torch.testing.assert_close(out["pbo_wright97"], pbo, equal_nan=True)
    torch.testing.assert_close(out["steric_height_wright97"], steric, equal_nan=True)
    assert out["pbo_wright97"][:, :, 0].isnan().all()  # land row
    assert out["pbo_wright97"][:, :, 1:].isfinite().all()
    # partial bottom cell: row 2 differs from a full-cell column
    full = build_optimized_derived_variables(
        [OptimizedDerivedVariableConfig(name="pbo_wright97", stds={"pbo_wright97": 1})],
        vertical_coordinate=DepthCoordinate(IDEPTH, _mask()),
        network_normalizer=_normalizer(),
        loss_normalizer=_normalizer(),
        loss_names=COLUMN_NAMES,
        gridded_operations=_ops(),
    )(data)["pbo_wright97"]
    assert not torch.allclose(full[:, :, 2], out["pbo_wright97"][:, :, 2])


def test_column_global_mean_removed():
    derived = _build_column(("pbo_wright97", "steric_height_wright97"))
    data = _column_data()
    out = derived(data)
    w = _ops()._cpu_area.to(get_device()) * (_mask().to(get_device())[..., 0] > 0)
    for name in derived.names:
        mean = (out[name].nan_to_num() * w).sum((-2, -1)) / w.sum()
        torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-2, rtol=0)
    # uniform zos shift and a uniform NaN off-mask do not change pbo
    shifted = dict(data, zos=data["zos"] + 0.3)
    shifted["zos"][:, :, 0] = torch.nan
    torch.testing.assert_close(
        derived(shifted)["pbo_wright97"],
        out["pbo_wright97"],
        equal_nan=True,
        atol=1e-2,  # float32 at |P| ~ 1e3 Pa
        rtol=0,
    )


def test_column_gradient_reaches_zos_thetao_so():
    derived = _build_column(("pbo_wright97",))
    step_loss = StepLossConfig(
        type="MSE", weights={n: 0.0 for n in COLUMN_NAMES}
    ).build(
        gridded_ops=None,
        out_names=COLUMN_NAMES,
        normalizer=StandardNormalizer(
            means={n: torch.tensor(0.0) for n in COLUMN_NAMES},
            stds={n: torch.tensor(1.0) for n in COLUMN_NAMES},
        ),
        channel_dim=-3,
        derived=derived,
    )
    mask = _mask().to(get_device()) > 0
    target = _column_data(seed=0)
    for k in range(N_LEVELS):  # NaN off-mask, as in the ocean datasets
        for v in ("so", "thetao"):
            target[f"{v}_{k}"] = torch.where(
                mask[..., k], target[f"{v}_{k}"], torch.nan
            )
    target["zos"] = torch.where(mask[..., 0], target["zos"], torch.nan)
    predict = {k: v.clone().requires_grad_() for k, v in _column_data(seed=1).items()}
    with torch.no_grad():  # NaN predictions at masked points
        predict["thetao_0"][0, 0, 0, 0] = torch.nan
        predict["zos"][0, 0, 0, 1] = torch.nan
    output = step_loss(predict, target, step=0)
    assert "pbo_wright97" in output.get_channel_losses()
    total = output.total()
    assert total.isfinite() and total > 0
    total.backward()
    for name in COLUMN_NAMES:
        grad = predict[name].grad
        assert grad is not None and grad.isfinite().all(), name
        k = 0 if name == "zos" else int(name.split("_")[-1])
        valid = mask[..., k].expand_as(grad)
        assert (grad[valid] != 0).any(), name
        assert (grad[~valid] == 0).all(), name


def test_column_default_stds():
    derived = build_optimized_derived_variables(
        [
            OptimizedDerivedVariableConfig(name="pbo_wright97"),
            OptimizedDerivedVariableConfig(name="steric_height_wright97", weight=0.5),
        ],
        vertical_coordinate=DepthCoordinate(IDEPTH, _mask(), DEPTHO),
        network_normalizer=_normalizer(),
        loss_normalizer=_normalizer(),
        loss_names=COLUMN_NAMES,
        gridded_operations=_ops(),
    )
    assert derived.stds == {
        "pbo_wright97": PBO_WRIGHT97_STD,
        "steric_height_wright97": STERIC_HEIGHT_WRIGHT97_STD,
    }
    assert derived.weights == {"pbo_wright97": 1.0, "steric_height_wright97": 0.5}
    assert derived.means == {"pbo_wright97": 0.0, "steric_height_wright97": 0.0}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"loss_names": NAMES}, "zos"),
        ({"loss_names": ["zos", "so_0", "thetao_0", "so_1"]}, "thetao_1"),
        ({"ops": None}, "gridded"),
        ({"loss_names": COLUMN_NAMES + ["pbo_wright97"]}, "not unique"),
    ],
)
def test_column_build_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _build_column(**kwargs)


def test_steric_height_does_not_need_zos():
    derived = _build_column(("steric_height_wright97",), loss_names=NAMES)
    data = _column_data()
    del data["zos"]
    assert set(derived(data)) == {"steric_height_wright97"}


@pytest.mark.parametrize("name", ["pbo_wright97", "steric_height_wright97"])
def test_column_config_rejects_levels(name):
    with pytest.raises(ValueError, match="levels"):
        OptimizedDerivedVariableConfig(name=name, levels=[0])


# ---------------------------------------------------------------- layer_ohc
# layer_ohc_b = RHO_0 c_p sum_k thetao_k h_b,k,
# h_b,k = clamp(idepth[k] + dz_k, a, b) - clamp(idepth[k], a, b)

RHO_CP = DENSITY_OF_SEA_WATER_CM4 * SPECIFIC_HEAT_OF_SEA_WATER_CM4


def _build_ohc(
    bands=None,
    vertical_coordinate=None,
    loss_names=NAMES,
    ops="default",
    normalizer=None,
    **config_kwargs,
):
    return build_optimized_derived_variables(
        [
            OptimizedDerivedVariableConfig(
                name="layer_ohc", bands=bands, **config_kwargs
            )
        ],
        vertical_coordinate=vertical_coordinate
        if vertical_coordinate is not None
        else DepthCoordinate(IDEPTH, _mask(), DEPTHO),
        network_normalizer=_normalizer(),
        loss_normalizer=normalizer if normalizer is not None else _normalizer(),
        loss_names=list(loss_names),
        gridded_operations=_ops() if ops == "default" else ops,
    )


def test_layer_ohc_hand_computed():
    """Columns (idepth 0, 100, 300; theta 10, 4 degC) with a partial bottom cell,
    band edges inside layers 0 and 1, a land cell and a dry level 1."""
    idepth = torch.tensor([0.0, 100.0, 300.0])
    mask = torch.ones(2, 3, 2)
    mask[0, 0] = 0.0  # land
    mask[1, 0, 1] = 0.0  # level 1 dry
    deptho = torch.tensor([[0.0, 300.0, 250.0], [100.0, 300.0, 180.0]])
    derived = build_optimized_derived_variables(
        [
            OptimizedDerivedVariableConfig(
                name="layer_ohc", bands=[[0, 50], [50, 200], [200, None]]
            )
        ],
        vertical_coordinate=DepthCoordinate(idepth, mask, deptho),
        network_normalizer=_normalizer(),
        loss_normalizer=_normalizer(),
        loss_names=NAMES,
        gridded_operations=LatLonOperations(torch.ones(2, 3)),
    )
    assert derived.names == [
        "layer_ohc_0_50",
        "layer_ohc_50_200",
        "layer_ohc_200_bottom",
    ]
    data = {
        "thetao_0": torch.full((1, 2, 3), 10.0),
        "thetao_1": torch.full((1, 2, 3), 4.0),
    }
    data["thetao_1"][0, 1, 0] = torch.nan  # below the sea floor
    out = {k: v.cpu() for k, v in derived(data).items()}
    nan = float("nan")
    expected = {
        "layer_ohc_0_50": [[nan, 500.0, 500.0], [500.0, 500.0, 500.0]],
        "layer_ohc_50_200": [[nan, 900.0, 900.0], [500.0, 900.0, 820.0]],
        "layer_ohc_200_bottom": [[nan, 400.0, 200.0], [nan, 400.0, nan]],
    }
    for name, values in expected.items():
        torch.testing.assert_close(
            out[name], RHO_CP * torch.tensor([values]), equal_nan=True
        )


_DEEP_IDEPTH = torch.tensor([0.0, 50.0, 200.0, 1000.0, 3000.0, 5000.0])


def _deep_coordinate(with_deptho: bool) -> DepthCoordinate:
    nz = len(_DEEP_IDEPTH) - 1
    mask = torch.ones(N_LAT, N_LON, nz)
    mask[0] = 0.0  # land
    mask[1, :, 3:] = 0.0
    mask[2, :3, 4:] = 0.0
    deptho = torch.full((N_LAT, N_LON), 4600.0)
    deptho[1] = 700.0
    deptho[2, :3] = 2400.0
    return DepthCoordinate(_DEEP_IDEPTH, mask, deptho if with_deptho else None)


@pytest.mark.parametrize("with_deptho", [True, False])
@pytest.mark.parametrize("bands", ["default", "random"])
def test_layer_ohc_sum_is_ocean_heat_content(with_deptho, bands):
    if bands == "default":
        band_list = None
    else:
        g = torch.Generator().manual_seed(1)
        cuts = sorted((5000.0 * torch.rand(4, generator=g)).tolist())
        edges = [0.0, *cuts]
        band_list = [[a, b] for a, b in zip(edges[:-1], edges[1:])]
        band_list.append([edges[-1], None])
    nz = len(_DEEP_IDEPTH) - 1
    names = [f"thetao_{k}" for k in range(nz)]
    normalizer = StandardNormalizer(
        means={n: torch.tensor(0.0) for n in names},
        stds={n: torch.tensor(1.0) for n in names},
    )
    vc = _deep_coordinate(with_deptho)
    derived = build_optimized_derived_variables(
        [OptimizedDerivedVariableConfig(name="layer_ohc", bands=band_list)],
        vertical_coordinate=vc,
        network_normalizer=normalizer,
        loss_normalizer=normalizer,
        loss_names=names,
        gridded_operations=_ops(),
    )
    g = torch.Generator().manual_seed(0)
    data = {
        n: (-2.0 + 30.0 * torch.rand((2, N_LAT, N_LON), generator=g)).to(get_device())
        for n in names
    }
    for k in range(nz):  # NaN below the sea floor, as in the data
        data[f"thetao_{k}"] = torch.where(
            vc.mask[..., k].to(get_device()) > 0, data[f"thetao_{k}"], torch.nan
        )
    out = derived(data)
    total = sum(v.nan_to_num() for v in out.values())
    reference = OceanData(data, vc.to(get_device())).ocean_heat_content
    wet = vc.mask[..., 0].to(get_device()) > 0
    torch.testing.assert_close(total[:, wet], reference[:, wet], rtol=1e-5, atol=0.0)
    for v in out.values():
        assert v[:, ~wet].isnan().all()


def test_layer_ohc_masking_and_gradient():
    """Only layer_ohc carries loss weight: NaN off mask_0 and where a band is
    absent, finite loss and gradients with NaN below the bottom, gradient on
    thetao_k exactly for the levels a band overlaps, none on so_k."""
    # [0, 5] overlaps level 0 only; [20, bottom] level 1 only, absent in row 1
    cases: list[tuple[list[list[float | None]], set[int]]] = [
        ([[0.0, 5.0]], {0}),
        ([[20.0, None]], {1}),
        ([[0.0, 5.0], [20.0, None]], {0, 1}),
    ]
    for bands, levels in cases:
        derived = _build_ohc(bands=bands)
        step_loss = StepLossConfig(type="MSE", weights={n: 0.0 for n in NAMES}).build(
            gridded_ops=None,
            out_names=NAMES,
            normalizer=_normalizer(),
            channel_dim=-3,
            derived=derived,
        )
        mask = _mask().to(get_device()) > 0
        target = _data(seed=0)
        for k in range(N_LEVELS):
            for v in ("so", "thetao"):
                target[f"{v}_{k}"] = torch.where(
                    mask[..., k], target[f"{v}_{k}"], torch.nan
                )
        predict = {k: v.clone().requires_grad_() for k, v in _data(seed=1).items()}
        with torch.no_grad():  # NaN predictions off the mask
            predict["thetao_1"][0, 0, 1, 2] = torch.nan
            predict["thetao_0"][0, 0, 0, 0] = torch.nan
        out = derived(predict)
        for name, ohc in out.items():
            assert ohc[:, :, 0].isnan().all(), name  # land row
            if name.endswith("_bottom"):
                assert ohc[:, :, 1].isnan().all()  # level 1 dry: band absent
                assert ohc[:, :, 2:].isfinite().all()
            else:
                assert ohc[:, :, 1:].isfinite().all()
        total = step_loss(predict, target, step=0).total()
        assert total.isfinite() and total > 0
        total.backward()
        for k in range(N_LEVELS):
            grad = predict[f"thetao_{k}"].grad
            assert grad is not None and grad.isfinite().all()
            if k in levels:
                valid = mask[..., k].expand_as(grad)
                assert (grad[valid] != 0).any()
                assert (grad[~valid] == 0).all()
            else:
                assert (grad == 0).all()
            so_grad = predict[f"so_{k}"].grad
            assert so_grad is None or (so_grad == 0).all()


def test_layer_ohc_default_stds_and_override():
    """s_b = RHO_0 c_p sum_k hbar_b,k sigma_k, hbar over cells where b exists."""
    stds = {"so_0": 0.5, "so_1": 0.5, "thetao_0": 2.0, "thetao_1": 5.0}
    normalizer = StandardNormalizer(
        means={n: torch.tensor(0.0) for n in stds},
        stds={n: torch.tensor(v) for n, v in stds.items()},
    )
    bands: list[list[float | None]] = [[0.0, 5.0], [5.0, 100.0], [100.0, None]]
    derived = _build_ohc(bands=bands, normalizer=normalizer)
    area = _ops()._cpu_area
    mask = _mask()
    dz = DepthCoordinate(IDEPTH, mask, DEPTHO).dz
    for (a, b), name in zip(bands, derived.names):
        assert a is not None
        hi = float("inf") if b is None else b
        z_top = IDEPTH[:-1]
        h = (z_top + dz).clamp(a, hi) - z_top.expand_as(dz).clamp(a, hi)
        exists = (mask[..., 0] > 0) & (h.sum(-1) > 0)
        w = area * exists
        hbar = (h * w[..., None]).sum((0, 1)) / w.sum()
        expected = RHO_CP * float(hbar[0] * 2.0 + hbar[1] * 5.0)
        assert derived.stds[name] == pytest.approx(expected, rel=1e-6)
        assert derived.weights[name] == 1.0 and derived.means[name] == 0.0
    # band [0, 5] in a 10 m top layer: hbar_0 = 5 m
    assert derived.stds["layer_ohc_0_5"] == pytest.approx(RHO_CP * 5.0 * 2.0)
    override = _build_ohc(
        bands=bands, normalizer=normalizer, stds={"layer_ohc_5_100": 7.0}, weight=0.5
    )
    assert override.stds["layer_ohc_5_100"] == 7.0
    assert override.stds["layer_ohc_0_5"] == derived.stds["layer_ohc_0_5"]
    assert set(override.weights.values()) == {0.5}


def test_layer_ohc_linearized_std_hand_computed():
    """Grid of test_layer_ohc_hand_computed, unit area, sigma = (2, 5):
    hbar_b = (50, 0), (50, 76), (0, 250/3) m over the cells where b exists."""
    idepth = torch.tensor([0.0, 100.0, 300.0])
    mask = torch.ones(2, 3, 2)
    mask[0, 0] = 0.0
    mask[1, 0, 1] = 0.0
    deptho = torch.tensor([[0.0, 300.0, 250.0], [100.0, 300.0, 180.0]])
    stds = {"so_0": 0.5, "so_1": 0.5, "thetao_0": 2.0, "thetao_1": 5.0}
    normalizer = StandardNormalizer(
        means={n: torch.tensor(0.0) for n in stds},
        stds={n: torch.tensor(v) for n, v in stds.items()},
    )
    derived = build_optimized_derived_variables(
        [
            OptimizedDerivedVariableConfig(
                name="layer_ohc", bands=[[0, 50], [50, 200], [200, None]]
            )
        ],
        vertical_coordinate=DepthCoordinate(idepth, mask, deptho),
        network_normalizer=normalizer,
        loss_normalizer=normalizer,
        loss_names=NAMES,
        gridded_operations=LatLonOperations(torch.ones(2, 3)),
    )
    expected = {
        "layer_ohc_0_50": 50.0 * 2.0,
        "layer_ohc_50_200": 50.0 * 2.0 + 76.0 * 5.0,
        "layer_ohc_200_bottom": 250.0 / 3.0 * 5.0,
    }
    for name, value in expected.items():
        assert derived.stds[name] == pytest.approx(RHO_CP * value, rel=1e-6)


def test_layer_ohc_default_bands_use_data_stds():
    """Default band names take LAYER_OHC_DEFAULT_STDS; another band keeps the
    linearized bound."""
    vc = _deep_coordinate(with_deptho=True)
    nz = len(_DEEP_IDEPTH) - 1
    names = [f"thetao_{k}" for k in range(nz)]
    normalizer = StandardNormalizer(
        means={n: torch.tensor(0.0) for n in names},
        stds={n: torch.tensor(1.0) for n in names},
    )
    bands: list[list[float | None]] = [
        *LAYER_OHC_DEFAULT_BANDS[:-1],
        [2700.0, 4000.0],
    ]
    derived = build_optimized_derived_variables(
        [OptimizedDerivedVariableConfig(name="layer_ohc", bands=bands)],
        vertical_coordinate=vc,
        network_normalizer=normalizer,
        loss_normalizer=normalizer,
        loss_names=names,
        gridded_operations=_ops(),
    )
    for name in derived.names[:-1]:
        assert derived.stds[name] == LAYER_OHC_DEFAULT_STDS[name]
    assert derived.names[-1] == "layer_ohc_2700_4000"
    default = build_optimized_derived_variables(
        [OptimizedDerivedVariableConfig(name="layer_ohc")],
        vertical_coordinate=vc,
        network_normalizer=normalizer,
        loss_normalizer=normalizer,
        loss_names=names,
        gridded_operations=_ops(),
    )
    assert default.names == list(LAYER_OHC_DEFAULT_STDS)
    for name in default.names:
        assert default.stds[name] == LAYER_OHC_DEFAULT_STDS[name]
    # sigma = 1: RHO_0 c_p hbar summed over levels = RHO_0 c_p mean thickness
    assert 0 < derived.stds["layer_ohc_2700_4000"] <= RHO_CP * 1300.0


def test_layer_ohc_names_and_defaults():
    config = OptimizedDerivedVariableConfig(name="layer_ohc")
    assert config.bands == LAYER_OHC_DEFAULT_BANDS
    assert config.bands is not LAYER_OHC_DEFAULT_BANDS
    assert [layer_ohc_name(b) for b in LAYER_OHC_DEFAULT_BANDS] == [
        "layer_ohc_0_130",
        "layer_ohc_130_450",
        "layer_ohc_450_1200",
        "layer_ohc_1200_2700",
        "layer_ohc_2700_bottom",
    ]
    derived = _build_ohc(bands=[[0, 12.5], [12.5, None]])
    assert derived.names == ["layer_ohc_0_12p5", "layer_ohc_12p5_bottom"]
    assert set(derived(_data())) == set(derived.names)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "layer_ohc", "bands": []},
        {"name": "layer_ohc", "bands": [[0.0]]},
        {"name": "layer_ohc", "bands": [[-1.0, 10.0]]},
        {"name": "layer_ohc", "bands": [[10.0, 5.0]]},
        {"name": "layer_ohc", "bands": [[10.0, 10.0]]},
        {"name": "layer_ohc", "bands": [[None, 10.0]]},
        {"name": "layer_ohc", "bands": [[0.0, None], [100.0, 200.0]]},
        {"name": "layer_ohc", "bands": [[0.0, 10.0], [0.0, 10.0]]},
        {"name": "layer_ohc", "bands": [["0", 10.0]]},
        {"name": "layer_ohc", "bands": [[0.0, "10"]]},
        {"name": "layer_ohc", "levels": [0]},
        {"name": "layer_ohc", "thetao_clamp": [-2.0, 40.0]},
        {"name": "layer_ohc", "so_clamp": [0.0, 50.0]},
        {"name": "rho_wright97", "bands": [[0.0, 10.0]]},
        {"name": "pbo_wright97", "bands": [[0.0, 10.0]]},
    ],
)
def test_layer_ohc_config_validation(kwargs):
    with pytest.raises(ValueError):
        OptimizedDerivedVariableConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"loss_names": ["so_0", "so_1", "thetao_0"]}, "thetao_1"),
        ({"loss_names": ["thetao_0", "thetao_1"]}, None),  # so_k not needed
        ({"ops": None}, "gridded"),
        ({"bands": [[5000.0, None]]}, "no wet layer"),
        ({"stds": {"layer_ohc_0_1": 1.0}}, "layer_ohc_0_1"),
        ({"loss_names": NAMES + ["layer_ohc_0_130"]}, "not unique"),
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
def test_layer_ohc_build_validation(kwargs, match):
    kwargs = {"bands": [[0, 130], [130, None]], **kwargs}
    if match is None:
        _build_ohc(**kwargs)
        return
    with pytest.raises(ValueError, match=match):
        _build_ohc(**kwargs)
