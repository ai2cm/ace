import pytest
import torch

from fme.core import metrics
from fme.core.device import get_device
from fme.core.gridded_ops import GriddedOperations, LatLonOperations
from fme.core.loss import (
    AreaWeightedMSELoss,
    CRPSLoss,
    EnergyScoreLoss,
    EnsembleLoss,
    GlobalMeanLoss,
    LossConfig,
    StepLossConfig,
    VariableWeightingLoss,
    WeightedMappingLoss,
    _channel_dim_positive,
    _construct_weight_tensor,
    _reduce_to_per_channel,
)
from fme.core.normalizer import StandardNormalizer
from fme.core.packer import Packer


def _assert_per_channel_sum_matches(
    loss_fn,
    x: torch.Tensor,
    y: torch.Tensor,
    channel_dim: int,
    expected_scalar: torch.Tensor,
    **tol,
):
    """Assert ``_reduce_to_per_channel(loss(x, y)).sum()`` equals
    ``expected_scalar`` (the old global-mean scalar loss)."""
    raw = loss_fn(x, y)
    cdim = _channel_dim_positive(x.ndim, channel_dim)
    n_c = int(x.shape[cdim])
    per_ch = _reduce_to_per_channel(raw, cdim, n_c)
    assert per_ch.shape == (n_c,)
    torch.testing.assert_close(per_ch.sum(), expected_scalar, **tol)


@pytest.mark.parametrize("global_mean_type", [None, "LpLoss"])
def test_loss_builds_and_runs(global_mean_type):
    config = LossConfig(global_mean_type=global_mean_type)
    area = torch.randn(10, 1, device=get_device()).broadcast_to(size=(10, 10))
    loss = config.build(
        reduction="mean",
        gridded_operations=LatLonOperations(area),
    )
    x = torch.randn(10, 10, 10, 10, 10, device=get_device())
    y = torch.randn(10, 10, 10, 10, 10, device=get_device())
    result = loss(x, y)
    assert isinstance(result, torch.Tensor)


def test_spectral_energy_score(very_fast_only: bool):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    torch.manual_seed(0)
    DEVICE = get_device()
    n_lat, n_lon = 16, 32
    pred = torch.rand(10000, 2, n_lat, n_lon, device=DEVICE)
    target = torch.rand(10000, 2, n_lat, n_lon, device=DEVICE)
    sht = LatLonOperations(torch.ones((n_lat, n_lon), device=DEVICE)).get_real_sht()
    spectral_energy_score_loss = EnergyScoreLoss(sht=sht)
    crps_loss = CRPSLoss(alpha=0.95)
    score = spectral_energy_score_loss(pred, target)
    crps = crps_loss(pred, target)

    n_lat2, n_lon2 = 32, 64
    pred = torch.rand(10000, 2, n_lat2, n_lon2, device=DEVICE)
    target = torch.rand(10000, 2, n_lat2, n_lon2, device=DEVICE)
    sht = LatLonOperations(torch.ones((n_lat2, n_lon2), device=DEVICE)).get_real_sht()
    spectral_energy_score_loss = EnergyScoreLoss(sht=sht)
    larger_domain_score = spectral_energy_score_loss(pred, target)
    torch.testing.assert_close(larger_domain_score, score, rtol=0.05, atol=0.0)
    torch.testing.assert_close(score, crps, rtol=0.5, atol=0.0)


def test_loss_of_zeros_is_variance():
    torch.manual_seed(0)
    config = LossConfig(global_mean_type=None)
    loss = config.build(
        reduction="mean",
        gridded_operations=LatLonOperations(torch.ones(10, 10)),
    )
    x = torch.zeros(10, 10, 10, 10, 10, device=get_device())
    y = torch.randn(10, 10, 10, 10, 10, device=get_device())
    result = loss(x, y)
    assert isinstance(result, torch.Tensor)
    if str(get_device()).startswith("cuda"):
        tol = {"rtol": 1e-4, "atol": 1e-4}
    elif str(get_device()).startswith("mps"):
        tol = {"rtol": 1e-3, "atol": 1e-3}
    else:
        tol = {}
    torch.testing.assert_close(result.mean(), y.var(), **tol)


@pytest.mark.parametrize("global_mean_weight", [0.0, 1.0, 5.0])
def test_loss_of_zeros_is_one_plus_global_mean_weight(global_mean_weight: float):
    torch.manual_seed(0)
    config = LossConfig(
        global_mean_type="LpLoss", global_mean_weight=global_mean_weight
    )
    area = torch.randn(10, 1, device=get_device()).broadcast_to(size=(10, 10))
    loss = config.build(
        reduction="mean",
        gridded_operations=LatLonOperations(area),
    )
    x = torch.zeros(10, 10, 10, 10, 10, device=get_device())
    y = torch.randn(10, 10, 10, 10, 10, device=get_device())
    result = loss(x, y)
    assert isinstance(result, torch.Tensor)
    expected = torch.tensor(1.0 + global_mean_weight)
    tol = (
        {"atol": 0.015, "rtol": 0.01}
        if str(get_device()).startswith("cuda")
        else {"atol": 0.01, "rtol": 0.0}
    )
    torch.testing.assert_close(result.mean().cpu(), expected, **tol)


@pytest.mark.parametrize(
    "config",
    [
        LossConfig(type="AreaWeightedMSE"),
        LossConfig(global_mean_type="LpLoss"),
        LossConfig(
            type="EnsembleLoss", kwargs={"energy_score_weight": 1.0, "crps_weight": 0.0}
        ),
    ],
)
def test_loss_fails_when_gridded_operations_not_provided(
    config: LossConfig,
):
    with pytest.raises(ValueError):
        config.build(reduction="mean", gridded_operations=None)


def test_global_mean_loss():
    torch.manual_seed(0)
    area = torch.randn(10, 1, device=get_device()).broadcast_to(size=(10, 10))
    loss = GlobalMeanLoss(
        LatLonOperations(area).area_weighted_mean, loss=torch.nn.MSELoss()
    )
    x = torch.zeros(10, 10, 10, 10, 10, device=get_device())
    y = torch.randn(10, 10, 10, 10, 10, device=get_device())
    result = loss(x, y)
    assert result.shape == ()
    assert isinstance(result, torch.Tensor)

    def global_weighted_mean(tensor, area):
        return (tensor * area[None, None, None, :, :]).sum(dim=(-1, -2)) / area.sum()

    mse = torch.nn.MSELoss()
    expected = mse(global_weighted_mean(x, area), global_weighted_mean(y, area))
    torch.testing.assert_close(result, expected)


def test_area_weighted_mse():
    torch.manual_seed(0)
    x = torch.rand(10, 10).to(get_device())
    target = torch.rand(10, 10).to(get_device())
    area = torch.rand(10, 1, device=get_device()).broadcast_to(size=(10, 10))
    area_weighted_mse = AreaWeightedMSELoss(LatLonOperations(area).area_weighted_mean)
    result = area_weighted_mse(x, target)
    expected = metrics.weighted_mean(
        torch.nn.MSELoss(reduction="none")(x, target), weights=area, dim=(-2, -1)
    ).mean()
    torch.testing.assert_close(result, expected)


def test__construct_weight_tensor():
    out_names = ["a", "b", "c"]
    weights = {"a": 0.5, "c": 3.0}
    n_samples = 10
    nlat, nlon = 8, 4
    gen_data = torch.rand(n_samples, len(out_names), nlat, nlon, device=get_device())
    weight_tensor = _construct_weight_tensor(
        weights, out_names, n_dim=4, channel_dim=-3
    )
    weighted_gen_data = gen_data * weight_tensor
    assert weight_tensor.shape == (1, len(out_names), 1, 1)
    assert weighted_gen_data.shape == gen_data.shape
    for i, name in enumerate(out_names):
        if name in weights:
            assert torch.allclose(
                weighted_gen_data[:, i], weights[name] * gen_data[:, i]
            )
        else:
            assert torch.allclose(weighted_gen_data[:, i], gen_data[:, i])


@pytest.mark.parametrize("mean", [0.0, 1.0])
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_WeightedMappingLoss(mean, scale):
    loss = torch.nn.MSELoss(reduction="none")
    n_channels = 5
    packer = Packer([f"var_{i}" for i in range(n_channels)])
    out_names = [f"var_{i}" for i in range(n_channels)]
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(mean) for name in out_names},
        stds={name: torch.as_tensor(scale) for name in out_names},
    )
    mapping_loss = WeightedMappingLoss(
        loss,
        weights={},
        out_names=out_names,
        normalizer=normalizer,
    )
    x = torch.randn(15, n_channels, 10, 10).to(get_device(), dtype=torch.float)
    y = torch.randn(15, n_channels, 10, 10).to(get_device(), dtype=torch.float)
    x_mapping = {name: x[:, i, :, :] for i, name in enumerate(packer.names)}
    y_mapping = {name: y[:, i, :, :] for i, name in enumerate(packer.names)}
    result = mapping_loss(x_mapping, y_mapping)
    assert result.shape == (n_channels,)
    expected_scalar = torch.nn.MSELoss()(x, y) / scale**2
    torch.testing.assert_close(result.sum(), expected_scalar)


def test_VariableWeightingLoss():
    weights = torch.tensor(
        [
            4.0,
            1.0,
        ]
    ).to(get_device())
    mse_loss = torch.nn.MSELoss()
    weighted_loss = VariableWeightingLoss(weights=weights, loss=mse_loss)

    x = torch.tensor([[1.0, 2.0]]).to(get_device())
    y = torch.tensor([[2, 2.5]]).to(get_device())

    weighted_result = weighted_loss(x, y)
    assert weighted_result == ((16 + 0.25) / 2.0)


def test_StepLossConfig_no_weights():
    loss_config = LossConfig()
    n_channels = 5
    out_names = [f"var_{i}" for i in range(n_channels)]
    channel_dim = -3
    area = torch.ones(1, 1)  # area not used by this config
    gridded_operations: GriddedOperations = LatLonOperations(area)
    mapping_loss_config = StepLossConfig(sqrt_loss_step_decay_constant=0.0)
    loss = loss_config.build(reduction="mean", gridded_operations=gridded_operations)
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(0.0) for name in out_names},
        stds={name: torch.as_tensor(1.0) for name in out_names},
    )
    mapping_loss = mapping_loss_config.build(
        gridded_operations,
        out_names=out_names,
        channel_dim=channel_dim,
        normalizer=normalizer,
    )
    packer = Packer(out_names)

    x_mapping = {name: torch.randn(4, 5, 5).to(get_device()) for name in out_names}
    y_mapping = {name: torch.randn(4, 5, 5).to(get_device()) for name in out_names}
    x = packer.pack(x_mapping, axis=channel_dim)
    y = packer.pack(y_mapping, axis=channel_dim)

    torch.testing.assert_close(
        loss(x, y).mean(),
        mapping_loss(x_mapping, y_mapping, step=0).sum(),
    )
    torch.testing.assert_close(
        loss(x, y).mean(),
        mapping_loss(x_mapping, y_mapping, step=1).sum(),
    )


def test_StepLossConfig_weights():
    out_names = ["var_0", "var_1"]
    channel_dim = -3
    area = torch.ones(1, 1)  # area not used by this config
    mapping_loss_config = StepLossConfig(
        type="MSE", weights={"var_0": 4.0, "var_1": 1.0}
    )
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(0.0) for name in out_names},
        stds={name: torch.as_tensor(1.0) for name in out_names},
    )

    gridded_operations: GriddedOperations = LatLonOperations(area)

    mapping_loss = mapping_loss_config.build(
        gridded_operations,
        out_names=out_names,
        channel_dim=channel_dim,
        normalizer=normalizer,
    )

    x0 = torch.ones(4, 5, 5).to(get_device())
    x1 = 2.0 * x0
    y0 = 2.0 * x0
    y1 = 2.5 * x0

    x_mapping = {"var_0": x0, "var_1": x1}
    y_mapping = {"var_0": y0, "var_1": y1}

    result = mapping_loss(x_mapping, y_mapping, step=0)
    assert result.shape == (2,)
    torch.testing.assert_close(
        result.sum(),
        torch.tensor((16 + 0.25) / 2.0, device=get_device()),
    )


@pytest.mark.parametrize("sqrt_loss_step_decay_constant", [0.0, 0.1, 1.0])
def test_StepLossConfig_with_step_loss_decay(sqrt_loss_step_decay_constant):
    out_names = ["var_0", "var_1"]
    channel_dim = -3
    area = torch.ones(1, 1)  # area not used by this config
    mapping_loss_config = StepLossConfig(
        type="MSE",
        weights={"var_0": 4.0, "var_1": 1.0},
        sqrt_loss_step_decay_constant=sqrt_loss_step_decay_constant,
    )
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(0.0) for name in out_names},
        stds={name: torch.as_tensor(1.0) for name in out_names},
    )

    gridded_operations: GriddedOperations = LatLonOperations(area)

    mapping_loss = mapping_loss_config.build(
        gridded_operations,
        out_names=out_names,
        channel_dim=channel_dim,
        normalizer=normalizer,
    )

    x0 = torch.ones(4, 5, 5).to(get_device())
    x1 = 2.0 * x0
    y0 = 2.0 * x0
    y1 = 2.5 * x0

    x_mapping = {"var_0": x0, "var_1": x1}
    y_mapping = {"var_0": y0, "var_1": y1}

    torch.testing.assert_close(
        mapping_loss(x_mapping, y_mapping, step=0).sum(),
        mapping_loss(x_mapping, y_mapping, step=1).sum()
        * (1 + sqrt_loss_step_decay_constant) ** 0.5,
    )
    torch.testing.assert_close(
        mapping_loss(x_mapping, y_mapping, step=0).sum(),
        mapping_loss(x_mapping, y_mapping, step=2).sum()
        * (1 + sqrt_loss_step_decay_constant * 2) ** 0.5,
    )


@pytest.mark.parametrize("mean", [0.0, 1.0])
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_WeightedMappingLoss_with_ensemble_dim(mean, scale):
    loss = torch.nn.MSELoss(reduction="none")
    n_channels = 5
    n_ensemble = 3
    packer = Packer([f"var_{i}" for i in range(n_channels)])
    out_names = [f"var_{i}" for i in range(n_channels)]
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(mean) for name in out_names},
        stds={name: torch.as_tensor(scale) for name in out_names},
    )
    mapping_loss = WeightedMappingLoss(
        loss,
        weights={},
        out_names=out_names,
        normalizer=normalizer,
    )
    x = torch.randn(15, n_ensemble, n_channels, 10, 10).to(
        get_device(), dtype=torch.float
    )
    y = torch.randn(15, 1, n_channels, 10, 10).to(get_device(), dtype=torch.float)
    x_mapping = {name: x[:, :, i, :, :] for i, name in enumerate(packer.names)}
    y_mapping = {name: y[:, :, i, :, :] for i, name in enumerate(packer.names)}
    result = mapping_loss(x_mapping, y_mapping)
    assert result.shape == (n_channels,)
    expected_scalar = torch.nn.MSELoss()(x, y) / scale**2
    torch.testing.assert_close(result.sum(), expected_scalar)


def test_WeightedMappingLoss_with_target_nans():
    loss = torch.nn.MSELoss(reduction="none")
    n_channels = 5
    packer = Packer([f"var_{i}" for i in range(n_channels)])
    out_names = [f"var_{i}" for i in range(n_channels)]
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(0.0) for name in out_names},
        stds={name: torch.as_tensor(1.0) for name in out_names},
    )
    mapping_loss = WeightedMappingLoss(
        loss,
        weights={},
        out_names=out_names,
        normalizer=normalizer,
    )
    x = torch.randn(15, n_channels, 10, 10).to(get_device(), dtype=torch.float)
    y = torch.randn(15, n_channels, 10, 10).to(get_device(), dtype=torch.float)
    x_mapping = {name: x[:, i, :, :].clone() for i, name in enumerate(packer.names)}
    y_mapping = {name: y[:, i, :, :].clone() for i, name in enumerate(packer.names)}
    y_mapping[packer.names[0]][:, :, 0] = float("nan")
    x[:, 0, :, 0] = 0.0
    y[:, 0, :, 0] = 0.0
    result = mapping_loss(x_mapping, y_mapping)
    assert result.shape == (n_channels,)
    mse = torch.nn.MSELoss()
    torch.testing.assert_close(result.sum(), mse(x, y))


# ---------- Per-channel sum-equivalence tests ----------


@pytest.mark.parametrize("channel_dim", [-3, 1])
def test_mse_per_channel_sum_4d(channel_dim):
    torch.manual_seed(42)
    x = torch.randn(4, 5, 8, 8, device=get_device())
    y = torch.randn(4, 5, 8, 8, device=get_device())
    _assert_per_channel_sum_matches(
        torch.nn.MSELoss(reduction="none"),
        x,
        y,
        channel_dim,
        torch.nn.MSELoss()(x, y),
    )


def test_mse_per_channel_sum_5d():
    torch.manual_seed(42)
    x = torch.randn(4, 2, 5, 8, 8, device=get_device())
    y = torch.randn(4, 2, 5, 8, 8, device=get_device())
    _assert_per_channel_sum_matches(
        torch.nn.MSELoss(reduction="none"),
        x,
        y,
        -3,
        torch.nn.MSELoss()(x, y),
    )


def test_l1_per_channel_sum():
    torch.manual_seed(42)
    x = torch.randn(4, 5, 8, 8, device=get_device())
    y = torch.randn(4, 5, 8, 8, device=get_device())
    _assert_per_channel_sum_matches(
        torch.nn.L1Loss(reduction="none"),
        x,
        y,
        -3,
        torch.nn.L1Loss()(x, y),
    )


def test_ensemble_per_channel_sum_crps_only():
    torch.manual_seed(0)
    n_lat, n_lon = 8, 16
    n_channels = 3
    sht = LatLonOperations(
        torch.ones((n_lat, n_lon), device=get_device())
    ).get_real_sht()
    loss = EnsembleLoss(
        crps_weight=1.0,
        energy_score_weight=0.0,
        sht=sht,
    ).to(get_device())

    x = torch.randn(4, 2, n_channels, n_lat, n_lon, device=get_device())
    y = torch.randn(4, 1, n_channels, n_lat, n_lon, device=get_device())
    expected = CRPSLoss(alpha=0.95).to(get_device())(x, y)
    _assert_per_channel_sum_matches(loss, x, y, -3, expected)


def test_ensemble_per_channel_sum_mixed():
    torch.manual_seed(0)
    n_lat, n_lon = 8, 16
    n_channels = 3
    sht = LatLonOperations(
        torch.ones((n_lat, n_lon), device=get_device())
    ).get_real_sht()
    loss = EnsembleLoss(
        crps_weight=0.5,
        energy_score_weight=0.5,
        sht=sht,
    ).to(get_device())

    x = torch.randn(4, 2, n_channels, n_lat, n_lon, device=get_device())
    y = torch.randn(4, 1, n_channels, n_lat, n_lon, device=get_device())
    crps_scalar = CRPSLoss(alpha=0.95).to(get_device())(x, y)
    es_scalar = EnergyScoreLoss(sht=sht).to(get_device())(x, y)
    expected = 0.5 * crps_scalar + 0.5 * es_scalar
    _assert_per_channel_sum_matches(loss, x, y, -3, expected)


def test_area_weighted_mse_per_channel_sum():
    torch.manual_seed(42)
    x = torch.rand(4, 3, 8, 8, device=get_device())
    y = torch.rand(4, 3, 8, 8, device=get_device())
    area = torch.rand(8, 1, device=get_device()).broadcast_to(size=(8, 8))
    awm = LatLonOperations(area).area_weighted_mean
    loss_fn = AreaWeightedMSELoss(awm)
    original_scalar = torch.mean(awm((x - y) ** 2))
    _assert_per_channel_sum_matches(loss_fn, x, y, -3, original_scalar)
