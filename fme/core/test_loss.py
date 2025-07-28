import pytest
import torch

from fme.core import metrics
from fme.core.device import get_device
from fme.core.gridded_ops import GriddedOperations, LatLonOperations
from fme.core.loss import (
    AreaWeightedMSELoss,
    CRPSLoss,
    EnergyScoreLoss,
    GlobalMeanLoss,
    LossConfig,
    VariableWeightingLoss,
    WeightedMappingLoss,
    WeightedMappingLossConfig,
    _construct_weight_tensor,
)
from fme.core.normalizer import StandardNormalizer
from fme.core.packer import Packer


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
    assert result.shape == ()
    assert isinstance(result, torch.Tensor)


def test_spectral_energy_score():
    torch.manual_seed(0)
    DEVICE = get_device()
    n_lat, n_lon = 16, 32
    pred = torch.rand(10000, 2, n_lat, n_lon, device=DEVICE)
    target = torch.rand(10000, 2, n_lat, n_lon, device=DEVICE)
    sht = LatLonOperations(torch.ones((n_lat, n_lon), device=DEVICE)).get_real_sht(
        grid="legendre-gauss"
    )
    spectral_energy_score_loss = EnergyScoreLoss(sht=sht)
    crps_loss = CRPSLoss(alpha=0.95)
    score = spectral_energy_score_loss(pred, target)
    crps = crps_loss(pred, target)

    n_lat2, n_lon2 = 32, 64
    pred = torch.rand(10000, 2, n_lat2, n_lon2, device=DEVICE)
    target = torch.rand(10000, 2, n_lat2, n_lon2, device=DEVICE)
    sht = LatLonOperations(torch.ones((n_lat2, n_lon2), device=DEVICE)).get_real_sht(
        grid="legendre-gauss"
    )
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
    assert result.shape == ()
    assert isinstance(result, torch.Tensor)
    if str(get_device()).startswith("cuda"):
        tol = {"rtol": 1e-4, "atol": 1e-4}
    elif str(get_device()).startswith("mps"):
        tol = {"rtol": 1e-3, "atol": 1e-3}
    else:
        tol = {}
    torch.testing.assert_close(result, y.var(), **tol)


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
    assert result.shape == ()
    assert isinstance(result, torch.Tensor)
    expected = torch.tensor(1.0 + global_mean_weight)
    tol = (
        {"atol": 0.015, "rtol": 0.01}
        if str(get_device()).startswith("cuda")
        else {"atol": 0.01, "rtol": 0.0}
    )
    torch.testing.assert_close(result.cpu(), expected, **tol)


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
    loss = torch.nn.MSELoss()
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
    x = torch.randn(
        15,
        n_channels,
        10,
        10,
    ).to(get_device(), dtype=torch.float)
    y = torch.randn(
        15,
        n_channels,
        10,
        10,
    ).to(get_device(), dtype=torch.float)
    x_mapping = {name: x[:, i, :, :] for i, name in enumerate(packer.names)}
    y_mapping = {name: y[:, i, :, :] for i, name in enumerate(packer.names)}
    assert torch.allclose(mapping_loss(x_mapping, y_mapping), loss(x, y) / scale**2)


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


def test_WeightedMappingLossConfig_no_weights():
    loss_config = LossConfig()
    n_channels = 5
    out_names = [f"var_{i}" for i in range(n_channels)]
    channel_dim = -3
    area = torch.tensor([])  # area not used by this config
    gridded_operations: GriddedOperations = LatLonOperations(area)
    mapping_loss_config = WeightedMappingLossConfig()
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

    assert loss(x, y) == mapping_loss(x_mapping, y_mapping)


def test_WeightedMappingLossConfig_weights():
    out_names = ["var_0", "var_1"]
    channel_dim = -3
    area = torch.tensor([])  # area not used by this config
    mapping_loss_config = WeightedMappingLossConfig(
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

    assert mapping_loss(x_mapping, y_mapping) == ((16 + 0.25) / 2.0)


@pytest.mark.parametrize("mean", [0.0, 1.0])
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_WeightedMappingLoss_with_ensemble_dim(mean, scale):
    loss = torch.nn.MSELoss()
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
    x = torch.randn(
        15,
        n_ensemble,
        n_channels,
        10,
        10,
    ).to(get_device(), dtype=torch.float)
    y = torch.randn(
        15,
        1,  # only 1 ensemble member for target
        n_channels,
        10,
        10,
    ).to(get_device(), dtype=torch.float)
    x_mapping = {name: x[:, :, i, :, :] for i, name in enumerate(packer.names)}
    y_mapping = {name: y[:, :, i, :, :] for i, name in enumerate(packer.names)}
    assert torch.allclose(mapping_loss(x_mapping, y_mapping), loss(x, y) / scale**2)


def test_WeightedMappingLoss_with_target_nans():
    loss = torch.nn.MSELoss()
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
    x = torch.randn(
        15,
        n_channels,
        10,
        10,
    ).to(get_device(), dtype=torch.float)
    y = torch.randn(
        15,
        n_channels,
        10,
        10,
    ).to(get_device(), dtype=torch.float)
    x_mapping = {name: x[:, i, :, :].clone() for i, name in enumerate(packer.names)}
    y_mapping = {name: y[:, i, :, :].clone() for i, name in enumerate(packer.names)}
    y_mapping[packer.names[0]][:, :, 0] = float("nan")
    x[:, 0, :, 0] = 0.0
    y[:, 0, :, 0] = 0.0
    assert torch.allclose(mapping_loss(x_mapping, y_mapping), loss(x, y))
