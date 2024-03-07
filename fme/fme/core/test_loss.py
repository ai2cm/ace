from typing import Optional

import numpy as np
import pytest
import torch

from fme.core import metrics
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.device import get_device
from fme.core.loss import (
    AreaWeightedMSELoss,
    ConservationLossConfig,
    GlobalMeanLoss,
    LossConfig,
    construct_weight_tensor,
    get_dry_air_nonconservation,
)


@pytest.mark.parametrize("global_mean_type", [None, "LpLoss"])
def test_loss_builds_and_runs(global_mean_type):
    config = LossConfig(global_mean_type=global_mean_type)
    area = torch.randn(10, 10, device=get_device())
    loss = config.build(area)
    x = torch.randn(10, 10, 10, 10, 10, device=get_device())
    y = torch.randn(10, 10, 10, 10, 10, device=get_device())
    result = loss(x, y)
    assert result.shape == ()
    assert isinstance(result, torch.Tensor)


def test_loss_of_zeros_is_one():
    config = LossConfig(global_mean_type=None)
    area = torch.randn(10, 10, device=get_device())
    loss = config.build(area)
    x = torch.zeros(10, 10, 10, 10, 10, device=get_device())
    y = torch.randn(10, 10, 10, 10, 10, device=get_device())
    result = loss(x, y)
    assert result.shape == ()
    assert isinstance(result, torch.Tensor)
    assert result == 1.0


@pytest.mark.parametrize("global_mean_weight", [0.0, 1.0, 5.0])
def test_loss_of_zeros_is_one_plus_global_mean_weight(global_mean_weight: float):
    config = LossConfig(
        global_mean_type="LpLoss", global_mean_weight=global_mean_weight
    )
    area = torch.randn(10, 10, device=get_device())
    loss = config.build(area)
    x = torch.zeros(10, 10, 10, 10, 10, device=get_device())
    y = torch.randn(10, 10, 10, 10, 10, device=get_device())
    result = loss(x, y)
    assert result.shape == ()
    assert isinstance(result, torch.Tensor)
    assert result == 1.0 + global_mean_weight


def test_global_mean_loss():
    area = torch.randn(10, 10, device=get_device())
    loss = GlobalMeanLoss(area=area, loss=torch.nn.MSELoss())
    x = torch.zeros(10, 10, 10, 10, 10, device=get_device())
    y = torch.randn(10, 10, 10, 10, 10, device=get_device())
    result = loss(x, y)
    assert result.shape == ()
    assert isinstance(result, torch.Tensor)

    def global_weighted_mean(tensor, area):
        return (tensor * area[None, None, None, :, :]).sum(dim=(-1, -2)) / area.sum()

    mse = torch.nn.MSELoss()
    expected = mse(global_weighted_mean(x, area), global_weighted_mean(y, area))
    torch.testing.assert_allclose(result, expected)


def setup_single_level_conservation_loss(
    n_lat: int,
    n_lon: int,
    dry_air_penalty: float,
    area_weights: Optional[torch.Tensor] = None,
):
    if area_weights is None:
        area_weights = torch.ones(n_lat, n_lon).to(get_device())
    sigma_coordinates = SigmaCoordinates(
        ak=torch.zeros(size=[2]).to(get_device()),
        bk=torch.asarray([0.0, 1.0]).to(get_device()),
    )
    config = ConservationLossConfig(dry_air_penalty)
    conservation_loss = config.build(
        sigma_coordinates=sigma_coordinates, area_weights=area_weights
    )
    return conservation_loss


def test_dry_air_conservation_loss_single_level_no_change():
    n_lat = 5
    n_lon = 10
    conservation_loss = setup_single_level_conservation_loss(
        n_lat=n_lat, n_lon=n_lon, dry_air_penalty=1.0
    )
    data = {
        "PRESsfc": torch.ones(size=[1, 2, n_lat, n_lon]).to(get_device()),
        "specific_total_water_0": torch.zeros(size=[1, 2, n_lat, n_lon]).to(
            get_device()
        ),
    }
    metrics, loss = conservation_loss(data)
    assert len(metrics) == 1
    assert "dry_air_loss" in metrics
    assert metrics["dry_air_loss"] == 0.0
    assert loss == 0.0


@pytest.mark.parametrize("dry_air_penalty", [0.0, 1.0, 0.01])
def test_dry_air_conservation_loss_single_level_remove_all_air(dry_air_penalty: float):
    n_lat = 5
    n_lon = 10
    conservation_loss = setup_single_level_conservation_loss(
        n_lat=n_lat, n_lon=n_lon, dry_air_penalty=dry_air_penalty
    )
    data = {
        "PRESsfc": torch.ones(size=[1, 2, n_lat, n_lon]).to(get_device()),
        "specific_total_water_0": torch.zeros(size=[1, 2, n_lat, n_lon]).to(
            get_device()
        ),
    }
    data["PRESsfc"][:, 1, :, :] = 0.0
    metrics, loss = conservation_loss(data)
    assert len(metrics) == 1
    assert "dry_air_loss" in metrics
    assert metrics["dry_air_loss"] == dry_air_penalty
    assert loss == dry_air_penalty


@pytest.mark.parametrize("dry_air_penalty", [0.0, 1.0, 0.01])
def test_dry_air_conservation_loss_single_level(dry_air_penalty: float):
    torch.manual_seed(0)
    n_lat = 5
    n_lon = 10
    area_weights = torch.ones(n_lat, n_lon).to(get_device())
    conservation_loss = setup_single_level_conservation_loss(
        n_lat=n_lat,
        n_lon=n_lon,
        dry_air_penalty=dry_air_penalty,
        area_weights=area_weights,
    )
    data = {
        "PRESsfc": torch.ones(1, 2, n_lat, n_lon).to(get_device()),
        "specific_total_water_0": torch.rand(1, 2, n_lat, n_lon).to(get_device()),
    }
    dry_air = data["PRESsfc"] * (1.0 - data["specific_total_water_0"])
    dry_air_final = (dry_air[:, 1, :, :] * area_weights).sum() / area_weights.sum()
    dry_air_initial = (dry_air[:, 0, :, :] * area_weights).sum() / area_weights.sum()
    target_loss = torch.abs(dry_air_final - dry_air_initial) * dry_air_penalty
    metrics, loss = conservation_loss(data)
    assert len(metrics) == 1
    assert "dry_air_loss" in metrics
    np.testing.assert_almost_equal(
        metrics["dry_air_loss"].cpu().numpy(), target_loss.cpu().numpy()
    )
    np.testing.assert_almost_equal(loss.cpu().numpy(), target_loss.cpu().numpy())


def test_get_dry_air_conservation_loss_single_level():
    torch.manual_seed(0)
    n_lat = 5
    n_lon = 10
    area_weights = torch.ones(n_lat, n_lon).to(get_device())
    sigma_coordinates = SigmaCoordinates(
        ak=torch.zeros(size=[2]).to(get_device()),
        bk=torch.asarray([0.0, 1.0]).to(get_device()),
    )
    data = {
        "PRESsfc": torch.ones(1, 2, n_lat, n_lon).to(get_device()),
        "specific_total_water_0": torch.rand(1, 2, n_lat, n_lon).to(get_device()),
    }
    loss = get_dry_air_nonconservation(
        data, area_weights=area_weights, sigma_coordinates=sigma_coordinates
    )
    dry_air = data["PRESsfc"] * (1.0 - data["specific_total_water_0"])
    dry_air_final = (dry_air[:, 1, :, :] * area_weights).sum() / area_weights.sum()
    dry_air_initial = (dry_air[:, 0, :, :] * area_weights).sum() / area_weights.sum()
    target_loss = torch.abs(dry_air_final - dry_air_initial)
    np.testing.assert_almost_equal(loss.cpu().numpy(), target_loss.cpu().numpy())


def test_area_weighted_mse():
    torch.manual_seed(0)
    x = torch.rand(10, 10).to(get_device())
    target = torch.rand(10, 10).to(get_device())
    area = torch.rand(10, 10).to(get_device())
    area_weighted_mse = AreaWeightedMSELoss(area)
    result = area_weighted_mse(x, target)
    expected = metrics.weighted_mean(
        torch.nn.MSELoss(reduction="none")(x, target), weights=area, dim=(-2, -1)
    ).mean()
    torch.testing.assert_allclose(result, expected)


def test_construct_weight_tensor():
    out_names = ["a", "b", "c"]
    weights = {"a": 0.5, "c": 3.0}
    n_samples = 10
    nlat, nlon = 8, 4
    gen_data = torch.rand(n_samples, len(out_names), nlat, nlon, device=get_device())
    weight_tensor = construct_weight_tensor(weights, out_names, n_dim=4, channel_dim=-3)
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


def test_construct_weight_tensor_missing_key_error():
    out_names = ["a", "b", "c"]
    weights = {"a": 0.5, "c": 3.0, "d": 1.5}
    with pytest.raises(KeyError):
        construct_weight_tensor(weights, out_names, n_dim=4, channel_dim=-3)(
            out_names, n_dim=4, channel_dim=-3
        )
