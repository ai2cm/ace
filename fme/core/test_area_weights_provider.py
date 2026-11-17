import pytest
import torch

from fme.core.area_weights_provider import (
    AreaWeightsProvider,
    NullAreaWeightsProvider,
    area_weights_provider_from_state,
)
from fme.core.constants import SPHERE_AREA_M2
from fme.core.coordinates import HEALPixCoordinates, LatLonCoordinates
from fme.core.device import get_device
from fme.core.spatial_mask_provider import SpatialMaskProvider

NLAT, NLON = 4, 8


def _coordinates() -> LatLonCoordinates:
    return LatLonCoordinates(
        lat=torch.linspace(-67.5, 67.5, NLAT), lon=torch.arange(NLON) * 45.0
    )


def _statics() -> tuple[torch.Tensor, torch.Tensor]:
    """float64 areacello (m2) and sea_surface_fraction, 0 over land."""
    torch.manual_seed(0)
    areacello = 1e12 * (1.0 + torch.rand(NLAT, NLON, dtype=torch.float64))
    sea_surface_fraction = torch.rand(NLAT, NLON, dtype=torch.float64)
    sea_surface_fraction[0, :] = 1.0
    sea_surface_fraction[1, 2] = 0.0
    areacello[1, 2] = 0.0
    return areacello, sea_surface_fraction


def _provider() -> AreaWeightsProvider:
    return AreaWeightsProvider.from_cell_area_and_fraction(*_statics())


def _ocean_area_mean(
    x: torch.Tensor, mask: torch.Tensor, areacello, sea_surface_fraction
) -> torch.Tensor:
    """<x> = sum_J(a_J s_J m_J x_J) / sum_J(a_J s_J m_J), independent of fme."""
    w = areacello * sea_surface_fraction * mask
    return (w * x.nan_to_num()).sum(dim=(-2, -1)) / w.sum()


def test_from_cell_area_and_fraction():
    areacello, sea_surface_fraction = _statics()
    provider = AreaWeightsProvider.from_cell_area_and_fraction(
        areacello, sea_surface_fraction
    )
    torch.testing.assert_close(
        provider.area_weights * SPHERE_AREA_M2, areacello * sea_surface_fraction
    )


@pytest.mark.parametrize(
    "weights", [torch.ones(2, NLAT, NLON), torch.tensor([[float("nan")]])]
)
def test_provider_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        AreaWeightsProvider(weights)


def test_area_weighted_mean_matches_ocean_area_mean():
    areacello, sea_surface_fraction = _statics()
    mask_2d = (sea_surface_fraction > 0).double()
    mask_1 = mask_2d.clone()
    mask_1[2, :3] = 0.0  # shallower level has less ocean
    masks = SpatialMaskProvider({"mask_2d": mask_2d, "mask_1": mask_1})
    ops = _coordinates().get_gridded_operations(
        spatial_mask_provider=masks, area_weights_provider=_provider()
    )
    torch.manual_seed(1)
    x = torch.randn(3, NLAT, NLON, dtype=torch.float64)
    for name, mask in [("sst", mask_2d), ("thetao_1", mask_1)]:
        x_masked = x.where(mask > 0, float("nan"))
        torch.testing.assert_close(
            ops.area_weighted_mean(x_masked, name=name),
            _ocean_area_mean(x_masked, mask, areacello, sea_surface_fraction),
        )
    torch.testing.assert_close(
        ops.area_weighted_sum(x, name="sst") * SPHERE_AREA_M2,
        (areacello * sea_surface_fraction * mask_2d * x).sum(dim=(-2, -1)),
    )


def test_null_provider_keeps_latitude_weights():
    coordinates = _coordinates()
    x = torch.randn(2, NLAT, NLON)
    without = coordinates.get_gridded_operations()
    with_null = coordinates.get_gridded_operations(
        area_weights_provider=NullAreaWeightsProvider
    )
    torch.testing.assert_close(
        with_null.area_weighted_mean(x), without.area_weighted_mean(x)
    )
    with_provider = coordinates.get_gridded_operations(
        area_weights_provider=_provider()
    )
    assert not torch.allclose(
        with_provider.area_weighted_mean(x.double()),
        without.area_weighted_mean(x).double(),
    )


def test_provider_shape_must_match_grid():
    provider = AreaWeightsProvider(torch.ones(NLAT + 1, NLON))
    with pytest.raises(ValueError, match="shape"):
        _coordinates().get_gridded_operations(area_weights_provider=provider)


def test_healpix_rejects_provider():
    coordinates = HEALPixCoordinates(
        face=torch.arange(12), width=torch.arange(2), height=torch.arange(2)
    )
    with pytest.raises(NotImplementedError):
        coordinates.get_gridded_operations(area_weights_provider=_provider())


def test_merge():
    p = _provider()
    q = AreaWeightsProvider(2 * p.area_weights)
    assert NullAreaWeightsProvider.merge(NullAreaWeightsProvider) is (
        NullAreaWeightsProvider
    )
    assert NullAreaWeightsProvider.merge(p) is p
    assert p.merge(NullAreaWeightsProvider) is p
    assert p.merge(_provider()) is p
    with pytest.raises(ValueError, match="different area weights"):
        p.merge(q)


def test_state_round_trip():
    p = _provider()
    reloaded = area_weights_provider_from_state(p.get_state())
    assert reloaded == p
    assert reloaded.area_weights is not None
    assert reloaded.area_weights.device == get_device()
    assert area_weights_provider_from_state(None) is NullAreaWeightsProvider
