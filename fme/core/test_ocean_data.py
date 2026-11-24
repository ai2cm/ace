import math

import pytest
import torch

from fme.core.constants import (
    DENSITY_OF_SEA_WATER_CM4,
    REFERENCE_SALINITY,
    SPECIFIC_HEAT_OF_SEA_WATER_CM4,
)
from fme.core.coordinates import DepthCoordinate
from fme.core.ocean_data import OceanData


@pytest.mark.parametrize("has_depth_coordinate", [True, False])
def test_column_integrated_ocean_heat_content(has_depth_coordinate: bool):
    """Test column-integrated ocean heat content."""
    n_samples, n_time_steps, nlat, nlon, nlevels = 2, 2, 2, 2, 2
    shape_2d = (n_samples, n_time_steps, nlat, nlon)

    data = {
        "thetao_0": torch.ones(n_samples, n_time_steps, nlat, nlon),
        "thetao_1": torch.ones(n_samples, n_time_steps, nlat, nlon),
    }

    if has_depth_coordinate:
        idepth = torch.tensor([2.5, 10, 20])
        lev_thickness = idepth.diff(dim=-1)
        mask = torch.ones(n_samples, n_time_steps, nlat, nlon, nlevels)
        mask[:, :, 0, 0, 0] = 0.0
        mask[:, :, 0, 0, 1] = 0.0
        mask[:, :, 0, 1, 1] = 0.0

        expected_ohc = torch.tensor(
            SPECIFIC_HEAT_OF_SEA_WATER_CM4
            * DENSITY_OF_SEA_WATER_CM4
            * n_samples
            * n_time_steps
            * (
                nlat * nlon * lev_thickness.sum()
                - lev_thickness[0]
                - 2 * lev_thickness[1]
            )
        )
        depth_coordinate = DepthCoordinate(idepth, mask)
        ocean_data = OceanData(data, depth_coordinate)
        assert ocean_data.ocean_heat_content.shape == shape_2d
        assert torch.allclose(
            ocean_data.ocean_heat_content.nansum(),
            expected_ohc,
            atol=1e-10,
            equal_nan=True,
        )
    else:
        ocean_data = OceanData(data)
        with pytest.raises(ValueError, match="Depth coordinate must be provided"):
            _ = ocean_data.ocean_heat_content


@pytest.mark.parametrize("has_depth_coordinate", [True, False])
def test_column_integrated_ocean_salt_content(has_depth_coordinate: bool):
    """Test column-integrated ocean salt content, which is weighted by the sea
    surface fraction.
    """
    n_samples, n_time_steps, nlat, nlon, nlevels = 2, 2, 2, 2, 2
    shape_2d = (n_samples, n_time_steps, nlat, nlon)
    so_0, so_1, sea_surface_fraction = 34.0, 36.0, 0.5

    data = {
        "so_0": torch.full(shape_2d, so_0),
        "so_1": torch.full(shape_2d, so_1),
        "sea_surface_fraction": torch.full(shape_2d, sea_surface_fraction),
    }

    if has_depth_coordinate:
        idepth = torch.tensor([2.5, 10, 20])
        lev_thickness = idepth.diff(dim=-1)
        mask = torch.ones(n_samples, n_time_steps, nlat, nlon, nlevels)
        mask[:, :, 0, 0, 0] = 0.0
        mask[:, :, 0, 0, 1] = 0.0
        mask[:, :, 0, 1, 1] = 0.0

        # 3 of 4 columns are ocean at level 0, 2 of 4 at level 1
        expected_osc = (
            DENSITY_OF_SEA_WATER_CM4
            * sea_surface_fraction
            * n_samples
            * n_time_steps
            * (3 * lev_thickness[0] * so_0 + 2 * lev_thickness[1] * so_1)
        )
        depth_coordinate = DepthCoordinate(idepth, mask)
        ocean_data = OceanData(data, depth_coordinate)
        assert ocean_data.ocean_salt_content.shape == shape_2d
        torch.testing.assert_close(ocean_data.ocean_salt_content.nansum(), expected_osc)
    else:
        ocean_data = OceanData(data)
        with pytest.raises(ValueError, match="Depth coordinate must be provided"):
            _ = ocean_data.ocean_salt_content


@pytest.mark.parametrize("sfdsi", [2e-7, float("nan"), None])
def test_salt_fluxes_into_ocean(sfdsi: float | None):
    """Test the virtual and net salt fluxes, where the net salt flux is only
    defined when sfdsi is available.
    """
    shape = (1, 1, 1, 1)
    wfo, sea_surface_fraction = 1e-5, 0.5
    data = {
        "wfo": torch.full(shape, wfo),
        "sea_surface_fraction": torch.full(shape, sea_surface_fraction),
    }
    if sfdsi is not None:
        data["sfdsi"] = torch.full(shape, sfdsi)
    ocean_data = OceanData(data)

    expected_virtual = -REFERENCE_SALINITY * wfo * sea_surface_fraction
    torch.testing.assert_close(
        ocean_data.net_virtual_salt_flux_into_ocean,
        torch.full(shape, expected_virtual),
    )
    if sfdsi is None:
        with pytest.raises(KeyError, match="downward_sea_ice_basal_salt_flux"):
            _ = ocean_data.net_salt_flux_into_ocean
    else:
        sfdsi_flux = 0.0 if math.isnan(sfdsi) else sfdsi
        expected_net = expected_virtual + 1000.0 * sfdsi_flux * sea_surface_fraction
        torch.testing.assert_close(
            ocean_data.net_salt_flux_into_ocean, torch.full(shape, expected_net)
        )


def test_get_3d_fields():
    """Test getting 3D fields (fields with vertical levels)."""
    n_samples, n_time_steps, nlat, nlon, nlevels = 2, 3, 4, 8, 2
    shape_3d = (n_samples, n_time_steps, nlat, nlon, nlevels)

    data = {
        "thetao_0": torch.rand(n_samples, n_time_steps, nlat, nlon),
        "thetao_1": torch.rand(n_samples, n_time_steps, nlat, nlon),
        "so_0": torch.rand(n_samples, n_time_steps, nlat, nlon),
        "so_1": torch.rand(n_samples, n_time_steps, nlat, nlon),
        "uo_0": torch.rand(n_samples, n_time_steps, nlat, nlon),
        "uo_1": torch.rand(n_samples, n_time_steps, nlat, nlon),
        "vo_0": torch.rand(n_samples, n_time_steps, nlat, nlon),
        "vo_1": torch.rand(n_samples, n_time_steps, nlat, nlon),
    }

    ocean_data = OceanData(data)

    # Test shape of 3D fields
    assert ocean_data.sea_water_potential_temperature.shape == shape_3d
    assert ocean_data.sea_water_salinity.shape == shape_3d
    assert ocean_data.sea_water_x_velocity.shape == shape_3d
    assert ocean_data.sea_water_y_velocity.shape == shape_3d


def test_get_2d_fields():
    """Test getting 2D surface fields."""
    n_samples, n_time_steps, nlat, nlon = 2, 3, 4, 8
    shape_2d = (n_samples, n_time_steps, nlat, nlon)

    data = {
        "sst": torch.rand(n_samples, n_time_steps, nlat, nlon),
        "zos": torch.rand(n_samples, n_time_steps, nlat, nlon),
    }

    ocean_data = OceanData(data)

    # Test shape of 2D fields
    assert ocean_data.sea_surface_temperature.shape == shape_2d
    assert ocean_data.sea_surface_height_above_geoid.shape == shape_2d


def test_missing_field():
    """Test that accessing a missing field raises KeyError."""
    data = {"sst": torch.rand(2, 3, 4, 8)}
    ocean_data = OceanData(data)

    with pytest.raises(KeyError, match="thetao_"):
        _ = ocean_data.sea_water_potential_temperature


@pytest.mark.parametrize("missing_layer", [True, False])
def test_keyerror_when_missing_3d_layer(missing_layer: bool):
    """Test that missing a layer in a 3D field raises ValueError."""
    n_samples, n_time_steps, nlat, nlon = 2, 3, 4, 8

    def _get_data(missing_layer: bool):
        data = {
            "thetao_1": torch.rand(n_samples, n_time_steps, nlat, nlon),
        }
        if not missing_layer:
            data["thetao_0"] = torch.rand(n_samples, n_time_steps, nlat, nlon)
        return data

    ocean_data = OceanData(_get_data(missing_layer))

    if not missing_layer:
        assert ocean_data.sea_water_potential_temperature.shape == (
            n_samples,
            n_time_steps,
            nlat,
            nlon,
            2,
        )
    else:
        with pytest.raises(ValueError, match="Missing level 0 in thetao_ levels"):
            _ = ocean_data.sea_water_potential_temperature


def test_getitem():
    """Test the __getitem__ method."""
    data = {"sst": torch.rand(2, 3, 4, 8)}
    ocean_data = OceanData(data)

    assert torch.equal(
        ocean_data["sea_surface_temperature"], ocean_data.sea_surface_temperature
    )

    with pytest.raises(
        AttributeError, match="object has no attribute 'nonexistent_field'"
    ):
        _ = ocean_data["nonexistent_field"]


def _thickness_mean_data() -> dict[str, torch.Tensor]:
    # cells: ice, ice-free ocean, land
    return {
        "ocean_sea_ice_thickness_mean": torch.tensor([1.5, 0.0, float("nan")]),
        "ocean_sea_ice_fraction": torch.tensor([0.75, 0.0, float("nan")]),
    }


def test_sea_ice_thickness_from_thickness_mean():
    thickness = OceanData(_thickness_mean_data()).sea_ice_thickness
    torch.testing.assert_close(
        thickness, torch.tensor([2.0, 0.0, float("nan")]), equal_nan=True
    )


def test_sea_ice_thickness_from_thickness_mean_has_finite_gradients():
    data = _thickness_mean_data()
    thickness_mean = data["ocean_sea_ice_thickness_mean"].requires_grad_()
    frac = data["ocean_sea_ice_fraction"].requires_grad_()
    thickness = OceanData(data).sea_ice_thickness
    torch.nansum(thickness).backward()
    assert thickness_mean.grad is not None and frac.grad is not None
    assert torch.isfinite(thickness_mean.grad).all()
    assert torch.isfinite(frac.grad).all()
    torch.testing.assert_close(thickness_mean.grad, torch.tensor([1 / 0.75, 0, 0]))
    torch.testing.assert_close(frac.grad, torch.tensor([-1.5 / 0.75**2, 0, 0]))


def test_sea_ice_thickness_from_thickness_mean_nan_numerator_finite_gradients():
    thickness_mean = torch.tensor([float("nan")], requires_grad=True)
    frac = torch.tensor([0.5], requires_grad=True)
    data = {
        "ocean_sea_ice_thickness_mean": thickness_mean,
        "ocean_sea_ice_fraction": frac,
    }
    thickness = OceanData(data).sea_ice_thickness
    assert torch.isnan(thickness).all()
    torch.nansum(thickness).backward()
    assert frac.grad is not None and torch.isfinite(frac.grad).all()


def test_sea_ice_thickness_stored_takes_precedence():
    data = {**_thickness_mean_data(), "HI": torch.tensor([7.0, 8.0, 9.0])}
    torch.testing.assert_close(OceanData(data).sea_ice_thickness, data["HI"])


def test_sea_ice_thickness_mean_takes_precedence_over_volume():
    # the volume path would raise: no cell area provider
    data = {**_thickness_mean_data(), "sea_ice_volume": torch.ones(3)}
    thickness = OceanData(data).sea_ice_thickness
    torch.testing.assert_close(
        thickness, torch.tensor([2.0, 0.0, float("nan")]), equal_nan=True
    )


def test_sea_ice_thickness_mean_without_ocean_fraction_uses_volume():
    data = {
        "ocean_sea_ice_thickness_mean": torch.ones(3),
        "sea_ice_volume": torch.ones(3),
        "sea_ice_fraction": torch.ones(3),
        "land_fraction": torch.zeros(3),
    }
    with pytest.raises(ValueError, match="cell area provider"):
        OceanData(data).sea_ice_thickness


def test_sea_ice_thickness_missing_inputs():
    with pytest.raises(KeyError):
        OceanData({"ocean_sea_ice_thickness_mean": torch.ones(3)}).sea_ice_thickness


class _CellArea:
    def __init__(self, area_weights_m2: torch.Tensor):
        self.area_weights_m2 = area_weights_m2


def _volume_data() -> dict[str, torch.Tensor]:
    # cells: ice, ice-free ocean with stray thickness mean, land
    return {
        "ocean_sea_ice_thickness_mean": torch.tensor([1.5, 0.25, float("nan")]),
        "ocean_sea_ice_fraction": torch.tensor([0.75, 0.0, float("nan")]),
        "sea_surface_fraction": torch.tensor([0.5, 1.0, 0.0]),
    }


_CELL_AREA = _CellArea(torch.tensor([4.0, 2.0, 3.0]))


def test_sea_ice_volume_from_thickness_mean():
    volume = OceanData(_volume_data(), cell_area_provider=_CELL_AREA).sea_ice_volume
    torch.testing.assert_close(
        volume, torch.tensor([1.5 * 4.0 * 0.5, 0.0, float("nan")]), equal_nan=True
    )


def test_sea_ice_volume_from_thickness_mean_has_finite_gradients():
    data = _volume_data()
    thickness_mean = data["ocean_sea_ice_thickness_mean"].requires_grad_()
    sfrac = data["sea_surface_fraction"].requires_grad_()
    volume = OceanData(data, cell_area_provider=_CELL_AREA).sea_ice_volume
    torch.nansum(volume).backward()
    for tensor in (thickness_mean, sfrac):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    torch.testing.assert_close(thickness_mean.grad, torch.tensor([2.0, 0.0, 0.0]))
    torch.testing.assert_close(sfrac.grad, torch.tensor([6.0, 0.0, 0.0]))


def test_sea_ice_volume_stored_takes_precedence():
    data = {**_volume_data(), "sea_ice_volume": torch.tensor([7.0, 8.0, 9.0])}
    volume = OceanData(data, cell_area_provider=_CELL_AREA).sea_ice_volume
    torch.testing.assert_close(volume, data["sea_ice_volume"])


def test_sea_ice_thickness_and_volume_missing_inputs_no_recursion():
    data = {
        "ocean_sea_ice_fraction": torch.ones(3),
        "sea_surface_fraction": torch.ones(3),
    }
    ocean_data = OceanData(data, cell_area_provider=_CELL_AREA)
    with pytest.raises(KeyError):
        ocean_data.sea_ice_volume
    with pytest.raises(KeyError):
        ocean_data.sea_ice_thickness


def test_sea_ice_thickness_from_derived_volume_round_trip():
    data = _volume_data()
    from_mean = OceanData(data).sea_ice_thickness
    volume = OceanData(data, cell_area_provider=_CELL_AREA).sea_ice_volume
    volume_data = {
        "sea_ice_volume": volume,
        "ocean_sea_ice_fraction": data["ocean_sea_ice_fraction"],
        "sea_surface_fraction": data["sea_surface_fraction"],
    }
    from_volume = OceanData(
        volume_data, cell_area_provider=_CELL_AREA
    ).sea_ice_thickness
    has_ice = data["ocean_sea_ice_fraction"] > 0
    torch.testing.assert_close(from_volume[has_ice], from_mean[has_ice])
