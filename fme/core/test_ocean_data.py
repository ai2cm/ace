import pytest
import torch

from fme.core.constants import DENSITY_OF_WATER_CM4, SPECIFIC_HEAT_OF_WATER_CM4
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
            SPECIFIC_HEAT_OF_WATER_CM4
            * DENSITY_OF_WATER_CM4
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
    """Test column-integrated ocean salt content."""
    n_samples, n_time_steps, nlat, nlon, nlevels = 2, 2, 2, 2, 2
    shape_2d = (n_samples, n_time_steps, nlat, nlon)

    data = {
        "so_0": torch.ones(n_samples, n_time_steps, nlat, nlon),
        "so_1": torch.ones(n_samples, n_time_steps, nlat, nlon),
    }

    if has_depth_coordinate:
        idepth = torch.tensor([2.5, 10, 20])
        lev_thickness = idepth.diff(dim=-1)
        mask = torch.ones(n_samples, n_time_steps, nlat, nlon, nlevels)
        mask[:, :, 0, 0, 0] = 0.0
        mask[:, :, 0, 0, 1] = 0.0
        mask[:, :, 0, 1, 1] = 0.0

        expected_osc = torch.tensor(
            DENSITY_OF_WATER_CM4
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
        assert ocean_data.ocean_salt_content.shape == shape_2d
        assert torch.allclose(
            ocean_data.ocean_salt_content.nansum(),
            expected_osc,
            atol=1e-10,
            equal_nan=True,
        )
    else:
        ocean_data = OceanData(data)
        with pytest.raises(ValueError, match="Depth coordinate must be provided"):
            _ = ocean_data.ocean_salt_content


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


def _make_ohc_test_data(
    n_samples, n_time_steps, nlat, nlon, nlevels, deptho_value=None
):
    """Helper to build OceanData with uniform temperature and optional deptho."""
    shape_2d = (n_samples, n_time_steps, nlat, nlon)
    data = {f"thetao_{i}": torch.ones(shape_2d) for i in range(nlevels)}
    if deptho_value is not None:
        data["deptho"] = torch.full(shape_2d, deptho_value)
    idepth = torch.tensor([2.5, 10, 20])
    mask = torch.ones(n_samples, n_time_steps, nlat, nlon, nlevels)
    depth_coordinate = DepthCoordinate(idepth, mask)
    return OceanData(data, depth_coordinate), idepth


def test_ocean_heat_content_with_deep_sea_floor():
    """When deptho exceeds the deepest interface, OHC matches the no-deptho case."""
    n_samples, n_time_steps, nlat, nlon, nlevels = 2, 2, 3, 4, 2

    ocean_data_with, idepth = _make_ohc_test_data(
        n_samples, n_time_steps, nlat, nlon, nlevels, deptho_value=100.0
    )
    ocean_data_without, _ = _make_ohc_test_data(
        n_samples, n_time_steps, nlat, nlon, nlevels, deptho_value=None
    )

    ohc_with = ocean_data_with.ocean_heat_content
    ohc_without = ocean_data_without.ocean_heat_content

    assert ohc_with.shape == (n_samples, n_time_steps, nlat, nlon)
    torch.testing.assert_close(ohc_with, ohc_without)

    lev_thickness = idepth.diff(dim=-1)
    expected_total = torch.tensor(
        SPECIFIC_HEAT_OF_WATER_CM4
        * DENSITY_OF_WATER_CM4
        * n_samples
        * n_time_steps
        * nlat
        * nlon
        * lev_thickness.sum().item()
    )
    torch.testing.assert_close(ohc_with.nansum(), expected_total, rtol=1e-6, atol=0)


def test_ocean_heat_content_with_shallow_sea_floor():
    """When deptho is shallower than some layers, OHC is reduced."""
    n_samples, n_time_steps, nlat, nlon, nlevels = 2, 2, 3, 4, 2

    ocean_data_shallow, idepth = _make_ohc_test_data(
        n_samples, n_time_steps, nlat, nlon, nlevels, deptho_value=15.0
    )
    ocean_data_full, _ = _make_ohc_test_data(
        n_samples, n_time_steps, nlat, nlon, nlevels, deptho_value=None
    )

    ohc_shallow = ocean_data_shallow.ocean_heat_content
    ohc_full = ocean_data_full.ocean_heat_content

    assert ohc_shallow.shape == (n_samples, n_time_steps, nlat, nlon)

    # deptho = 15:
    # layer 0 dz = clamp(15, 2.5, 10) - 2.5 = 7.5,
    # layer 1 dz = clamp(15, 10, 20) - 10 = 5
    expected_dz = 7.5 + 5.0
    n_cells = n_samples * n_time_steps * nlat * nlon
    expected_total = torch.tensor(
        SPECIFIC_HEAT_OF_WATER_CM4 * DENSITY_OF_WATER_CM4 * n_cells * expected_dz
    )
    torch.testing.assert_close(ohc_shallow.nansum(), expected_total, rtol=1e-6, atol=0)

    assert ohc_shallow.nansum() < ohc_full.nansum()


def test_ocean_heat_content_without_deptho_fallback():
    """When deptho is absent, ocean_heat_content falls back to full-column integral."""
    n_samples, n_time_steps, nlat, nlon, nlevels = 2, 2, 3, 4, 2

    ocean_data, idepth = _make_ohc_test_data(
        n_samples, n_time_steps, nlat, nlon, nlevels, deptho_value=None
    )

    ohc = ocean_data.ocean_heat_content
    assert ohc.shape == (n_samples, n_time_steps, nlat, nlon)

    lev_thickness = idepth.diff(dim=-1)
    expected_total = torch.tensor(
        SPECIFIC_HEAT_OF_WATER_CM4
        * DENSITY_OF_WATER_CM4
        * n_samples
        * n_time_steps
        * nlat
        * nlon
        * lev_thickness.sum().item()
    )
    torch.testing.assert_close(ohc.nansum(), expected_total, rtol=1e-6, atol=0)
