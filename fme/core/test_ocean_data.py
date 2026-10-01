import pytest
import torch

from fme.core.constants import DENSITY_OF_SEA_WATER_CM4, SPECIFIC_HEAT_OF_SEA_WATER_CM4
from fme.core.coordinates import DepthCoordinate
from fme.core.ocean_data import OceanData
from fme.core.ocean_eos import DELTA_RHO_THRESHOLD, MLD_REF_LAYER, wright97_anomaly


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


def _mld_case():
    """Two columns, four levels: column 0 ocean with the threshold crossed
    between the centres of levels 1 and 2; column 1 land (mask_0 == 0)."""
    idepth = torch.tensor([0.0, 10.0, 30.0, 60.0, 100.0], dtype=torch.float64)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    thetao = torch.tensor([[20.0, 20.0, 18.0, 10.0], [20.0, 20.0, 18.0, 10.0]])
    so = torch.full_like(thetao, 35.0)
    data = {f"thetao_{k}": thetao[:, k].double() for k in range(4)}
    data.update({f"so_{k}": so[:, k].double() for k in range(4)})
    return idepth, mask, thetao.double(), so.double(), data


def test_mld_wright97_known_profile_and_land():
    idepth, mask, thetao, so, data = _mld_case()
    coord = DepthCoordinate(idepth=idepth, mask=mask)
    mld = OceanData(data, coord).mld_wright97
    # by hand from the inputs: rho at p=0, d_k = rho_k - rho_ref, crossing
    # between the centres of MLD_REF_LAYER and MLD_REF_LAYER + 1
    rho = wright97_anomaly(so[0], thetao[0], torch.zeros_like(thetao[0]))
    d = rho - rho[MLD_REF_LAYER]
    zc = 0.5 * (idepth[:-1] + idepth[1:])
    k = MLD_REF_LAYER + 1
    assert d[k] > DELTA_RHO_THRESHOLD
    expected = zc[k - 1] + DELTA_RHO_THRESHOLD / d[k] * (zc[k] - zc[k - 1])
    torch.testing.assert_close(mld[0], expected, atol=1e-6, rtol=0)
    assert zc[k - 1] < mld[0] < zc[k]
    # land: NaN exactly where depth_integral is NaN
    ohc_nan = torch.isnan(coord.depth_integral(thetao))
    torch.testing.assert_close(torch.isnan(mld), ohc_nan)
    assert torch.isnan(mld[1])


def test_mld_wright97_no_crossing_is_sea_floor():
    idepth, mask, _, _, _ = _mld_case()
    thetao = torch.full((1, 4), 20.0, dtype=torch.float64)
    so = torch.full_like(thetao, 35.0)
    data = {f"thetao_{k}": thetao[:, k] for k in range(4)}
    data.update({f"so_{k}": so[:, k] for k in range(4)})
    deptho = torch.tensor([80.0], dtype=torch.float64)
    mld = OceanData(
        data, DepthCoordinate(idepth=idepth, mask=mask[:1], deptho=deptho)
    ).mld_wright97
    torch.testing.assert_close(mld, deptho)
    mld = OceanData(data, DepthCoordinate(idepth=idepth, mask=mask[:1])).mld_wright97
    torch.testing.assert_close(mld, idepth[-1:])


def test_mld_wright97_missing_depth_coordinate_raises_value_error():
    _, _, _, _, data = _mld_case()
    with pytest.raises(ValueError, match="depth coordinate"):
        _ = OceanData(data).mld_wright97


def test_mld_wright97_depth_coordinate_without_idepth_raises_value_error():
    class _IntegralOnly:
        def depth_integral(self, integrand):
            return integrand.sum(dim=-1)

    _, _, _, _, data = _mld_case()
    with pytest.raises(ValueError, match="idepth and mask"):
        _ = OceanData(data, _IntegralOnly()).mld_wright97


def test_mld_wright97_missing_salinity_raises_key_error():
    idepth, mask, _, _, data = _mld_case()
    data = {k: v for k, v in data.items() if not k.startswith("so_")}
    with pytest.raises(KeyError):
        _ = OceanData(data, DepthCoordinate(idepth=idepth, mask=mask)).mld_wright97


def test_mld_wright97_too_few_levels_raises_key_error():
    nz = MLD_REF_LAYER + 1
    data = {f"thetao_{k}": torch.rand(2) for k in range(nz)}
    data.update({f"so_{k}": torch.rand(2) for k in range(nz)})
    coord = DepthCoordinate(
        idepth=torch.arange(nz + 1, dtype=torch.float32), mask=torch.ones(nz)
    )
    with pytest.raises(KeyError, match="levels"):
        _ = OceanData(data, coord).mld_wright97
