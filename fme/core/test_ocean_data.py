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
from fme.core.ocean_eos import (
    DELTA_RHO_THRESHOLD,
    G_EARTH,
    MLD_REF_LAYER,
    RHO_0,
    wright97_anomaly,
)


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
    with pytest.raises(ValueError, match="idepth"):
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


N_LAT, N_LON, N_LEVELS = 4, 6, 2
IDEPTH = torch.tensor([0.0, 10.0, 1000.0])
DEPTHO = torch.full((N_LAT, N_LON), 1000.0)
DEPTHO[2, :] = 400.0  # partial bottom cell at level 1
DEPTHO[1, :] = 10.0  # level 1 dry (mask[1, :, 1] == 0)


class _CellArea:
    def __init__(self, area_weights_m2: torch.Tensor):
        self.area_weights_m2 = area_weights_m2


def _wright97_mask() -> torch.Tensor:
    mask = torch.ones(N_LAT, N_LON, N_LEVELS)
    mask[0, :, :] = 0.0  # land
    mask[1, :, 1] = 0.0  # below the sea floor at level 1
    return mask


def _cell_area() -> _CellArea:
    # non-uniform in latitude, uniform in longitude
    return _CellArea(torch.linspace(1.0, 2.0, N_LAT)[:, None].expand(N_LAT, N_LON))


def _wright97_data(seed=0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    shape = (2, 1, N_LAT, N_LON)
    data = {}
    for k in range(N_LEVELS):
        data[f"so_{k}"] = 34.0 + 2.0 * torch.rand(shape, generator=g)
        data[f"thetao_{k}"] = 2.0 + 20.0 * torch.rand(shape, generator=g)
    data["zos"] = 0.5 * torch.randn(shape, generator=g)
    return data


def _wright97_ocean_data(data, deptho=DEPTHO, cell_area=True) -> OceanData:
    return OceanData(
        data,
        DepthCoordinate(IDEPTH, _wright97_mask(), deptho),
        cell_area_provider=_cell_area() if cell_area else None,
    )


def _expected_pbo(data, deptho=DEPTHO) -> torch.Tensor:
    """P - <P> by hand: dz from idepth and deptho, area mean over mask_0."""
    mask = _wright97_mask()
    C = torch.zeros_like(data["zos"])
    for k in range(N_LEVELS):
        z_top, z_bot = float(IDEPTH[k]), float(IDEPTH[k + 1])
        dz = (deptho.clamp(z_top, z_bot) - z_top) * mask[..., k]
        p = torch.tensor(RHO_0 * G_EARTH * 0.5 * (z_top + z_bot))
        rho = wright97_anomaly(data[f"so_{k}"], data[f"thetao_{k}"], p)
        C = C + torch.where(mask[..., k] > 0, rho * dz, 0.0)
    wet = (mask[..., 0] > 0).expand_as(C)
    w = _cell_area().area_weights_m2 * wet
    P = RHO_0 * G_EARTH * data["zos"] + G_EARTH * C
    mean = (P * w).sum((-2, -1), keepdim=True) / w.sum((-2, -1), keepdim=True)
    return torch.where(wet, P - mean, torch.nan)


def test_rho_wright97_values_and_mask():
    data = _wright97_data()
    data["thetao_0"][0, 0, 2, 3] = torch.nan
    rho = _wright97_ocean_data(data).rho_wright97
    assert rho.shape == (2, 1, N_LAT, N_LON, N_LEVELS)
    mask = _wright97_mask() > 0
    for k in range(N_LEVELS):
        p = RHO_0 * G_EARTH * 0.5 * float(IDEPTH[k] + IDEPTH[k + 1])
        expected = wright97_anomaly(
            data[f"so_{k}"], data[f"thetao_{k}"], torch.tensor(p)
        )
        valid = mask[..., k] & expected.isfinite()
        torch.testing.assert_close(rho[..., k][valid], expected[valid])
        assert rho[..., k][~valid.expand_as(expected)].isnan().all()
    assert rho[0, 0, 2, 3, 0].isnan()
    assert rho[:, :, 1, :, 0].isfinite().all()
    assert rho[:, :, 1, :, 1].isnan().all()


def test_pbo_wright97_values_and_mask():
    data = _wright97_data()
    pbo = _wright97_ocean_data(data).pbo_wright97
    torch.testing.assert_close(pbo, _expected_pbo(data), equal_nan=True)
    assert pbo[:, :, 0].isnan().all()  # land row
    assert pbo[:, :, 1:].isfinite().all()
    # partial bottom cell: row 2 differs from a full-cell column
    full = _wright97_ocean_data(data, deptho=None).pbo_wright97
    assert not torch.allclose(full[:, :, 2], pbo[:, :, 2])


def test_pbo_wright97_global_mean_removed():
    data = _wright97_data()
    pbo = _wright97_ocean_data(data).pbo_wright97
    w = _cell_area().area_weights_m2 * (_wright97_mask()[..., 0] > 0)
    mean = (pbo.nan_to_num() * w).sum((-2, -1)) / w.sum()
    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-2, rtol=0)
    # uniform zos shift and a uniform NaN off-mask do not change pbo
    shifted = dict(data, zos=data["zos"] + 0.3)
    shifted["zos"][:, :, 0] = torch.nan
    torch.testing.assert_close(
        _wright97_ocean_data(shifted).pbo_wright97,
        pbo,
        equal_nan=True,
        atol=1e-2,  # float32 at |P| ~ 1e3 Pa
        rtol=0,
    )


@pytest.mark.parametrize("missing", ["zos", "cell_area"])
def test_pbo_wright97_missing_inputs_raise_key_error(missing):
    data = _wright97_data()
    if missing == "zos":
        del data["zos"]
    ocean_data = _wright97_ocean_data(data, cell_area=missing != "cell_area")
    with pytest.raises(KeyError):
        _ = ocean_data.pbo_wright97


@pytest.mark.parametrize("name", ["rho_wright97", "pbo_wright97"])
def test_wright97_without_layer_geometry_raises_value_error(name):
    class _IntegralOnly:
        def depth_integral(self, integrand):
            return integrand.sum(dim=-1)

    data = _wright97_data()
    for coord in (None, _IntegralOnly()):
        ocean_data = OceanData(data, coord, cell_area_provider=_cell_area())
        with pytest.raises(ValueError, match="depth coordinate"):
            _ = getattr(ocean_data, name)
