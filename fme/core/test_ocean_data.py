import math

import pytest
import torch

from fme.core.area_weights_provider import AreaWeightsProvider
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


class _IntegralOnly:
    """A depth coordinate without layer geometry."""

    def depth_integral(self, integrand):
        return integrand.sum(dim=-1)


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


@pytest.mark.parametrize("sst_name", ["sst", "SST_degK"])
def test_sea_surface_temperature_name_variants(sst_name: str):
    """Either SST name resolves, for get and set."""
    sst = torch.rand(2, 3, 4, 8) + 273.15
    ocean_data = OceanData({sst_name: sst})
    assert torch.equal(ocean_data.sea_surface_temperature, sst)

    new_sst = sst + 1.0
    ocean_data._set("sea_surface_temperature", new_sst)
    assert list(ocean_data.data) == [sst_name]
    assert torch.equal(ocean_data.data[sst_name], new_sst)
    assert torch.equal(ocean_data.sea_surface_temperature, new_sst)


def test_sea_surface_temperature_sst_wins_over_sst_degk():
    """When both names are present, "sst" is used for get and set."""
    sst = torch.full((1, 1, 2, 2), 280.0)
    sst_degk = torch.full((1, 1, 2, 2), 290.0)
    ocean_data = OceanData({"SST_degK": sst_degk, "sst": sst})
    assert torch.equal(ocean_data.sea_surface_temperature, sst)

    new_sst = sst + 1.0
    ocean_data._set("sea_surface_temperature", new_sst)
    assert torch.equal(ocean_data.data["sst"], new_sst)
    assert torch.equal(ocean_data.data["SST_degK"], sst_degk)


def test_sea_surface_temperature_missing():
    ocean_data = OceanData({"zos": torch.rand(1, 1, 2, 2)})
    with pytest.raises(KeyError, match="sea_surface_temperature"):
        _ = ocean_data.sea_surface_temperature


def test_ocean_area_weights():
    weights = torch.rand(2, 3)
    assert OceanData({}).ocean_area_weights is None
    ocean_data = OceanData({}, area_weights_provider=AreaWeightsProvider(weights))
    assert ocean_data.ocean_area_weights is weights


def _mld_inputs(thetao: torch.Tensor, so: torch.Tensor) -> dict[str, torch.Tensor]:
    data = {f"thetao_{k}": thetao[:, k] for k in range(thetao.shape[-1])}
    data.update({f"so_{k}": so[:, k] for k in range(so.shape[-1])})
    return data


def _mld_case():
    """Three columns, four levels: column 0 ocean with the threshold crossed
    between the centres of levels 1 and 2 (levels 2 and 3 for a level-1
    reference); column 1 land (mask_0 == 0); column 2 with level 3 masked and
    a crossing fill value there."""
    idepth = torch.tensor([0.0, 10.0, 30.0, 60.0, 100.0], dtype=torch.float64)
    mask = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]
    )
    thetao = torch.tensor(
        [
            [20.0, 19.92, 19.85, 18.0],
            [20.0, 19.92, 19.85, 18.0],
            [20.0, 20.0, 20.0, 0.0],
        ],
        dtype=torch.float64,
    )
    so = torch.full_like(thetao, 35.0)
    return idepth, mask, thetao, so, _mld_inputs(thetao, so)


def test_mld_wright97_known_profile_land_and_masked_level():
    idepth, mask, thetao, so, data = _mld_case()
    coord = DepthCoordinate(idepth=idepth, mask=mask)
    mld = OceanData(data, coord).mld_wright97
    assert MLD_REF_LAYER == 0
    # by hand from the inputs: rho at p=0, density_diff_k = rho_k - rho_0,
    # crossing between the centres of levels 1 and 2
    rho = wright97_anomaly(S=so[0], theta=thetao[0], p=torch.zeros_like(thetao[0]))
    density_diff = rho - rho[0]
    center_depth = 0.5 * (idepth[:-1] + idepth[1:])
    assert density_diff[1] < DELTA_RHO_THRESHOLD < density_diff[2]
    expected = center_depth[1] + (DELTA_RHO_THRESHOLD - density_diff[1]) / (
        density_diff[2] - density_diff[1]
    ) * (center_depth[2] - center_depth[1])
    torch.testing.assert_close(mld[0], expected, atol=1e-6, rtol=0)
    # land: NaN exactly where depth_integral is NaN
    ohc_nan = torch.isnan(coord.depth_integral(thetao))
    torch.testing.assert_close(torch.isnan(mld), ohc_nan)
    assert torch.isnan(mld[1])
    # masked level 3 does not cross: deepest unmasked interface
    torch.testing.assert_close(mld[2], idepth[3])


def _expected_mld(thetao, so, z_prev, z_k, k):
    """Linear interpolation of rho - rho_0 at p=0 between depths z_prev and
    z_k [m] of levels k - 1 and k."""
    rho = wright97_anomaly(S=so, theta=thetao, p=torch.zeros_like(thetao))
    d_prev, d_k = rho[k - 1] - rho[0], rho[k] - rho[0]
    assert d_prev <= DELTA_RHO_THRESHOLD < d_k
    return z_prev + (DELTA_RHO_THRESHOLD - d_prev) / (d_k - d_prev) * (z_k - z_prev)


def test_mld_wright97_crossing_below_reference_level():
    idepth, mask, _, _, _ = _mld_case()
    thetao = torch.tensor([[20.0, 19.0, 18.0, 17.0]], dtype=torch.float64)
    so = torch.full_like(thetao, 35.0)
    mld = OceanData(
        _mld_inputs(thetao, so), DepthCoordinate(idepth=idepth, mask=mask[:1])
    ).mld_wright97
    assert MLD_REF_LAYER == 0
    # centres of levels 0 and 1: 5 m and 20 m
    expected = _expected_mld(thetao[0], so[0], 5.0, 20.0, k=1)
    torch.testing.assert_close(mld[0], expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("thetao_2", [19.85, 19.8])
def test_mld_wright97_partial_bottom_cell_uses_wet_centre(thetao_2):
    idepth = torch.tensor([0.0, 10.0, 20.0, 1000.0], dtype=torch.float64)
    deptho = torch.tensor([30.0], dtype=torch.float64)
    thetao = torch.tensor([[20.0, 20.0, thetao_2]], dtype=torch.float64)
    so = torch.full_like(thetao, 35.0)
    mld = OceanData(
        _mld_inputs(thetao, so),
        DepthCoordinate(idepth=idepth, mask=torch.ones(1, 3), deptho=deptho),
    ).mld_wright97
    # level 2 is wet over [20, 30] m: centre 25 m, not the full cell's 510 m
    expected = _expected_mld(thetao[0], so[0], 15.0, 25.0, k=2)
    torch.testing.assert_close(mld[0], expected, atol=1e-6, rtol=0)
    assert mld[0] <= deptho[0]


def test_mld_wright97_non_finite_wet_input_is_nan():
    idepth, _, _, _, _ = _mld_case()
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]] * 3 + [[1.0, 1.0, 1.0, 0.0]])
    thetao = torch.full((4, 4), 20.0, dtype=torch.float64)
    so = torch.full_like(thetao, 35.0)
    thetao[0, 0] = float("nan")  # reference level
    so[1, 2] = float("nan")  # wet level, no crossing
    thetao[2, 1] = 10.0  # crossing above a non-finite wet level
    thetao[2, 3] = float("nan")
    thetao[3, 3] = float("nan")  # masked level
    deptho = torch.full((4,), 80.0, dtype=torch.float64)
    mld = OceanData(
        _mld_inputs(thetao, so),
        DepthCoordinate(idepth=idepth, mask=mask, deptho=deptho),
    ).mld_wright97
    assert torch.isnan(mld[:3]).all()
    torch.testing.assert_close(mld[3], deptho[3])


def test_mld_wright97_no_crossing_is_sea_floor():
    idepth, mask, _, _, _ = _mld_case()
    thetao = torch.full((1, 4), 20.0, dtype=torch.float64)
    so = torch.full_like(thetao, 35.0)
    data = _mld_inputs(thetao, so)
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


def _expected_column_density(data, deptho=DEPTHO, dtype=torch.float32) -> torch.Tensor:
    """C = sum_k rho_k * dz_k by hand: dz from idepth and deptho."""
    mask = _wright97_mask().to(dtype)
    C = torch.zeros_like(data["thetao_0"], dtype=dtype)
    for k in range(N_LEVELS):
        z_top, z_bot = float(IDEPTH[k]), float(IDEPTH[k + 1])
        dz = (deptho.to(dtype).clamp(z_top, z_bot) - z_top) * mask[..., k]
        p = torch.tensor(RHO_0 * G_EARTH * 0.5 * (z_top + z_bot), dtype=dtype)
        rho = wright97_anomaly(
            S=data[f"so_{k}"].to(dtype), theta=data[f"thetao_{k}"].to(dtype), p=p
        )
        C = C + torch.where((mask[..., k] > 0) & rho.isfinite(), rho * dz, 0.0)
    return C


def _expected_eta_mano(data, deptho=DEPTHO) -> torch.Tensor:
    """eta - <eta>, eta = zos + C / RHO_0, by hand: area mean over mask_0 > 0
    and finite zos.
    """
    mask = _wright97_mask()
    C = _expected_column_density(data, deptho)
    wet = (mask[..., 0] > 0) & data["zos"].isfinite()
    w = _cell_area().area_weights_m2 * wet
    eta = data["zos"].nan_to_num() + C / RHO_0
    mean = (eta * w).sum((-2, -1), keepdim=True) / w.sum((-2, -1), keepdim=True)
    return torch.where(wet, eta - mean, torch.nan)


def test_rho_wright97_values_and_mask():
    data = _wright97_data()
    data["thetao_0"][0, 0, 2, 3] = torch.nan
    rho = _wright97_ocean_data(data).rho_wright97
    assert rho.shape == (2, 1, N_LAT, N_LON, N_LEVELS)
    mask = _wright97_mask() > 0
    for k in range(N_LEVELS):
        p = RHO_0 * G_EARTH * 0.5 * float(IDEPTH[k] + IDEPTH[k + 1])
        expected = wright97_anomaly(
            S=data[f"so_{k}"], theta=data[f"thetao_{k}"], p=torch.tensor(p)
        )
        valid = mask[..., k] & expected.isfinite()
        torch.testing.assert_close(rho[..., k][valid], expected[valid])
        assert rho[..., k][~valid.expand_as(expected)].isnan().all()
    assert rho[0, 0, 2, 3, 0].isnan()
    assert rho[:, :, 1, :, 0].isfinite().all()
    assert rho[:, :, 1, :, 1].isnan().all()


def test_manometric_sea_level_wright97_values_and_mask():
    data = _wright97_data()
    eta_mano = _wright97_ocean_data(data).manometric_sea_level_wright97
    torch.testing.assert_close(eta_mano, _expected_eta_mano(data), equal_nan=True)
    assert eta_mano[:, :, 0].isnan().all()  # land row
    assert eta_mano[:, :, 1:].isfinite().all()
    # partial bottom cell: row 2 differs from a full-cell column
    full = _wright97_ocean_data(data, deptho=None).manometric_sea_level_wright97
    assert not torch.allclose(full[:, :, 2], eta_mano[:, :, 2])


def test_manometric_sea_level_wright97_global_mean_removed():
    data = _wright97_data()
    eta_mano = _wright97_ocean_data(data).manometric_sea_level_wright97
    w = _cell_area().area_weights_m2 * (_wright97_mask()[..., 0] > 0)
    mean = (eta_mano.nan_to_num() * w).sum((-2, -1)) / w.sum()
    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-6, rtol=0)
    # a per-sample uniform zos offset (SSH vs. zos) and NaN off-mask cancel
    offset = torch.tensor([0.3, -1.2]).reshape(2, 1, 1, 1)
    shifted = dict(data, zos=data["zos"] + offset)
    shifted["zos"][:, :, 0] = torch.nan
    torch.testing.assert_close(
        _wright97_ocean_data(shifted).manometric_sea_level_wright97,
        eta_mano,
        equal_nan=True,
        atol=1e-6,  # float32 at |zos + C / RHO_0| ~ 10 m
        rtol=0,
    )


def test_manometric_sea_level_wright97_nan_zos_in_wet_cell():
    data = _wright97_data()
    data["zos"][0, 0, 2, 3] = torch.nan
    eta_mano = _wright97_ocean_data(data).manometric_sea_level_wright97
    torch.testing.assert_close(eta_mano, _expected_eta_mano(data), equal_nan=True)
    assert eta_mano[0, 0, 2, 3].isnan()
    assert eta_mano[1, 0, 2, 3].isfinite()
    assert eta_mano[0, 0, 1:].isfinite().sum() == (N_LAT - 1) * N_LON - 1


def test_manometric_sea_level_wright97_nan_wet_level_counts_as_zero():
    data = _wright97_data()
    data["thetao_1"][0, 0, 3, 2] = torch.nan
    ocean_data = _wright97_ocean_data(data)
    eta_mano = ocean_data.manometric_sea_level_wright97
    assert eta_mano[0, 0, 3, 2].isfinite()
    torch.testing.assert_close(eta_mano, _expected_eta_mano(data), equal_nan=True)
    # D4: the MLD of the same cell is NaN
    assert ocean_data.mld_wright97[0, 0, 3, 2].isnan()


def test_wright97_one_dimensional_mask_matches_expanded():
    data = _wright97_data()
    mask_1d = torch.ones(N_LEVELS)
    cell_area = _cell_area()
    out_1d = OceanData(
        data, DepthCoordinate(IDEPTH, mask_1d), cell_area_provider=cell_area
    )
    out_3d = OceanData(
        data,
        DepthCoordinate(IDEPTH, mask_1d.expand(N_LAT, N_LON, N_LEVELS)),
        cell_area_provider=cell_area,
    )
    for name in (
        "rho_wright97",
        "mld_wright97",
        "manometric_sea_level_wright97",
        "steric_sea_level_wright97",
    ):
        torch.testing.assert_close(out_1d[name], out_3d[name])
        assert out_1d[name].isfinite().all()


def test_steric_sea_level_wright97_values_and_mask():
    data = _wright97_data()
    eta_steric = _wright97_ocean_data(data).steric_sea_level_wright97
    expected = -_expected_column_density(data, dtype=torch.float64) / RHO_0
    land = _wright97_mask()[..., 0] == 0
    torch.testing.assert_close(
        eta_steric[..., ~land].double(),
        expected[..., ~land],
        atol=1e-5,  # float32 against float64 at |C / RHO_0| ~ 5 m
        rtol=0,
    )
    assert eta_steric[..., land].isnan().all()
    # the dry level 1 in row 1 contributes 0
    single = OceanData(
        {k: v for k, v in data.items() if k.endswith("_0")},
        DepthCoordinate(IDEPTH[:2], _wright97_mask()[..., :1], DEPTHO),
    ).steric_sea_level_wright97
    torch.testing.assert_close(eta_steric[:, :, 1], single[:, :, 1])


def test_steric_sea_level_wright97_nan_wet_level_counts_as_zero():
    data = _wright97_data()
    data["thetao_1"][0, 0, 3, 2] = torch.nan
    eta_steric = _wright97_ocean_data(data).steric_sea_level_wright97
    assert eta_steric[0, 0, 3, 2].isfinite()
    expected = -_expected_column_density(data, dtype=torch.float64) / RHO_0
    torch.testing.assert_close(
        eta_steric[0, 0, 3, 2].double(), expected[0, 0, 3, 2], atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("perturbation", ["warm", "fresh"])
def test_steric_sea_level_wright97_sign(perturbation):
    data = _wright97_data()
    eta_steric = _wright97_ocean_data(data).steric_sea_level_wright97
    perturbed = dict(data)
    for k in range(N_LEVELS):
        if perturbation == "warm":
            perturbed[f"thetao_{k}"] = data[f"thetao_{k}"] + 1.0
        else:
            perturbed[f"so_{k}"] = data[f"so_{k}"] - 1.0
    change = _wright97_ocean_data(perturbed).steric_sea_level_wright97 - eta_steric
    assert (change[:, :, 1:] > 0).all()


def test_steric_plus_manometric_recovers_zos():
    data = _wright97_data()
    data["zos"][0, 0, 2, 3] = torch.nan
    ocean_data = _wright97_ocean_data(data)
    eta_steric = ocean_data.steric_sea_level_wright97
    wet = (_wright97_mask()[..., 0] > 0) & data["zos"].isfinite()
    w = _cell_area().area_weights_m2 * wet

    def demeaned(x):
        mean = (x.nan_to_num() * w).sum((-2, -1), keepdim=True) / w.sum(
            (-2, -1), keepdim=True
        )
        return torch.where(wet, x - mean, torch.nan)

    torch.testing.assert_close(
        demeaned(data["zos"]),
        demeaned(eta_steric) + ocean_data.manometric_sea_level_wright97,
        equal_nan=True,
        atol=1e-6,  # float32 at |C / RHO_0| ~ 10 m
        rtol=0,
    )


@pytest.mark.parametrize("missing", ["zos", "cell_area"])
def test_steric_sea_level_wright97_without_cell_area_or_zos(missing):
    data = _wright97_data()
    expected = _wright97_ocean_data(data).steric_sea_level_wright97
    if missing == "zos":
        del data["zos"]
    eta_steric = _wright97_ocean_data(
        data, cell_area=missing != "cell_area"
    ).steric_sea_level_wright97
    torch.testing.assert_close(eta_steric, expected, equal_nan=True)


@pytest.mark.parametrize("missing", ["zos", "cell_area", "area_weights_m2"])
def test_manometric_sea_level_wright97_missing_inputs_raise_key_error(missing):
    data = _wright97_data()
    if missing == "zos":
        del data["zos"]
    ocean_data = _wright97_ocean_data(data, cell_area=missing != "cell_area")
    if missing == "area_weights_m2":  # as HEALPixCoordinates
        ocean_data = OceanData(
            data,
            DepthCoordinate(IDEPTH, _wright97_mask(), DEPTHO),
            # HEALPixCoordinates.area_weights_m2 is None, outside the Protocol
            cell_area_provider=_CellArea(None),  # type: ignore[arg-type]
        )
    with pytest.raises(KeyError):
        _ = ocean_data.manometric_sea_level_wright97


@pytest.mark.parametrize(
    "name",
    [
        "rho_wright97",
        "mld_wright97",
        "manometric_sea_level_wright97",
        "steric_sea_level_wright97",
    ],
)
def test_wright97_extra_levels_raise_value_error(name):
    data = _wright97_data()
    data[f"thetao_{N_LEVELS}"] = data["thetao_0"].clone()
    with pytest.raises(ValueError, match="more than"):
        _ = getattr(_wright97_ocean_data(data), name)


@pytest.mark.parametrize(
    "name",
    [
        "rho_wright97",
        "mld_wright97",
        "manometric_sea_level_wright97",
        "steric_sea_level_wright97",
    ],
)
def test_wright97_missing_profiles_raise_key_error_before_geometry(name):
    data = {"zos": torch.zeros(1, 1, N_LAT, N_LON)}
    for coord in (None, _IntegralOnly()):
        ocean_data = OceanData(data, coord, cell_area_provider=_cell_area())
        with pytest.raises(KeyError):
            _ = getattr(ocean_data, name)


@pytest.mark.parametrize(
    "name",
    [
        "rho_wright97",
        "manometric_sea_level_wright97",
        "steric_sea_level_wright97",
    ],
)
def test_wright97_without_layer_geometry_raises_value_error(name):
    data = _wright97_data()
    for coord in (None, _IntegralOnly()):
        ocean_data = OceanData(data, coord, cell_area_provider=_cell_area())
        with pytest.raises(ValueError, match="depth coordinate"):
            _ = getattr(ocean_data, name)
