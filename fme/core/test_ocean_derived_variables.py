import datetime
import math

import pytest
import torch

from fme.core.area_weights_provider import (
    AreaWeightsProvider,
    AreaWeightsProviderABC,
    NullAreaWeightsProvider,
)
from fme.core.constants import (
    DENSITY_OF_SEA_WATER_CM4,
    REFERENCE_SALINITY,
    SPECIFIC_HEAT_OF_SEA_WATER_CM4,
)
from fme.core.coordinates import DepthCoordinate, LatLonCoordinates
from fme.core.derived_variables import get_derived_variable_metadata
from fme.core.ocean_data import OceanData
from fme.core.ocean_derived_variables import (
    HI,
    _compute_ocean_derived_variable,
    compute_ocean_derived_quantities,
    get_ocean_derived_variable_metadata,
)
from fme.core.typing_ import TensorDict, TensorMapping

TIMESTEP = datetime.timedelta(hours=5 * 24)


def test_compute_ocean_derived_variable():
    """Test computing a single ocean derived variable."""
    fake_data = {
        "thetao_0": torch.tensor([29.0]),
        "thetao_1": torch.tensor([10.0]),
    }

    idepth = torch.tensor([2.5, 10, 20])
    lev_thickness = idepth.diff(dim=-1)
    depth_coordinate = DepthCoordinate(
        idepth=idepth,
        mask=torch.ones(2),
    )

    def _derived_variable_func(data: OceanData, *_) -> torch.Tensor:
        return data.ocean_heat_content

    output_data = _compute_ocean_derived_variable(
        fake_data,
        depth_coordinate,
        TIMESTEP,
        "ocean_heat_content",
        _derived_variable_func,
    )
    assert "ocean_heat_content" in output_data
    torch.testing.assert_close(
        output_data["ocean_heat_content"],
        torch.tensor(
            [
                SPECIFIC_HEAT_OF_SEA_WATER_CM4
                * DENSITY_OF_SEA_WATER_CM4
                * (
                    lev_thickness[0] * fake_data["thetao_0"]
                    + lev_thickness[1] * fake_data["thetao_1"]
                )
            ]
        ),
    )


def test_compute_ocean_derived_variable_raises_value_error_when_overwriting():
    """Test that attempting to overwrite an existing variable raises an error."""
    fake_data = {
        "thetao_0": torch.tensor([29.0]),
        "thetao_1": torch.tensor([29.0]),
    }
    depth_coordinate = DepthCoordinate(
        idepth=torch.tensor([0.0, 5.0, 15.0]),
        mask=torch.ones(2),
    )

    def compute_ohc(data: OceanData, *_) -> torch.Tensor:
        return data.ocean_heat_content

    with pytest.raises(ValueError, match="already exists"):
        _compute_ocean_derived_variable(
            fake_data, depth_coordinate, TIMESTEP, "thetao_0", compute_ohc
        )


def test_compute_ocean_derived_variable_existing_variable():
    """Test that attempting to overwrite an existing variable raises an error."""
    fake_data = {
        "sea_ice_fraction": torch.tensor([1.0]),
    }
    depth_coordinate = DepthCoordinate(
        idepth=torch.tensor([0.0, 5.0, 15.0]),
        mask=torch.ones(2),
    )

    def modify_sea_ice_fraction(data: OceanData, *_) -> torch.Tensor:
        return data.sea_ice_fraction - 1

    new_data = _compute_ocean_derived_variable(
        fake_data,
        depth_coordinate,
        TIMESTEP,
        "sea_ice_fraction",
        modify_sea_ice_fraction,
        exists_ok=True,
    )
    torch.testing.assert_close(
        fake_data["sea_ice_fraction"],
        new_data["sea_ice_fraction"],
        msg=(
            "Existing variables should not be modified by "
            "_compute_ocean_derived_variable"
        ),
    )


def test_compute_ocean_derived_quantities():
    """Test computing all registered ocean derived variables."""
    torch.manual_seed(0)

    fake_data = {
        "thetao_0": torch.rand(2, 3, 4, 8),  # [batch, time, lat, lon]
        "thetao_1": torch.rand(2, 3, 4, 8),
        "ocean_sea_ice_fraction": torch.rand(2, 3, 4, 8),
        "land_fraction": torch.rand(2, 3, 4, 8),
    }
    gen_data = fake_data.copy()
    depth_coordinate = DepthCoordinate(
        idepth=torch.tensor([0.0, 5.0, 15.0]),
        mask=torch.ones(2, 3, 4, 8, 2),
    )

    def derive_func(data: TensorMapping, forcing_data: TensorMapping) -> TensorDict:
        updated = compute_ocean_derived_quantities(
            dict(data),
            depth_coordinate=depth_coordinate,
            timestep=TIMESTEP,
            forcing_data=dict(forcing_data),
        )
        return updated

    out_data = derive_func(gen_data, fake_data)

    # Test that ocean_heat_content was computed
    assert "ocean_heat_content" in out_data
    assert out_data["ocean_heat_content"].shape == (2, 3, 4, 8)

    # Test that sea_ice_fraction was computed
    assert "sea_ice_fraction" in out_data
    assert out_data["sea_ice_fraction"].shape == (2, 3, 4, 8)


def test_metadata_registry():
    """Test that the metadata registry contains expected entries."""
    metadata = get_ocean_derived_variable_metadata()
    assert metadata["ocean_heat_content"].units == "J/m**2"
    assert (
        metadata["ocean_heat_content"].long_name
        == "Column-integrated ocean heat content per unit ocean area"
    )


def _compute_salt_budget(
    wfo: float, sfdsi: float | None, sea_surface_fraction: float
) -> tuple[TensorDict, float]:
    """Computes the ocean derived quantities for a single-level column whose
    salinity changes only through the surface salt fluxes, where sfdsi of None
    means it is missing from the data.

    Returns the derived quantities and the surface salt flux in g/m2/s per
    unit ocean area.
    """
    dz = 10.0
    initial_salinity = 35.0
    # sfdsi is in kg/m2/s
    sfdsi_flux = 0.0 if sfdsi is None or math.isnan(sfdsi) else sfdsi
    salt_flux = -REFERENCE_SALINITY * wfo + 1000.0 * sfdsi_flux
    salinity_change = (
        salt_flux * TIMESTEP.total_seconds() / (DENSITY_OF_SEA_WATER_CM4 * dz)
    )
    shape = (1, 2, 1, 1)
    data = {
        "so_0": torch.tensor(
            [initial_salinity, initial_salinity + salinity_change],
            dtype=torch.float64,
        ).reshape(shape),
        "wfo": torch.full(shape, wfo, dtype=torch.float64),
        "sea_surface_fraction": torch.full(
            shape, sea_surface_fraction, dtype=torch.float64
        ),
    }
    if sfdsi is not None:
        data["sfdsi"] = torch.full(shape, sfdsi, dtype=torch.float64)
    depth_coordinate = DepthCoordinate(
        idepth=torch.tensor([0.0, dz], dtype=torch.float64),
        mask=torch.ones(*shape, 1, dtype=torch.float64),
    )
    out = compute_ocean_derived_quantities(
        data, depth_coordinate=depth_coordinate, timestep=TIMESTEP
    )
    return out, salt_flux


@pytest.mark.parametrize(
    "wfo, sfdsi, sea_surface_fraction",
    [
        pytest.param(1e-5, 0.0, 1.0, id="wfo"),
        pytest.param(0.0, 2e-7, 0.5, id="sfdsi-partial-ocean"),
        pytest.param(1e-5, 2e-7, 0.5, id="wfo-and-sfdsi-partial-ocean"),
        pytest.param(1e-5, float("nan"), 0.5, id="wfo-sfdsi-nan-ice-free"),
    ],
)
def test_salt_budget_closes(wfo: float, sfdsi: float, sea_surface_fraction: float):
    """A salinity change set by the surface salt fluxes leaves no implied
    advection, including in a cell that is partly land and when sfdsi is NaN.
    """
    out, salt_flux = _compute_salt_budget(wfo, sfdsi, sea_surface_fraction)
    expected_flux = torch.full(
        (1, 1, 1), salt_flux * sea_surface_fraction, dtype=torch.float64
    )
    torch.testing.assert_close(out["ocean_salt_content_tendency"][:, 1], expected_flux)
    torch.testing.assert_close(
        out["net_salt_flux_into_ocean_column"][:, 1], expected_flux
    )
    torch.testing.assert_close(
        out["implied_tendency_of_ocean_salt_content_due_to_advection"][:, 1],
        torch.zeros((1, 1, 1), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_salt_budget_without_sfdsi():
    """Without sfdsi, the net salt flux is not computed, and the implied
    advection closes with the virtual salt flux alone.
    """
    wfo, sea_surface_fraction = 1e-5, 0.5
    out, _ = _compute_salt_budget(wfo, None, sea_surface_fraction)
    assert "net_salt_flux_into_ocean_column" not in out
    torch.testing.assert_close(
        out["implied_tendency_of_ocean_salt_content_due_to_advection"][:, 1],
        torch.zeros((1, 1, 1), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        out["net_virtual_salt_flux_into_ocean_column"],
        torch.full(
            (1, 2, 1, 1),
            -REFERENCE_SALINITY * wfo * sea_surface_fraction,
            dtype=torch.float64,
        ),
    )


def _compute_heat_budget(
    hfds: float, hfgeou: float, sea_surface_fraction: float
) -> TensorDict:
    """Computes the ocean derived quantities for a single-level column whose
    temperature changes only through the surface and geothermal heat fluxes,
    given in W/m2 per unit ocean area.
    """
    dz = 10.0
    initial_temperature = 10.0
    temperature_change = (
        (hfds + hfgeou)
        * TIMESTEP.total_seconds()
        / (DENSITY_OF_SEA_WATER_CM4 * SPECIFIC_HEAT_OF_SEA_WATER_CM4 * dz)
    )
    shape = (1, 2, 1, 1)
    data = {
        "thetao_0": torch.tensor(
            [initial_temperature, initial_temperature + temperature_change],
            dtype=torch.float64,
        ).reshape(shape),
        "hfds": torch.full(shape, hfds, dtype=torch.float64),
        "hfgeou": torch.full(shape, hfgeou, dtype=torch.float64),
        "sea_surface_fraction": torch.full(
            shape, sea_surface_fraction, dtype=torch.float64
        ),
    }
    depth_coordinate = DepthCoordinate(
        idepth=torch.tensor([0.0, dz], dtype=torch.float64),
        mask=torch.ones(*shape, 1, dtype=torch.float64),
    )
    return compute_ocean_derived_quantities(
        data, depth_coordinate=depth_coordinate, timestep=TIMESTEP
    )


@pytest.mark.parametrize(
    "hfds, hfgeou, sea_surface_fraction",
    [
        pytest.param(100.0, 0.0, 1.0, id="hfds"),
        pytest.param(100.0, 0.0, 0.5, id="hfds-partial-ocean"),
        pytest.param(100.0, 0.05, 0.5, id="hfds-and-hfgeou-partial-ocean"),
    ],
)
def test_heat_budget_closes(hfds: float, hfgeou: float, sea_surface_fraction: float):
    """A temperature change set by the heat fluxes leaves no implied
    advection, including in a cell that is partly land.
    """
    out = _compute_heat_budget(hfds, hfgeou, sea_surface_fraction)
    ohc_tendency = out["ocean_heat_content_tendency"][:, 1]
    net_flux = out["net_energy_flux_into_ocean_column"][:, 1]
    implied_advection = out["implied_tendency_of_ocean_heat_content_due_to_advection"][
        :, 1
    ]
    torch.testing.assert_close(
        ohc_tendency, torch.full((1, 1, 1), hfds + hfgeou, dtype=torch.float64)
    )
    torch.testing.assert_close(
        net_flux,
        torch.full(
            (1, 1, 1), (hfds + hfgeou) * sea_surface_fraction, dtype=torch.float64
        ),
    )
    torch.testing.assert_close(
        implied_advection,
        torch.zeros((1, 1, 1), dtype=torch.float64),
        atol=1e-9,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("ocean_sea_ice_fraction", id="try-path"),
        pytest.param("sea_ice_fraction_and_land_fraction", id="except-path"),
    ],
)
def test_sea_ice_thickness_derived_variable(case):
    """Test recovering sea ice thickness (HI) from sea ice volume."""
    n_lat, n_lon = 4, 8
    horizontal_coordinates = LatLonCoordinates(
        lat=torch.linspace(-60, 60, n_lat),
        lon=torch.linspace(0, 360, n_lon + 1)[:-1],  # avoid wrapping
    )
    cell_area = horizontal_coordinates.area_weights_m2

    thickness_in_m = torch.full((1, 1, n_lat, n_lon), 2.0)
    thickness_in_m[:, :, 1, 1] = float("nan")
    sea_surface_frac = torch.full((1, 1, n_lat, n_lon), 0.7)

    if case == "sea_ice_fraction_and_land_fraction":
        sea_ice_frac = torch.full((1, 1, n_lat, n_lon), 0.6)
        land_frac = 1 - sea_surface_frac
        effective_sea_ice_frac = sea_ice_frac * sea_surface_frac / (1 - land_frac)
        fake_data = {
            "sea_ice_volume": thickness_in_m * cell_area * effective_sea_ice_frac,
            "sea_ice_fraction": sea_ice_frac,
            "land_fraction": land_frac,
            "sea_surface_fraction": sea_surface_frac,
        }
    else:
        ocean_sea_ice_frac = torch.full((1, 1, n_lat, n_lon), 0.6)
        effective_sea_ice_frac = ocean_sea_ice_frac * sea_surface_frac
        fake_data = {
            "sea_ice_volume": thickness_in_m * cell_area * effective_sea_ice_frac,
            "ocean_sea_ice_fraction": ocean_sea_ice_frac,
            "sea_surface_fraction": sea_surface_frac,
        }

    ocean_data = OceanData(fake_data, cell_area_provider=horizontal_coordinates)

    recovered_thickness = ocean_data.sea_ice_thickness
    torch.testing.assert_close(
        recovered_thickness,
        thickness_in_m,
        equal_nan=True,
        msg="Recovered sea ice thickness should match original",
    )


def test_HI_derived_from_ocean_sea_ice_thickness_mean():
    data = {
        "ocean_sea_ice_thickness_mean": torch.tensor([[1.5, 0.0, float("nan")]]),
        "ocean_sea_ice_fraction": torch.tensor([[0.75, 0.0, float("nan")]]),
    }
    out = _compute_ocean_derived_variable(
        data,
        depth_coordinate=None,
        timestep=TIMESTEP,
        label="HI",
        derived_variable_func=HI,
    )
    torch.testing.assert_close(
        out["HI"], torch.tensor([[2.0, 0.0, float("nan")]]), equal_nan=True
    )


def _zos_inputs(n_lat: int = 4, n_lon: int = 8) -> tuple[LatLonCoordinates, TensorDict]:
    torch.manual_seed(0)
    horizontal_coordinates = LatLonCoordinates(
        lat=torch.linspace(-60, 60, n_lat),
        lon=torch.linspace(0, 360, n_lon + 1)[:-1],
    )
    shape = (2, 3, n_lat, n_lon)
    sea_surface_fraction = torch.rand(shape)
    ssh = torch.randn(shape) + 0.5
    land = torch.zeros(shape, dtype=torch.bool)
    land[..., 0, :2] = True
    ssh[land] = float("nan")
    sea_surface_fraction[land] = 0.0
    data = {"SSH": ssh, "sea_surface_fraction": sea_surface_fraction}
    return horizontal_coordinates, data


def _derive(
    data: TensorDict,
    horizontal_coordinates: LatLonCoordinates | None,
    area_weights_provider: AreaWeightsProviderABC = NullAreaWeightsProvider,
) -> TensorDict:
    return compute_ocean_derived_quantities(
        dict(data),
        depth_coordinate=DepthCoordinate(
            idepth=torch.tensor([0.0, 5.0]), mask=torch.ones(1)
        ),
        timestep=TIMESTEP,
        cell_area_provider=horizontal_coordinates,
        area_weights_provider=area_weights_provider,
    )


def _coarse_grained_ssh(
    n_lat: int = 4, n_lon: int = 8, n_sub: int = 3
) -> tuple[LatLonCoordinates, TensorDict, AreaWeightsProvider, torch.Tensor]:
    """Target cells J each hold n_sub native cells i; data are native-area means.

    Returns coordinates, target data (SSH, sea_surface_fraction), the provider
    (areacello_J * ssf_J) and <SSH>_native per sample.
    """
    torch.manual_seed(0)
    horizontal_coordinates = LatLonCoordinates(
        lat=torch.linspace(-60, 60, n_lat),
        lon=torch.linspace(0, 360, n_lon + 1)[:-1],
    )
    areacello_i = torch.rand(n_lat, n_lon, n_sub, dtype=torch.float64) + 0.5
    wet_i = torch.rand(n_lat, n_lon, n_sub) > 0.3
    wet_i[0, :2] = False
    ocean_area_i = areacello_i * wet_i
    ssh_i = torch.randn(2, 3, n_lat, n_lon, n_sub, dtype=torch.float64)
    areacello_J = areacello_i.sum(dim=-1)
    ocean_area_J = ocean_area_i.sum(dim=-1)
    ssf_J = ocean_area_J / areacello_J
    ssh_J = (ocean_area_i * ssh_i).sum(dim=-1) / ocean_area_J  # NaN on all-land J
    native_mean = (ocean_area_i * ssh_i).sum(dim=(-3, -2, -1)) / ocean_area_i.sum()
    data = {"SSH": ssh_J, "sea_surface_fraction": ssf_J.expand_as(ssh_J)}
    provider = AreaWeightsProvider.from_cell_area_and_fraction(areacello_J, ssf_J)
    return horizontal_coordinates, data, provider, native_mean[..., None, None]


def test_zos_with_area_weights_provider_matches_native_mean():
    horizontal_coordinates, data, provider, native_mean = _coarse_grained_ssh()
    out = _derive(data, horizontal_coordinates, provider)
    torch.testing.assert_close(out["zos"], data["SSH"] - native_mean, equal_nan=True)


def test_zos_with_area_weights_provider_does_not_need_sea_surface_fraction():
    horizontal_coordinates, data, provider, native_mean = _coarse_grained_ssh()
    del data["sea_surface_fraction"]
    out = _derive(data, None, provider)
    torch.testing.assert_close(out["zos"], data["SSH"] - native_mean, equal_nan=True)


def test_zos_without_area_weights_provider_falls_back_to_cell_area():
    horizontal_coordinates, data, _, native_mean = _coarse_grained_ssh()
    out = _derive(data, horizontal_coordinates)
    ssh = data["SSH"]
    weights = torch.where(
        torch.isfinite(ssh),
        horizontal_coordinates.area_weights_m2.double() * data["sea_surface_fraction"],
        0.0,
    )
    fallback_mean = (weights * ssh.nan_to_num()).sum(
        dim=(-2, -1), keepdim=True
    ) / weights.sum(dim=(-2, -1), keepdim=True)
    torch.testing.assert_close(out["zos"], ssh - fallback_mean, equal_nan=True)
    assert not torch.allclose(fallback_mean, native_mean)


def test_zos_derived_from_ssh():
    horizontal_coordinates, data = _zos_inputs()
    out = _derive(data, horizontal_coordinates)
    ssh = data["SSH"].double()
    finite = torch.isfinite(ssh)
    weights = (
        horizontal_coordinates.area_weights_m2.double()
        * data["sea_surface_fraction"].double()
        * finite
    )
    expected_mean = (weights * ssh.nan_to_num()).sum(
        dim=(-2, -1), keepdim=True
    ) / weights.sum(dim=(-2, -1), keepdim=True)
    torch.testing.assert_close(out["zos"].double(), ssh - expected_mean, equal_nan=True)
    zos_mean = (weights * out["zos"].double().nan_to_num()).sum(
        dim=(-2, -1)
    ) / weights.sum(dim=(-2, -1))
    torch.testing.assert_close(
        zos_mean, torch.zeros_like(zos_mean), atol=1e-6, rtol=0.0
    )
    assert torch.equal(torch.isnan(out["zos"]), ~finite)


def test_zos_land_ssh_values_do_not_change_mean():
    horizontal_coordinates, data = _zos_inputs()
    out_nan_land = _derive(data, horizontal_coordinates)
    filled = dict(data)
    filled["SSH"] = data["SSH"].nan_to_num(nan=1.0e3)
    out_filled_land = _derive(filled, horizontal_coordinates)
    ocean = torch.isfinite(data["SSH"])
    torch.testing.assert_close(
        out_nan_land["zos"][ocean], out_filled_land["zos"][ocean]
    )


def test_zos_mean_excludes_nan_ssh_cells_with_positive_fraction():
    horizontal_coordinates, data = _zos_inputs()
    nan_ssh = torch.isnan(data["SSH"])
    data["sea_surface_fraction"] = data["sea_surface_fraction"].masked_fill(
        nan_ssh, 1.0
    )
    out = _derive(data, horizontal_coordinates)
    weights = torch.where(
        nan_ssh,
        0.0,
        horizontal_coordinates.area_weights_m2 * data["sea_surface_fraction"],
    )
    zos_mean = (weights * out["zos"].nan_to_num()).sum(dim=(-2, -1))
    torch.testing.assert_close(
        zos_mean / weights.sum(dim=(-2, -1)),
        torch.zeros_like(zos_mean),
        atol=1e-6,
        rtol=0.0,
    )


def test_zos_not_derived_without_ssh():
    horizontal_coordinates, data = _zos_inputs()
    del data["SSH"]
    out = _derive(data, horizontal_coordinates)
    assert "zos" not in out


def test_zos_not_derived_without_cell_area_provider():
    _, data = _zos_inputs()
    out = _derive(data, None)
    assert "zos" not in out


def test_zos_not_derived_without_sea_surface_fraction():
    horizontal_coordinates, data = _zos_inputs()
    del data["sea_surface_fraction"]
    out = _derive(data, horizontal_coordinates)
    assert "zos" not in out


class _NoCellArea:
    area_weights_m2 = None


def test_zos_not_derived_when_cell_area_is_none():
    _, data = _zos_inputs()
    out = compute_ocean_derived_quantities(
        dict(data),
        depth_coordinate=DepthCoordinate(
            idepth=torch.tensor([0.0, 5.0]), mask=torch.ones(1)
        ),
        timestep=TIMESTEP,
        cell_area_provider=_NoCellArea(),
    )
    assert "zos" not in out


def test_zos_in_data_is_untouched():
    horizontal_coordinates, data = _zos_inputs()
    stored_zos = torch.randn_like(data["SSH"])
    data["zos"] = stored_zos.clone()
    out = _derive(data, horizontal_coordinates)
    assert torch.equal(out["zos"], stored_zos)


def test_zos_registered_before_variables_that_may_read_it():
    labels = list(get_ocean_derived_variable_metadata())
    assert labels.index("zos") < labels.index("sea_ice_fraction")
    assert get_ocean_derived_variable_metadata()["zos"].units == "m"


def test_mld_wright97_derived_variable():
    torch.manual_seed(0)
    shape = (2, 3, 4, 8)
    nz = 4
    data = {f"thetao_{k}": 20.0 - 3.0 * k * torch.rand(*shape) for k in range(nz)}
    data.update({f"so_{k}": 34.0 + torch.rand(*shape) for k in range(nz)})
    mask = torch.ones(4, 8, nz)
    mask[0, 0, :] = 0.0
    depth_coordinate = DepthCoordinate(
        idepth=torch.tensor([0.0, 10.0, 30.0, 60.0, 100.0]), mask=mask
    )
    out = compute_ocean_derived_quantities(
        dict(data), depth_coordinate=depth_coordinate, timestep=TIMESTEP
    )
    assert "mld_wright97" in out
    expected = OceanData(data, depth_coordinate).mld_wright97
    torch.testing.assert_close(out["mld_wright97"], expected, equal_nan=True)
    assert torch.isnan(out["mld_wright97"][..., 0, 0]).all()
    assert torch.isfinite(out["mld_wright97"][..., 1:, :]).all()
    metadata = get_ocean_derived_variable_metadata()["mld_wright97"]
    assert metadata.units == "m"
    assert metadata.long_name == ("Mixed layer depth, Wright (1997) density threshold")


def test_mld_wright97_skipped_without_salinity():
    data = {f"thetao_{k}": torch.rand(1, 1, 2, 2) for k in range(3)}
    depth_coordinate = DepthCoordinate(
        idepth=torch.tensor([0.0, 10.0, 30.0, 60.0]), mask=torch.ones(2, 2, 3)
    )
    out = compute_ocean_derived_quantities(
        dict(data), depth_coordinate=depth_coordinate, timestep=TIMESTEP
    )
    assert "mld_wright97" not in out
    assert "ocean_heat_content" in out


def test_wright97_metadata_has_units():
    metadata = get_derived_variable_metadata()
    assert metadata["mld_wright97"].units == "m"
    assert metadata["pbo_wright97"].units == "Pa"


def _wright97_inputs():
    n_lat, n_lon, nz = 4, 8, 3
    horizontal_coordinates = LatLonCoordinates(
        lat=torch.linspace(-60, 60, n_lat),
        lon=torch.linspace(0, 360, n_lon + 1)[:-1],
    )
    data = {
        f"thetao_{k}": 20.0 - 3.0 * k * torch.rand(1, 1, n_lat, n_lon)
        for k in range(nz)
    }
    data.update({f"so_{k}": 34.0 + torch.rand(1, 1, n_lat, n_lon) for k in range(nz)})
    data["zos"] = 0.1 * torch.randn(1, 1, n_lat, n_lon)
    depth_coordinate = DepthCoordinate(
        idepth=torch.tensor([0.0, 10.0, 30.0, 60.0]),
        mask=torch.ones(n_lat, n_lon, nz),
    )
    return data, depth_coordinate, horizontal_coordinates


def test_pbo_wright97_derived_variable():
    data, depth_coordinate, horizontal_coordinates = _wright97_inputs()
    out = compute_ocean_derived_quantities(
        dict(data),
        depth_coordinate=depth_coordinate,
        timestep=TIMESTEP,
        cell_area_provider=horizontal_coordinates,
    )
    expected = OceanData(
        data, depth_coordinate, cell_area_provider=horizontal_coordinates
    ).pbo_wright97
    torch.testing.assert_close(out["pbo_wright97"], expected)


@pytest.mark.parametrize("missing", ["cell_area", "zos"])
def test_pbo_wright97_skipped_without_cell_area_or_zos(missing):
    data, depth_coordinate, horizontal_coordinates = _wright97_inputs()
    if missing == "zos":
        del data["zos"]
    out = compute_ocean_derived_quantities(
        dict(data),
        depth_coordinate=depth_coordinate,
        timestep=TIMESTEP,
        cell_area_provider=None if missing == "cell_area" else horizontal_coordinates,
    )
    assert "mld_wright97" in out
    assert "pbo_wright97" not in out
