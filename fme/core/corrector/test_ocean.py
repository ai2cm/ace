import datetime

import pytest
import torch

from fme import get_device
from fme.core.constants import (
    FREEZING_TEMPERATURE_KELVIN,
    LATENT_HEAT_OF_FREEZING,
    LATENT_HEAT_OF_VAPORIZATION,
    SPECIFIC_HEAT_OF_SEA_WATER_CM4,
)
from fme.core.coordinates import DepthCoordinate
from fme.core.corrector.ocean import (
    OceanCorrector,
    OceanCorrectorConfig,
    OceanHeatContentBudgetConfig,
    SeaIceFractionConfig,
    SurfaceEnergyFluxCorrectionConfig,
    _compute_ocean_net_surface_energy_flux,
)
from fme.core.gridded_ops import LatLonOperations
from fme.core.mask_provider import MaskProvider
from fme.core.ocean_data import OceanData
from fme.core.typing_ import TensorMapping

DEVICE = get_device()
IMG_SHAPE = (5, 5)
NZ = 2

_MASK = torch.ones(*IMG_SHAPE, NZ, device=DEVICE)
_LAT, _LON = 2, 2
_MASK[_LAT, _LON, :] = 0.0


class _MockDepth:
    def depth_integral(self, integrand: torch.Tensor) -> torch.Tensor:
        idepth = torch.tensor([0, 5, 15], device=DEVICE)
        thickness = idepth.diff(dim=-1)
        return torch.nansum(_MASK * integrand * thickness, dim=-1)


_VERTICAL_COORD = _MockDepth()


def test_ocean_corrector_force_positive():
    """"""
    torch.manual_seed(0)
    config = OceanCorrectorConfig(force_positive_names=["so_0", "so_1"])
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = OceanCorrector(config, ops, _VERTICAL_COORD, timestep)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    corrected_gen = corrector(input_data, gen_data, {})
    for name in ["so_0", "so_1"]:
        x = corrected_gen[name].clone()
        x[_LAT, _LON] = 0.0
        assert torch.all(x >= 0.0)


def test_ocean_corrector_has_no_negative_ocean_fraction():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    input_data["land_fraction"] = torch.ones(IMG_SHAPE, device=DEVICE) * 0.8
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data["sea_ice_fraction"] = torch.randn(IMG_SHAPE, device=DEVICE) * 0.5
    gen_data["sea_ice_fraction"][_LAT, _LON] = -0.5
    corrector = OceanCorrector(config, ops, None, timestep)
    violation = (input_data["land_fraction"] + gen_data["sea_ice_fraction"]) > 1.0
    assert violation.any()
    negative_sea_ice_fraction = gen_data["sea_ice_fraction"] < 0.0
    assert negative_sea_ice_fraction.any()

    next_step_input_data: TensorMapping = {}
    gen_data_corrected = corrector(input_data, gen_data, next_step_input_data)
    corrected_violation = (
        input_data["land_fraction"] + gen_data_corrected["sea_ice_fraction"]
    ) > 1.0
    assert not corrected_violation.any()
    assert not (gen_data_corrected["sea_ice_fraction"] < 0.0).any()


def test_ocean_corrector_has_negative_ocean_fraction():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            remove_negative_ocean_fraction=False,
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    input_data["land_fraction"] = torch.ones(IMG_SHAPE, device=DEVICE) * 0.8
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data["sea_ice_fraction"] = torch.randn(IMG_SHAPE, device=DEVICE) * 0.5
    gen_data["sea_ice_fraction"][_LAT, _LON] = -0.5
    corrector = OceanCorrector(config, ops, None, timestep)
    violation = (input_data["land_fraction"] + gen_data["sea_ice_fraction"]) > 1.0
    assert violation.any()
    negative_sea_ice_fraction = gen_data["sea_ice_fraction"] < 0.0
    assert negative_sea_ice_fraction.any()

    next_step_input_data: TensorMapping = {}
    gen_data_corrected = corrector(input_data, gen_data, next_step_input_data)
    corrected_violation = (
        input_data["land_fraction"] + gen_data_corrected["sea_ice_fraction"]
    ) > 1.0
    assert corrected_violation.any()
    # sea_ice_fraction values are still clamped to [0, 1]
    assert not (gen_data_corrected["sea_ice_fraction"] < 0.0).any()


def test_zero_where_ice_free_names():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            zero_where_ice_free_names=["HI"],
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {"land_fraction": torch.ones(IMG_SHAPE, device=DEVICE)}
    input_data["land_fraction"][:3, :3] = torch.rand(3, 3, device=DEVICE)
    gen_data = {
        "sea_ice_fraction": torch.rand(IMG_SHAPE, device=DEVICE),
        "HI": torch.rand(IMG_SHAPE, device=DEVICE) * 10,
    }
    corrector = OceanCorrector(config, ops, None, timestep)
    gen_data_corrected = corrector(input_data, gen_data, {})
    sea_ice_zero = gen_data_corrected["sea_ice_fraction"] == 0.0
    thickness = gen_data_corrected["HI"]
    torch.testing.assert_close(
        torch.where(sea_ice_zero, thickness, 0.0), torch.zeros_like(thickness)
    )


def test_zero_where_ice_free_names_multiple_variables():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            zero_where_ice_free_names=["HI", "HS"],
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {"land_fraction": torch.ones(IMG_SHAPE, device=DEVICE)}
    input_data["land_fraction"][:3, :3] = torch.rand(3, 3, device=DEVICE)
    gen_data = {
        "sea_ice_fraction": torch.rand(IMG_SHAPE, device=DEVICE),
        "HI": torch.rand(IMG_SHAPE, device=DEVICE) * 10,
        "HS": torch.rand(IMG_SHAPE, device=DEVICE) * 5,
    }
    corrector = OceanCorrector(config, ops, None, timestep)
    gen_data_corrected = corrector(input_data, gen_data, {})
    sea_ice_zero = gen_data_corrected["sea_ice_fraction"] == 0.0
    for name in ["HI", "HS"]:
        values = gen_data_corrected[name]
        torch.testing.assert_close(
            torch.where(sea_ice_zero, values, 0.0), torch.zeros_like(values)
        )


def test_from_state_migrates_sea_ice_thickness_name():
    state = {
        "sea_ice_fraction_correction": {
            "sea_ice_fraction_name": "ocean_sea_ice_fraction",
            "land_fraction_name": "land_fraction",
            "sea_ice_thickness_name": "HI",
            "remove_negative_ocean_fraction": False,
        },
    }
    config = OceanCorrectorConfig.from_state(state)
    assert config.sea_ice_fraction_correction is not None
    assert config.sea_ice_fraction_correction.zero_where_ice_free_names == ["HI"]


def test_from_state_migrates_sea_ice_thickness_name_none():
    state = {
        "sea_ice_fraction_correction": {
            "sea_ice_fraction_name": "ocean_sea_ice_fraction",
            "land_fraction_name": "land_fraction",
            "sea_ice_thickness_name": None,
            "remove_negative_ocean_fraction": False,
        },
    }
    config = OceanCorrectorConfig.from_state(state)
    assert config.sea_ice_fraction_correction is not None
    assert config.sea_ice_fraction_correction.zero_where_ice_free_names == []


def _make_atmos_forcing_data(shape, device=DEVICE):
    """Build atmosphere forcing tensors needed for the surface energy flux
    correction tests."""
    return {
        "DSWRFsfc": torch.full(shape, 200.0, device=device),
        "USWRFsfc": torch.full(shape, 50.0, device=device),
        "DLWRFsfc": torch.full(shape, 300.0, device=device),
        "ULWRFsfc": torch.full(shape, 350.0, device=device),
        "LHTFLsfc": torch.full(shape, 100.0, device=device),
        "SHTFLsfc": torch.full(shape, 20.0, device=device),
        "PRATEsfc": torch.full(shape, 1e-4, device=device),
        "total_frozen_precipitation_rate": torch.full(shape, 1e-5, device=device),
    }


def test_surface_energy_flux_correction_resid():
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="residual_prediction"
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = OceanCorrector(config, ops, None, timestep)

    sst = torch.full(IMG_SHAPE, 300.0, device=DEVICE)
    gen_hfds = torch.full(IMG_SHAPE, 5.0, device=DEVICE)
    sea_ice_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    sea_ice_fraction[0, :] = 0.3
    land_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    land_fraction[-1, :] = 1.0

    gen_data = {
        "sst": sst,
        "hfds": gen_hfds,
        "sea_ice_fraction": sea_ice_fraction,
    }
    forcing_data = {
        "land_fraction": land_fraction,
        **_make_atmos_forcing_data(IMG_SHAPE),
    }
    input_data = {**forcing_data, **gen_data}

    ocean_fraction = 1 - land_fraction - sea_ice_fraction
    expected_net_flux = _compute_ocean_net_surface_energy_flux(input_data, sst)
    expected_hfds = gen_hfds + ocean_fraction * expected_net_flux

    corrected = corrector(input_data, gen_data, forcing_data)
    torch.testing.assert_close(corrected["hfds"], expected_hfds)
    # on land ocean_fraction is 0, so hfds is unchanged
    torch.testing.assert_close(corrected["hfds"][-1, :], gen_hfds[-1, :])
    # with sea ice, correction is reduced relative to ice-free rows
    ice_row_correction = (corrected["hfds"][0, 0] - gen_hfds[0, 0]).abs()
    open_row_correction = (corrected["hfds"][1, 0] - gen_hfds[1, 0]).abs()
    assert ice_row_correction < open_row_correction


def test_surface_energy_flux_correction_prescribed():
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed"
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = OceanCorrector(config, ops, None, timestep)

    sst = torch.full(IMG_SHAPE, 300.0, device=DEVICE)
    gen_hfds = torch.full(IMG_SHAPE, 5.0, device=DEVICE)
    sea_ice_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    sea_ice_fraction[0, :] = 0.3
    land_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    land_fraction[-1, :] = 1.0

    gen_data = {
        "sst": sst,
        "hfds": gen_hfds,
        "sea_ice_fraction": sea_ice_fraction,
    }
    forcing_data = {
        "land_fraction": land_fraction,
        **_make_atmos_forcing_data(IMG_SHAPE),
    }
    input_data = {**forcing_data, **gen_data}

    ocean_fraction = 1 - land_fraction - sea_ice_fraction
    net_flux = _compute_ocean_net_surface_energy_flux(input_data, sst)
    expected_hfds = net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)

    corrected = corrector(input_data, gen_data, forcing_data)
    torch.testing.assert_close(corrected["hfds"], expected_hfds)
    # on land (ocean_fraction=0), hfds equals gen_hfds
    torch.testing.assert_close(corrected["hfds"][-1, :], gen_hfds[-1, :])
    # in open ocean (no ice, no land), hfds equals net_flux
    open_ocean_row = 1
    torch.testing.assert_close(
        corrected["hfds"][open_ocean_row, :], net_flux[open_ocean_row, :]
    )


_DEFAULT_GRANULAR_NAMES = [
    "sfc_down_sw_radiative_flux",
    "sfc_up_sw_radiative_flux",
    "sfc_down_lw_radiative_flux",
    "sfc_up_lw_radiative_flux",
    "latent_heat_flux",
    "sensible_heat_flux",
    "precipitation_rate",
    "frozen_precipitation_rate",
]


def _surface_flux_test_setup():
    """Return (input_data, gen_data, forcing_data, ocean_fraction, sst)
    suitable for surface_energy_flux_correction tests."""
    sst = torch.full(IMG_SHAPE, 300.0, device=DEVICE)
    gen_hfds = torch.full(IMG_SHAPE, 5.0, device=DEVICE)
    sea_ice_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    sea_ice_fraction[0, :] = 0.3
    land_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    land_fraction[-1, :] = 1.0
    gen_data = {
        "sst": sst,
        "hfds": gen_hfds,
        "sea_ice_fraction": sea_ice_fraction,
    }
    forcing_data = {
        "land_fraction": land_fraction,
        **_make_atmos_forcing_data(IMG_SHAPE),
    }
    input_data = {**forcing_data, **gen_data}
    ocean_fraction = 1 - land_fraction - sea_ice_fraction
    return input_data, gen_data, forcing_data, ocean_fraction, sst


def test_surface_energy_flux_correction_default_omitted_kwarg():
    """Omitting `names` defaults to "default" and matches the legacy formula."""
    config = SurfaceEnergyFluxCorrectionConfig(method="prescribed")
    assert config.names == "default"


def test_surface_energy_flux_correction_full_granular_list_matches_default():
    """Listing all 8 granular names reproduces the legacy default. Tolerances
    accommodate fp32 reassociation between the legacy single-expression sum
    and the atomic per-term sum."""
    input_data, gen_data, forcing_data, _, sst = _surface_flux_test_setup()
    default_flux = _compute_ocean_net_surface_energy_flux(
        forcing_data, sst, names="default"
    )
    granular_flux = _compute_ocean_net_surface_energy_flux(
        forcing_data, sst, names=_DEFAULT_GRANULAR_NAMES
    )
    torch.testing.assert_close(granular_flux, default_flux, rtol=1e-5, atol=1e-4)


def test_surface_energy_flux_correction_buckets_plus_mass_terms_matches_default():
    """`net_surface_energy_flux` plus the three mass-flux variables also
    reproduces the legacy default. Validates atomic-contribution dedup of the
    `lhf_turbulent` and `frozen_precip_latent` terms shared between the bucket
    and the granular names."""
    input_data, gen_data, forcing_data, _, sst = _surface_flux_test_setup()
    default_flux = _compute_ocean_net_surface_energy_flux(
        forcing_data, sst, names="default"
    )
    bucket_flux = _compute_ocean_net_surface_energy_flux(
        forcing_data,
        sst,
        names=[
            "net_surface_energy_flux",
            "precipitation_rate",
            "latent_heat_flux",
            "frozen_precipitation_rate",
        ],
    )
    torch.testing.assert_close(bucket_flux, default_flux, rtol=1e-5, atol=1e-4)


def test_surface_energy_flux_correction_subset_radiative_only():
    """Listing only the four radiative names yields sw_down - sw_up + lw_down -
    lw_up applied via the prescribed-method correction."""
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed",
            names=[
                "sfc_down_sw_radiative_flux",
                "sfc_up_sw_radiative_flux",
                "sfc_down_lw_radiative_flux",
                "sfc_up_lw_radiative_flux",
            ],
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = OceanCorrector(config, ops, None, timestep)
    input_data, gen_data, forcing_data, ocean_fraction, _ = _surface_flux_test_setup()
    expected_net_flux = (
        forcing_data["DSWRFsfc"]
        - forcing_data["USWRFsfc"]
        + forcing_data["DLWRFsfc"]
        - forcing_data["ULWRFsfc"]
    )
    expected_hfds = expected_net_flux * ocean_fraction + gen_data["hfds"] * (
        1 - ocean_fraction
    )
    corrected = corrector(input_data, gen_data, forcing_data)
    torch.testing.assert_close(corrected["hfds"], expected_hfds)


def test_surface_energy_flux_correction_bucket_only():
    """`names=["net_surface_energy_flux"]` matches `atmos.net_surface_energy_flux`
    with no mass-heat contributions."""
    from fme.core.atmosphere_data import AtmosphereData

    input_data, gen_data, forcing_data, _, sst = _surface_flux_test_setup()
    flux = _compute_ocean_net_surface_energy_flux(
        forcing_data, sst, names=["net_surface_energy_flux"]
    )
    expected = AtmosphereData(forcing_data).net_surface_energy_flux
    torch.testing.assert_close(flux, expected)


def test_surface_energy_flux_correction_bucket_without_frozen_precip():
    """`names=["net_surface_energy_flux_without_frozen_precip"]` matches the
    corresponding AtmosphereData property."""
    from fme.core.atmosphere_data import AtmosphereData

    input_data, gen_data, forcing_data, _, sst = _surface_flux_test_setup()
    flux = _compute_ocean_net_surface_energy_flux(
        forcing_data,
        sst,
        names=["net_surface_energy_flux_without_frozen_precip"],
    )
    expected = AtmosphereData(
        forcing_data
    ).net_surface_energy_flux_without_frozen_precip
    torch.testing.assert_close(flux, expected)


def test_surface_energy_flux_correction_subset_latent_only():
    """`names=["latent_heat_flux"]` includes both turbulent and mass-heat terms."""
    input_data, gen_data, forcing_data, _, sst = _surface_flux_test_setup()
    flux = _compute_ocean_net_surface_energy_flux(
        forcing_data, sst, names=["latent_heat_flux"]
    )
    lhf = forcing_data["LHTFLsfc"]
    expected = -lhf - SPECIFIC_HEAT_OF_SEA_WATER_CM4 * (
        lhf / LATENT_HEAT_OF_VAPORIZATION
    ) * (sst - FREEZING_TEMPERATURE_KELVIN)
    torch.testing.assert_close(flux, expected)


def test_surface_energy_flux_correction_subset_frozen_precip_only():
    """`names=["frozen_precipitation_rate"]` includes both the latent-of-fusion
    and mass-heat terms."""
    input_data, gen_data, forcing_data, _, sst = _surface_flux_test_setup()
    flux = _compute_ocean_net_surface_energy_flux(
        forcing_data, sst, names=["frozen_precipitation_rate"]
    )
    fp = forcing_data["total_frozen_precipitation_rate"]
    expected = -LATENT_HEAT_OF_FREEZING * fp + SPECIFIC_HEAT_OF_SEA_WATER_CM4 * fp * (
        sst - FREEZING_TEMPERATURE_KELVIN
    )
    torch.testing.assert_close(flux, expected)


def test_surface_energy_flux_correction_missing_forcing_variable_raises():
    """If a name is requested but the underlying atmosphere variable is absent
    from the forcing data, raise a RuntimeError that names the missing
    variable, the atomic term, and the requested names list."""
    _, _, forcing_data, _, sst = _surface_flux_test_setup()
    forcing_data_missing = {k: v for k, v in forcing_data.items() if k != "DSWRFsfc"}
    with pytest.raises(
        RuntimeError, match="missing variable 'sfc_down_sw_radiative_flux'"
    ):
        _compute_ocean_net_surface_energy_flux(
            forcing_data_missing,
            sst,
            names=["sfc_down_sw_radiative_flux"],
        )


def test_surface_energy_flux_correction_unknown_name_raises():
    """Bogus name is rejected by __post_init__ with a 'not recognized' message."""
    with pytest.raises(ValueError, match="not recognized atmospheric surface"):
        SurfaceEnergyFluxCorrectionConfig(
            method="prescribed", names=["not_a_real_flux_name"]
        )


def test_surface_energy_flux_correction_recognized_but_unsupported_raises():
    """A recognized atmospheric surface variable that is not a flux term is
    rejected with a 'not yet supported' message."""
    with pytest.raises(ValueError, match="not yet supported as flux terms"):
        SurfaceEnergyFluxCorrectionConfig(
            method="prescribed", names=["surface_pressure"]
        )


@pytest.mark.parametrize(
    "hfds_type",
    [
        pytest.param("input", id="hfds_in_input"),
        pytest.param("gen", id="hfds_in_gen"),
        pytest.param("total_area", id="hfds_total_area_in_gen"),
    ],
)
def test_ocean_heat_content_correction(hfds_type):
    config = OceanCorrectorConfig(
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method="scaled_temperature",
            constant_unaccounted_heating=0.1,
        )
    )
    timestep = datetime.timedelta(seconds=5 * 24 * 3600)
    nsamples, nlat, nlon, nlevels = 4, 3, 3, 2
    mask = torch.ones(nsamples, nlat, nlon, nlevels)
    mask[:, 0, 0, 0] = 0.0
    mask[:, 0, 0, 1] = 0.0
    mask[:, 0, 1, 1] = 0.0
    masks = {
        "mask_0": mask[:, :, :, 0],
        "mask_1": mask[:, :, :, 1],
        "mask_2d": mask[:, :, :, 0],
    }
    mask_provider = MaskProvider(masks)
    ops = LatLonOperations(torch.ones(size=[3, 3]), mask_provider)

    idepth = torch.tensor([2.5, 10, 20])
    depth_coordinate = DepthCoordinate(idepth, mask)

    sea_surface_fraction = mask[:, :, :, 0]

    input_data_dict = {
        "thetao_0": torch.ones(nsamples, nlat, nlon),
        "thetao_1": torch.ones(nsamples, nlat, nlon),
        "sst": torch.ones(nsamples, nlat, nlon) + 273.15,
    }
    gen_data_dict = {
        "thetao_0": torch.ones(nsamples, nlat, nlon) * 2,
        "thetao_1": torch.ones(nsamples, nlat, nlon) * 2,
        "sst": torch.ones(nsamples, nlat, nlon) * 2 + 273.15,
    }
    if hfds_type == "gen":
        gen_data_dict["hfds"] = torch.ones(nsamples, nlat, nlon)
    elif hfds_type == "total_area":
        # hfds_total_area is already weighted by sea_surface_fraction also
        # include hfds with a different value to verify hfds_total_area takes
        # priority
        gen_data_dict["hfds"] = (
            torch.ones(nsamples, nlat, nlon) * 100
        )  # should be ignored
        gen_data_dict["hfds_total_area"] = (
            torch.ones(nsamples, nlat, nlon) * sea_surface_fraction
        )
    else:
        input_data_dict["hfds"] = torch.ones(nsamples, nlat, nlon)
    forcing_data_dict = {
        "hfgeou": torch.ones(nsamples, nlat, nlon),
        "sea_surface_fraction": sea_surface_fraction,
    }
    input_data = OceanData(input_data_dict, depth_coordinate)
    gen_data = OceanData(gen_data_dict, depth_coordinate)
    corrector = OceanCorrector(config, ops, depth_coordinate, timestep)
    gen_data_corrected_dict = corrector(
        input_data_dict, gen_data_dict, forcing_data_dict
    )

    input_ohc = input_data.ocean_heat_content.nanmean(dim=(-1, -2), keepdim=True)
    gen_ohc = gen_data.ocean_heat_content.nanmean(dim=(-1, -2), keepdim=True)
    torch.testing.assert_close(
        gen_ohc,
        input_ohc * 2,
        equal_nan=True,
    )
    ohc_change = (
        2.1 * timestep.total_seconds()
    )  # 2.1 because of hfds + hfgeou + unaccounted heating
    corrector_ratio = (input_ohc + ohc_change) / gen_ohc
    expected_gen_data_dict = {
        key: value * corrector_ratio if key.startswith("thetao") else value
        for key, value in gen_data_dict.items()
    }
    expected_gen_data_dict["sst"] = (
        gen_data_dict["sst"] - 273.15
    ) * corrector_ratio + 273.15

    torch.testing.assert_close(
        gen_data_corrected_dict["sst"],
        expected_gen_data_dict["sst"],
    )

    expected_gen_data = OceanData(expected_gen_data_dict, depth_coordinate)
    gen_data_corrected = OceanData(gen_data_corrected_dict, depth_coordinate)
    torch.testing.assert_close(
        expected_gen_data.ocean_heat_content,
        gen_data_corrected.ocean_heat_content,
        equal_nan=True,
    )
