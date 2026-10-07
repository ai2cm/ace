import dataclasses
import datetime
import pathlib

import pytest
import torch
import xarray as xr

from fme import get_device
from fme.core.coordinates import (
    DepthCoordinate,
    LatLonCoordinates,
    NullVerticalCoordinate,
)
from fme.core.corrector.ocean import (
    OceanCorrectorConfig,
    OceanHeatContentBudgetConfig,
    RunoffHeatFluxConfig,
    SeaIceFractionConfig,
    SurfaceEnergyFluxCorrectionConfig,
    UnderIceHeatFluxConfig,
    _compute_ocean_net_surface_energy_flux,
)
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import LatLonOperations
from fme.core.ocean_data import OceanData
from fme.core.registry.corrector import CorrectorSelector
from fme.core.spatial_mask_provider import SpatialMaskProvider
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
    corrector = config._build(ops, _VERTICAL_COORD, timestep)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    corrected_gen = corrector(input_data, gen_data, {}, None).corrected
    for name in ["so_0", "so_1"]:
        x = corrected_gen[name].clone()
        x[_LAT, _LON] = 0.0
        assert torch.all(x >= 0.0)


def test_sea_ice_fraction_keep_gradient_passes_gradient_through_clamp():
    config = SeaIceFractionConfig(
        sea_ice_fraction_name="sea_ice_fraction",
        land_fraction_name="land_fraction",
        remove_negative_ocean_fraction=False,
    )
    input_data = {"land_fraction": torch.zeros(IMG_SHAPE, device=DEVICE)}
    # values both below 0 and above 1 so the clamp saturates at both ends
    raw = torch.tensor([-0.5, 0.3, 1.5], device=DEVICE)

    sif_plain = raw.clone().requires_grad_(True)
    config({"sea_ice_fraction": sif_plain}, input_data)[
        "sea_ice_fraction"
    ].sum().backward()
    # plain clamp: zero gradient where saturated, one in the interior
    torch.testing.assert_close(
        sif_plain.grad, torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    )

    sif_ste = raw.clone().requires_grad_(True)
    out = config({"sea_ice_fraction": sif_ste}, input_data, keep_gradient=True)
    # forward value is still clamped to [0, 1]
    torch.testing.assert_close(
        out["sea_ice_fraction"], torch.tensor([0.0, 0.3, 1.0], device=DEVICE)
    )
    out["sea_ice_fraction"].sum().backward()
    torch.testing.assert_close(sif_ste.grad, torch.ones_like(raw))


def test_ocean_corrector_keep_gradient_through_clamps_forward_unchanged():
    # The straight-through flag must not change forward values; only gradients.
    torch.manual_seed(0)
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    sif = SeaIceFractionConfig(
        sea_ice_fraction_name="sea_ice_fraction",
        land_fraction_name="land_fraction",
    )
    input_data = {
        "land_fraction": torch.ones(IMG_SHAPE, device=DEVICE) * 0.3,
    }
    gen_data = {
        "so_0": torch.randn(IMG_SHAPE, device=DEVICE),
        "sea_ice_fraction": torch.randn(IMG_SHAPE, device=DEVICE),
    }
    baseline = (
        OceanCorrectorConfig(
            force_positive_names=["so_0"], sea_ice_fraction_correction=sif
        )
        ._build(ops, None, timestep)(input_data, gen_data, {}, None)
        .corrected
    )
    ste = (
        OceanCorrectorConfig(
            force_positive_names=["so_0"],
            sea_ice_fraction_correction=sif,
            keep_gradient_through_clamps=True,
        )
        ._build(ops, None, timestep)(input_data, gen_data, {}, None)
        .corrected
    )
    for name in baseline:
        torch.testing.assert_close(baseline[name], ste[name])


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
    corrector = config._build(ops, None, timestep)
    violation = (input_data["land_fraction"] + gen_data["sea_ice_fraction"]) > 1.0
    assert violation.any()
    negative_sea_ice_fraction = gen_data["sea_ice_fraction"] < 0.0
    assert negative_sea_ice_fraction.any()

    next_step_input_data: TensorMapping = {}
    gen_data_corrected = corrector(
        input_data, gen_data, next_step_input_data, None
    ).corrected
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
    corrector = config._build(ops, None, timestep)
    violation = (input_data["land_fraction"] + gen_data["sea_ice_fraction"]) > 1.0
    assert violation.any()
    negative_sea_ice_fraction = gen_data["sea_ice_fraction"] < 0.0
    assert negative_sea_ice_fraction.any()

    next_step_input_data: TensorMapping = {}
    gen_data_corrected = corrector(
        input_data, gen_data, next_step_input_data, None
    ).corrected
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
    corrector = config._build(ops, None, timestep)
    gen_data_corrected = corrector(input_data, gen_data, {}, None).corrected
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
    corrector = config._build(ops, None, timestep)
    gen_data_corrected = corrector(input_data, gen_data, {}, None).corrected
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
    corrector = config._build(ops, None, timestep)

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

    corrected = corrector(input_data, gen_data, forcing_data, None).corrected
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
    corrector = config._build(ops, None, timestep)

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

    corrected = corrector(input_data, gen_data, forcing_data, None).corrected
    torch.testing.assert_close(corrected["hfds"], expected_hfds)
    # on land (ocean_fraction=0), hfds equals gen_hfds
    torch.testing.assert_close(corrected["hfds"][-1, :], gen_hfds[-1, :])
    # in open ocean (no ice, no land), hfds equals net_flux
    open_ocean_row = 1
    torch.testing.assert_close(
        corrected["hfds"][open_ocean_row, :], net_flux[open_ocean_row, :]
    )


@pytest.mark.parametrize(
    "hfds_name",
    [
        pytest.param("hfds", id="hfds_in_gen"),
        pytest.param("hfds_total_area", id="hfds_total_area_in_gen"),
    ],
)
def test_surface_energy_flux_correction_prescribed_open_ocean(hfds_name):
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed_open_ocean"
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = config._build(ops, None, timestep)

    sst = torch.full(IMG_SHAPE, 300.0, device=DEVICE)
    gen_hfds = torch.full(IMG_SHAPE, 5.0, device=DEVICE)
    sea_ice_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    sea_ice_fraction[0, :] = 0.3
    land_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    land_fraction[-1, :] = 1.0

    gen_data = {
        "sst": sst,
        hfds_name: gen_hfds,
        "sea_ice_fraction": sea_ice_fraction,
    }
    forcing_data = {
        "land_fraction": land_fraction,
        **_make_atmos_forcing_data(IMG_SHAPE),
    }
    input_data = {**forcing_data, **gen_data}

    ocean_fraction = 1 - land_fraction - sea_ice_fraction
    net_flux = _compute_ocean_net_surface_energy_flux(input_data, sst)
    if hfds_name == "hfds_total_area":
        net_flux = net_flux * (1 - land_fraction)
    expected_hfds = torch.where(ocean_fraction == 1, net_flux, gen_hfds)

    corrected = corrector(input_data, gen_data, forcing_data, None).corrected
    torch.testing.assert_close(corrected[hfds_name], expected_hfds)
    # open ocean (ocean_fraction exactly 1): hfds is the prescribed net flux
    open_ocean_row = 1
    torch.testing.assert_close(
        corrected[hfds_name][open_ocean_row, :], net_flux[open_ocean_row, :]
    )
    # partial ocean under sea ice: gen_hfds passes through unweighted, unlike
    # the "prescribed" method which would blend it with the net flux
    torch.testing.assert_close(corrected[hfds_name][0, :], gen_hfds[0, :])
    # land: gen_hfds passes through
    torch.testing.assert_close(corrected[hfds_name][-1, :], gen_hfds[-1, :])


# Rows of the 5x5 grid used by the "prescribed_cell_mean" tests.
_ROW_ICE = 0  # all sea, 30% covered by sea ice
_ROW_OPEN = 1  # all sea, ice-free
_ROW_COAST = 2  # half land, ice-free
_ROW_COAST_ICE = 3  # half land, 20% of the cell under sea ice
_ROW_LAND = 4  # all land


def _make_cell_mean_case():
    """Fractions and data for a grid with open-ocean, coastal, ice-covered,
    coastal-and-icy, and all-land rows, with sea_surface_fraction supplied
    explicitly (so the 1 - land_fraction fallback is not exercised)."""
    land_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    land_fraction[_ROW_COAST, :] = 0.5
    land_fraction[_ROW_COAST_ICE, :] = 0.5
    land_fraction[_ROW_LAND, :] = 1.0
    sea_ice_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    sea_ice_fraction[_ROW_ICE, :] = 0.3
    sea_ice_fraction[_ROW_COAST_ICE, :] = 0.2
    sea_surface_fraction = 1 - land_fraction
    sst = torch.full(IMG_SHAPE, 300.0, device=DEVICE)
    gen_hfds = torch.full(IMG_SHAPE, 5.0, device=DEVICE)
    gen_data = {
        "sst": sst,
        "hfds_total_area": gen_hfds,
        "sea_ice_fraction": sea_ice_fraction,
    }
    forcing_data = {
        "land_fraction": land_fraction,
        "sea_surface_fraction": sea_surface_fraction,
        **_make_atmos_forcing_data(IMG_SHAPE),
    }
    input_data = {**forcing_data, **gen_data}
    net_flux = _compute_ocean_net_surface_energy_flux(input_data, sst)
    return input_data, gen_data, forcing_data, net_flux


def _write_runoff_file(path, values: torch.Tensor) -> str:
    """Write a (lat, lon) runoff heat map to netCDF, returning its path."""
    ds = xr.Dataset({"hfrunoffds": (("lat", "lon"), values.cpu().numpy())})
    filename = str(path / "time-mean.nc")
    ds.to_netcdf(filename)
    return filename


def _build_cell_mean_corrector(runoff_heat_flux: RunoffHeatFluxConfig | None):
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed_cell_mean",
            runoff_heat_flux=runoff_heat_flux,
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    return config._build(ops, None, timestep)


def test_surface_energy_flux_correction_prescribed_cell_mean(tmp_path):
    input_data, gen_data, forcing_data, net_flux = _make_cell_mean_case()
    gen_hfds = gen_data["hfds_total_area"]
    sea_ice_fraction = gen_data["sea_ice_fraction"]
    sea_surface_fraction = forcing_data["sea_surface_fraction"]
    # runoff heat per unit sea area: in the coastal rows only, NaN on land
    # as the ocean model's diagnostic has it
    runoff = torch.zeros(IMG_SHAPE)
    runoff[_ROW_COAST, :] = 10.0
    runoff[_ROW_COAST_ICE, :] = 10.0
    runoff[_ROW_LAND, :] = float("nan")
    runoff_path = _write_runoff_file(tmp_path, runoff)
    corrector = _build_cell_mean_corrector(RunoffHeatFluxConfig(path=runoff_path))

    corrected = corrector(input_data, gen_data, forcing_data, None).corrected
    out = corrected["hfds_total_area"]

    runoff_cell = torch.nan_to_num(runoff.to(DEVICE)) * sea_surface_fraction
    expected = (net_flux * (1 - sea_ice_fraction) + runoff_cell) * (
        sea_surface_fraction > 0
    ) + gen_hfds * sea_ice_fraction
    torch.testing.assert_close(out, expected)
    # open ocean: the cell mean exactly, no network share
    torch.testing.assert_close(out[_ROW_OPEN, :], net_flux[_ROW_OPEN, :])
    # coast: the unscaled cell mean plus runoff heat, no sea-fraction scaling
    # of the atmosphere's flux and no network share
    torch.testing.assert_close(out[_ROW_COAST, :], net_flux[_ROW_COAST, :] + 10.0 * 0.5)
    # under ice the network keeps the ice-covered share only
    torch.testing.assert_close(
        out[_ROW_ICE, :], 0.7 * net_flux[_ROW_ICE, :] + 0.3 * gen_hfds[_ROW_ICE, :]
    )
    torch.testing.assert_close(
        out[_ROW_COAST_ICE, :],
        0.8 * net_flux[_ROW_COAST_ICE, :]
        + 10.0 * 0.5
        + 0.2 * gen_hfds[_ROW_COAST_ICE, :],
    )
    # all land: no sea, no flux (the NaN runoff cell reads as zero)
    torch.testing.assert_close(out[_ROW_LAND, :], torch.zeros_like(out[_ROW_LAND, :]))


def test_surface_energy_flux_correction_prescribed_cell_mean_no_runoff():
    input_data, gen_data, forcing_data, net_flux = _make_cell_mean_case()
    corrector = _build_cell_mean_corrector(None)
    out = corrector(input_data, gen_data, forcing_data, None).corrected[
        "hfds_total_area"
    ]
    torch.testing.assert_close(out[_ROW_COAST, :], net_flux[_ROW_COAST, :])
    torch.testing.assert_close(out[_ROW_OPEN, :], net_flux[_ROW_OPEN, :])
    torch.testing.assert_close(out[_ROW_LAND, :], torch.zeros_like(out[_ROW_LAND, :]))


def test_prescribed_cell_mean_runoff_per_unit_cell_area(tmp_path):
    input_data, gen_data, forcing_data, net_flux = _make_cell_mean_case()
    runoff = torch.zeros(IMG_SHAPE)
    runoff[_ROW_COAST, :] = 4.0
    runoff_path = _write_runoff_file(tmp_path, runoff)
    corrector = _build_cell_mean_corrector(
        RunoffHeatFluxConfig(path=runoff_path, per_unit_sea_area=False)
    )
    out = corrector(input_data, gen_data, forcing_data, None).corrected[
        "hfds_total_area"
    ]
    # used as is: not multiplied by the 0.5 sea surface fraction
    torch.testing.assert_close(out[_ROW_COAST, :], net_flux[_ROW_COAST, :] + 4.0)


def test_prescribed_cell_mean_requires_hfds_total_area():
    input_data, gen_data, forcing_data, _ = _make_cell_mean_case()
    gen_data = dict(gen_data)
    gen_data["hfds"] = gen_data.pop("hfds_total_area")
    corrector = _build_cell_mean_corrector(None)
    with pytest.raises(NotImplementedError, match="hfds_total_area"):
        corrector(input_data, gen_data, forcing_data, None)


def test_prescribed_cell_mean_runoff_map_shape_mismatch(tmp_path):
    input_data, gen_data, forcing_data, _ = _make_cell_mean_case()
    runoff_path = _write_runoff_file(tmp_path, torch.zeros((3, 3)))
    corrector = _build_cell_mean_corrector(RunoffHeatFluxConfig(path=runoff_path))
    with pytest.raises(ValueError, match="shape"):
        corrector(input_data, gen_data, forcing_data, None)


_CELL_AREA = 1.0e10  # m**2
_TIMESTEP = datetime.timedelta(days=5)
_RHO_L = 905.0 * 334000.0


def _build_under_ice_corrector(under_ice: UnderIceHeatFluxConfig):
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed_cell_mean", under_ice=under_ice
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    area = torch.full(IMG_SHAPE, _CELL_AREA, device=DEVICE)
    return config._build(ops, None, _TIMESTEP, cell_area_m2=area)


def _with_ice_volume(input_data, gen_data, v_in, v_out):
    input_data = {**input_data, "sea_ice_volume": v_in}
    gen_data = {**gen_data, "sea_ice_volume": v_out}
    return input_data, gen_data


def test_prescribed_cell_mean_under_ice_storage():
    input_data, gen_data, forcing_data, net_flux = _make_cell_mean_case()
    has_sea = forcing_data["sea_surface_fraction"] > 0
    v_in = torch.full(IMG_SHAPE, 1.0e9, device=DEVICE)  # m**3 per cell
    v_out = v_in.clone()
    v_out[_ROW_ICE, :] = 2.0e9  # growth
    v_out[_ROW_COAST_ICE, :] = 0.5e9  # melt
    input_data, gen_data = _with_ice_volume(input_data, gen_data, v_in, v_out)
    corrector = _build_under_ice_corrector(UnderIceHeatFluxConfig())
    out = corrector(input_data, gen_data, forcing_data, None).corrected[
        "hfds_total_area"
    ]
    release = _RHO_L * (v_out - v_in) / (_CELL_AREA * _TIMESTEP.total_seconds())
    torch.testing.assert_close(out, (net_flux + release) * has_sea)
    # growth releases latent heat: the ocean loses less than the atmosphere took
    assert torch.all(out[_ROW_ICE, :] > net_flux[_ROW_ICE, :])
    # melt takes heat before it reaches the water
    assert torch.all(out[_ROW_COAST_ICE, :] < net_flux[_ROW_COAST_ICE, :])
    # no volume change: the atmosphere's flux, with no network share anywhere
    torch.testing.assert_close(out[_ROW_OPEN, :], net_flux[_ROW_OPEN, :])
    gen_data = {**gen_data, "hfds_total_area": gen_data["hfds_total_area"] + 100.0}
    out2 = corrector(input_data, gen_data, forcing_data, None).corrected[
        "hfds_total_area"
    ]
    torch.testing.assert_close(out2, out)
    # all land: nothing
    torch.testing.assert_close(out[_ROW_LAND, :], torch.zeros_like(out[_ROW_LAND, :]))


@pytest.mark.parametrize("gradient", [True, False])
def test_under_ice_gradient_through_ice_volume(gradient):
    input_data, gen_data, forcing_data, _ = _make_cell_mean_case()
    v_in = torch.full(IMG_SHAPE, 1.0e9, device=DEVICE)
    v_out = torch.full(IMG_SHAPE, 1.5e9, device=DEVICE, requires_grad=True)
    input_data, gen_data = _with_ice_volume(input_data, gen_data, v_in, v_out)
    corrector = _build_under_ice_corrector(
        UnderIceHeatFluxConfig(gradient_through_ice_volume=gradient)
    )
    out = corrector(input_data, gen_data, forcing_data, None).corrected[
        "hfds_total_area"
    ]
    if gradient:
        out.sum().backward()
        assert v_out.grad is not None and torch.any(v_out.grad != 0)
    else:
        assert not out.requires_grad


def test_under_ice_requires_ice_volume():
    input_data, gen_data, forcing_data, _ = _make_cell_mean_case()
    corrector = _build_under_ice_corrector(UnderIceHeatFluxConfig())
    with pytest.raises(KeyError, match="sea_ice_volume"):
        corrector(input_data, gen_data, forcing_data, None)


def test_under_ice_requires_cell_area():
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed_cell_mean", under_ice=UnderIceHeatFluxConfig()
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    with pytest.raises(ValueError, match="cell areas"):
        config._build(ops, None, _TIMESTEP)


def test_under_ice_cell_area_from_dataset_info():
    # the production path: cell areas come from the dataset's lat-lon grid
    lat = torch.linspace(-80.0, 80.0, IMG_SHAPE[0])
    lon = torch.linspace(0.0, 288.0, IMG_SHAPE[1])
    coords = LatLonCoordinates(lat=lat, lon=lon)
    dataset_info = DatasetInfo(
        horizontal_coordinates=coords,
        vertical_coordinate=NullVerticalCoordinate(),
        timestep=_TIMESTEP,
    )
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed_cell_mean", under_ice=UnderIceHeatFluxConfig()
        ),
    )
    corrector = config._get_corrector(dataset_info)
    input_data, gen_data, forcing_data, net_flux = _make_cell_mean_case()
    v_in = torch.zeros(IMG_SHAPE, device=DEVICE)
    v_out = torch.full(IMG_SHAPE, 1.0e9, device=DEVICE)
    input_data, gen_data = _with_ice_volume(input_data, gen_data, v_in, v_out)
    out = corrector(input_data, gen_data, forcing_data, None).corrected[
        "hfds_total_area"
    ]
    area = coords.area_weights_m2.to(DEVICE)
    release = _RHO_L * 1.0e9 / (area * _TIMESTEP.total_seconds())
    has_sea = forcing_data["sea_surface_fraction"] > 0
    torch.testing.assert_close(out, (net_flux + release) * has_sea)


def test_under_ice_only_with_prescribed_cell_mean():
    with pytest.raises(ValueError, match="prescribed_cell_mean"):
        SurfaceEnergyFluxCorrectionConfig(
            method="prescribed", under_ice=UnderIceHeatFluxConfig()
        )


def test_under_ice_config_round_trip():
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed_cell_mean",
            under_ice=UnderIceHeatFluxConfig(gradient_through_ice_volume=False),
        ),
    )
    state = dataclasses.asdict(config)
    assert OceanCorrectorConfig.from_state(state) == config


def test_runoff_heat_flux_only_with_prescribed_cell_mean():
    with pytest.raises(ValueError, match="prescribed_cell_mean"):
        SurfaceEnergyFluxCorrectionConfig(
            method="prescribed",
            runoff_heat_flux=RunoffHeatFluxConfig(path="unused.nc"),
        )


def test_ocean_corrector_config_round_trip_with_runoff_heat_flux():
    # the serialized form in a checkpoint must load back, with the new nested
    # config intact and absent fields defaulting
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed_cell_mean",
            runoff_heat_flux=RunoffHeatFluxConfig(path="gs://bucket/time-mean.nc"),
        ),
    )
    state = dataclasses.asdict(config)
    assert OceanCorrectorConfig.from_state(state) == config
    # a checkpoint written before runoff_heat_flux existed
    state["surface_energy_flux_correction"] = {"method": "prescribed"}
    loaded = OceanCorrectorConfig.from_state(state)
    assert loaded.surface_energy_flux_correction is not None
    assert loaded.surface_energy_flux_correction.runoff_heat_flux is None


def _coastal_runoff_map() -> torch.Tensor:
    """Runoff heat per unit sea area in the coastal rows, NaN over land."""
    runoff = torch.zeros(IMG_SHAPE)
    runoff[_ROW_COAST, :] = 10.0
    runoff[_ROW_COAST_ICE, :] = 10.0
    runoff[_ROW_LAND, :] = float("nan")
    return runoff


def _cell_mean_dataset_info(img_shape=IMG_SHAPE) -> DatasetInfo:
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.linspace(-80.0, 80.0, img_shape[0]),
            lon=torch.linspace(0.0, 288.0, img_shape[1]),
        ),
        vertical_coordinate=NullVerticalCoordinate(),
        timestep=_TIMESTEP,
    )


def test_runoff_heat_flux_load_embeds_map(tmp_path):
    runoff = _coastal_runoff_map()
    runoff_path = _write_runoff_file(tmp_path, runoff)
    config = RunoffHeatFluxConfig(path=runoff_path)
    config.load()
    assert config.path is None
    assert config.values is not None
    torch.testing.assert_close(
        torch.tensor(config.values), torch.nan_to_num(runoff), rtol=0, atol=0
    )
    config.load()  # idempotent once loaded
    assert config.path is None


def test_runoff_heat_flux_needs_path_or_values():
    with pytest.raises(ValueError, match="exactly one"):
        RunoffHeatFluxConfig()
    with pytest.raises(ValueError, match="exactly one"):
        RunoffHeatFluxConfig(path="unused.nc", values=[[0.0]])


def test_runoff_heat_map_shape_validated_at_build(tmp_path):
    runoff_path = _write_runoff_file(tmp_path, torch.zeros((3, 3)))
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed_cell_mean",
            runoff_heat_flux=RunoffHeatFluxConfig(path=runoff_path),
        ),
    )
    with pytest.raises(ValueError, match="shape"):
        config._get_corrector(_cell_mean_dataset_info())


def test_loaded_runoff_heat_flux_survives_missing_file(tmp_path):
    """The serialized corrector config holds the runoff map after load, so a
    corrector built from it reproduces the correction once the file is gone.
    """
    runoff_path = _write_runoff_file(tmp_path, _coastal_runoff_map())
    selector = CorrectorSelector(
        "ocean_corrector",
        dataclasses.asdict(
            OceanCorrectorConfig(
                surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
                    method="prescribed_cell_mean",
                    runoff_heat_flux=RunoffHeatFluxConfig(path=runoff_path),
                ),
            )
        ),
    )
    dataset_info = _cell_mean_dataset_info()
    input_data, gen_data, forcing_data, _ = _make_cell_mean_case()
    expected = selector.get_corrector(dataset_info)(
        input_data, gen_data, forcing_data, None
    ).corrected["hfds_total_area"]
    selector.load()
    state = dataclasses.asdict(selector)
    pathlib.Path(runoff_path).unlink()
    reloaded = CorrectorSelector.from_state(state)
    out = reloaded.get_corrector(dataset_info)(
        input_data, gen_data, forcing_data, None
    ).corrected["hfds_total_area"]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    # the runoff term is in the result: coast is not the bare atmosphere flux
    no_runoff = _build_cell_mean_corrector(None)(
        input_data, gen_data, forcing_data, None
    ).corrected["hfds_total_area"]
    assert torch.all(out[_ROW_COAST, :] > no_runoff[_ROW_COAST, :])


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
    spatial_mask_provider = SpatialMaskProvider(masks)
    ops = LatLonOperations(torch.ones(size=[3, 3]), spatial_mask_provider)

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
    corrector = config._build(ops, depth_coordinate, timestep)
    result = corrector(input_data_dict, gen_data_dict, forcing_data_dict, None)
    gen_data_corrected_dict = result.corrected

    # the OHC correction writes every potential-temperature level and the SST;
    # the heat-flux fields are read but not written, so they stay out of the set
    assert set(result.modified_names) == {"thetao_0", "thetao_1", "sst"}
    for name, delta in result.diagnostics.delta.items():
        torch.testing.assert_close(
            delta, result.corrected[name] - gen_data_dict[name], equal_nan=True
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


def test_ocean_corrector_config_fields_are_known():
    # Staleness guard: if a new corrector option is added to
    # OceanCorrectorConfig this fails, flagging that the corrector delta/
    # modified-return tests need to exercise it.
    expected = {
        "force_positive_names",
        "sea_ice_fraction_correction",
        "surface_energy_flux_correction",
        "ocean_heat_content_correction",
        "keep_gradient_through_clamps",
        "corrector_disabled_epochs",  # inherited epoch-scheduling field
    }
    actual = {f.name for f in dataclasses.fields(OceanCorrectorConfig)}
    assert actual == expected, (
        "OceanCorrectorConfig fields changed; update the corrector delta tests "
        f"to cover the new option(s): {actual ^ expected}"
    )


def test_ocean_corrector_delta_matches_modified_returns():
    torch.manual_seed(0)
    config = OceanCorrectorConfig(
        force_positive_names=["so_0"],
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            zero_where_ice_free_names=["HI", "HS"],
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = config._build(ops, None, timestep)
    input_data = {"land_fraction": torch.rand(IMG_SHAPE, device=DEVICE)}
    gen_data = {
        "so_0": torch.randn(IMG_SHAPE, device=DEVICE),
        "so_1": torch.randn(IMG_SHAPE, device=DEVICE),  # uncorrected field
        "sea_ice_fraction": torch.rand(IMG_SHAPE, device=DEVICE),
        "HI": torch.rand(IMG_SHAPE, device=DEVICE) * 10,
        "HS": torch.rand(IMG_SHAPE, device=DEVICE) * 5,
    }
    result = corrector(input_data, gen_data, {}, None)
    # delta keys are exactly the corrector's modified names
    assert set(result.diagnostics.delta) == set(result.modified_names)
    for name, delta in result.diagnostics.delta.items():
        torch.testing.assert_close(delta, result.corrected[name] - gen_data[name])
    assert set(result.modified_names) == {"so_0", "sea_ice_fraction", "HI", "HS"}
    # the uncorrected field passes through unchanged and is absent from the set
    assert "so_1" not in result.modified_names
    torch.testing.assert_close(result.corrected["so_1"], gen_data["so_1"])


def test_ocean_corrector_empty_delta_when_nothing_modified():
    # A corrector with no field-modifying option emits an empty delta and an
    # unchanged copy of gen_data.
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = OceanCorrectorConfig()._build(ops, None, timestep)
    gen_data = {"so_0": torch.randn(IMG_SHAPE, device=DEVICE)}
    result = corrector({}, gen_data, {}, None)
    assert dict(result.diagnostics.delta) == {}
    assert set(result.modified_names) == set()
    torch.testing.assert_close(result.corrected["so_0"], gen_data["so_0"])


def test_ocean_corrector_is_per_member_under_ensemble_folding():
    """Ensemble training folds the ensemble members into the batch dimension, so
    the corrector sees several members at once. Every correction must act
    per-member: one that coupled across the batch dim (e.g. a global mean taken
    over samples too) would tie the members together and silently collapse the
    ensemble spread the proper scoring rule is meant to reward.
    """
    torch.manual_seed(0)
    n_members, nlat, nlon, nlevels = 2, 3, 3, 2
    config = OceanCorrectorConfig(
        force_positive_names=["so_0", "so_1"],
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            zero_where_ice_free_names=["sea_ice_thickness"],
        ),
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method="scaled_temperature",
            constant_unaccounted_heating=0.1,
        ),
    )
    timestep = datetime.timedelta(seconds=5 * 24 * 3600)
    mask = torch.ones(nlat, nlon, nlevels)
    mask[0, 0, :] = 0.0
    masks = {
        "mask_0": mask[:, :, 0],
        "mask_1": mask[:, :, 1],
        "mask_2d": mask[:, :, 0],
    }
    # non-uniform in latitude only, as the area weights require
    area = torch.tensor([0.5, 1.0, 1.5]).unsqueeze(-1).expand(nlat, nlon)
    ops = LatLonOperations(area, SpatialMaskProvider(masks))
    depth_coordinate = DepthCoordinate(torch.tensor([2.5, 10.0, 20.0]), mask)
    corrector = config._build(ops, depth_coordinate, timestep)

    def randoms(shape):
        return torch.randn(shape)

    input_data = {
        "thetao_0": randoms((n_members, nlat, nlon)) + 2.0,
        "thetao_1": randoms((n_members, nlat, nlon)) + 2.0,
        "sst": randoms((n_members, nlat, nlon)) + 275.0,
        "land_fraction": torch.zeros(n_members, nlat, nlon),
    }
    # members differ in every generated field, as they would under different
    # noise draws
    gen_data = {
        "thetao_0": randoms((n_members, nlat, nlon)) + 2.0,
        "thetao_1": randoms((n_members, nlat, nlon)) + 2.0,
        "sst": randoms((n_members, nlat, nlon)) + 275.0,
        "so_0": randoms((n_members, nlat, nlon)),
        "so_1": randoms((n_members, nlat, nlon)),
        # spans the clamp range at both ends so the sea-ice rebalance engages
        "sea_ice_fraction": randoms((n_members, nlat, nlon)) * 0.8 + 0.5,
        "sea_ice_thickness": randoms((n_members, nlat, nlon)),
        "hfds": randoms((n_members, nlat, nlon)),
    }
    forcing_data = {
        "hfgeou": randoms((n_members, nlat, nlon)),
        "sea_surface_fraction": mask[:, :, 0].expand(n_members, nlat, nlon),
    }

    folded = corrector(input_data, gen_data, forcing_data, None).corrected
    assert set(folded) >= {"so_0", "sea_ice_fraction", "thetao_0", "sst"}

    for member in range(n_members):

        def slice_member(data, member=member):
            return {name: value[member : member + 1] for name, value in data.items()}

        alone = corrector(
            slice_member(input_data),
            slice_member(gen_data),
            slice_member(forcing_data),
            None,
        ).corrected
        for name, value in alone.items():
            torch.testing.assert_close(
                folded[name][member : member + 1],
                value,
                msg=lambda m, name=name, member=member: (
                    f"{name} for member {member} depends on the other members: {m}"
                ),
            )

    # and the members really are distinct after correction, so the comparison
    # above is not vacuous
    for name in folded:
        assert not torch.allclose(folded[name][0], folded[name][1])
