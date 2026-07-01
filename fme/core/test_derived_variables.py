import datetime

import pytest
import torch

from fme.core.atmosphere_data import AtmosphereData
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.derived_variables import (
    _DERIVED_VARIABLE_REGISTRY,
    _compute_derived_variable,
    compute_derived_quantities,
    get_derived_variable_metadata,
)
from fme.core.typing_ import TensorDict, TensorMapping

TIMESTEP = datetime.timedelta(hours=6)


def test_compute_derived_variable():
    fake_data = {"PRESsfc": torch.tensor([1.0]), "PRATEsfc": torch.tensor([2.0])}
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.tensor([0.0, 0.0]), bk=torch.tensor([0.0, 1.0])
    )

    def _derived_variable_func(data: AtmosphereData, *_) -> torch.Tensor:
        return data.surface_pressure + data.precipitation_rate

    output_data = _compute_derived_variable(
        fake_data, vertical_coordinate, TIMESTEP, "c", _derived_variable_func
    )
    torch.testing.assert_close(output_data["c"], torch.tensor([3.0]))


def test_compute_derived_variable_keeps_existing_value():
    """A stored variable takes precedence over its derived form — the derived
    computation is skipped (not raised) so a variable can be stored on one side
    (target) and derived-when-absent on the other (prediction). See zg
    reconstruction from predicted layer thicknesses."""
    fake_data = {"PRESsfc": torch.tensor([1.0]), "PRATEsfc": torch.tensor([2.0])}
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.tensor([0.0, 0.0]), bk=torch.tensor([0.0, 1.0])
    )

    def add_surface_pressure_and_precipitation(
        data: AtmosphereData, *_
    ) -> torch.Tensor:
        return data.surface_pressure + data.precipitation_rate

    output_data = _compute_derived_variable(
        fake_data,
        vertical_coordinate,
        TIMESTEP,
        "PRATEsfc",
        add_surface_pressure_and_precipitation,
    )
    # the stored value is preserved, not overwritten by the derived (1+2=3)
    torch.testing.assert_close(output_data["PRATEsfc"], torch.tensor([2.0]))


def _zg_func(level: int):
    return _DERIVED_VARIABLE_REGISTRY[f"zg{level}"][0]


def test_zg_reconstruction_from_thickness():
    """zg<level> is reconstructed by integrating predicted layer thicknesses
    upward from the surface height."""
    data = {
        "HGTsfc": torch.full((1, 1, 2, 2), 50.0),
        "thickness_surface_1000": torch.full((1, 1, 2, 2), 150.0),
        "thickness_1000_850": torch.full((1, 1, 2, 2), 1300.0),
        "thickness_850_700": torch.full((1, 1, 2, 2), 1500.0),
    }
    # zg1000 = 50 + 150 = 200
    out = _compute_derived_variable(data, None, TIMESTEP, "zg1000", _zg_func(1000))
    torch.testing.assert_close(out["zg1000"], torch.full((1, 1, 2, 2), 200.0))
    # zg850 = 50 + 150 + 1300 = 1500
    out = _compute_derived_variable(data, None, TIMESTEP, "zg850", _zg_func(850))
    torch.testing.assert_close(out["zg850"], torch.full((1, 1, 2, 2), 1500.0))
    # zg700 = 1500 + 1500 = 3000
    out = _compute_derived_variable(data, None, TIMESTEP, "zg700", _zg_func(700))
    torch.testing.assert_close(out["zg700"], torch.full((1, 1, 2, 2), 3000.0))
    # zg500 lacks its thickness input (thickness_700_500) -> not computed
    out = _compute_derived_variable(data, None, TIMESTEP, "zg500", _zg_func(500))
    assert "zg500" not in out


def test_zg_reconstruction_skipped_when_zg_stored():
    """On the target side (real zg stored) the reconstruction is skipped and
    the stored zg is preserved, so eval compares reconstructed-gen vs
    stored-target under the same name."""
    data = {
        "HGTsfc": torch.full((1, 1, 2, 2), 50.0),
        "thickness_surface_1000": torch.full((1, 1, 2, 2), 150.0),
        "zg1000": torch.full((1, 1, 2, 2), 999.0),  # real stored target zg
    }
    out = _compute_derived_variable(data, None, TIMESTEP, "zg1000", _zg_func(1000))
    torch.testing.assert_close(out["zg1000"], torch.full((1, 1, 2, 2), 999.0))


@pytest.mark.parametrize("dataset", ["fv3", "e3sm"])
def test_compute_derived_quantities(dataset: str):
    torch.manual_seed(0)

    if dataset == "fv3":
        fake_data = {
            "PRESsfc": 10.0 + torch.rand(2, 3, 4, 8),
            "specific_total_water_0": torch.rand(2, 3, 4, 8),
            "specific_total_water_1": torch.rand(2, 3, 4, 8),
            "PRATEsfc": torch.rand(2, 3, 4, 8),
            "LHTFLsfc": torch.rand(2, 3, 4, 8),
            "tendency_of_total_water_path_due_to_advection": torch.rand(2, 3, 4, 8),
            "DSWRFtoa": torch.rand(2, 3, 4, 8),
            "USWRFtoa": torch.rand(2, 3, 4, 8),
            "ULWRFtoa": torch.rand(2, 3, 4, 8),
        }
        gen_data = fake_data.copy()
        del gen_data["DSWRFtoa"]

    if dataset == "e3sm":
        fake_data = {
            "PS": 10.0 + torch.rand(2, 3, 4, 8),
            "specific_total_water_0": torch.rand(2, 3, 4, 8),
            "specific_total_water_1": torch.rand(2, 3, 4, 8),
            "surface_precipitation_rate": torch.rand(2, 3, 4, 8),
            "LHFLX": torch.rand(2, 3, 4, 8),
            "tendency_of_total_water_path_due_to_advection": torch.rand(2, 3, 4, 8),
            "SOLIN": torch.rand(2, 3, 4, 8),
            "top_of_atmos_upward_shortwave_flux": torch.rand(2, 3, 4, 8),
            "FLUT": torch.rand(2, 3, 4, 8),
        }
        gen_data = fake_data.copy()
        del gen_data["SOLIN"]

    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.tensor([0.0, 0.5, 0.0]),
        bk=torch.tensor([0.0, 0.5, 1.0]),
    )

    def derive_func(data: TensorMapping, forcing_data: TensorMapping) -> TensorDict:
        updated = compute_derived_quantities(
            dict(data),
            vertical_coordinate=vertical_coordinate,
            timestep=TIMESTEP,
            forcing_data=dict(forcing_data),
        )
        return updated

    out_gen_data = derive_func(gen_data, fake_data)
    out_target_data = derive_func(fake_data, fake_data)
    for name in (
        "total_water_path_budget_residual",
        "total_water_path",
        "surface_pressure_due_to_dry_air",
        "surface_pressure_due_to_dry_air_absolute_tendency",
        "net_energy_flux_toa_into_atmosphere",
    ):
        assert name in out_gen_data
        assert name in out_target_data
        assert out_gen_data[name].shape == (2, 3, 4, 8)
        assert out_target_data[name].shape == (2, 3, 4, 8)


def test_metadata_registry():
    metadata = get_derived_variable_metadata()
    assert metadata["total_water_path"].units == "kg/m**2"
    assert metadata["total_water_path"].long_name == "Total water path"
