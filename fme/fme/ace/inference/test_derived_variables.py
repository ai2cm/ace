import datetime

import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.stepper import TrainOutput
from fme.core.climate_data import ClimateData
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.typing_ import TensorDict, TensorMapping

from .derived_variables import _compute_derived_variable, compute_derived_quantities

TIMESTEP = datetime.timedelta(hours=6)


def test_compute_derived_variable():
    fake_data = {"PRESsfc": torch.tensor([1.0]), "PRATEsfc": torch.tensor([2.0])}
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.tensor([0.0, 0.0]), bk=torch.tensor([0.0, 1.0])
    )

    def _derived_variable_func(data: ClimateData, *_) -> torch.Tensor:
        return data.surface_pressure + data.precipitation_rate

    output_data = _compute_derived_variable(
        fake_data, vertical_coordinate, TIMESTEP, "c", _derived_variable_func
    )
    torch.testing.assert_close(output_data["c"], torch.tensor([3.0]))


def test_compute_derived_variable_raises_value_error_when_overwriting():
    fake_data = {"PRESsfc": torch.tensor([1.0]), "PRATEsfc": torch.tensor([2.0])}
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.tensor([0.0, 0.0]), bk=torch.tensor([0.0, 1.0])
    )

    def add_surface_pressure_and_precipitation(data: ClimateData, *_) -> torch.Tensor:
        return data.surface_pressure + data.precipitation_rate

    derived_variable_func = add_surface_pressure_and_precipitation
    with pytest.raises(ValueError):
        _compute_derived_variable(
            fake_data, vertical_coordinate, TIMESTEP, "PRATEsfc", derived_variable_func
        )


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

    data = TrainOutput(
        metrics={"loss": torch.tensor(0.0)},
        gen_data=gen_data,
        target_data=fake_data,
        time=xr.DataArray(np.zeros((2, 3)), dims=["sample", "time"]),
        normalize=lambda x: x,
        derive_func=derive_func,
    )
    out_data = data.compute_derived_variables()
    for name in (
        "total_water_path_budget_residual",
        "total_water_path",
        "surface_pressure_due_to_dry_air",
        "surface_pressure_due_to_dry_air_absolute_tendency",
        "net_energy_flux_toa_into_atmosphere",
    ):
        assert name in out_data.gen_data
        assert name in out_data.target_data
        assert out_data.gen_data[name].shape == (2, 3, 4, 8)
        assert out_data.target_data[name].shape == (2, 3, 4, 8)
