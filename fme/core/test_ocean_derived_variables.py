import datetime

import pytest
import torch

from fme.core.constants import DENSITY_OF_WATER_CM4, SPECIFIC_HEAT_OF_WATER_CM4
from fme.core.coordinates import DepthCoordinate
from fme.core.ocean_data import OceanData
from fme.core.ocean_derived_variables import (
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
                SPECIFIC_HEAT_OF_WATER_CM4
                * DENSITY_OF_WATER_CM4
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
        == "Column-integrated ocean heat content"
    )
