import datetime
from typing import Callable, Optional, Tuple

import numpy as np
import pytest
import torch

from fme.ace.inference.derived_variables import total_water_path_budget_residual
from fme.core import ClimateData, metrics
from fme.core.climate_data import compute_dry_air_absolute_differences
from fme.core.corrector import (
    _force_conserve_dry_air,
    _force_conserve_moisture,
    _force_positive,
    _force_zero_global_mean_moisture_advection,
)
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.gridded_ops import GriddedOperations, HEALPixOperations, LatLonOperations
from fme.core.typing_ import TensorMapping

TIMESTEP = datetime.timedelta(hours=6)


def get_dry_air_nonconservation(
    data: TensorMapping,
    area_weighted_mean: Callable[[torch.Tensor], torch.Tensor],
    sigma_coordinates: SigmaCoordinates,
):
    """
    Computes the time-average one-step absolute difference in surface pressure due to
    changes in globally integrated dry air.

    Args:
        data: A mapping from variable name to tensor of shape
            [sample, time, lat, lon], in physical units. specific_total_water in kg/kg
            and surface_pressure in Pa must be present.
        area_weighted_mean: Computes the area-weighted mean of a tensor, removing the
            horizontal dimensions.
        sigma_coordinates: The sigma coordinates of the model.
    """
    return compute_dry_air_absolute_differences(
        ClimateData(data),
        area_weighted_mean=area_weighted_mean,
        sigma_coordinates=sigma_coordinates,
    ).mean()


def test_force_no_global_mean_moisture_advection():
    torch.random.manual_seed(0)
    data = {
        "tendency_of_total_water_path_due_to_advection": torch.rand(size=(3, 2, 5, 5)),
    }
    area_weights = 1.0 + torch.rand(size=(5, 5))
    original_mean = metrics.weighted_mean(
        data["tendency_of_total_water_path_due_to_advection"],
        weights=area_weights,
        dim=[-2, -1],
    )
    assert (original_mean.abs() > 0.0).all()
    fixed_data = _force_zero_global_mean_moisture_advection(
        data,
        area_weighted_mean=LatLonOperations(area_weights).area_weighted_mean,
    )
    new_mean = metrics.weighted_mean(
        fixed_data["tendency_of_total_water_path_due_to_advection"],
        weights=area_weights,
        dim=[-2, -1],
    )
    assert (new_mean.abs() < original_mean.abs()).all()
    np.testing.assert_almost_equal(new_mean.cpu().numpy(), 0.0, decimal=6)


@pytest.mark.parametrize(
    "size, use_area",
    [
        pytest.param((3, 2, 5, 5), True, id="latlon"),
        pytest.param((3, 12, 2, 3, 3), False, id="healpix"),
    ],
)
def test_force_conserve_dry_air(size: Tuple[int, ...], use_area: bool):
    torch.random.manual_seed(0)
    data = {
        "PRESsfc": 10.0 + torch.rand(size=size),
        "specific_total_water_0": torch.rand(size=size),
        "specific_total_water_1": torch.rand(size=size),
    }
    sigma_coordinates = SigmaCoordinates(
        ak=torch.asarray([3.0, 1.0, 0.0]), bk=torch.asarray([0.0, 0.6, 1.0])
    )
    if use_area:
        area_weights: Optional[torch.Tensor] = 1.0 + torch.rand(
            size=(size[-2], size[-1])
        )
    else:
        area_weights = None
    if area_weights is not None:
        gridded_operations: GriddedOperations = LatLonOperations(area_weights)
    else:
        gridded_operations = HEALPixOperations()
    original_nonconservation = get_dry_air_nonconservation(
        data,
        sigma_coordinates=sigma_coordinates,
        area_weighted_mean=gridded_operations.area_weighted_mean,
    )
    assert original_nonconservation > 0.0
    in_data = {k: v.select(dim=1, index=0) for k, v in data.items()}
    out_data = {k: v.select(dim=1, index=1) for k, v in data.items()}
    fixed_out_data = _force_conserve_dry_air(
        in_data,
        out_data,
        sigma_coordinates=sigma_coordinates,
        area_weighted_mean=gridded_operations.area_weighted_mean,
    )
    new_data = {
        k: torch.stack([v, fixed_out_data[k]], dim=1) for k, v in in_data.items()
    }
    new_nonconservation = get_dry_air_nonconservation(
        new_data,
        sigma_coordinates=sigma_coordinates,
        area_weighted_mean=gridded_operations.area_weighted_mean,
    )
    assert new_nonconservation < original_nonconservation
    np.testing.assert_almost_equal(new_nonconservation.cpu().numpy(), 0.0, decimal=6)


@pytest.mark.parametrize("fv3_data", [True, False])
@pytest.mark.parametrize(
    "global_only, terms_to_modify",
    [
        (True, "precipitation"),
        (True, "evaporation"),
        (False, "advection_and_precipitation"),
        (False, "advection_and_evaporation"),
    ],
)
def test_force_conserve_moisture(fv3_data: bool, global_only: bool, terms_to_modify):
    torch.random.manual_seed(0)
    if fv3_data:
        data = {
            "PRESsfc": 10.0 + torch.rand(size=(3, 2, 5, 5)),
            "specific_total_water_0": torch.rand(size=(3, 2, 5, 5)),
            "specific_total_water_1": torch.rand(size=(3, 2, 5, 5)),
            "PRATEsfc": torch.rand(size=(3, 2, 5, 5)),
            "LHTFLsfc": torch.rand(size=(3, 2, 5, 5)),
            "tendency_of_total_water_path_due_to_advection": torch.rand(
                size=(3, 2, 5, 5)
            ),
        }
    else:
        data = {
            "PS": 10.0 + torch.rand(size=(3, 2, 5, 5)),
            "specific_total_water_0": torch.rand(size=(3, 2, 5, 5)),
            "specific_total_water_1": torch.rand(size=(3, 2, 5, 5)),
            "surface_precipitation_rate": torch.rand(size=(3, 2, 5, 5)),
            "LHFLX": torch.rand(size=(3, 2, 5, 5)),
            "tendency_of_total_water_path_due_to_advection": torch.rand(
                size=(3, 2, 5, 5)
            ),
        }
    sigma_coordinates = SigmaCoordinates(
        ak=torch.asarray([3.0, 1.0, 0.0]), bk=torch.asarray([0.0, 0.6, 1.0])
    )
    area_weights = 1.0 + torch.rand(size=(5, 5))
    data["tendency_of_total_water_path_due_to_advection"] -= metrics.weighted_mean(
        data["tendency_of_total_water_path_due_to_advection"],
        weights=area_weights,
        dim=[-2, -1],
    )[..., None, None]
    original_budget_residual = total_water_path_budget_residual(
        ClimateData(data),
        sigma_coordinates=sigma_coordinates,
        timestep=TIMESTEP,
    )[
        :, 1
    ]  # no meaning for initial value data, want first timestep
    if global_only:
        original_budget_residual = metrics.weighted_mean(
            original_budget_residual, weights=area_weights, dim=[-2, -1]
        )
    original_budget_residual = original_budget_residual.cpu().numpy()
    original_dry_air = (
        ClimateData(data)
        .surface_pressure_due_to_dry_air(sigma_coordinates)
        .cpu()
        .numpy()
    )
    assert np.any(np.abs(original_budget_residual) > 0.0)
    in_data = {k: v.select(dim=1, index=0) for k, v in data.items()}
    out_data = {k: v.select(dim=1, index=1) for k, v in data.items()}
    fixed_out_data = _force_conserve_moisture(
        in_data,
        out_data,
        sigma_coordinates=sigma_coordinates,
        area_weighted_mean=LatLonOperations(area_weights).area_weighted_mean,
        timestep=TIMESTEP,
        terms_to_modify=terms_to_modify,
    )
    new_data = {
        k: torch.stack([v, fixed_out_data[k]], dim=1) for k, v in in_data.items()
    }
    new_budget_residual = total_water_path_budget_residual(
        ClimateData(new_data),
        sigma_coordinates=sigma_coordinates,
        timestep=TIMESTEP,
    )[
        :, 1
    ]  # no meaning for initial value data, want first timestep
    new_dry_air = (
        ClimateData(data)
        .surface_pressure_due_to_dry_air(sigma_coordinates)
        .cpu()
        .numpy()
    )

    global_budget_residual = (
        metrics.weighted_mean(new_budget_residual, weights=area_weights, dim=[-2, -1])
        .cpu()
        .numpy()
    )
    np.testing.assert_almost_equal(global_budget_residual, 0.0, decimal=6)

    if not global_only:
        new_budget_residual = new_budget_residual.cpu().numpy()
        assert np.all(np.abs(new_budget_residual) < np.abs(original_budget_residual))
        np.testing.assert_almost_equal(new_budget_residual, 0.0, decimal=6)

    np.testing.assert_almost_equal(new_dry_air, original_dry_air, decimal=6)


def test__force_positive():
    data = {
        "foo": torch.tensor([[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]]),
        "bar": torch.tensor([[-1.0, 0.0], [0.0, -3.0], [1.0, 2.0]]),
    }
    original_min = torch.min(data["foo"])
    assert original_min < 0.0
    fixed_data = _force_positive(data, ["foo"])
    new_min = torch.min(fixed_data["foo"])
    # Ensure the minimum value of 'foo' is now 0
    torch.testing.assert_close(new_min, torch.tensor(0.0))
    # Ensure other variables are not modified
    torch.testing.assert_close(fixed_data["bar"], data["bar"])
