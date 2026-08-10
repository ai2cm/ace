import dataclasses

import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData
from fme.core.device import get_device
from fme.core.step.step_diagnostics import StepDiagnostics
from fme.coupled.data_loading.batch_data import CoupledBatchData, CoupledPairedData

N_SAMPLES = 2
IMG_SHAPE = (5, 5)


def _get_batch_data(
    names: list[str],
    n_time: int,
    n_samples: int = N_SAMPLES,
    n_ensemble: int = 1,
    time_offset: int = 0,
) -> BatchData:
    data = {
        name: torch.rand(n_samples, n_time, *IMG_SHAPE, device=get_device())
        for name in names
    }
    return BatchData.new_on_device(
        data=data,
        time=xr.DataArray(
            np.full((n_samples, n_time), time_offset),
            dims=["sample", "time"],
        ),
        n_ensemble=n_ensemble,
    )


def _get_step_diagnostics(name: str, n_time: int, value: float) -> StepDiagnostics:
    return StepDiagnostics(
        delta={
            name: torch.full(
                (N_SAMPLES, n_time, *IMG_SHAPE), value, device=get_device()
            )
        }
    )


def _get_coupled_batch_data(
    n_time_ocean: int = 2,
    n_time_atmosphere: int = 4,
    ocean_diagnostics: StepDiagnostics | None = None,
    atmosphere_diagnostics: StepDiagnostics | None = None,
    n_ensemble: int = 1,
    time_offset: int = 0,
) -> CoupledBatchData:
    ocean_data = _get_batch_data(
        ["sst"], n_time_ocean, n_ensemble=n_ensemble, time_offset=time_offset
    )
    atmosphere_data = _get_batch_data(
        ["a_prog"], n_time_atmosphere, n_ensemble=n_ensemble, time_offset=time_offset
    )
    return CoupledBatchData(
        ocean_data=dataclasses.replace(ocean_data, step_diagnostics=ocean_diagnostics),
        atmosphere_data=dataclasses.replace(
            atmosphere_data, step_diagnostics=atmosphere_diagnostics
        ),
    )


@pytest.mark.parametrize(
    "ocean_set, atmosphere_set",
    [(True, False), (False, True), (True, True)],
)
def test_from_coupled_batch_data_forwards_step_diagnostics(
    ocean_set: bool, atmosphere_set: bool
):
    ocean_diagnostics = (
        _get_step_diagnostics("sst", n_time=2, value=1.5) if ocean_set else None
    )
    atmosphere_diagnostics = (
        _get_step_diagnostics("a_prog", n_time=4, value=-0.5)
        if atmosphere_set
        else None
    )
    prediction = _get_coupled_batch_data(
        ocean_diagnostics=ocean_diagnostics,
        atmosphere_diagnostics=atmosphere_diagnostics,
    )
    reference = _get_coupled_batch_data()
    paired = CoupledPairedData.from_coupled_batch_data(prediction, reference)
    assert paired.ocean_data.step_diagnostics is ocean_diagnostics
    assert paired.atmosphere_data.step_diagnostics is atmosphere_diagnostics


def test_from_coupled_batch_data_forwards_n_ensemble():
    prediction = _get_coupled_batch_data(n_ensemble=2)
    reference = _get_coupled_batch_data(n_ensemble=2)
    paired = CoupledPairedData.from_coupled_batch_data(prediction, reference)
    assert paired.ocean_data.n_ensemble == 2
    assert paired.atmosphere_data.n_ensemble == 2


@pytest.mark.parametrize("mismatched_realm", ["ocean", "atmosphere"])
def test_from_coupled_batch_data_rejects_mismatched_time(mismatched_realm: str):
    prediction = _get_coupled_batch_data()
    reference = _get_coupled_batch_data()
    shifted = _get_coupled_batch_data(time_offset=1)
    if mismatched_realm == "ocean":
        reference = CoupledBatchData(
            ocean_data=shifted.ocean_data,
            atmosphere_data=reference.atmosphere_data,
        )
    else:
        reference = CoupledBatchData(
            ocean_data=reference.ocean_data,
            atmosphere_data=shifted.atmosphere_data,
        )
    with pytest.raises(ValueError, match="time coordinate must be the same"):
        CoupledPairedData.from_coupled_batch_data(prediction, reference)


@pytest.mark.parametrize(
    "ocean_set, atmosphere_set",
    [(True, False), (False, True), (True, True), (False, False)],
)
def test_with_step_diagnostics_attaches_per_realm(
    ocean_set: bool, atmosphere_set: bool
):
    batch = _get_coupled_batch_data()
    ocean_diagnostics = (
        _get_step_diagnostics("sst", n_time=2, value=1.5) if ocean_set else None
    )
    atmosphere_diagnostics = (
        _get_step_diagnostics("a_prog", n_time=4, value=-0.5)
        if atmosphere_set
        else None
    )
    result = batch.with_step_diagnostics(
        ocean=ocean_diagnostics, atmosphere=atmosphere_diagnostics
    )
    assert result.ocean_data.step_diagnostics is ocean_diagnostics
    assert result.atmosphere_data.step_diagnostics is atmosphere_diagnostics
    # the input batch is unmutated and the data is shared, not copied
    assert batch.ocean_data.step_diagnostics is None
    assert batch.atmosphere_data.step_diagnostics is None
    assert result.ocean_data.data is batch.ocean_data.data
    assert result.atmosphere_data.data is batch.atmosphere_data.data


def test_with_step_diagnostics_validates_sample_dim():
    batch = _get_coupled_batch_data()
    bad_diagnostics = StepDiagnostics(
        delta={"sst": torch.zeros(N_SAMPLES + 1, 2, *IMG_SHAPE, device=get_device())}
    )
    with pytest.raises(ValueError, match="leading dim"):
        batch.with_step_diagnostics(ocean=bad_diagnostics, atmosphere=None)
