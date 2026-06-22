import numpy as np
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.coupled.data_loading.batch_data import (
    CoupledBatchData,
    CoupledPairedData,
    CoupledPrognosticState,
)


def _batch(n_samples: int, n_times: int = 3) -> BatchData:
    return BatchData(
        data={"x": torch.randn(n_samples, n_times, 4, 8)},
        time=xr.DataArray(np.random.rand(n_samples, n_times), dims=["sample", "time"]),
        horizontal_dims=["lat", "lon"],
    )


def _paired(n_samples: int, n_times: int = 3) -> PairedData:
    return PairedData(
        prediction={"x": torch.randn(n_samples, n_times, 4, 8)},
        reference={"x": torch.randn(n_samples, n_times, 4, 8)},
        time=xr.DataArray(np.random.rand(n_samples, n_times), dims=["sample", "time"]),
    )


def test_coupled_batch_data_cat_and_split_roundtrip():
    a = CoupledBatchData(ocean_data=_batch(2), atmosphere_data=_batch(2, n_times=5))
    b = CoupledBatchData(ocean_data=_batch(3), atmosphere_data=_batch(3, n_times=5))
    cat = CoupledBatchData.cat([a, b])
    assert cat.ocean_data.time.shape == (5, 3)
    assert cat.atmosphere_data.time.shape == (5, 5)
    pieces = cat.split([2, 3])
    assert len(pieces) == 2
    torch.testing.assert_close(pieces[0].ocean_data.data["x"], a.ocean_data.data["x"])
    torch.testing.assert_close(
        pieces[1].atmosphere_data.data["x"], b.atmosphere_data.data["x"]
    )


def test_coupled_prognostic_state_cat_and_split_roundtrip():
    a = CoupledPrognosticState(
        ocean_data=PrognosticState(_batch(2)),
        atmosphere_data=PrognosticState(_batch(2, n_times=5)),
    )
    b = CoupledPrognosticState(
        ocean_data=PrognosticState(_batch(3)),
        atmosphere_data=PrognosticState(_batch(3, n_times=5)),
    )
    cat = CoupledPrognosticState.cat([a, b])
    pieces = cat.split([2, 3])
    assert len(pieces) == 2
    torch.testing.assert_close(
        pieces[0].ocean_data.as_batch_data().data["x"],
        a.ocean_data.as_batch_data().data["x"],
    )
    torch.testing.assert_close(
        pieces[1].atmosphere_data.as_batch_data().data["x"],
        b.atmosphere_data.as_batch_data().data["x"],
    )


def test_coupled_paired_data_cat_and_split_roundtrip():
    a = CoupledPairedData(ocean_data=_paired(2), atmosphere_data=_paired(2, n_times=5))
    b = CoupledPairedData(ocean_data=_paired(3), atmosphere_data=_paired(3, n_times=5))
    cat = CoupledPairedData.cat([a, b])
    assert cat.ocean_data.time.shape == (5, 3)
    pieces = cat.split([2, 3])
    assert len(pieces) == 2
    torch.testing.assert_close(
        pieces[0].ocean_data.prediction["x"], a.ocean_data.prediction["x"]
    )
    torch.testing.assert_close(
        pieces[1].atmosphere_data.reference["x"], b.atmosphere_data.reference["x"]
    )
