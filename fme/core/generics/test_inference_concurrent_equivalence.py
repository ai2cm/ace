"""End-to-end equivalence test: concurrent inline inference == sequential.

Builds one small *real* ACE stepper and runs the same inference tasks through
both the sequential path (``run_inference``) and the concurrent path
(``run_concurrent_inference`` via ``BatchedPredictor``), then asserts the
predicted fields, derived variables, and aggregator metric logs match up to
rounding error.

This locks in the concurrent==sequential invariant. The two concurrent-only
divergences seen in production -- the #1279 window-shape (heterogeneous-length)
cat crash and the #1282 broadcast_ensemble time-coordinate ordering -- were
caught only empirically on beaker, because the production configs are
effectively fully concurrent and so never exercise the sequential reference
path. The two scenarios below cover exactly those cases:

* a heterogeneous-length pair -- a short task reaching its final window in the
  same round a longer task submits a full window, and
* an ensemble task (``n_ensemble_per_ic > 1``) on distinct start dates.
"""

import dataclasses
import datetime
import unittest.mock

import numpy as np
import torch
import xarray as xr

import fme
from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.ace.stepper.single_module import StepperConfig
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.generics.aggregator import InferenceAggregatorABC, InferenceSummary
from fme.core.generics.data import SimpleInferenceData
from fme.core.generics.inference import (
    BatchedPredictor,
    ConcurrentInferenceEntry,
    run_concurrent_inference,
    run_inference,
)
from fme.core.registry.module import ModuleSelector
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector
from fme.core.testing import trivial_network_and_loss_normalization
from fme.core.timing import GlobalTimer

IN_NAMES = ["forcing", "prognostic"]
OUT_NAMES = ["prognostic", "diagnostic"]
ALL_NAMES = list(set(IN_NAMES + OUT_NAMES))
FORCING_NAMES = sorted(set(IN_NAMES) - set(OUT_NAMES))
N_LAT, N_LON, NZ = 2, 4, 3


class _DeterministicStep(torch.nn.Module):
    """Like the test_looper ChannelSum, but with a deterministic diagnostic."""

    def forward(self, x):
        summed = torch.sum(x, dim=-3, keepdim=True)
        return torch.concat([x, summed], dim=-3)


def _make_stepper():
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="prebuilt", config={"module": _DeterministicStep()}
                    ),
                    in_names=IN_NAMES,
                    out_names=OUT_NAMES,
                    normalization=trivial_network_and_loss_normalization(ALL_NAMES),
                ),
            ),
        ),
    )
    return config.get_stepper(
        dataset_info=DatasetInfo(
            horizontal_coordinates=LatLonCoordinates(
                lat=torch.zeros(N_LAT), lon=torch.zeros(N_LON)
            ),
            vertical_coordinate=HybridSigmaPressureCoordinate(
                ak=torch.arange(NZ), bk=torch.arange(NZ)
            ),
            timestep=datetime.timedelta(hours=6),
        ),
    )


def _time_coord(n_samples: int, n_timesteps: int, start: int) -> xr.DataArray:
    base = np.arange(start, start + n_timesteps)
    return xr.DataArray(
        np.repeat(base[None, :], n_samples, axis=0),
        dims=["sample", "time"],
    )


def _make_windows(
    n_samples: int,
    window_timesteps: list[int],
    start_offsets: list[int] | None = None,
):
    """Build a deterministic list of forcing windows + the initial condition.

    ``window_timesteps`` gives the number of timesteps in each successive
    window (allowing a short final window). Windows carry all variables
    (prediction targets + forcing) so the paired output has reference data for
    every channel. Consecutive windows advance the time coordinate by one step
    of overlap. ``start_offsets`` (length ``n_samples``) gives a per-sample
    additive time offset, used to place samples on distinct start dates.
    """
    device = fme.get_device()
    windows: list[BatchData] = []
    start = 0
    for n_timesteps in window_timesteps:
        if start_offsets is not None:
            time = xr.concat(
                [_time_coord(1, n_timesteps, start + off) for off in start_offsets],
                dim="sample",
            )
        else:
            time = _time_coord(n_samples, n_timesteps, start)
        data = {
            name: torch.rand(n_samples, n_timesteps, N_LAT, N_LON, device=device)
            for name in ALL_NAMES
        }
        windows.append(BatchData.new_on_device(data=data, time=time, labels=None))
        start += n_timesteps - 1
    initial_condition = windows[0].get_start(
        prognostic_names=["prognostic"], n_ic_timesteps=1
    )
    return initial_condition, windows


class _RecordingAggregator(InferenceAggregatorABC):
    """Captures the initial condition and every recorded PairedData batch.

    ``get_summary`` returns a deterministic metric derived from the recorded
    predictions, so that comparing summaries exercises the metric-log path too.
    """

    def __init__(self):
        self.initial_condition: PrognosticState | None = None
        self.batches: list[PairedData] = []

    def record_initial_condition(self, initial_condition):
        self.initial_condition = initial_condition
        return []

    def record_batch(self, data: PairedData):
        self.batches.append(data)
        return []

    def get_summary(self) -> InferenceSummary:
        logs: dict[str, float] = {}
        for name in OUT_NAMES:
            stacked = torch.cat([b.prediction[name] for b in self.batches], dim=1)
            logs[f"time_mean/{name}"] = float(stacked.mean().item())
        return InferenceSummary(logs=logs, loss=logs["time_mean/prognostic"])

    def flush_diagnostics(self, subdir):
        pass


def _make_predictor(base_predict) -> BatchedPredictor:
    return BatchedPredictor(
        base_predict=base_predict,
        concat_ic=PrognosticState.cat,
        concat_forcing=BatchData.cat,
        split_output=lambda sd, sizes: sd.split(sizes),
        split_state=lambda ps, sizes: ps.split(sizes),
        sample_size_of_state=lambda ic: ic.as_batch_data().time.shape[0],
        batch_key_of_forcing=lambda forcing: forcing.n_timesteps,
    )


def _counting_predict(stepper):
    """Wrap predict_paired in a Mock that delegates to the real method."""
    return unittest.mock.MagicMock(side_effect=stepper.predict_paired)


def _run_sequential(stepper, tasks):
    summaries = {}
    aggregators = {}
    predict = _counting_predict(stepper)
    for name, (ic, windows) in tasks.items():
        agg = _RecordingAggregator()
        aggregators[name] = agg
        with torch.no_grad(), GlobalTimer():
            run_inference(
                predict=predict,
                data=SimpleInferenceData(ic, list(windows)),
                aggregator=agg,
                record_logs=lambda logs: None,
            )
        summaries[name] = agg.get_summary()
    return aggregators, summaries, predict.call_count


def _run_concurrent(stepper, tasks):
    aggregators = {name: _RecordingAggregator() for name in tasks}
    entries = [
        ConcurrentInferenceEntry(
            name=name,
            data=SimpleInferenceData(ic, list(windows)),
            aggregator=aggregators[name],
            record_logs=lambda logs: None,
        )
        for name, (ic, windows) in tasks.items()
    ]
    predict = _counting_predict(stepper)
    predictor = _make_predictor(predict)
    with torch.no_grad(), GlobalTimer():
        run_concurrent_inference(predictor, entries)
    summaries = {name: aggregators[name].get_summary() for name in tasks}
    return aggregators, summaries, predict.call_count


def _assert_paired_close(a: PairedData, b: PairedData):
    assert set(a.prediction) == set(b.prediction)
    assert set(a.reference) == set(b.reference)
    for name in a.prediction:
        torch.testing.assert_close(
            a.prediction[name], b.prediction[name], rtol=1e-6, atol=1e-6
        )
    for name in a.reference:
        torch.testing.assert_close(
            a.reference[name], b.reference[name], rtol=1e-6, atol=1e-6
        )
    assert a.time.equals(b.time)


def _assert_tasks_equivalent(seq_aggs, seq_summaries, con_aggs, con_summaries):
    assert set(seq_aggs) == set(con_aggs)
    for name in seq_aggs:
        seq_batches = seq_aggs[name].batches
        con_batches = con_aggs[name].batches
        assert len(seq_batches) == len(con_batches), name
        for sb, cb in zip(seq_batches, con_batches):
            _assert_paired_close(sb, cb)
        assert set(seq_summaries[name].logs) == set(con_summaries[name].logs)
        for key in seq_summaries[name].logs:
            np.testing.assert_allclose(
                seq_summaries[name].logs[key],
                con_summaries[name].logs[key],
                rtol=1e-6,
                atol=1e-6,
            )
        assert seq_summaries[name].loss == con_summaries[name].loss


def test_concurrent_equals_sequential_heterogeneous_lengths():
    """A short task and a long task share a group; the short one ends first.

    With one step of overlap per window, the short task submits a single short
    window while the long task submits full windows, reproducing the #1279
    mismatched-window-shape situation the concurrent cat must tolerate.
    """
    stepper = _make_stepper()
    # Both tasks submit a full 3-timestep window in round 1 (batched together),
    # then the short task submits a 2-timestep final window in round 2 while the
    # long task submits another full window (different shapes, same round), then
    # the long task finishes alone in round 3.
    short_ic, short_windows = _make_windows(n_samples=2, window_timesteps=[3, 2])
    long_ic, long_windows = _make_windows(n_samples=2, window_timesteps=[3, 3, 3])
    tasks = {
        "short": (short_ic, short_windows),
        "long": (long_ic, long_windows),
    }
    seq_aggs, seq_summaries, seq_calls = _run_sequential(stepper, tasks)
    con_aggs, con_summaries, con_calls = _run_concurrent(stepper, tasks)
    _assert_tasks_equivalent(seq_aggs, seq_summaries, con_aggs, con_summaries)
    # Sequential: 2 + 3 = 5 forward passes. Concurrent batches the matching
    # round-1 windows into one pass (4 total), proving the tasks really shared a
    # forward pass rather than silently falling back to per-task inference.
    assert seq_calls == 5
    assert con_calls == 4


def test_concurrent_equals_sequential_ensemble_distinct_start_dates():
    """A weather-like task with n_ensemble_per_ic > 1 on distinct start dates.

    Distinct per-sample start times exercise the broadcast_ensemble
    time-coordinate ordering fixed in #1282, which the concurrent cat/split
    must preserve.
    """
    stepper = _make_stepper()
    # Two distinct initial-condition dates, each broadcast to two ensemble
    # members (n_ic=2, n_ensemble_per_ic=2 => 4 samples). The per-sample start
    # offsets mirror the broadcast_ensemble layout (repeat_interleave over the
    # date dimension), so the cat/split round trip must preserve each sample's
    # distinct start date.
    ic, windows = _make_windows(
        n_samples=4,
        window_timesteps=[2, 2],
        start_offsets=[0, 0, 7, 7],
    )
    tasks = {"weather": (ic, windows)}
    seq_aggs, seq_summaries, _ = _run_sequential(stepper, tasks)
    con_aggs, con_summaries, _ = _run_concurrent(stepper, tasks)
    _assert_tasks_equivalent(seq_aggs, seq_summaries, con_aggs, con_summaries)
