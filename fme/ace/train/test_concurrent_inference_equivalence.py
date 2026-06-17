"""Concurrent inline inference must match the sequential reference path.

The production inline-inference configs are effectively fully concurrent, so the
sequential reference path (``inference_one_epoch`` -> ``run_inference``) is never
exercised on beaker. The two historical concurrent-only divergences were caught
only empirically there:

  - ace#1279: a shorter run reaches its short final window in the same round a
    longer run still submits a full window. Those windows have different shapes
    (n_timesteps) and must not be concatenated; ``batch_key_of_forcing`` splits
    such a round into one forward pass per shape.
  - ace#1282: ``broadcast_ensemble`` time-coordinate handling for
    ``n_ensemble_per_ic > 1``.

This module pins ``concurrent == sequential`` for both paths using one real
(small) ``Stepper``: it runs the same stepper and data through the sequential
path and through ``run_concurrent_inference`` and asserts the predicted fields,
derived/diagnostic variables, time coordinates, and the aggregator's summary
metric logs are equal up to rounding error.

Cross-window ``stepper_state`` (corrector state) propagation through the batched
cat/split path is exercised separately by the BatchData cat/split round-trip
tests in ``fme/ace/data_loading/test_batch_data.py``; the small stepper here has
no corrector, so its ``stepper_state`` stays ``None`` throughout.
"""

import dataclasses
import datetime

import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.aggregator.inference.main import InferenceEvaluatorAggregatorConfig
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.stepper.single_module import Stepper, StepperConfig
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.generics.aggregator import InferenceAggregatorABC
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

N_LAT, N_LON, NZ = 4, 8, 3
_TIMES = xr.date_range("2000-01-01", freq="6h", periods=400, use_cftime=True)


class _DeterministicChannelSum(torch.nn.Module):
    """Deterministic module: passes inputs through and appends a diagnostic
    channel equal to half the channel sum. Determinism is required so the two
    inference paths are comparable bit-for-bit."""

    def forward(self, x):
        summed = torch.sum(x, dim=-3, keepdim=True)
        return torch.concat([x, 0.5 * summed], dim=-3)


def _get_stepper() -> tuple[Stepper, DatasetInfo, list[str]]:
    in_names = ["forcing", "prognostic"]
    out_names = ["prognostic", "diagnostic"]
    all_names = list(set(in_names + out_names))
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="prebuilt",
                        config={"module": _DeterministicChannelSum()},
                    ),
                    in_names=in_names,
                    out_names=out_names,
                    normalization=trivial_network_and_loss_normalization(all_names),
                ),
            ),
        ),
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(N_LAT), lon=torch.zeros(N_LON)
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            torch.arange(NZ), torch.arange(NZ)
        ),
        timestep=datetime.timedelta(hours=6),
    )
    return config.get_stepper(dataset_info=dataset_info), dataset_info, all_names


def _make_window(start: int, n_timesteps: int, names, n_samples: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    idx = _TIMES[start : start + n_timesteps]
    time = xr.DataArray(
        np.stack([np.asarray(idx)] * n_samples), dims=["sample", "time"]
    )
    data = {
        n: torch.rand(n_samples, n_timesteps, N_LAT, N_LON, generator=g).to(
            get_device()
        )
        for n in names
    }
    return BatchData.new_on_device(data=data, time=time, labels=None)


def _make_ic(stepper: Stepper, n_samples: int, n_ensemble: int, seed: int):
    """Returns (initial_condition, initial_time). initial_time is the per-IC time
    before any ensemble broadcast, as the aggregator expects."""
    ic_window = _make_window(0, 1, stepper.prognostic_names, n_samples, seed)
    initial_time = ic_window.time
    ic = ic_window.get_start(
        prognostic_names=stepper.prognostic_names, n_ic_timesteps=1
    )
    if n_ensemble > 1:
        ic = PrognosticState(ic.as_batch_data().broadcast_ensemble(n_ensemble))
    return ic, initial_time


def _make_loader(names, window_forward_steps, n_samples: int, seed: int):
    """One forcing ``BatchData`` per window. ``window_forward_steps`` gives the
    forward-step count of each window (n_ic_timesteps is 1); consecutive windows
    overlap by the one initial-condition timestep."""
    windows = []
    start = 0
    for i, nf in enumerate(window_forward_steps):
        windows.append(_make_window(start, 1 + nf, names, n_samples, seed + i))
        start += nf
    return windows


def _build_predictor(base_predict) -> BatchedPredictor:
    return BatchedPredictor(
        base_predict=base_predict,
        concat_ic=PrognosticState.cat,
        concat_forcing=BatchData.cat,
        split_output=lambda sd, sizes: sd.split(sizes),
        split_state=lambda ps, sizes: ps.split(sizes),
        sample_size_of_state=lambda ic: ic.as_batch_data().time.shape[0],
        batch_key_of_forcing=lambda forcing: forcing.n_timesteps,
    )


class _CapturingAggregator(InferenceAggregatorABC):
    """Records each batch's prediction/reference/time and delegates to a real
    inference aggregator, so both the raw fields and the summary metric logs can
    be compared between the sequential and concurrent paths."""

    def __init__(self, inner: InferenceAggregatorABC):
        self._inner = inner
        self.predictions: list[dict] = []
        self.references: list[dict] = []
        self.times: list[xr.DataArray] = []

    def record_initial_condition(self, initial_condition):
        return self._inner.record_initial_condition(initial_condition)

    def record_batch(self, data):
        self.predictions.append(
            {k: v.detach().clone() for k, v in data.prediction.items()}
        )
        self.references.append(
            {k: v.detach().clone() for k, v in data.reference.items()}
        )
        self.times.append(data.time.copy())
        return self._inner.record_batch(data)

    def get_summary(self):
        return self._inner.get_summary()

    def flush_diagnostics(self, subdir=None):
        self._inner.flush_diagnostics(subdir)


def _make_agg(
    stepper, dataset_info, total_forward, initial_time, n_ensemble
) -> _CapturingAggregator:
    inner = InferenceEvaluatorAggregatorConfig().build(
        dataset_info=dataset_info,
        n_ic_steps=stepper.n_ic_timesteps,
        n_forward_steps=total_forward,
        initial_time=initial_time,
        normalize=stepper.normalizer.normalize,
        channel_mean_names=stepper.loss_names,
        save_diagnostics=False,
        n_ensemble_per_ic=n_ensemble,
        enable_time_series=False,
    )
    return _CapturingAggregator(inner)


@dataclasses.dataclass
class _Task:
    name: str
    ic: PrognosticState
    loader: list
    total_forward: int
    n_ensemble: int
    initial_time: xr.DataArray


def _run_sequential(stepper, dataset_info, task: _Task) -> _CapturingAggregator:
    agg = _make_agg(
        stepper, dataset_info, task.total_forward, task.initial_time, task.n_ensemble
    )
    data = SimpleInferenceData(initial_condition=task.ic, loader=list(task.loader))
    stepper.set_eval()
    with torch.no_grad(), GlobalTimer():
        run_inference(predict=stepper.predict_paired, data=data, aggregator=agg)
    agg.flush_diagnostics(subdir=None)
    return agg


def _run_concurrent(stepper, dataset_info, tasks: list[_Task]):
    aggs: dict[str, _CapturingAggregator] = {}
    entries = []
    for task in tasks:
        agg = _make_agg(
            stepper,
            dataset_info,
            task.total_forward,
            task.initial_time,
            task.n_ensemble,
        )
        aggs[task.name] = agg
        entries.append(
            ConcurrentInferenceEntry(
                name=task.name,
                data=SimpleInferenceData(
                    initial_condition=task.ic, loader=list(task.loader)
                ),
                aggregator=agg,
                record_logs=lambda logs: None,
            )
        )
    counting_predict = _CountingPredict(stepper.predict_paired)
    stepper.set_eval()
    with torch.no_grad(), GlobalTimer():
        run_concurrent_inference(_build_predictor(counting_predict), entries)
    for agg in aggs.values():
        agg.flush_diagnostics(subdir=None)
    return aggs, counting_predict.count


class _CountingPredict:
    def __init__(self, predict):
        self._predict = predict
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self._predict(*args, **kwargs)


def _assert_equivalent(seq_agg, conc_agg):
    assert len(seq_agg.predictions) == len(conc_agg.predictions)
    assert len(seq_agg.predictions) > 0
    for sp, cp in zip(seq_agg.predictions, conc_agg.predictions):
        assert set(sp) == set(cp)
        for k in sp:
            torch.testing.assert_close(sp[k], cp[k], rtol=1e-6, atol=1e-6)
    for sr, cr in zip(seq_agg.references, conc_agg.references):
        assert set(sr) == set(cr)
        for k in sr:
            torch.testing.assert_close(sr[k], cr[k], rtol=1e-6, atol=1e-6)
    for st, ct in zip(seq_agg.times, conc_agg.times):
        xr.testing.assert_equal(st, ct)
    s_logs = seq_agg.get_summary().logs
    c_logs = conc_agg.get_summary().logs
    s_scalar = {k: v for k, v in s_logs.items() if isinstance(v, int | float)}
    c_scalar = {k: v for k, v in c_logs.items() if isinstance(v, int | float)}
    assert set(s_scalar) == set(c_scalar)
    assert len(s_scalar) > 0
    for k in s_scalar:
        assert s_scalar[k] == pytest.approx(
            c_scalar[k], rel=1e-6, abs=1e-6, nan_ok=True
        )


def test_concurrent_matches_sequential_mismatched_window_shapes():
    """ace#1279 path: in the final round the short task submits a short window
    while the long task submits a full window. Those windows cannot be
    concatenated, so the round splits into one forward pass per shape; each
    task's output must still match running it alone sequentially."""
    stepper, dataset_info, all_names = _get_stepper()
    ic_a, it_a = _make_ic(stepper, n_samples=2, n_ensemble=1, seed=1)
    ic_b, it_b = _make_ic(stepper, n_samples=2, n_ensemble=1, seed=2)
    task_a = _Task(
        "ten_year_like", ic_a, _make_loader(all_names, [2, 2, 1], 2, 10), 5, 1, it_a
    )
    task_b = _Task(
        "forty_six_year_like",
        ic_b,
        _make_loader(all_names, [2, 2, 2], 2, 20),
        6,
        1,
        it_b,
    )
    seq_a = _run_sequential(stepper, dataset_info, task_a)
    seq_b = _run_sequential(stepper, dataset_info, task_b)
    conc, n_forward_passes = _run_concurrent(stepper, dataset_info, [task_a, task_b])
    _assert_equivalent(seq_a, conc["ten_year_like"])
    _assert_equivalent(seq_b, conc["forty_six_year_like"])
    # 6 windows total: rounds 1-2 batch both tasks into one pass each; the
    # mismatched final round splits into two passes (n_timesteps 2 vs 3).
    assert n_forward_passes == 4


def test_concurrent_matches_sequential_with_ensemble():
    """ace#1282 path: an ``n_ensemble_per_ic > 1`` task on distinct start dates.
    Running it concurrently (batched with a sibling ensemble task) must match
    running it alone sequentially, including the broadcast-ensemble time coords
    consumed by the aggregator."""
    stepper, dataset_info, all_names = _get_stepper()
    ic_c, it_c = _make_ic(stepper, n_samples=2, n_ensemble=3, seed=3)
    ic_d, it_d = _make_ic(stepper, n_samples=2, n_ensemble=3, seed=4)
    task_c = _Task(
        "weather_a", ic_c, _make_loader(all_names, [2, 2], 2, 30), 4, 3, it_c
    )
    task_d = _Task(
        "weather_b", ic_d, _make_loader(all_names, [2, 2], 2, 40), 4, 3, it_d
    )
    seq_c = _run_sequential(stepper, dataset_info, task_c)
    conc, n_forward_passes = _run_concurrent(stepper, dataset_info, [task_c, task_d])
    _assert_equivalent(seq_c, conc["weather_a"])
    # 4 windows total across the two tasks' two shared rounds; one batched
    # forward pass per round.
    assert n_forward_passes == 2
