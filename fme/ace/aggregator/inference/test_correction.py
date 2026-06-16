"""Tests for correction inference metrics.

The corrector applies ``output = uncorrected + correction``. With an
identity normalizer (``normalize = lambda x: dict(x)``) the normalized
correction equals the raw correction, so metric values are hand-computable.
"""

import datetime

import numpy as np
import torch
import xarray as xr

from fme.ace.aggregator.inference.correction import (
    CorrectionMeanAggregator,
    CorrectionTimeMeanAggregator,
    compute_correction_norm,
)
from fme.ace.aggregator.inference.data import make_dummy_time
from fme.ace.aggregator.inference.main import (
    InferenceAggregatorConfig,
    TimeMeanMetricConfig,
    build_inference_evaluator_aggregator,
)
from fme.ace.data_loading.batch_data import BatchData, PairedData
from fme.ace.data_loading.step_diagnostics import StepDiagnostics
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.wandb import Image

BATCH, N_FORWARD, NX, NY = 2, 3, 4, 8


def get_ds_info(nx: int = NX, ny: int = NY) -> DatasetInfo:
    coords = LatLonCoordinates(lon=torch.arange(nx), lat=torch.arange(ny))
    return DatasetInfo(
        horizontal_coordinates=coords, timestep=datetime.timedelta(hours=6)
    )


def _const(value: float, n_time: int = N_FORWARD) -> torch.Tensor:
    # spatial dims are [lat, lon] = [NY, NX]
    return torch.full([BATCH, n_time, NY, NX], value, device=get_device())


def _identity(x):
    return dict(x)


# ---------------------------------------------------------------------------
# Direct aggregator unit tests with analytically known corrections
# ---------------------------------------------------------------------------


def test_compute_correction_norm_uses_normalized_difference():
    # normalize halves; correction_norm should be 0.5*(output) - 0.5*(uncorrected)
    prediction = {"a": _const(4.0), "b": _const(1.0)}
    uncorrected = {"a": _const(1.0)}
    correction = compute_correction_norm(
        prediction, uncorrected, normalize=lambda x: {k: v * 0.5 for k, v in x.items()}
    )
    assert set(correction) == {"a"}  # only corrector-modified variables
    np.testing.assert_allclose(correction["a"].cpu().numpy(), 0.5 * (4.0 - 1.0))


def test_compute_correction_norm_empty_when_no_corrector():
    assert compute_correction_norm({"a": _const(1.0)}, {}, _identity) == {}


def test_correction_time_mean_constant_offset():
    ops = get_ds_info().gridded_operations
    agg = CorrectionTimeMeanAggregator(gridded_operations=ops)
    # constant correction => time-mean |correction| == |delta| everywhere.
    agg.record_batch({"a": _const(2.0), "b": _const(-4.0)}, i_time_start=1)

    logs = agg.get_logs(label="time_mean_norm")
    np.testing.assert_allclose(logs["time_mean_norm/correction_magnitude/a"], 2.0)
    np.testing.assert_allclose(logs["time_mean_norm/correction_magnitude/b"], 4.0)
    # channel_mean is over corrected variables only.
    np.testing.assert_allclose(
        logs["time_mean_norm/correction_magnitude/channel_mean"], 3.0
    )
    assert isinstance(logs["time_mean_norm/correction_map/a"], Image)

    ds = agg.get_dataset()
    assert "correction_map-a" in ds
    # signed time-mean map of a constant correction is the constant itself.
    np.testing.assert_allclose(ds["correction_map-a"].values, 2.0)


def test_correction_time_mean_silent_without_data():
    ops = get_ds_info().gridded_operations
    agg = CorrectionTimeMeanAggregator(gridded_operations=ops)
    assert agg.get_logs(label="time_mean_norm") == {}
    assert len(agg.get_dataset()) == 0
    # recording an empty correction (no corrector ran) keeps it silent.
    agg.record_batch({}, i_time_start=1)
    assert agg.get_logs(label="time_mean_norm") == {}


def test_correction_mean_series_constant_offset_has_zero_std():
    ops = get_ds_info().gridded_operations
    agg = CorrectionMeanAggregator(gridded_operations=ops, n_timesteps=N_FORWARD + 1)
    agg.record_batch({"a": _const(2.0)}, i_time_start=1)

    logs = agg.get_logs(label="mean_norm", step_slice=slice(1, 1 + N_FORWARD))
    table = logs["mean_norm/correction_series"]
    columns = table.columns
    mag_col = columns.index("weighted_correction_magnitude/a")
    std_col = columns.index("weighted_correction_std/a")
    for row in table.data:
        np.testing.assert_allclose(row[mag_col], 2.0)
        np.testing.assert_allclose(row[std_col], 0.0, atol=1e-6)


def test_correction_mean_series_spatially_varying_std_positive():
    ops = get_ds_info().gridded_operations
    agg = CorrectionMeanAggregator(gridded_operations=ops, n_timesteps=N_FORWARD + 1)
    varying = torch.zeros([BATCH, N_FORWARD, NY, NX], device=get_device())
    varying[..., 0, :] = 1.0  # non-uniform across space
    agg.record_batch({"a": varying}, i_time_start=1)
    expected_std = (
        ops.area_weighted_std_dict({"a": varying})["a"].mean(dim=0).cpu().numpy()
    )

    logs = agg.get_logs(label="mean_norm", step_slice=slice(1, 1 + N_FORWARD))
    table = logs["mean_norm/correction_series"]
    std_col = table.columns.index("weighted_correction_std/a")
    stds = np.array([row[std_col] for row in table.data])
    assert np.all(stds > 0)
    np.testing.assert_allclose(stds, expected_std, rtol=1e-5)


def test_correction_mean_silent_without_data():
    ops = get_ds_info().gridded_operations
    agg = CorrectionMeanAggregator(gridded_operations=ops, n_timesteps=N_FORWARD)
    assert agg.get_logs(label="mean_norm", step_slice=slice(0, N_FORWARD)) == {}
    assert len(agg.get_dataset()) == 0


# ---------------------------------------------------------------------------
# Integration via the evaluator aggregator
# ---------------------------------------------------------------------------


def _paired_with_uncorrected(uncorrected: dict | None) -> PairedData:
    # The prediction returned by ``predict`` spans the forward steps only (the
    # initial condition is recorded separately), aligned with ``uncorrected``.
    time = make_dummy_time(n_sample=BATCH, n_time=N_FORWARD)
    prediction = {"a": _const(2.0), "b": _const(5.0)}
    target = {"a": _const(1.0), "b": _const(3.0)}
    step_diagnostics = (
        None if uncorrected is None else StepDiagnostics(uncorrected=uncorrected)
    )
    gen_batch = BatchData(data=prediction, time=time, step_diagnostics=step_diagnostics)
    target_batch = BatchData(data=target, time=time)
    return PairedData.from_batch_data(prediction=gen_batch, reference=target_batch)


def _build_evaluator(log_correction_metrics=True, enable_time_series=True):
    return build_inference_evaluator_aggregator(
        metrics=[TimeMeanMetricConfig(target="norm")],
        dataset_info=get_ds_info(),
        n_ic_steps=1,
        n_forward_steps=N_FORWARD,
        initial_time=xr.DataArray(np.zeros((BATCH,)), dims=["sample"]),
        normalize=_identity,
        save_diagnostics=False,
        enable_time_series=enable_time_series,
        log_correction_metrics=log_correction_metrics,
    )


def test_evaluator_logs_correction_metrics():
    agg = _build_evaluator()
    # uncorrected only for "a": correction = output - uncorrected = 2 - 0.5 = 1.5
    step_logs = agg.record_batch(
        _paired_with_uncorrected({"a": _const(0.5, N_FORWARD)})
    )
    summary = agg.get_summary_logs()

    np.testing.assert_allclose(summary["time_mean_norm/correction_magnitude/a"], 1.5)
    np.testing.assert_allclose(
        summary["time_mean_norm/correction_magnitude/channel_mean"], 1.5
    )
    # existing norm time-mean metrics are unaffected.
    assert "time_mean_norm/rmse/a" in summary

    step_keys: set[str] = set()
    for d in step_logs:
        step_keys.update(d.keys())
    assert "mean_norm/weighted_correction_magnitude/a" in step_keys
    assert "mean_norm/weighted_correction_std/a" in step_keys
    # only the corrector-modified variable appears.
    assert "mean_norm/weighted_correction_magnitude/b" not in step_keys


def test_evaluator_correction_flag_off_logs_nothing():
    agg = _build_evaluator(log_correction_metrics=False)
    step_logs = agg.record_batch(
        _paired_with_uncorrected({"a": _const(0.5, N_FORWARD)})
    )
    summary = agg.get_summary_logs()
    assert not any("correction" in key for key in summary)
    for d in step_logs:
        assert not any("correction" in key for key in d)


def test_evaluator_silent_when_no_corrector():
    agg = _build_evaluator()
    agg.record_batch(_paired_with_uncorrected({}))
    summary = agg.get_summary_logs()
    assert not any("correction" in key for key in summary)


def test_evaluator_inline_training_drops_correction_series():
    # enable_time_series=False mirrors inline training-time inference.
    agg = _build_evaluator(enable_time_series=False)
    step_logs = agg.record_batch(
        _paired_with_uncorrected({"a": _const(0.5, N_FORWARD)})
    )
    summary = agg.get_summary_logs()
    # time-mean correction metric present...
    assert "time_mean_norm/correction_magnitude/a" in summary
    # ...but per-step time series absent.
    step_keys: set[str] = set()
    for d in step_logs:
        step_keys.update(d.keys())
    assert not any("weighted_correction" in key for key in step_keys)


# ---------------------------------------------------------------------------
# Integration via the no-target inference aggregator
# ---------------------------------------------------------------------------


def test_no_target_aggregator_logs_correction_metrics():
    agg = InferenceAggregatorConfig().build(
        dataset_info=get_ds_info(),
        n_timesteps=1 + N_FORWARD,
        save_diagnostics=False,
        normalize=_identity,
    )
    step_logs = agg.record_batch(
        _paired_with_uncorrected({"a": _const(0.5, N_FORWARD)})
    )
    summary = agg.get_summary_logs()
    np.testing.assert_allclose(summary["time_mean_norm/correction_magnitude/a"], 1.5)
    step_keys: set[str] = set()
    for d in step_logs:
        step_keys.update(d.keys())
    assert "mean_norm/weighted_correction_magnitude/a" in step_keys


def test_no_target_aggregator_skips_correction_without_normalizer():
    # Correction metrics are normalized-space, so without a normalizer they are
    # silently skipped (preserving backward compatibility for callers that do
    # not pass one).
    agg = InferenceAggregatorConfig().build(
        dataset_info=get_ds_info(),
        n_timesteps=1 + N_FORWARD,
        save_diagnostics=False,
    )
    agg.record_batch(_paired_with_uncorrected({"a": _const(0.5, N_FORWARD)}))
    assert not any("correction" in key for key in agg.get_summary_logs())


def test_no_target_aggregator_correction_metrics_can_be_disabled():
    agg = InferenceAggregatorConfig(log_correction_metrics=False).build(
        dataset_info=get_ds_info(),
        n_timesteps=1 + N_FORWARD,
        save_diagnostics=False,
    )
    agg.record_batch(_paired_with_uncorrected({"a": _const(0.5, N_FORWARD)}))
    assert not any("correction" in key for key in agg.get_summary_logs())


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    from fme.ace.aggregator.inference.main import InferenceEvaluatorAggregatorConfig

    assert InferenceEvaluatorAggregatorConfig().log_correction_metrics is True
    assert InferenceAggregatorConfig().log_correction_metrics is True
