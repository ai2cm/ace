import datetime

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.aggregator.inference.main import (
    InferenceAggregatorConfig,
    InferenceEvaluatorAggregatorConfig,
    MeanMetricConfig,
    TimeMeanMetricConfig,
    build_inference_evaluator_aggregator,
)
from fme.ace.aggregator.inference.step_diagnostics import (
    CorrectionDeltaAggregator,
    CorrectionDeltaMeanAggregator,
    CorrectionDeltaTimeMeanAggregator,
    StepDiagnosticsMetricConfig,
)
from fme.ace.data_loading.batch_data import PairedData
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.normalizer import StandardNormalizer
from fme.core.spatial_mask_provider import SpatialMaskProvider
from fme.core.step.step_diagnostics import StepDiagnostics

NY, NX = 4, 8
TIMESTEP = datetime.timedelta(hours=6)


def identity_normalize(tensors, apply_mean: bool = True):
    return dict(tensors)


def get_ops(masks: dict[str, torch.Tensor] | None = None) -> LatLonOperations:
    area_weights = torch.ones(NY, NX)
    if masks is None:
        return LatLonOperations(area_weights)
    provider = SpatialMaskProvider(masks).to(get_device())
    return LatLonOperations(area_weights, provider)


def get_ds_info() -> DatasetInfo:
    coords = LatLonCoordinates(lon=torch.arange(NX), lat=torch.arange(NY))
    return DatasetInfo(horizontal_coordinates=coords, timestep=TIMESTEP)


def make_time(n_sample: int, n_time: int) -> xr.DataArray:
    base = cftime.DatetimeProlepticGregorian(2000, 1, 1)
    times = [base + i * TIMESTEP for i in range(n_time)]
    return xr.DataArray([times] * n_sample, dims=["sample", "time"])


def constant_tensor(value: float, n_sample: int = 2, n_time: int = 3) -> torch.Tensor:
    return torch.full((n_sample, n_time, NY, NX), value, device=get_device())


def build_delta_aggregator(
    normalize=identity_normalize,
    ops: LatLonOperations | None = None,
    n_timesteps: int = 3,
    record_maps: bool = False,
    time_series: bool = True,
) -> CorrectionDeltaAggregator:
    ops = ops or get_ops()
    return CorrectionDeltaAggregator(
        normalize=normalize,
        time_mean=CorrectionDeltaTimeMeanAggregator(ops, record_maps=record_maps),
        time_series=(
            CorrectionDeltaMeanAggregator(ops, n_timesteps=n_timesteps)
            if time_series
            else None
        ),
    )


def test_correction_delta_aggregator_normalizes_without_mean():
    normalizer = StandardNormalizer(
        means={"a": torch.tensor(2.0)}, stds={"a": torch.tensor(4.0)}
    )
    agg = build_delta_aggregator(normalize=normalizer.normalize)
    delta = {"a": constant_tensor(8.0)}
    agg.record_batch(
        prediction=delta,
        step_diagnostics=StepDiagnostics(delta=delta),
        i_time_start=0,
    )
    logs = agg.summary_logs()
    # delta / std, with no mean subtraction: 8 / 4, not (8 - 2) / 4
    assert logs["time_mean_norm/correction_magnitude/a"] == pytest.approx(2.0)


def test_correction_delta_aggregator_raises_on_unknown_delta_key():
    normalizer = StandardNormalizer(
        means={"a": torch.tensor(0.0)}, stds={"a": torch.tensor(1.0)}
    )
    agg = build_delta_aggregator(normalize=normalizer.normalize)
    delta = {"b": constant_tensor(1.0)}
    with pytest.raises(ValueError, match="b"):
        agg.record_batch(
            prediction=delta,
            step_diagnostics=StepDiagnostics(delta=delta),
            i_time_start=0,
        )


def test_correction_delta_aggregator_raises_on_time_dim_mismatch():
    agg = build_delta_aggregator()
    prediction = {"a": constant_tensor(1.0, n_time=3)}
    delta = {"a": constant_tensor(1.0, n_time=2)}
    with pytest.raises(ValueError, match="time-aligned"):
        agg.record_batch(
            prediction=prediction,
            step_diagnostics=StepDiagnostics(delta=delta),
            i_time_start=0,
        )


def test_time_mean_aggregator_constant_offset():
    agg = CorrectionDeltaTimeMeanAggregator(get_ops(), record_maps=True)
    agg.record_batch({"a": constant_tensor(-3.0)})
    logs = agg.get_logs(label="")
    assert logs["correction_magnitude/a"] == pytest.approx(3.0)
    assert "correction_map/a" in logs
    ds = agg.get_dataset()
    np.testing.assert_allclose(ds["correction_map-a"].values, -3.0)


def test_time_mean_aggregator_masked_cells_do_not_poison_scalars():
    mask = torch.ones(NY, NX)
    mask[1, 1] = 0
    agg = CorrectionDeltaTimeMeanAggregator(get_ops(masks={"mask_a": mask}))
    correction = constant_tensor(2.0)
    correction[:, :, 1, 1] = float("nan")
    agg.record_batch({"a": correction})
    logs = agg.get_logs(label="")
    assert logs["correction_magnitude/a"] == pytest.approx(2.0)
    assert np.isfinite(logs["correction_magnitude/a"])


def test_time_mean_aggregator_null_masking_matches_unmasked_hand_computation():
    torch.manual_seed(0)
    correction = torch.randn(2, 3, NY, NX, device=get_device())
    agg = CorrectionDeltaTimeMeanAggregator(get_ops())
    agg.record_batch({"a": correction})
    logs = agg.get_logs(label="")
    # uniform area weights: the scalar is the plain mean over cells of the
    # sample- and time-mean magnitude
    expected = correction.abs().mean(dim=(0, 1)).mean().item()
    assert logs["correction_magnitude/a"] == pytest.approx(expected, rel=1e-5)


def test_time_mean_aggregator_maps_off_by_default():
    agg = CorrectionDeltaTimeMeanAggregator(get_ops())
    agg.record_batch({"a": constant_tensor(1.0)})
    logs = agg.get_logs(label="")
    assert "correction_magnitude/a" in logs
    assert not any(key.startswith("correction_map") for key in logs)
    assert len(agg.get_dataset()) == 0


def test_mean_aggregator_constant_offset_series():
    n_time = 3
    ops = get_ops()
    agg = CorrectionDeltaMeanAggregator(ops, n_timesteps=n_time)
    agg.record_batch({"a": constant_tensor(2.0, n_time=n_time)}, i_time_start=0)
    ds = agg.get_dataset()
    np.testing.assert_allclose(ds["weighted_correction_magnitude-a"].values, 2.0)
    np.testing.assert_allclose(ds["weighted_correction_std-a"].values, 0.0, atol=1e-6)

    torch.manual_seed(0)
    varying = torch.randn(2, n_time, NY, NX, device=get_device())
    agg = CorrectionDeltaMeanAggregator(ops, n_timesteps=n_time)
    agg.record_batch({"a": varying}, i_time_start=0)
    ds = agg.get_dataset()
    expected_std = (
        ops.area_weighted_std_dict({"a": varying})["a"].mean(dim=0).cpu().numpy()
    )
    np.testing.assert_allclose(
        ds["weighted_correction_std-a"].values, expected_std, rtol=1e-5
    )


@pytest.mark.parametrize(
    "make_aggregator",
    [
        lambda: CorrectionDeltaTimeMeanAggregator(get_ops(), record_maps=True),
        lambda: CorrectionDeltaMeanAggregator(get_ops(), n_timesteps=3),
    ],
    ids=["time_mean", "time_series"],
)
def test_aggregators_silent_without_data(make_aggregator):
    agg = make_aggregator()
    assert agg.get_logs(label="") == {}
    assert len(agg.get_dataset()) == 0


def test_metric_config_build_granularity():
    def build(config, enable_time_series=True, normalize=identity_normalize):
        return config.build(
            gridded_operations=get_ops(),
            n_timesteps=3,
            variable_metadata=None,
            enable_time_series=enable_time_series,
            normalize=normalize,
        )

    default_agg = build(StepDiagnosticsMetricConfig())
    assert default_agg is not None
    assert default_agg.log_time_series

    no_series = build(StepDiagnosticsMetricConfig(), enable_time_series=False)
    assert no_series is not None
    assert not no_series.log_time_series

    maps_only = build(
        StepDiagnosticsMetricConfig(correction_scalars=False, correction_maps=True)
    )
    assert maps_only is not None
    assert not maps_only.log_time_series

    assert (
        build(
            StepDiagnosticsMetricConfig(correction_scalars=False, correction_maps=False)
        )
        is None
    )
    assert build(StepDiagnosticsMetricConfig(), normalize=None) is None


def build_evaluator_aggregator(
    tmp_path=None,
    step_diagnostics: StepDiagnosticsMetricConfig | None = None,
    enable_time_series: bool = True,
):
    n_sample = 2
    return build_inference_evaluator_aggregator(
        metrics=[
            MeanMetricConfig(target="norm"),
            TimeMeanMetricConfig(target="norm"),
        ],
        dataset_info=get_ds_info(),
        n_ic_steps=0,
        n_forward_steps=3,
        initial_time=xr.DataArray(np.zeros((n_sample,)), dims=["sample"]),
        normalize=identity_normalize,
        save_diagnostics=tmp_path is not None,
        output_dir=str(tmp_path) if tmp_path is not None else None,
        enable_time_series=enable_time_series,
        step_diagnostics=step_diagnostics,
    )


def make_paired_data(
    delta: dict[str, torch.Tensor] | None, n_time: int = 3
) -> PairedData:
    data = {
        "a": constant_tensor(1.0, n_time=n_time),
        "b": constant_tensor(2.0, n_time=n_time),
    }
    return PairedData(
        prediction=data,
        reference={name: tensor + 1.0 for name, tensor in data.items()},
        time=make_time(n_sample=2, n_time=n_time),
        step_diagnostics=(StepDiagnostics(delta=delta) if delta is not None else None),
    )


def test_evaluator_aggregator_logs_correction_metrics(tmp_path):
    agg = build_evaluator_aggregator(tmp_path=tmp_path)
    inference_logs = agg.record_batch(
        make_paired_data(delta={"a": constant_tensor(0.5)})
    )

    summary = agg.get_summary_logs()
    assert summary["time_mean_norm/correction_magnitude/a"] == pytest.approx(0.5)
    assert "time_mean_norm/correction_magnitude/b" not in summary
    # existing metrics are unaffected
    assert "time_mean_norm/rmse/a" in summary

    series_keys = {key for log in inference_logs for key in log}
    assert "mean_norm/weighted_correction_magnitude/a" in series_keys
    assert "mean_norm/weighted_correction_std/a" in series_keys
    assert "mean_norm/weighted_correction_magnitude/b" not in series_keys
    # the existing mean_norm series is still reported
    assert "mean_norm/weighted_mean_gen/a" in series_keys

    agg.flush_diagnostics()
    assert (tmp_path / "mean_norm_correction_diagnostics.nc").exists()


@pytest.mark.parametrize(
    ("step_diagnostics_config", "with_delta"),
    [
        (
            StepDiagnosticsMetricConfig(
                correction_scalars=False, correction_maps=False
            ),
            True,
        ),
        (None, False),  # no corrector ran: step_diagnostics is None
    ],
    ids=["config_off", "no_corrector"],
)
def test_evaluator_aggregator_silent_paths(
    tmp_path, step_diagnostics_config, with_delta
):
    delta = {"a": constant_tensor(0.5)} if with_delta else None
    agg = build_evaluator_aggregator(
        tmp_path=tmp_path, step_diagnostics=step_diagnostics_config
    )
    inference_logs = agg.record_batch(make_paired_data(delta=delta))

    summary = agg.get_summary_logs()
    assert not any("correction" in key for key in summary)
    series_keys = {key for log in inference_logs for key in log}
    assert not any("correction" in key for key in series_keys)

    agg.flush_diagnostics()
    assert not list(tmp_path.glob("*correction*"))


def test_evaluator_aggregator_maps_opt_in():
    agg = build_evaluator_aggregator(
        step_diagnostics=StepDiagnosticsMetricConfig(correction_maps=True)
    )
    agg.record_batch(make_paired_data(delta={"a": constant_tensor(0.5)}))
    summary = agg.get_summary_logs()
    assert "time_mean_norm/correction_map/a" in summary
    assert "time_mean_norm/correction_magnitude/a" in summary


def test_inline_training_drops_correction_series():
    agg = build_evaluator_aggregator(enable_time_series=False)
    inference_logs = agg.record_batch(
        make_paired_data(delta={"a": constant_tensor(0.5)})
    )
    summary = agg.get_summary_logs()
    assert "time_mean_norm/correction_magnitude/a" in summary
    series_keys = {key for log in inference_logs for key in log}
    assert not any("correction" in key for key in series_keys)


def test_no_target_aggregator_logs_correction_metrics():
    agg = InferenceAggregatorConfig().build(
        dataset_info=get_ds_info(),
        n_timesteps=3,
        save_diagnostics=False,
        normalize=identity_normalize,
    )
    inference_logs = agg.record_batch(
        make_paired_data(delta={"a": constant_tensor(0.5)})
    )
    summary = agg.get_summary_logs()
    assert summary["time_mean_norm/correction_magnitude/a"] == pytest.approx(0.5)
    series_keys = {key for log in inference_logs for key in log}
    assert "mean_norm/weighted_correction_magnitude/a" in series_keys


def test_no_target_aggregator_skips_without_normalizer():
    agg = InferenceAggregatorConfig().build(
        dataset_info=get_ds_info(),
        n_timesteps=3,
        save_diagnostics=False,
    )
    inference_logs = agg.record_batch(
        make_paired_data(delta={"a": constant_tensor(0.5)})
    )
    summary = agg.get_summary_logs()
    assert not any("correction" in key for key in summary)
    series_keys = {key for log in inference_logs for key in log}
    assert not any("correction" in key for key in series_keys)


def test_config_defaults():
    step_diagnostics_configs = [
        InferenceEvaluatorAggregatorConfig().step_diagnostics,
        InferenceAggregatorConfig().step_diagnostics,
    ]
    for config in step_diagnostics_configs:
        assert config.correction_scalars is True
        assert config.correction_maps is False
