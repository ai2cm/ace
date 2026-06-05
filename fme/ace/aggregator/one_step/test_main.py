import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.aggregator.one_step import (
    LegacyFlagOneStepAggregatorConfig,
    OneStepAggregatorConfig,
    build_one_step_aggregator,
    spectrum,
)
from fme.ace.aggregator.one_step.build_context import MetricNotSupportedError
from fme.ace.aggregator.one_step.main import (
    OneStepMeanMetricConfig,
    OneStepSnapshotMetricConfig,
    OneStepSpectrumMetricConfig,
)
from fme.ace.aggregator.one_step.map import OneStepMapMetricConfig
from fme.ace.stepper import TrainOutput
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.typing_ import EnsembleTensorDict


def get_ds_info(nx: int, ny: int) -> DatasetInfo:
    coords = LatLonCoordinates(lon=torch.arange(nx), lat=torch.arange(ny))
    return DatasetInfo(horizontal_coordinates=coords)


def test_labels_exist():
    batch_size = 10
    n_ensemble = 2
    n_time = 3
    nx, ny = 2, 2
    loss = 1.0
    ds_info = get_ds_info(nx, ny)
    agg = OneStepAggregatorConfig().build(ds_info, save_diagnostics=False)
    target_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, 1, n_time, nx, ny, device=get_device())},
    )
    gen_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, n_ensemble, n_time, nx, ny, device=get_device())},
    )
    agg.record_batch(
        batch=TrainOutput(
            metrics={"loss": loss},
            target_data=target_data,
            gen_data=gen_data,
            time=xr.DataArray(np.zeros((batch_size, n_time)), dims=["sample", "time"]),
            normalize=lambda x: x,
        ),
    )
    logs = agg.get_logs(label="test")
    expected_keys = [
        "test/mean/loss",
        "test/mean/weighted_rmse/a",
        "test/mean/weighted_bias/a",
        "test/mean/weighted_grad_mag_percent_diff/a",
        "test/mean_norm/weighted_rmse/a",
        "test/mean_norm/weighted_rmse/channel_mean",
        "test/snapshot/image-full-field/a",
        "test/snapshot/image-residual/a",
        "test/snapshot/image-error/a",
        "test/mean_map/image-full-field/a",
        "test/mean_map/image-error/a",
        "test/power_spectrum/positive_norm_bias/a",
        "test/power_spectrum/negative_norm_bias/a",
        "test/power_spectrum/mean_abs_norm_bias/a",
        "test/power_spectrum/smallest_scale_norm_bias/a",
        "test/ensemble/crps/a",
        "test/ensemble/crps/mean_map/a",
        "test/ensemble/ssr_bias/a",
        "test/ensemble/ssr_bias/mean_map/a",
        "test/ensemble/ensemble_mean_rmse/mean_map/a",
        "test/ensemble/ensemble_mean_rmse/a",
    ]
    assert set(logs.keys()) == set(expected_keys)


def test_aggregator_raises_on_no_data():
    """
    Basic test the aggregator combines loss correctly
    with multiple batches and no distributed training.
    """
    ds_info = get_ds_info(2, 2)
    agg = OneStepAggregatorConfig().build(ds_info, save_diagnostics=False)
    with pytest.raises(ValueError) as excinfo:
        agg.record_batch(
            batch=TrainOutput(
                metrics={"loss": 1.0},
                target_data=EnsembleTensorDict({}),
                gen_data=EnsembleTensorDict({}),
                time=xr.DataArray(np.zeros((0, 0)), dims=["sample", "time"]),
                normalize=lambda x: x,
            ),
        )
        # check that the raised exception contains the right substring
        assert "No data" in str(excinfo.value)


@pytest.mark.parametrize(
    "epoch", [pytest.param(None, id="no epoch"), pytest.param(2, id="epoch 2")]
)
def test_flush_diagnostics(tmpdir, epoch):
    nx, ny, batch_size, n_ensemble, n_time = 3, 3, 10, 2, 3
    ds_info = get_ds_info(nx, ny)
    agg = OneStepAggregatorConfig().build(ds_info, output_dir=str(tmpdir / "val"))
    target_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, 1, n_time, nx, ny, device=get_device())}
    )
    gen_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, n_ensemble, n_time, nx, ny, device=get_device())}
    )
    time = xr.DataArray(np.zeros((batch_size, n_time)), dims=["sample", "time"])
    agg.record_batch(
        batch=TrainOutput(
            metrics={"loss": 1.0},
            target_data=target_data,
            gen_data=gen_data,
            time=time,
            normalize=lambda x: x,
        ),
    )
    if epoch is not None:
        agg.flush_diagnostics(subdir=f"epoch_{epoch:04d}")
        output_dir = tmpdir / "val" / f"epoch_{epoch:04d}"
    else:
        agg.flush_diagnostics()
        output_dir = tmpdir / "val"
    expected_files = [
        "mean",
        "snapshot",
        "mean_map",
    ]
    for file in expected_files:
        assert (output_dir / f"{file}_diagnostics.nc").exists()


def test_agg_raises_without_output_dir():
    ds_info = get_ds_info(nx=2, ny=2)
    with pytest.raises(
        ValueError, match="Output directory must be set to save diagnostics"
    ):
        OneStepAggregatorConfig().build(ds_info, save_diagnostics=True, output_dir=None)


def test_explicit_metrics_build():
    ds_info = get_ds_info(nx=2, ny=2)
    agg = build_one_step_aggregator(
        metrics=[
            OneStepMeanMetricConfig(target="denorm"),
            OneStepMeanMetricConfig(
                target="norm",
                include_bias=False,
                include_grad_mag_percent_diff=False,
            ),
        ],
        dataset_info=ds_info,
        save_diagnostics=False,
    )
    target_data = EnsembleTensorDict(
        {"a": torch.randn(2, 1, 3, 2, 2, device=get_device())},
    )
    gen_data = EnsembleTensorDict(
        {"a": torch.randn(2, 1, 3, 2, 2, device=get_device())},
    )
    agg.record_batch(
        batch=TrainOutput(
            metrics={"loss": 1.0},
            target_data=target_data,
            gen_data=gen_data,
            time=xr.DataArray(np.zeros((2, 3)), dims=["sample", "time"]),
            normalize=lambda x: x,
        ),
    )
    logs = agg.get_logs(label="test")
    assert "test/mean/weighted_rmse/a" in logs
    assert "test/snapshot/image-full-field/a" not in logs


def test_duplicate_metric_names_rejected():
    ds_info = get_ds_info(nx=2, ny=2)
    with pytest.raises(ValueError, match="Duplicate metric names"):
        build_one_step_aggregator(
            metrics=[
                OneStepMeanMetricConfig(target="denorm"),
                OneStepMeanMetricConfig(target="denorm"),
            ],
            dataset_info=ds_info,
            save_diagnostics=False,
        )


def test_legacy_config_produces_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        LegacyFlagOneStepAggregatorConfig()


def test_legacy_config_builds_same_log_keys():
    batch_size = 10
    n_ensemble = 2
    n_time = 3
    nx, ny = 2, 2
    ds_info = get_ds_info(nx, ny)

    typed_agg = OneStepAggregatorConfig().build(ds_info, save_diagnostics=False)

    with pytest.warns(DeprecationWarning):
        legacy_agg = LegacyFlagOneStepAggregatorConfig().build(
            ds_info, save_diagnostics=False
        )

    target_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, 1, n_time, nx, ny, device=get_device())},
    )
    gen_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, n_ensemble, n_time, nx, ny, device=get_device())},
    )
    batch = TrainOutput(
        metrics={"loss": 1.0},
        target_data=target_data,
        gen_data=gen_data,
        time=xr.DataArray(np.zeros((batch_size, n_time)), dims=["sample", "time"]),
        normalize=lambda x: x,
    )

    typed_agg.record_batch(batch=batch)
    legacy_agg.record_batch(batch=batch)

    typed_logs = typed_agg.get_logs(label="test")
    legacy_logs = legacy_agg.get_logs(label="test")

    assert set(typed_logs.keys()) == set(legacy_logs.keys())
    for key in typed_logs:
        typed_val = typed_logs[key]
        legacy_val = legacy_logs[key]
        if isinstance(typed_val, float):
            assert typed_val == pytest.approx(legacy_val), f"Value mismatch for {key}"


def test_hierarchical_defaults_build():
    """Hierarchical config with all defaults builds and includes core metrics."""
    nx, ny = 2, 2
    ds_info = get_ds_info(nx, ny)
    agg = OneStepAggregatorConfig().build(ds_info, save_diagnostics=False)
    target_data = EnsembleTensorDict(
        {"a": torch.randn(2, 1, 3, nx, ny, device=get_device())},
    )
    gen_data = EnsembleTensorDict(
        {"a": torch.randn(2, 1, 3, nx, ny, device=get_device())},
    )
    agg.record_batch(
        batch=TrainOutput(
            metrics={"loss": 1.0},
            target_data=target_data,
            gen_data=gen_data,
            time=xr.DataArray(np.zeros((2, 3)), dims=["sample", "time"]),
            normalize=lambda x: x,
        ),
    )
    logs = agg.get_logs(label="test")
    for expected in ["mean", "mean_norm", "power_spectrum", "snapshot", "mean_map"]:
        assert any(expected in k for k in logs), f"Expected {expected} in log keys"


def test_hierarchical_enable_disable():
    """Disabling a default metric removes it; other metrics still present."""
    nx, ny = 2, 2
    ds_info = get_ds_info(nx, ny)
    agg = OneStepAggregatorConfig(
        snapshot=OneStepSnapshotMetricConfig(enabled=False),
    ).build(ds_info, save_diagnostics=False)
    target_data = EnsembleTensorDict(
        {"a": torch.randn(2, 1, 3, nx, ny, device=get_device())},
    )
    gen_data = EnsembleTensorDict(
        {"a": torch.randn(2, 1, 3, nx, ny, device=get_device())},
    )
    agg.record_batch(
        batch=TrainOutput(
            metrics={"loss": 1.0},
            target_data=target_data,
            gen_data=gen_data,
            time=xr.DataArray(np.zeros((2, 3)), dims=["sample", "time"]),
            normalize=lambda x: x,
        ),
    )
    logs = agg.get_logs(label="test")
    assert not any("snapshot" in k for k in logs)
    assert any("mean" in k for k in logs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        pytest.param(
            dict(
                mean_denorm=OneStepMeanMetricConfig(
                    target="norm",
                    include_bias=False,
                    include_grad_mag_percent_diff=False,
                )
            ),
            "mean_denorm.target must be 'denorm'",
            id="mean_denorm_wrong_target",
        ),
        pytest.param(
            dict(mean_norm=OneStepMeanMetricConfig(target="denorm")),
            "mean_norm.target must be 'norm'",
            id="mean_norm_wrong_target",
        ),
    ],
)
def test_hierarchical_rejects_mismatched_target(kwargs, match):
    with pytest.raises(ValueError, match=match):
        OneStepAggregatorConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        pytest.param(
            dict(
                mean_denorm=OneStepMeanMetricConfig(target="denorm", enabled=False),
            ),
            "mean_denorm cannot be disabled",
            id="mean_denorm_disabled",
        ),
        pytest.param(
            dict(
                mean_norm=OneStepMeanMetricConfig(
                    target="norm",
                    include_bias=False,
                    include_grad_mag_percent_diff=False,
                    enabled=False,
                ),
            ),
            "mean_norm cannot be disabled",
            id="mean_norm_disabled",
        ),
    ],
)
def test_hierarchical_rejects_disabled_required_metrics(kwargs, match):
    with pytest.raises(ValueError, match=match):
        OneStepAggregatorConfig(**kwargs)


def test_legacy_disabled_flags_match_typed_config():
    """Legacy config with log_snapshots=False and log_mean_maps=False produces
    the same keys as the equivalent OneStepAggregatorConfig."""
    batch_size = 10
    nx, ny = 2, 2
    ds_info = get_ds_info(nx, ny)

    typed_agg = OneStepAggregatorConfig(
        snapshot=OneStepSnapshotMetricConfig(enabled=False),
        mean_map=OneStepMapMetricConfig(enabled=False),
    ).build(ds_info, save_diagnostics=False)

    with pytest.warns(DeprecationWarning):
        legacy_agg = LegacyFlagOneStepAggregatorConfig(
            log_snapshots=False, log_mean_maps=False
        ).build(ds_info, save_diagnostics=False)

    target_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, 1, 3, nx, ny, device=get_device())},
    )
    gen_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, 1, 3, nx, ny, device=get_device())},
    )
    batch = TrainOutput(
        metrics={"loss": 1.0},
        target_data=target_data,
        gen_data=gen_data,
        time=xr.DataArray(np.zeros((batch_size, 3)), dims=["sample", "time"]),
        normalize=lambda x: x,
    )

    typed_agg.record_batch(batch=batch)
    legacy_agg.record_batch(batch=batch)

    typed_logs = typed_agg.get_logs(label="test")
    legacy_logs = legacy_agg.get_logs(label="test")

    assert set(typed_logs.keys()) == set(legacy_logs.keys())
    assert not any("snapshot" in k for k in typed_logs)
    assert not any("mean_map" in k for k in typed_logs)


def test_empty_metrics_list_raises():
    """An empty metrics list raises because mean_norm is required."""
    ds_info = get_ds_info(nx=2, ny=2)
    with pytest.raises(ValueError, match="mean_norm"):
        build_one_step_aggregator(
            metrics=[],
            dataset_info=ds_info,
            save_diagnostics=False,
        )


def test_disabled_ensemble_produces_no_ensemble_logs():
    """Disabling the ensemble metric produces no ensemble-related log keys."""
    from fme.ace.aggregator.one_step.ensemble import OneStepEnsembleMetricConfig

    nx, ny = 2, 2
    ds_info = get_ds_info(nx, ny)
    agg = OneStepAggregatorConfig(
        ensemble_denorm=OneStepEnsembleMetricConfig(target="denorm", enabled=False),
    ).build(ds_info, save_diagnostics=False)

    n_ensemble = 2
    target_data = EnsembleTensorDict(
        {"a": torch.randn(2, 1, 3, nx, ny, device=get_device())},
    )
    gen_data = EnsembleTensorDict(
        {"a": torch.randn(2, n_ensemble, 3, nx, ny, device=get_device())},
    )
    agg.record_batch(
        batch=TrainOutput(
            metrics={"loss": 1.0},
            target_data=target_data,
            gen_data=gen_data,
            time=xr.DataArray(np.zeros((2, 3)), dims=["sample", "time"]),
            normalize=lambda x: x,
        ),
    )
    logs = agg.get_logs(label="test")
    assert not any("crps" in k for k in logs)
    assert not any("ssr_bias" in k for k in logs)
    assert not any("ensemble_mean_rmse" in k for k in logs)


def test_both_norm_and_denorm_ensemble_metrics_coexist():
    """Configuring both norm and denorm ensemble metrics emits both sets of
    log keys, with the norm side also producing a channel_mean scalar."""
    from fme.ace.aggregator.one_step.ensemble import OneStepEnsembleMetricConfig

    nx, ny = 2, 2
    ds_info = get_ds_info(nx, ny)
    agg = OneStepAggregatorConfig(
        ensemble_denorm=OneStepEnsembleMetricConfig(
            target="denorm", log_mean_maps=False
        ),
        ensemble_norm=OneStepEnsembleMetricConfig(
            target="norm", enabled=True, log_mean_maps=False
        ),
    ).build(ds_info, save_diagnostics=False)

    n_ensemble = 2
    names = ["a", "b"]
    target_data = EnsembleTensorDict(
        {n: torch.randn(2, 1, 3, nx, ny, device=get_device()) for n in names},
    )
    gen_data = EnsembleTensorDict(
        {n: torch.randn(2, n_ensemble, 3, nx, ny, device=get_device()) for n in names},
    )
    agg.record_batch(
        batch=TrainOutput(
            metrics={"loss": 1.0},
            target_data=target_data,
            gen_data=gen_data,
            time=xr.DataArray(np.zeros((2, 3)), dims=["sample", "time"]),
            normalize=lambda x: {k: v * 0.5 for k, v in x.items()},
        ),
    )
    logs = agg.get_logs(label="test")
    for metric in ("crps", "ssr_bias", "ensemble_mean_rmse"):
        for var in names:
            assert f"test/ensemble/{metric}/{var}" in logs
            assert f"test/ensemble_norm/{metric}/{var}" in logs
        assert f"test/ensemble_norm/{metric}/channel_mean" in logs
        assert f"test/ensemble/{metric}/channel_mean" not in logs
    # The denorm and norm sides see different inputs (the normalize callback
    # scales by 0.5), so scale-sensitive metric values must differ. SSR-bias
    # is scale-invariant and is excluded from this differential check.
    for metric in ("crps", "ensemble_mean_rmse"):
        for var in names:
            assert (
                logs[f"test/ensemble/{metric}/{var}"]
                != logs[f"test/ensemble_norm/{metric}/{var}"]
            )


def test_raise_on_unsupported_true_raises(monkeypatch):
    """Explicit build with raise_on_unsupported=True raises for unsupported metrics."""

    def failing_build(self, ctx):
        raise MetricNotSupportedError("test: spectrum unsupported")

    monkeypatch.setattr(spectrum.OneStepSpectrumMetricConfig, "build", failing_build)

    ds_info = get_ds_info(nx=2, ny=2)
    with pytest.raises(MetricNotSupportedError):
        build_one_step_aggregator(
            metrics=[OneStepSpectrumMetricConfig()],
            dataset_info=ds_info,
            save_diagnostics=False,
        )


def test_raise_on_unsupported_false_skips(monkeypatch):
    """Explicit build with raise_on_unsupported=False skips unsupported metrics."""

    def failing_build(self, ctx):
        raise MetricNotSupportedError("test: spectrum unsupported")

    monkeypatch.setattr(spectrum.OneStepSpectrumMetricConfig, "build", failing_build)

    ds_info = get_ds_info(nx=2, ny=2)
    agg = build_one_step_aggregator(
        metrics=[
            OneStepMeanMetricConfig(target="denorm"),
            OneStepMeanMetricConfig(
                target="norm",
                include_bias=False,
                include_grad_mag_percent_diff=False,
            ),
            OneStepSpectrumMetricConfig(),
        ],
        dataset_info=ds_info,
        save_diagnostics=False,
        raise_on_unsupported=False,
    )
    target_data = EnsembleTensorDict(
        {"a": torch.randn(2, 1, 3, 2, 2, device=get_device())},
    )
    gen_data = EnsembleTensorDict(
        {"a": torch.randn(2, 1, 3, 2, 2, device=get_device())},
    )
    agg.record_batch(
        batch=TrainOutput(
            metrics={"loss": 1.0},
            target_data=target_data,
            gen_data=gen_data,
            time=xr.DataArray(np.zeros((2, 3)), dims=["sample", "time"]),
            normalize=lambda x: x,
        ),
    )
    logs = agg.get_logs(label="test")
    assert any("mean" in k for k in logs)
    assert not any("power_spectrum" in k for k in logs)


def test_strict_metric_raises_even_when_not_raise_on_unsupported(monkeypatch):
    """A metric with strict=True raises even when raise_on_unsupported=False."""

    def failing_build(self, ctx):
        raise MetricNotSupportedError("test: spectrum unsupported")

    monkeypatch.setattr(spectrum.OneStepSpectrumMetricConfig, "build", failing_build)

    ds_info = get_ds_info(nx=2, ny=2)
    with pytest.raises(MetricNotSupportedError):
        build_one_step_aggregator(
            metrics=[OneStepSpectrumMetricConfig(strict=True)],
            dataset_info=ds_info,
            save_diagnostics=False,
            raise_on_unsupported=False,
        )
