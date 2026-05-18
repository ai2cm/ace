import dataclasses

import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.aggregator.one_step import (
    LegacyFlagOneStepAggregatorConfig,
    OneStepAggregatorConfig,
)
from fme.ace.aggregator.one_step.build_context import (
    MetricNotSupportedError,
    OneStepBuildContext,
    OneStepMetricBuildResult,
)
from fme.ace.aggregator.one_step.main import OneStepMeanMetricConfig
from fme.ace.stepper import TrainOutput
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.typing_ import EnsembleTensorDict


@dataclasses.dataclass
class _UnsupportedMetricConfig:
    name: str = "unsupported"

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: OneStepBuildContext) -> OneStepMetricBuildResult:
        raise MetricNotSupportedError("not supported")


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
        "test/crps/a",
        "test/crps/mean_map/a",
        "test/ssr_bias/a",
        "test/ssr_bias/mean_map/a",
        "test/ensemble_mean_rmse/mean_map/a",
        "test/ensemble_mean_rmse/a",
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
    config = OneStepAggregatorConfig(
        metrics=[
            OneStepMeanMetricConfig(target="denorm"),
            OneStepMeanMetricConfig(
                target="norm",
                include_bias=False,
                include_grad_mag_percent_diff=False,
            ),
        ]
    )
    agg = config.build(ds_info, save_diagnostics=False)
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
    with pytest.raises(ValueError, match="Duplicate metric names"):
        OneStepAggregatorConfig(
            metrics=[
                OneStepMeanMetricConfig(target="denorm"),
                OneStepMeanMetricConfig(target="denorm"),
            ]
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


def test_explicit_unsupported_metric_raises():
    ds_info = get_ds_info(nx=2, ny=2)
    config = OneStepAggregatorConfig(
        metrics=[
            OneStepMeanMetricConfig(target="denorm"),
            OneStepMeanMetricConfig(
                target="norm",
                include_bias=False,
                include_grad_mag_percent_diff=False,
            ),
            _UnsupportedMetricConfig(),  # type: ignore[list-item]
        ]
    )
    with pytest.raises(MetricNotSupportedError):
        config.build(ds_info, save_diagnostics=False)


def test_default_unsupported_metric_skipped_with_warning():
    nx, ny = 2, 2
    ds_info = get_ds_info(nx, ny)
    config_with_unsupported = OneStepAggregatorConfig(metrics=None)
    import fme.ace.aggregator.one_step.main as main_mod

    original_default = main_mod._default_metrics

    def _patched_defaults():
        return [*original_default(), _UnsupportedMetricConfig()]

    main_mod._default_metrics = _patched_defaults
    try:
        agg = config_with_unsupported.build(ds_info, save_diagnostics=False)
        batch_size, n_ensemble, n_time = 2, 2, 3
        agg.record_batch(
            batch=TrainOutput(
                metrics={"loss": 1.0},
                target_data=EnsembleTensorDict(
                    {
                        "a": torch.randn(
                            batch_size, 1, n_time, nx, ny, device=get_device()
                        )
                    },
                ),
                gen_data=EnsembleTensorDict(
                    {
                        "a": torch.randn(
                            batch_size, n_ensemble, n_time, nx, ny, device=get_device()
                        )
                    },
                ),
                time=xr.DataArray(
                    np.zeros((batch_size, n_time)), dims=["sample", "time"]
                ),
                normalize=lambda x: x,
            ),
        )
        logs = agg.get_logs(label="test")
        assert "test/unsupported" not in str(logs.keys())
    finally:
        main_mod._default_metrics = original_default


def test_fallback_ensemble_aggregator_with_explicit_metrics():
    ds_info = get_ds_info(nx=2, ny=2)
    config = OneStepAggregatorConfig(
        metrics=[
            OneStepMeanMetricConfig(target="denorm"),
            OneStepMeanMetricConfig(
                target="norm",
                include_bias=False,
                include_grad_mag_percent_diff=False,
            ),
        ]
    )
    agg = config.build(ds_info, save_diagnostics=False)
    batch_size, n_ensemble, n_time, nx, ny = 2, 2, 3, 2, 2
    target_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, 1, n_time, nx, ny, device=get_device())},
    )
    gen_data = EnsembleTensorDict(
        {"a": torch.randn(batch_size, n_ensemble, n_time, nx, ny, device=get_device())},
    )
    agg.record_batch(
        batch=TrainOutput(
            metrics={"loss": 1.0},
            target_data=target_data,
            gen_data=gen_data,
            time=xr.DataArray(np.zeros((batch_size, n_time)), dims=["sample", "time"]),
            normalize=lambda x: x,
        ),
    )
    logs = agg.get_logs(label="test")
    assert "test/crps/a" in logs
