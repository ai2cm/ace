import datetime
import math

import dacite
import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from fme.ace.data_loading.batch_data import PairedData
from fme.core import get_device
from fme.core.constants import EARTH_RADIUS
from fme.core.coordinates import HEALPixCoordinates, LatLonCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed
from fme.core.wandb import Image

from .build_context import MetricBuildContext, MetricNotSupportedError
from .data import InferenceBatchData
from .main import (
    InferenceEvaluatorAggregatorConfig,
    LegacyFlagInferenceEvaluatorAggregatorConfig,
)
from .regional import RegionalMeanMetricConfig, RegionalTimeMeanMetricConfig
from .utils import hemisphere_weights


def test_hemisphere_weights_partition_at_equator():
    coords = LatLonCoordinates(
        lat=torch.tensor([-45.0, 0.0, 45.0]), lon=torch.tensor([0.0, 180.0])
    )
    north = hemisphere_weights(coords, "north")
    south = hemisphere_weights(coords, "south")
    torch.testing.assert_close(north, torch.tensor([[0.0], [0.0], [1.0]]))
    torch.testing.assert_close(south, torch.tensor([[1.0], [1.0], [0.0]]))


def test_hemisphere_weights_unknown_hemisphere():
    coords = LatLonCoordinates(lat=torch.tensor([0.0]), lon=torch.tensor([0.0]))
    with pytest.raises(ValueError):
        hemisphere_weights(coords, "east")  # type: ignore[arg-type]


def test_hemisphere_weights_healpix_not_supported():
    coords = HEALPixCoordinates(
        face=torch.arange(12), height=torch.arange(4), width=torch.arange(4)
    )
    with pytest.raises(MetricNotSupportedError):
        hemisphere_weights(coords, "north")


@pytest.mark.parallel
@pytest.mark.parametrize("hemisphere", ["north", "south"])
def test_hemisphere_regional_sum_matches_global_reference(hemisphere):
    dist = Distributed.get_instance()
    nlat, nlon = 8, 16
    coords = LatLonCoordinates(
        lat=torch.linspace(-78.75, 78.75, nlat),
        lon=torch.linspace(0.0, 337.5, nlon),
    )
    global_mask = coords.lat > 0 if hemisphere == "north" else coords.lat <= 0
    gen = torch.Generator(device="cpu").manual_seed(0)
    global_data = torch.randn(2, nlat, nlon, generator=gen)
    expected = (
        global_mask.float().unsqueeze(-1) * coords.area_weights * global_data
    ).sum(dim=(-2, -1))

    h_slice, w_slice = dist.get_local_slices((nlat, nlon))
    weights = hemisphere_weights(coords, hemisphere)
    torch.testing.assert_close(weights.squeeze(-1).bool(), global_mask[h_slice])

    ops = coords.get_gridded_operations()
    local_data = global_data[..., h_slice, w_slice].to(get_device())
    result = ops.regional_area_weighted_sum(local_data, weights)
    torch.testing.assert_close(result.cpu(), expected)


TIMESTEP = datetime.timedelta(hours=6)
HEMISPHERIC_INTEGRALS_YAML = """
regional_means:
  - variables: [frozen_mass_total_area, sea_ice_fraction]
    reduction: sum
"""


def _latlon_coords(nlat: int = 6, nlon: int = 8) -> LatLonCoordinates:
    return LatLonCoordinates(
        lat=torch.linspace(-75.0, 75.0, nlat),
        lon=torch.linspace(0.0, 315.0, nlon),
    )


def _latlon_ds_info(nlat: int = 6, nlon: int = 8) -> DatasetInfo:
    return DatasetInfo(
        horizontal_coordinates=_latlon_coords(nlat, nlon), timestep=TIMESTEP
    )


def _build_context(ds_info: DatasetInfo, n_timesteps: int) -> MetricBuildContext:
    return MetricBuildContext(
        ops=ds_info.gridded_operations,
        horizontal_coordinates=ds_info.horizontal_coordinates,
        n_timesteps=n_timesteps,
        n_ic_steps=1,
        timestep=ds_info.timestep,
        variable_metadata={"m": VariableMetadata(units="kg/m**2", long_name="mass")},
        channel_mean_names=None,
        monthly_reference_data=None,
        time_mean_reference_data=None,
        initial_time=xr.DataArray(np.zeros(2), dims=["sample"]),
    )


def _random_fields(names, n_sample, n_time, nlat, nlon) -> dict[str, torch.Tensor]:
    return {
        name: torch.randn(n_sample, n_time, nlat, nlon, device=get_device())
        for name in names
    }


def _paired_data(names, n_sample=2, n_time=3, nlat=6, nlon=8) -> PairedData:
    return PairedData.new_on_device(
        prediction=_random_fields(names, n_sample, n_time, nlat, nlon),
        reference=_random_fields(names, n_sample, n_time, nlat, nlon),
        time=xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"]),
        labels=None,
    )


def _hemisphere_reference(
    data: torch.Tensor, coords: LatLonCoordinates, region: str, reduction: str
) -> torch.Tensor:
    """Per-step, batch-mean reduction of (sample, time, lat, lon) data."""
    if region == "global":
        mask = torch.ones_like(coords.lat)
    elif region == "north":
        mask = (coords.lat > 0).float()
    else:
        mask = (coords.lat <= 0).float()
    weights = (mask.unsqueeze(-1) * coords.area_weights).to(data.device)
    total = (weights * data).sum(dim=(-2, -1))
    if reduction == "sum":
        result = 4 * math.pi * EARTH_RADIUS**2 * total
    else:
        result = total / weights.expand(data.shape[-2:]).sum()
    return result.mean(dim=0).cpu()


def _record_one_batch(config: RegionalMeanMetricConfig, data: PairedData):
    ds_info = _latlon_ds_info()
    n_time = data.time.shape[1]
    agg = config.build(_build_context(ds_info, n_timesteps=n_time)).aggregator
    assert agg is not None
    agg.record_batch(
        InferenceBatchData(
            prediction=data.prediction,
            prediction_norm=data.prediction,
            target=data.reference,
            target_norm=data.reference,
            time=data.time,
            i_time_start=0,
        )
    )
    return agg, ds_info.horizontal_coordinates


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_regional_mean_aggregator_matches_reference(reduction):
    data = _paired_data(["m"])
    config = RegionalMeanMetricConfig(variables=["m"], reduction=reduction)
    agg, coords = _record_one_batch(config, data)
    table = agg.get_logs(label="regional")["regional/series"]
    gen_metric, target_metric = (
        f"weighted_{reduction}_gen",
        f"weighted_{reduction}_target",
    )
    bias_metric = "weighted_bias" if reduction == "mean" else "weighted_sum_bias"
    for region in ["global", "north", "south"]:
        for metric, field in [
            (gen_metric, data.prediction["m"]),
            (target_metric, data.reference["m"]),
            (bias_metric, data.prediction["m"] - data.reference["m"]),
        ]:
            column = table.columns.index(f"{region}/{metric}/m")
            logged = torch.tensor([row[column] for row in table.data])
            expected = _hemisphere_reference(field, coords, region, reduction)
            torch.testing.assert_close(logged, expected, rtol=1e-4, atol=0)
    ds = agg.get_dataset()
    assert f"{gen_metric}-north-m" in ds.data_vars
    expected_units = "kg/m**2" if reduction == "mean" else "(kg/m**2) m**2"
    assert ds[f"{gen_metric}-north-m"].attrs["units"] == expected_units


def test_regional_mean_aggregator_rmse_matches_reference():
    data = _paired_data(["m"])
    agg, coords = _record_one_batch(RegionalMeanMetricConfig(variables=["m"]), data)
    table = agg.get_logs(label="regional")["regional/series"]
    column = table.columns.index("north/weighted_rmse/m")
    logged = torch.tensor([row[column] for row in table.data])
    squared_error = (data.prediction["m"] - data.reference["m"]) ** 2
    per_sample = torch.stack(
        [
            _hemisphere_reference(squared_error[i : i + 1], coords, "north", "mean")
            for i in range(squared_error.shape[0])
        ]
    )
    torch.testing.assert_close(logged, per_sample.sqrt().mean(dim=0), rtol=1e-4, atol=0)


def test_regional_mean_aggregator_skips_absent_variables():
    data = _paired_data(["a"])
    config = RegionalMeanMetricConfig(variables=["a", "missing"], reduction="sum")
    agg, _ = _record_one_batch(config, data)
    columns = agg.get_logs(label="regional")["regional/series"].columns
    assert "north/weighted_sum_gen/a" in columns
    assert not any("missing" in c for c in columns)
    no_match, _ = _record_one_batch(
        RegionalMeanMetricConfig(variables=["missing"]), data
    )
    assert no_match.get_logs(label="r")["r/series"].columns == ["forecast_step"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(variables=[]), "non-empty"),
        (dict(variables=["a"], regions=["east"]), "Unknown regions"),
        (dict(variables=["a"], regions=["north", "north"]), "unique"),
        (dict(variables=["a"], reduction="sum", target="norm"), "denorm"),
    ],
)
def test_regional_mean_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RegionalMeanMetricConfig(**kwargs)


def test_regional_mean_config_default_names():
    assert RegionalMeanMetricConfig(variables=["a"]).get_name() == "regional_mean"
    assert (
        RegionalMeanMetricConfig(variables=["a"], reduction="sum").get_name()
        == "regional_sum"
    )
    assert (
        RegionalMeanMetricConfig(variables=["a"], target="norm").get_name()
        == "regional_mean_norm"
    )


def test_regional_mean_config_healpix_hemispheres_not_supported():
    ds_info = DatasetInfo(
        horizontal_coordinates=HEALPixCoordinates(
            face=torch.arange(12), height=torch.arange(4), width=torch.arange(4)
        ),
        timestep=TIMESTEP,
    )
    with pytest.raises(MetricNotSupportedError):
        RegionalMeanMetricConfig(variables=["a"]).build(
            _build_context(ds_info, n_timesteps=2)
        )


def _build_evaluator(
    config: InferenceEvaluatorAggregatorConfig
    | LegacyFlagInferenceEvaluatorAggregatorConfig,
    n_time: int,
    output_dir: str | None = None,
    enable_time_series: bool = True,
):
    return config.build(
        dataset_info=_latlon_ds_info(),
        n_ic_steps=1,
        n_forward_steps=n_time - 1,
        initial_time=xr.DataArray(np.zeros(2), dims=["sample"]),
        normalize=lambda x, apply_mean=True: dict(x),
        output_dir=output_dir,
        save_diagnostics=output_dir is not None,
        enable_time_series=enable_time_series,
    )


def _load_hemispheric_integrals_config(data_class):
    data = yaml.safe_load(HEMISPHERIC_INTEGRALS_YAML)
    if data_class is LegacyFlagInferenceEvaluatorAggregatorConfig:
        data["log_step_means"] = []  # default step 20 exceeds the test run
    return dacite.from_dict(data_class, data, config=dacite.Config(strict=True))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "data_class",
    [InferenceEvaluatorAggregatorConfig, LegacyFlagInferenceEvaluatorAggregatorConfig],
)
def test_hemispheric_integrals_from_config(tmp_path, data_class):
    names = ["frozen_mass_total_area", "sea_ice_fraction", "other"]
    n_time = 3
    config = _load_hemispheric_integrals_config(data_class)
    agg = _build_evaluator(config, n_time, output_dir=str(tmp_path))
    logs = agg.record_batch(_paired_data(names, n_time=n_time))
    assert len(logs) == n_time
    for region in ["global", "north", "south"]:
        assert f"regional_sum/{region}/weighted_sum_gen/sea_ice_fraction" in logs[0]
        assert (
            f"regional_sum/{region}/weighted_sum_bias/frozen_mass_total_area"
            in (logs[-1])
        )
    assert not any(k.startswith("regional_sum/") and "other" in k for k in logs[0])
    agg.flush_diagnostics()
    ds = xr.open_dataset(tmp_path / "regional_sum_diagnostics.nc")
    assert ds["weighted_sum_target-south-sea_ice_fraction"].sizes == {
        "forecast_step": n_time
    }


def test_regional_means_off_without_time_series():
    config = _load_hemispheric_integrals_config(InferenceEvaluatorAggregatorConfig)
    agg = _build_evaluator(config, n_time=3, enable_time_series=False)
    logs = agg.record_batch(_paired_data(["sea_ice_fraction"]))
    assert not any(k.startswith("regional_sum/") for log in logs for k in log)


@pytest.mark.parallel
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_regional_mean_aggregator_matches_global_reference(reduction):
    dist = Distributed.get_instance()
    n_sample, n_time, nlat, nlon = 2, 3, 8, 16
    coords = _latlon_coords(nlat, nlon)
    ds_info = DatasetInfo(horizontal_coordinates=coords, timestep=TIMESTEP)
    gen = torch.Generator(device="cpu").manual_seed(0)
    global_data = torch.randn(n_sample, n_time, nlat, nlon, generator=gen)
    h_slice, w_slice = dist.get_local_slices((nlat, nlon))
    local = {"m": global_data[..., h_slice, w_slice].to(get_device())}
    config = RegionalMeanMetricConfig(variables=["m"], reduction=reduction)
    agg = config.build(_build_context(ds_info, n_timesteps=n_time)).aggregator
    assert agg is not None
    agg.record_batch(
        InferenceBatchData(
            prediction=local,
            target=local,
            time=xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"]),
            i_time_start=0,
        )
    )
    table = agg.get_logs(label="r")["r/series"]
    for region in ["global", "north", "south"]:
        column = table.columns.index(f"{region}/weighted_{reduction}_gen/m")
        logged = torch.tensor([row[column] for row in table.data])
        expected = _hemisphere_reference(global_data, coords, region, reduction)
        torch.testing.assert_close(logged, expected, rtol=1e-4, atol=0)


def _record_time_mean_batch(config: RegionalTimeMeanMetricConfig, data: PairedData):
    ds_info = _latlon_ds_info()
    agg = config.build(_build_context(ds_info, n_timesteps=data.time.shape[1]))
    assert agg.aggregator is not None
    agg.aggregator.record_batch(
        InferenceBatchData(
            prediction=data.prediction,
            prediction_norm=data.prediction,
            target=data.reference,
            target_norm=data.reference,
            time=data.time,
            i_time_start=0,
        )
    )
    return agg.aggregator, ds_info.horizontal_coordinates


def _time_mean_scalars_reference(
    bias: torch.Tensor, coords: LatLonCoordinates, region: str
) -> tuple[float, float]:
    """Hemispheric area-weighted mean and RMS of a (lat, lon) bias map."""
    mask = coords.lat > 0 if region == "north" else coords.lat <= 0
    weights = (mask.float().unsqueeze(-1) * coords.area_weights).expand(bias.shape)
    mean = (weights * bias).sum() / weights.sum()
    rms = torch.sqrt((weights * bias**2).sum() / weights.sum())
    return float(mean), float(rms)


def test_regional_time_mean_matches_reference():
    data = _paired_data(["m", "other"])
    config = RegionalTimeMeanMetricConfig(variables=["m"], lat_min=40.0)
    agg, coords = _record_time_mean_batch(config, data)
    logs = agg.get_logs(label="regional_time_mean")

    # initial condition step is excluded from the time mean
    gen = data.prediction["m"][:, 1:].mean(dim=(0, 1)).cpu()
    target = data.reference["m"][:, 1:].mean(dim=(0, 1)).cpu()
    bias = gen - target
    for region, crop in [
        ("north", coords.lat > 40.0),
        ("south", coords.lat < -40.0),
    ]:
        expected_bias, expected_rmse = _time_mean_scalars_reference(
            bias, coords, region
        )
        prefix = "regional_time_mean"
        assert logs[f"{prefix}/bias/{region}/m"] == pytest.approx(
            float(expected_bias), rel=1e-5
        )
        assert logs[f"{prefix}/rmse/{region}/m"] == pytest.approx(
            float(expected_rmse), rel=1e-5
        )
        assert isinstance(logs[f"{prefix}/gen_map/{region}/m"], Image)
        assert isinstance(logs[f"{prefix}/bias_map/{region}/m"], Image)
        assert int(crop.sum()) == 2
    assert not any("other" in key for key in logs)

    ds = agg.get_dataset()
    torch.testing.assert_close(
        torch.as_tensor(ds["bias_map-north-m"].values),
        bias[coords.lat > 40.0],
    )
    np.testing.assert_allclose(ds["lat_south"].values, [-75.0, -45.0])
    assert ds["rmse-south-m"].values == pytest.approx(
        logs["regional_time_mean/rmse/south/m"]
    )


def test_regional_time_mean_no_listed_variables_logs_nothing():
    config = RegionalTimeMeanMetricConfig(variables=["absent"])
    agg, _ = _record_time_mean_batch(config, _paired_data(["m"]))
    assert agg.get_logs(label="regional_time_mean") == {}
    assert len(agg.get_dataset().data_vars) == 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"variables": []}, "non-empty"),
        ({"variables": ["a"], "regions": ["global"]}, "Unknown regions"),
        ({"variables": ["a"], "regions": ["north", "north"]}, "unique"),
        ({"variables": ["a"], "lat_min": 90.0}, "lat_min"),
    ],
)
def test_regional_time_mean_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RegionalTimeMeanMetricConfig(**kwargs)


def test_regional_time_mean_healpix_not_supported():
    ds_info = DatasetInfo(
        horizontal_coordinates=HEALPixCoordinates(
            face=torch.arange(12), height=torch.arange(4), width=torch.arange(4)
        ),
        timestep=TIMESTEP,
    )
    with pytest.raises(MetricNotSupportedError):
        RegionalTimeMeanMetricConfig(variables=["a"]).build(
            _build_context(ds_info, n_timesteps=2)
        )


HEMISPHERIC_MAPS_YAML = """
regional_time_means:
  - variables: [sea_ice_fraction]
    lat_min: 40
"""


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "data_class",
    [InferenceEvaluatorAggregatorConfig, LegacyFlagInferenceEvaluatorAggregatorConfig],
)
def test_hemispheric_maps_from_config(tmp_path, data_class):
    data = yaml.safe_load(HEMISPHERIC_MAPS_YAML)
    if data_class is LegacyFlagInferenceEvaluatorAggregatorConfig:
        data["log_step_means"] = []  # default step 20 exceeds the test run
    config = dacite.from_dict(data_class, data, config=dacite.Config(strict=True))
    n_time = 3
    agg = _build_evaluator(config, n_time, output_dir=str(tmp_path))
    agg.record_batch(_paired_data(["sea_ice_fraction", "other"], n_time=n_time))
    logs = agg.get_summary_logs()
    for region in ["north", "south"]:
        for key in ["gen_map", "bias_map", "bias", "rmse"]:
            assert f"regional_time_mean/{key}/{region}/sea_ice_fraction" in logs
    assert not any(k.startswith("regional_time_mean/") and "other" in k for k in logs)
    agg.flush_diagnostics()
    ds = xr.open_dataset(tmp_path / "regional_time_mean_diagnostics.nc")
    assert ds["gen_map-north-sea_ice_fraction"].dims == ("lat_north", "lon")
    np.testing.assert_allclose(ds["lon"].values, _latlon_coords().lon.numpy())


@pytest.mark.parallel
def test_regional_time_mean_scalars_match_global_reference():
    dist = Distributed.get_instance()
    n_sample, n_time, nlat, nlon = 2, 3, 8, 16
    coords = _latlon_coords(nlat, nlon)
    ds_info = DatasetInfo(horizontal_coordinates=coords, timestep=TIMESTEP)
    gen = torch.Generator(device="cpu").manual_seed(0)
    global_data = torch.randn(n_sample, n_time, nlat, nlon, generator=gen)
    h_slice, w_slice = dist.get_local_slices((nlat, nlon))
    local = global_data[..., h_slice, w_slice].to(get_device())
    config = RegionalTimeMeanMetricConfig(variables=["m"])
    agg = config.build(_build_context(ds_info, n_timesteps=n_time)).aggregator
    assert agg is not None
    agg.record_batch(
        InferenceBatchData(
            prediction={"m": local},
            target={"m": torch.zeros_like(local)},
            time=xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"]),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="r")
    bias = global_data[:, 1:].mean(dim=(0, 1))
    for region in ["north", "south"]:
        expected_bias, expected_rmse = _time_mean_scalars_reference(
            bias, coords, region
        )
        assert logs[f"r/bias/{region}/m"] == pytest.approx(expected_bias, rel=1e-4)
        assert logs[f"r/rmse/{region}/m"] == pytest.approx(expected_rmse, rel=1e-4)
