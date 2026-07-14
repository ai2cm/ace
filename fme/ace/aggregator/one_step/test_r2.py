import numpy as np
import torch

from fme.ace.aggregator.one_step.r2 import OneStepR2MetricConfig, R2Aggregator
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.wandb import Image

DIMS = ["lat", "lon"]


def _record(agg: R2Aggregator, target: torch.Tensor, gen: torch.Tensor):
    """Record a single batch of shape [sample, time, lat, lon] for variable 'a'."""
    agg.record_batch(
        loss=1.0,
        target_data={"a": target},
        gen_data={"a": gen},
        target_data_norm={"a": target},
        gen_data_norm={"a": gen},
    )


def _batch(sample: int, nx: int, ny: int, n_time: int = 3) -> torch.Tensor:
    return torch.randn(sample, n_time, nx, ny, device=get_device(), dtype=torch.double)


def test_perfect_prediction_gives_r2_one():
    torch.manual_seed(0)
    agg = R2Aggregator(DIMS)
    target = _batch(sample=8, nx=3, ny=4)
    _record(agg, target=target, gen=target.clone())
    r2 = agg.get_dataset()["r2-a"].values
    np.testing.assert_allclose(r2, 1.0, atol=1e-10)


def test_sample_mean_prediction_gives_r2_zero():
    torch.manual_seed(0)
    agg = R2Aggregator(DIMS)
    target = _batch(sample=16, nx=3, ny=4)
    # predict the per-gridcell sample mean at the scored step for every sample
    mean_at_step = target[:, 1].mean(dim=0)  # [lat, lon]
    gen = torch.zeros_like(target)
    gen[:, 1] = mean_at_step  # broadcast across samples
    _record(agg, target=target, gen=gen)
    r2 = agg.get_dataset()["r2-a"].values
    np.testing.assert_allclose(r2, 0.0, atol=1e-10)


def test_analytic_intermediate_value():
    # One gridcell, hand-computed. Samples of target and prediction at step 1.
    target = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
    gen = torch.tensor([1.5, 1.5, 3.5, 3.5], dtype=torch.double)
    mse = float(((gen - target) ** 2).mean())
    var = float(target.var(unbiased=False))
    expected = 1 - mse / var

    agg = R2Aggregator(DIMS)
    # shape [sample, time=2, lat=1, lon=1]; step 0 is unused input.
    t = torch.zeros(4, 2, 1, 1, dtype=torch.double)
    g = torch.zeros(4, 2, 1, 1, dtype=torch.double)
    t[:, 1, 0, 0] = target
    g[:, 1, 0, 0] = gen
    _record(agg, target=t, gen=g)
    r2 = agg.get_dataset()["r2-a"].values
    np.testing.assert_allclose(r2[0, 0], expected, atol=1e-12)


def test_zero_variance_target_gives_nan():
    agg = R2Aggregator(DIMS)
    t = torch.zeros(6, 2, 1, 2, dtype=torch.double)
    g = torch.randn(6, 2, 1, 2, dtype=torch.double)
    # first cell: target constant across samples -> zero variance -> NaN
    t[:, 1, 0, 0] = 5.0
    # second cell: target varies across samples -> finite R2
    t[:, 1, 0, 1] = torch.arange(6, dtype=torch.double)
    _record(agg, target=t, gen=g)
    r2 = agg.get_dataset()["r2-a"].values
    assert np.isnan(r2[0, 0])
    assert not np.isnan(r2[0, 1])


def test_only_first_forecast_step_is_used():
    torch.manual_seed(0)
    target = _batch(sample=8, nx=2, ny=2)
    gen = _batch(sample=8, nx=2, ny=2)

    agg_full = R2Aggregator(DIMS)
    _record(agg_full, target=target, gen=gen)

    # corrupt every step except the scored one (index 1); result must not change
    target_corrupt = target.clone()
    gen_corrupt = gen.clone()
    target_corrupt[:, 0] = 1e6
    target_corrupt[:, 2] = -1e6
    gen_corrupt[:, 0] = -1e6
    gen_corrupt[:, 2] = 1e6
    agg_corrupt = R2Aggregator(DIMS)
    _record(agg_corrupt, target=target_corrupt, gen=gen_corrupt)

    np.testing.assert_allclose(
        agg_full.get_dataset()["r2-a"].values,
        agg_corrupt.get_dataset()["r2-a"].values,
    )


def test_multi_batch_accumulation_matches_single_batch():
    torch.manual_seed(0)
    target = _batch(sample=8, nx=3, ny=3)
    gen = _batch(sample=8, nx=3, ny=3)

    single = R2Aggregator(DIMS)
    _record(single, target=target, gen=gen)

    split = R2Aggregator(DIMS)
    # equal-sized halves so the mean-of-batch-means equals the overall mean
    _record(split, target=target[:4], gen=gen[:4])
    _record(split, target=target[4:], gen=gen[4:])

    np.testing.assert_allclose(
        single.get_dataset()["r2-a"].values,
        split.get_dataset()["r2-a"].values,
        atol=1e-12,
    )


def test_get_logs_and_dataset_shapes():
    torch.manual_seed(0)
    nx, ny = 3, 4
    agg = R2Aggregator(DIMS)
    target = _batch(sample=8, nx=nx, ny=ny)
    _record(agg, target=target, gen=_batch(sample=8, nx=nx, ny=ny))

    logs = agg.get_logs(label="val")
    assert set(logs.keys()) == {"val/a"}
    assert isinstance(logs["val/a"], Image)

    ds = agg.get_dataset()
    assert list(ds["r2-a"].dims) == DIMS
    assert ds["r2-a"].shape == (nx, ny)


def test_metric_config_builds_aggregator():
    coords = LatLonCoordinates(lon=torch.arange(4), lat=torch.arange(3))
    ds_info = DatasetInfo(horizontal_coordinates=coords)
    ctx_result = OneStepR2MetricConfig().build(
        # OneStepBuildContext is constructed by the builder; mimic its fields.
        _build_context(ds_info)
    )
    assert isinstance(ctx_result.deterministic, R2Aggregator)


def _build_context(ds_info: DatasetInfo):
    from fme.ace.aggregator.one_step.build_context import OneStepBuildContext

    return OneStepBuildContext(
        ops=ds_info.gridded_operations,
        horizontal_coordinates=ds_info.horizontal_coordinates,
        variable_metadata=ds_info.variable_metadata,
        channel_mean_names=None,
    )
