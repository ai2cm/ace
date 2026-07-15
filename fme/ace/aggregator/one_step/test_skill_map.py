import numpy as np
import torch

from fme.ace.aggregator.one_step.skill_map import (
    OneStepSkillMapMetricConfig,
    SkillMapAggregator,
)
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.wandb import Image

DIMS = ["lat", "lon"]


def _record(agg: SkillMapAggregator, target: torch.Tensor, gen: torch.Tensor):
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


def test_perfect_prediction_gives_r2_one_and_rmse_zero():
    torch.manual_seed(0)
    agg = SkillMapAggregator(DIMS)
    target = _batch(sample=8, nx=3, ny=4)
    _record(agg, target=target, gen=target.clone())
    ds = agg.get_dataset()
    np.testing.assert_allclose(ds["r2-a"].values, 1.0, atol=1e-10)
    np.testing.assert_allclose(ds["rmse-a"].values, 0.0, atol=1e-10)


def test_sample_mean_prediction_gives_r2_zero_and_rmse_std():
    torch.manual_seed(0)
    agg = SkillMapAggregator(DIMS)
    target = _batch(sample=16, nx=3, ny=4)
    # predict the per-gridcell sample mean at the scored step for every sample
    mean_at_step = target[:, 1].mean(dim=0)  # [lat, lon]
    gen = torch.zeros_like(target)
    gen[:, 1] = mean_at_step  # broadcast across samples
    _record(agg, target=target, gen=gen)
    ds = agg.get_dataset()
    np.testing.assert_allclose(ds["r2-a"].values, 0.0, atol=1e-10)
    # when the prediction is the sample mean, RMSE equals the target std
    expected_std = target[:, 1].std(dim=0, unbiased=False).cpu().numpy()
    np.testing.assert_allclose(ds["rmse-a"].values, expected_std, atol=1e-10)


def test_analytic_intermediate_values():
    # One gridcell, hand-computed. Samples of target and prediction at step 1.
    target = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
    gen = torch.tensor([1.5, 1.5, 3.5, 3.5], dtype=torch.double)
    mse = float(((gen - target) ** 2).mean())
    var = float(target.var(unbiased=False))
    expected_r2 = 1 - mse / var
    expected_rmse = mse**0.5

    agg = SkillMapAggregator(DIMS)
    # shape [sample, time=2, lat=1, lon=1]; step 0 is unused input.
    t = torch.zeros(4, 2, 1, 1, dtype=torch.double)
    g = torch.zeros(4, 2, 1, 1, dtype=torch.double)
    t[:, 1, 0, 0] = target
    g[:, 1, 0, 0] = gen
    _record(agg, target=t, gen=g)
    ds = agg.get_dataset()
    np.testing.assert_allclose(ds["r2-a"].values[0, 0], expected_r2, atol=1e-12)
    np.testing.assert_allclose(ds["rmse-a"].values[0, 0], expected_rmse, atol=1e-12)


def test_rmse_scales_with_data_but_r2_is_invariant():
    # R2 is affine-invariant; RMSE is in physical units and scales with the data.
    torch.manual_seed(0)
    target = _batch(sample=12, nx=2, ny=2)
    gen = _batch(sample=12, nx=2, ny=2)

    base = SkillMapAggregator(DIMS)
    _record(base, target=target, gen=gen)

    scaled = SkillMapAggregator(DIMS)
    _record(scaled, target=3.0 * target, gen=3.0 * gen)

    base_ds = base.get_dataset()
    scaled_ds = scaled.get_dataset()
    np.testing.assert_allclose(
        base_ds["r2-a"].values, scaled_ds["r2-a"].values, atol=1e-10
    )
    np.testing.assert_allclose(
        3.0 * base_ds["rmse-a"].values, scaled_ds["rmse-a"].values, atol=1e-10
    )


def test_zero_variance_target_gives_nan_r2_but_finite_rmse():
    agg = SkillMapAggregator(DIMS)
    t = torch.zeros(6, 2, 1, 2, dtype=torch.double)
    g = torch.randn(6, 2, 1, 2, dtype=torch.double)
    # first cell: target constant across samples -> zero variance -> NaN R2
    t[:, 1, 0, 0] = 5.0
    # second cell: target varies across samples -> finite R2
    t[:, 1, 0, 1] = torch.arange(6, dtype=torch.double)
    _record(agg, target=t, gen=g)
    ds = agg.get_dataset()
    assert np.isnan(ds["r2-a"].values[0, 0])
    assert not np.isnan(ds["r2-a"].values[0, 1])
    # RMSE is defined regardless of target variance
    assert np.isfinite(ds["rmse-a"].values).all()


def test_only_first_forecast_step_is_used():
    torch.manual_seed(0)
    target = _batch(sample=8, nx=2, ny=2)
    gen = _batch(sample=8, nx=2, ny=2)

    agg_full = SkillMapAggregator(DIMS)
    _record(agg_full, target=target, gen=gen)

    # corrupt every step except the scored one (index 1); result must not change
    target_corrupt = target.clone()
    gen_corrupt = gen.clone()
    target_corrupt[:, 0] = 1e6
    target_corrupt[:, 2] = -1e6
    gen_corrupt[:, 0] = -1e6
    gen_corrupt[:, 2] = 1e6
    agg_corrupt = SkillMapAggregator(DIMS)
    _record(agg_corrupt, target=target_corrupt, gen=gen_corrupt)

    for var in ("r2-a", "rmse-a"):
        np.testing.assert_allclose(
            agg_full.get_dataset()[var].values,
            agg_corrupt.get_dataset()[var].values,
        )


def test_multi_batch_accumulation_matches_single_batch():
    torch.manual_seed(0)
    target = _batch(sample=8, nx=3, ny=3)
    gen = _batch(sample=8, nx=3, ny=3)

    single = SkillMapAggregator(DIMS)
    _record(single, target=target, gen=gen)

    split = SkillMapAggregator(DIMS)
    # equal-sized halves so the mean-of-batch-means equals the overall mean
    _record(split, target=target[:4], gen=gen[:4])
    _record(split, target=target[4:], gen=gen[4:])

    for var in ("r2-a", "rmse-a"):
        np.testing.assert_allclose(
            single.get_dataset()[var].values,
            split.get_dataset()[var].values,
            atol=1e-12,
        )


def test_get_logs_and_dataset_shapes():
    torch.manual_seed(0)
    nx, ny = 3, 4
    agg = SkillMapAggregator(DIMS)
    target = _batch(sample=8, nx=nx, ny=ny)
    _record(agg, target=target, gen=_batch(sample=8, nx=nx, ny=ny))

    logs = agg.get_logs(label="skill_map")
    assert set(logs.keys()) == {"skill_map/r2/a", "skill_map/rmse/a"}
    assert all(isinstance(v, Image) for v in logs.values())

    ds = agg.get_dataset()
    for var in ("r2-a", "rmse-a"):
        assert list(ds[var].dims) == DIMS
        assert ds[var].shape == (nx, ny)


def test_include_flags_gate_each_statistic():
    torch.manual_seed(0)
    target = _batch(sample=8, nx=2, ny=2)
    gen = _batch(sample=8, nx=2, ny=2)

    r2_only = SkillMapAggregator(DIMS, include_rmse=False)
    _record(r2_only, target=target, gen=gen)
    assert set(r2_only.get_logs(label="skill_map")) == {"skill_map/r2/a"}
    assert list(r2_only.get_dataset().data_vars) == ["r2-a"]

    rmse_only = SkillMapAggregator(DIMS, include_r2=False)
    _record(rmse_only, target=target, gen=gen)
    assert set(rmse_only.get_logs(label="skill_map")) == {"skill_map/rmse/a"}
    assert list(rmse_only.get_dataset().data_vars) == ["rmse-a"]


def test_r2_panel_uses_fixed_diverging_scale(monkeypatch):
    from fme.ace.aggregator.one_step import skill_map as skill_map_mod

    calls = []

    def fake_plot_paneled_data(
        data, diverging, caption=None, roll_lon=True, vmin=None, vmax=None
    ):
        calls.append(dict(diverging=diverging, vmin=vmin, vmax=vmax, caption=caption))
        return object()  # stand-in for the wandb Image

    monkeypatch.setattr(skill_map_mod, "plot_paneled_data", fake_plot_paneled_data)

    agg = SkillMapAggregator(DIMS)
    _record(agg, target=_batch(sample=8, nx=3, ny=4), gen=_batch(sample=8, nx=3, ny=4))
    logs = agg.get_logs(label="skill_map")
    assert set(logs) == {"skill_map/r2/a", "skill_map/rmse/a"}

    r2_call = next(c for c in calls if "determination" in c["caption"])
    rmse_call = next(c for c in calls if "root-mean-squared" in c["caption"])
    # R² gets a fixed diverging [-1, 1] scale for cross-epoch comparability
    assert r2_call["diverging"] is True
    assert r2_call["vmin"] == -1.0
    assert r2_call["vmax"] == 1.0
    # RMSE keeps auto-scaling (no natural bound)
    assert rmse_call["diverging"] is False
    assert rmse_call["vmin"] is None
    assert rmse_call["vmax"] is None


def test_metric_config_builds_aggregator():
    coords = LatLonCoordinates(lon=torch.arange(4), lat=torch.arange(3))
    ds_info = DatasetInfo(horizontal_coordinates=coords)
    result = OneStepSkillMapMetricConfig().build(_build_context(ds_info))
    assert isinstance(result.deterministic, SkillMapAggregator)


def _build_context(ds_info: DatasetInfo):
    from fme.ace.aggregator.one_step.build_context import OneStepBuildContext

    return OneStepBuildContext(
        ops=ds_info.gridded_operations,
        horizontal_coordinates=ds_info.horizontal_coordinates,
        variable_metadata=ds_info.variable_metadata,
        channel_mean_names=None,
    )
