import datetime

import cftime
import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.normalizer import NormalizationConfig
from fme.downscaling.data.datasets import (
    PairedVideoBatchData,
    VideoBatchData,
    VideoBatchItem,
)
from fme.downscaling.data.time_encoding import compute_calendar_features
from fme.downscaling.predictors.video_composite import VideoPatchPredictor
from fme.downscaling.video_models import VideoDiffusionModelConfig

OUT_NAMES = ["var0", "var1"]


def _times(n):
    base = cftime.DatetimeProlepticGregorian(2013, 1, 2)
    return xr.DataArray(
        [base + datetime.timedelta(hours=3 * i) for i in range(n)], dims=["time"]
    )


def _video_item(n_times, height, width):
    data = {v: torch.rand(n_times, height, width) for v in OUT_NAMES}
    time = _times(n_times)
    coords = LatLonCoordinates(
        lat=torch.linspace(-10.0, 10.0, height),
        lon=torch.linspace(0.0, 30.0, width),
    )
    doy, sod = compute_calendar_features(time)
    return VideoBatchItem(data, time, coords, doy, sod)


def _spatial_paired_batch(
    batch_size, n_times, fine_height, fine_width, downscale_factor
):
    fine_items = [
        _video_item(n_times, fine_height, fine_width) for _ in range(batch_size)
    ]
    coarse_items = [
        _video_item(
            n_times, fine_height // downscale_factor, fine_width // downscale_factor
        )
        for _ in range(batch_size)
    ]
    fine = VideoBatchData.from_sequence(fine_items)
    coarse = VideoBatchData.from_sequence(coarse_items)
    return PairedVideoBatchData(fine=fine, coarse=coarse)


def _coarse_endpoints_only_model(n_times, full_fine_coords=None, downscale_factor=None):
    """Tiny endpoints_observed=False + coarse_endpoints_only=True model --
    the single-stage LR-endpoints-in/HR-full-out architecture, matching the
    real deployment configs this predictor is built for."""
    config = VideoDiffusionModelConfig(
        out_names=OUT_NAMES,
        n_timesteps=n_times,
        normalization=NormalizationConfig(
            means={"var0": 0.0, "var1": 0.0}, stds={"var0": 1.0, "var1": 1.0}
        ),
        coarse_normalization=NormalizationConfig(
            means={"var0": 0.0, "var1": 0.0}, stds={"var0": 1.0, "var1": 1.0}
        ),
        endpoints_observed=False,
        coarse_endpoints_only=True,
        num_diffusion_generation_steps=4,
        model_channels=16,
        n_heads=2,
        num_freqs=3,
    )
    return config.build(
        full_fine_coords=full_fine_coords, downscale_factor=downscale_factor
    )


def test_video_patch_predictor_generate_matches_full_extent():
    # fine 16x16 / coarse 4x4 (downscale_factor=4), patched into 2x2 coarse
    # patches (4 patches, no overlap) -- output must cover the full 16x16
    # fine extent with no gaps.
    n_times, fine_hw, downscale_factor = 5, 16, 4
    model = _coarse_endpoints_only_model(n_times, downscale_factor=downscale_factor)
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_hw,
        fine_width=fine_hw,
        downscale_factor=downscale_factor,
    )
    predictor = VideoPatchPredictor(
        model,
        coarse_yx_patch_extent=(2, 2),
        downscale_factor=downscale_factor,
        coarse_horizontal_overlap=0,
    )

    generated = predictor.generate(batch, n_samples=2)
    for name in OUT_NAMES:
        assert generated[name].shape == (2, 2, n_times, fine_hw, fine_hw)
        assert torch.isfinite(generated[name]).all()


def test_video_patch_predictor_matches_unpatched_single_patch():
    # With one patch covering the whole domain (patch extent == full
    # extent), the patched predictor's output must exactly reproduce a
    # direct (unpatched) model.generate() call given the same noise -- the
    # patching/compositing machinery should be a no-op in this case.
    n_times, fine_hw, downscale_factor = 5, 8, 4
    model = _coarse_endpoints_only_model(n_times, downscale_factor=downscale_factor)
    batch = _spatial_paired_batch(
        batch_size=2,
        n_times=n_times,
        fine_height=fine_hw,
        fine_width=fine_hw,
        downscale_factor=downscale_factor,
    )
    predictor = VideoPatchPredictor(
        model,
        coarse_yx_patch_extent=(2, 2),
        downscale_factor=downscale_factor,
        coarse_horizontal_overlap=0,
    )

    torch.manual_seed(0)
    direct = model.generate(batch, n_samples=2)
    torch.manual_seed(0)
    patched = predictor.generate(batch, n_samples=2)
    for name in OUT_NAMES:
        assert torch.allclose(direct[name], patched[name], atol=1e-5)


def _midpoint_grid(n, width):
    """Genuinely nested cell-center grid (matches real lat/lon convention):
    n cells of the given width, centers at width/2, 3*width/2, ... -- unlike
    torch.linspace, subdividing by an exact factor gives an exact nested
    subset relationship, which adjust_fine_coord_range (used by
    get_fine_coords_for_batch, and hence generate_on_batch_no_target) needs.
    Copied from test_video_models.py's helper of the same name."""
    return torch.arange(n, dtype=torch.float32) * width + width / 2


def test_video_patch_predictor_generate_on_batch_no_target():
    # A 4x4 coarse subset (tiled into four 2x2 patches, no overlap) carved
    # out of the interior of a larger 20x20 coarse grid -- gives every patch
    # real margin within full_fine_coords, same pattern
    # test_generate_on_batch_no_target_runs_with_endpoint_super_resolution
    # uses in test_video_models.py.
    n_times, downscale_factor, n_coarse_full, coarse_width = 5, 4, 20, 5.0
    full_coarse = _midpoint_grid(n_coarse_full, coarse_width)
    full_fine = _midpoint_grid(
        n_coarse_full * downscale_factor, coarse_width / downscale_factor
    )
    coarse_slice = slice(8, 12)  # interior 4x4 subset
    coarse_coord = full_coarse[coarse_slice]
    fine_hw = len(coarse_coord) * downscale_factor

    model = _coarse_endpoints_only_model(
        n_times,
        full_fine_coords=LatLonCoordinates(lat=full_fine, lon=full_fine),
        downscale_factor=downscale_factor,
    )

    def _coarse_item(n_t):
        data = {
            v: torch.rand(n_t, len(coarse_coord), len(coarse_coord)) for v in OUT_NAMES
        }
        time = _times(n_t)
        coords = LatLonCoordinates(lat=coarse_coord, lon=coarse_coord)
        doy, sod = compute_calendar_features(time)
        return VideoBatchItem(data, time, coords, doy, sod)

    coarse = VideoBatchData.from_sequence(
        [_coarse_item(n_times), _coarse_item(n_times)]
    )
    predictor = VideoPatchPredictor(
        model,
        coarse_yx_patch_extent=(2, 2),
        downscale_factor=downscale_factor,
        coarse_horizontal_overlap=0,
    )

    generated = predictor.generate_on_batch_no_target(coarse, n_samples=2)
    for name in OUT_NAMES:
        assert generated[name].shape == (2, 2, n_times, fine_hw, fine_hw)
        assert torch.isfinite(generated[name]).all()
