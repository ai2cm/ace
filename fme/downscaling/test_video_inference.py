import datetime
import os
from unittest.mock import MagicMock

import cftime
import numpy as np
import pytest
import zarr

from fme.core.logging_utils import LoggingConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.writer import ZarrWriter
from fme.downscaling.data import PairedDataLoaderConfig
from fme.downscaling.data.config import XarrayDataConfig
from fme.downscaling.data.utils import ClosedInterval
from fme.downscaling.video_inference import (
    DIMS,
    ENSEMBLE_NAME,
    LAT_NAME,
    LON_NAME,
    TIME_NAME,
    VideoInferenceConfig,
    _all_slices_written,
)
from fme.downscaling.video_models import VideoDiffusionModelConfig

OUT_NAMES = ["var0", "var1"]


def _base_kwargs(**overrides):
    norm = NormalizationConfig(
        means={"var0": 0.0, "var1": 0.0}, stds={"var0": 1.0, "var1": 1.0}
    )
    kwargs = dict(
        checkpoint_path="/dev/null",
        model=VideoDiffusionModelConfig(
            out_names=OUT_NAMES,
            n_timesteps=5,
            normalization=norm,
            coarse_normalization=norm,
            endpoints_observed=False,
            coarse_endpoints_only=True,
        ),
        data=PairedDataLoaderConfig(
            fine=[XarrayDataConfig("/dev/null")],
            coarse=[XarrayDataConfig("/dev/null")],
            batch_size=2,
            num_data_workers=0,
            strict_ensemble=False,
            lat_extent=ClosedInterval(0, 8),
            lon_extent=ClosedInterval(0, 8),
            n_timesteps=5,
        ),
        output_path="/dev/null",
        experiment_dir="/dev/null",
        logging=LoggingConfig(project="p", entity="e", name="n"),
    )
    kwargs.update(overrides)
    return kwargs


def test_divide_generation_requires_coarse_patch_extent():
    with pytest.raises(ValueError, match="coarse_patch_extent"):
        VideoInferenceConfig(**_base_kwargs(divide_generation=True))


def test_divide_generation_with_coarse_patch_extent_is_valid():
    config = VideoInferenceConfig(
        **_base_kwargs(divide_generation=True, coarse_patch_extent=[4, 4])
    )
    assert config.coarse_patch_extent == [4, 4]


def test_coarse_patch_extent_must_be_length_two():
    with pytest.raises(ValueError, match="lat, lon"):
        VideoInferenceConfig(
            **_base_kwargs(divide_generation=True, coarse_patch_extent=[4, 4, 4])
        )


def test_coarse_patch_extent_without_divide_generation_is_allowed():
    # Not an error -- a stray coarse_patch_extent with divide_generation=False
    # is simply unused, not a silent behavior change, so nothing to guard
    # against here.
    config = VideoInferenceConfig(**_base_kwargs(coarse_patch_extent=[4, 4]))
    assert config.divide_generation is False


def test_resume_and_overwrite_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        VideoInferenceConfig(**_base_kwargs(overwrite=True, resume=True))


def test_resume_alone_is_valid():
    config = VideoInferenceConfig(**_base_kwargs(resume=True))
    assert config.resume is True
    assert config.overwrite is False


def _mock_writer(is_slice_written_results: list[bool]) -> MagicMock:
    writer = MagicMock()
    writer.is_slice_written.side_effect = is_slice_written_results
    return writer


def test_all_slices_written_true_when_every_clip_is_written():
    writer = _mock_writer([True, True, True])
    slices = [slice(0, 5), slice(4, 9), slice(8, 13)]
    assert _all_slices_written(writer, slices, ensemble_slice=slice(0, 4)) is True
    assert writer.is_slice_written.call_count == 3
    writer.is_slice_written.assert_any_call(
        {TIME_NAME: slices[0], ENSEMBLE_NAME: slice(0, 4)}
    )


def test_all_slices_written_false_when_any_clip_is_not_written():
    # Second clip not written -- overall result is False regardless of the
    # third (short-circuits, matching `all()`'s laziness).
    writer = _mock_writer([True, False])
    slices = [slice(0, 5), slice(4, 9), slice(8, 13)]
    assert _all_slices_written(writer, slices, ensemble_slice=slice(0, 4)) is False
    assert writer.is_slice_written.call_count == 2


def test_all_slices_written_true_for_empty_batch():
    # Vacuously true -- never occurs in practice (a raw batch always has at
    # least one clip), but `all([])` should not be mistaken for "nothing to
    # skip."
    writer = _mock_writer([])
    assert _all_slices_written(writer, [], ensemble_slice=slice(0, 4)) is True
    writer.is_slice_written.assert_not_called()


def _make_real_writer(path, n_times=8, n_ens=4, mode="w-"):
    times = np.array(
        [
            cftime.DatetimeJulian(2020, 1, 1, 0) + datetime.timedelta(hours=3 * i)
            for i in range(n_times)
        ]
    )
    coords = {
        TIME_NAME: times,
        ENSEMBLE_NAME: np.arange(n_ens),
        LAT_NAME: np.arange(4, dtype=np.float32),
        LON_NAME: np.arange(4, dtype=np.float32),
    }
    return ZarrWriter(
        path=path,
        dims=DIMS,
        coords=coords,
        data_vars=["var"],
        mode=mode,
        overwrite_check=(mode != "a"),
        time_calendar="julian",
    )


def test_resume_against_real_writer_does_not_wipe_prior_data(tmp_path):
    """End-to-end (no mocks) check of the exact sequence a killed-and-
    restarted video_inference.py run performs: a first process writes some
    batches and gets killed; a second process opens the SAME store with
    mode="a" (what config.resume selects), calls initialize_store() again
    (as run_inference always does, unconditionally) and record_batch() for
    a later batch. Verifies the first writer's data survives untouched and
    _all_slices_written correctly distinguishes written vs. unwritten
    batches using the real ZarrWriter, not a mock."""
    path = os.path.join(tmp_path, "out.zarr")
    n_ens = 4
    ensemble_slice = slice(0, n_ens)

    # "Process 1": writes batch 0 (frames 0-3), then gets killed -- batch 1
    # (frames 4-7) never happens.
    writer1 = _make_real_writer(path, mode="w-")
    batch0_slice = slice(0, 4)
    original_batch0 = np.random.rand(4, n_ens, 4, 4).astype(np.float32)
    writer1.initialize_store(data_dtype=np.float32)
    writer1.record_batch(
        {"var": original_batch0},
        position_slices={TIME_NAME: batch0_slice, ENSEMBLE_NAME: ensemble_slice},
    )

    # "Process 2": resumed run. Constructs a fresh ZarrWriter exactly like
    # run_inference does when config.resume=True (mode="a"), and -- just
    # like run_inference -- unconditionally calls initialize_store() again
    # before touching any batch.
    writer2 = _make_real_writer(path, mode="a")
    writer2.initialize_store(data_dtype=np.float32)  # must be a no-op, not a wipe

    batch1_slice = slice(4, 8)
    assert _all_slices_written(writer2, [batch0_slice], ensemble_slice) is True, (
        "already-written batch must be detected as written after resume"
    )
    assert _all_slices_written(writer2, [batch1_slice], ensemble_slice) is False, (
        "not-yet-written batch must be detected as unwritten after resume"
    )

    # The resumed process writes the batch that was actually missing.
    new_batch1 = np.random.rand(4, n_ens, 4, 4).astype(np.float32)
    writer2.record_batch(
        {"var": new_batch1},
        position_slices={TIME_NAME: batch1_slice, ENSEMBLE_NAME: ensemble_slice},
    )

    # The critical assertion: process 1's data was NOT wiped by process 2's
    # initialize_store() call, and both batches are now present correctly.
    root = zarr.open_group(path, mode="r")
    np.testing.assert_array_equal(root["var"][batch0_slice], original_batch0)
    np.testing.assert_array_equal(root["var"][batch1_slice], new_batch1)
