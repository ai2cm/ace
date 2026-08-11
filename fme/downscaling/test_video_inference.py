import pytest

from fme.core.logging_utils import LoggingConfig
from fme.core.normalizer import NormalizationConfig
from fme.downscaling.data import PairedDataLoaderConfig
from fme.downscaling.data.config import XarrayDataConfig
from fme.downscaling.data.utils import ClosedInterval
from fme.downscaling.video_inference import VideoInferenceConfig
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
