import dataclasses
import datetime

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
import zarr

from fme.core.dataset.time import TimeSlice
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.distributed.non_distributed import DummyWrapper
from fme.core.ema import EMAConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import OptimizationConfig
from fme.downscaling.data.config import PairedVideoLoaderConfig
from fme.downscaling.test_video_train import _trainer_config
from fme.downscaling.video_inference import (
    VideoInferenceConfig,
    _bare_module,
    _clip_write_slice,
    _splice_observed_endpoints,
    _validate_contiguous_clips,
    _warn_if_idle_ranks,
)
from fme.downscaling.video_models import VideoDiffusionModelConfig
from fme.downscaling.video_train import VideoTrainerConfig, _save_checkpoint

_OUT_NAMES = ["var0", "var1"]


def _state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _inference_config(trainer_config, ckpt_path, tmp_path, use_ema):
    return VideoInferenceConfig(
        checkpoint_path=ckpt_path,
        model=trainer_config.model,
        data=trainer_config.test_data,
        output_path=str(tmp_path / "out.zarr"),
        experiment_dir=str(tmp_path / "exp"),
        logging=LoggingConfig(
            log_to_screen=False, log_to_wandb=False, log_to_file=False
        ),
        use_ema=use_ema,
    )


@pytest.mark.medium_duration
def test_build_model_round_trip(tmp_path):
    """``VideoInferenceConfig.build_model`` is the actual production
    checkpoint->inference path (distinct from, and previously untested by,
    ``VideoTrainer``'s own resume mechanism). It hand-parses the checkpoint's
    ``module``/``ema`` dict and strips a ``module.`` prefix -- this proves
    that round trip preserves weights exactly, for both EMA and raw loading.

    A checkpoint is saved explicitly (rather than reusing ``best.ckpt``) right
    after capturing the expected state, so the comparison isn't confounded by
    ``best.ckpt`` reflecting an earlier epoch than the trainer's final state.
    """
    trainer_config = _trainer_config(tmp_path)
    trainer = trainer_config.build()
    trainer.train()

    with trainer._ema_context():
        expected_ema_state = _state_dict_cpu(_bare_module(trainer.model.module))
    expected_raw_state = _state_dict_cpu(_bare_module(trainer.model.module))

    ckpt_path = str(tmp_path / "round_trip.ckpt")
    _save_checkpoint(trainer, ckpt_path)

    loaded_ema = _inference_config(
        trainer_config, ckpt_path, tmp_path, use_ema=True
    ).build_model()
    actual_ema_state = _state_dict_cpu(_bare_module(loaded_ema.module))
    assert actual_ema_state.keys() == expected_ema_state.keys()
    for key, expected in expected_ema_state.items():
        torch.testing.assert_close(actual_ema_state[key], expected)

    loaded_raw = _inference_config(
        trainer_config, ckpt_path, tmp_path, use_ema=False
    ).build_model()
    actual_raw_state = _state_dict_cpu(_bare_module(loaded_raw.module))
    assert actual_raw_state.keys() == expected_raw_state.keys()
    for key, expected in expected_raw_state.items():
        torch.testing.assert_close(actual_raw_state[key], expected)


def test_bare_module_unwraps_dummy_wrapper():
    inner = torch.nn.Linear(2, 2)
    assert _bare_module(DummyWrapper(inner)) is inner


def test_bare_module_raises_informative_error_when_unwrapped():
    with pytest.raises(RuntimeError, match="checkpoint loading assumptions"):
        _bare_module(torch.nn.Linear(2, 2))


def test_splice_observed_endpoints_overwrites_only_first_and_last_frame():
    n_ensemble = 3
    generated = {"x": torch.full((1, n_ensemble, 4, 2, 2), -1.0)}
    truth = {"x": torch.arange(4).float().view(1, 4, 1, 1).expand(1, 4, 2, 2)}

    spliced = _splice_observed_endpoints(generated, truth, n_ensemble)

    assert torch.all(spliced["x"][:, :, 0] == 0)
    assert torch.all(spliced["x"][:, :, -1] == 3)
    assert torch.all(spliced["x"][:, :, 1:-1] == -1.0)  # interior left generated


def test_clip_write_slice_middle_clip_excludes_shared_boundary_frame():
    # 19 reference times; a 9-frame clip starting at index 8 is not the last
    # clip (that one starts at 16), so it should write only frames [8, 16)
    # -- not its own trailing endpoint, which the next clip owns as its
    # start frame.
    time = np.arange(19)
    result = _clip_write_slice(8, time, last_clip_start_idx=16, n_timesteps=9)
    assert result == slice(8, 16)


def test_clip_write_slice_last_clip_includes_trailing_endpoint():
    # 17 reference times; a 9-frame clip starting at index 8 is the last
    # clip, so it must also write its own trailing endpoint.
    time = np.arange(17)
    result = _clip_write_slice(8, time, last_clip_start_idx=8, n_timesteps=9)
    assert result == slice(8, 17)


def test_clip_write_slice_raises_on_start_time_mismatch():
    time = np.array(
        [cftime.DatetimeProlepticGregorian(2020, 1, d) for d in range(1, 5)]
    )
    bad_start = cftime.DatetimeProlepticGregorian(2020, 2, 1)
    with pytest.raises(ValueError, match="not found in the reference test time axis"):
        _clip_write_slice(bad_start, time, last_clip_start_idx=0, n_timesteps=2)


def test_warn_if_idle_ranks_warns_when_clips_short_of_ranks(caplog):
    _warn_if_idle_ranks(n_clips=1, world_size=2)
    assert "Only 1 clip" in caplog.text


def test_warn_if_idle_ranks_silent_when_clips_cover_ranks(caplog):
    _warn_if_idle_ranks(n_clips=2, world_size=2)
    _warn_if_idle_ranks(n_clips=3, world_size=2)
    assert caplog.text == ""


def test_validate_contiguous_clips_passes_for_tumbling_starts():
    # 3 clips of 9 frames, tumbling (stride 8) -> 25-frame axis.
    _validate_contiguous_clips(np.array([0, 8, 16]), n_time=25, n_timesteps=9)


def test_validate_contiguous_clips_raises_for_gap():
    # A gap between clip starts (9 instead of 8) would leave frame 8 unwritten.
    with pytest.raises(ValueError, match="do not tile"):
        _validate_contiguous_clips(np.array([0, 9, 17]), n_time=26, n_timesteps=9)


def _valid_inference_kwargs(trainer_config, tmp_path):
    return dict(
        checkpoint_path=str(tmp_path / "unused.ckpt"),
        model=trainer_config.model,
        data=trainer_config.test_data,
        output_path=str(tmp_path / "out.zarr"),
        experiment_dir=str(tmp_path / "exp"),
        logging=LoggingConfig(
            log_to_screen=False, log_to_wandb=False, log_to_file=False
        ),
    )


def test_post_init_rejects_n_timesteps_mismatch(tmp_path):
    trainer_config = _trainer_config(tmp_path)
    kwargs = _valid_inference_kwargs(trainer_config, tmp_path)
    kwargs["data"] = dataclasses.replace(kwargs["data"], n_timesteps=3)
    with pytest.raises(ValueError, match="data.n_timesteps"):
        VideoInferenceConfig(**kwargs)


def test_post_init_rejects_non_tumbling_stride(tmp_path):
    trainer_config = _trainer_config(tmp_path)
    kwargs = _valid_inference_kwargs(trainer_config, tmp_path)
    kwargs["data"] = dataclasses.replace(kwargs["data"], time_stride=1)
    with pytest.raises(ValueError, match="tumbling clips"):
        VideoInferenceConfig(**kwargs)


def test_post_init_rejects_repeated_data(tmp_path):
    trainer_config = _trainer_config(tmp_path)
    kwargs = _valid_inference_kwargs(trainer_config, tmp_path)
    kwargs["data"] = dataclasses.replace(kwargs["data"], repeat=2)
    with pytest.raises(ValueError, match="data.repeat"):
        VideoInferenceConfig(**kwargs)


def test_post_init_rejects_sample_with_replacement(tmp_path):
    trainer_config = _trainer_config(tmp_path)
    kwargs = _valid_inference_kwargs(trainer_config, tmp_path)
    kwargs["data"] = dataclasses.replace(kwargs["data"], sample_with_replacement=4)
    with pytest.raises(ValueError, match="sample_with_replacement"):
        VideoInferenceConfig(**kwargs)


def test_post_init_rejects_non_positive_ensemble_chunk_size(tmp_path):
    trainer_config = _trainer_config(tmp_path)
    kwargs = _valid_inference_kwargs(trainer_config, tmp_path)
    kwargs["ensemble_chunk_size"] = 0
    with pytest.raises(ValueError, match="ensemble_chunk_size"):
        VideoInferenceConfig(**kwargs)


def _write_video_zarr(tmp_path, num_timesteps: int, lat: int = 4, lon: int = 4):
    # Julian (a non-standard CF calendar) keeps xarray decoding this as
    # cftime rather than datetime64, matching the real X-SHiELD data and the
    # rest of the downscaling data-loading pipeline.
    time = [
        cftime.DatetimeJulian(2000, 1, 1) + datetime.timedelta(days=i)
        for i in range(num_timesteps)
    ]
    data_vars: dict[str, object] = {
        name: (
            ("time", "lat", "lon"),
            np.random.randn(num_timesteps, lat, lon).astype(np.float32),
        )
        for name in _OUT_NAMES
    }
    for i in range(7):
        data_vars[f"ak_{i}"] = float(i)
        data_vars[f"bk_{i}"] = float(i + 1)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": time,
            "lat": np.linspace(0.0, 8.0, lat, dtype=np.float32),
            "lon": np.linspace(0.0, 8.0, lon, dtype=np.float32),
        },
    )
    ds.to_zarr(tmp_path / "data.zarr")


@pytest.mark.medium_duration
def test_build_and_run_end_to_end(tmp_path):
    """Exercises ``VideoInferenceConfig.build().run()`` as a whole, since
    unit tests of its helpers don't cover the entrypoint's own wiring
    (dacite-loadable config -> built model/loader -> written zarr store).
    """
    num_timesteps = 9
    n_timesteps = 5
    _write_video_zarr(tmp_path, num_timesteps)

    model_config = VideoDiffusionModelConfig(
        out_names=_OUT_NAMES,
        n_timesteps=n_timesteps,
        normalization=NormalizationConfig(
            means={name: 0.0 for name in _OUT_NAMES},
            stds={name: 1.0 for name in _OUT_NAMES},
        ),
        model_channels=8,
        n_heads=1,
        num_freqs=2,
        num_diffusion_generation_steps=2,
    )
    train_data_config = PairedVideoLoaderConfig(
        fine=[XarrayDataConfig(str(tmp_path), file_pattern="data.zarr", engine="zarr")],
        coarse=[
            XarrayDataConfig(str(tmp_path), file_pattern="data.zarr", engine="zarr")
        ],
        batch_size=2,
        num_data_workers=0,
        strict_ensemble=False,
        n_timesteps=n_timesteps,
    )
    trainer_config = VideoTrainerConfig(
        model=model_config,
        optimization=OptimizationConfig(lr=1e-3),
        train_data=train_data_config,
        validation_data=train_data_config,
        max_epochs=1,
        experiment_dir=str(tmp_path / "train_exp"),
        logging=LoggingConfig(
            log_to_screen=False, log_to_wandb=False, log_to_file=False
        ),
        ema=EMAConfig(decay=0.99),
    )
    # No need to actually train for an inference test -- build() already
    # initializes model/EMA/optimization state.
    trainer = trainer_config.build()
    ckpt_path = str(tmp_path / "e2e.ckpt")
    _save_checkpoint(trainer, ckpt_path)

    inference_data_config = dataclasses.replace(
        train_data_config,
        fine=[
            XarrayDataConfig(
                str(tmp_path),
                file_pattern="data.zarr",
                engine="zarr",
                subset=TimeSlice(),
            )
        ],
    )
    inference_config = VideoInferenceConfig(
        checkpoint_path=ckpt_path,
        model=model_config,
        data=inference_data_config,
        output_path=str(tmp_path / "out.zarr"),
        experiment_dir=str(tmp_path / "inf_exp"),
        logging=LoggingConfig(
            log_to_screen=False, log_to_wandb=False, log_to_file=False
        ),
        n_ensemble=2,
        ensemble_chunk_size=2,
        use_ema=False,
    )
    inference_config.build().run()

    ds = xr.open_zarr(inference_config.output_path)
    assert ds.sizes["ensemble"] == 2
    assert ds.sizes["time"] == num_timesteps
    for name in _OUT_NAMES:
        assert ds[name].shape == (num_timesteps, 2, 4, 4)
    np.testing.assert_array_equal(
        ds["frame_source"].values, np.array([0, 1, 1, 1, 0, 1, 1, 1, 0])
    )

    # Sharded to one clip's full write: (n_timesteps - 1, n_ensemble, *spatial).
    group = zarr.open_group(inference_config.output_path, mode="r")
    assert group[_OUT_NAMES[0]].shards == (n_timesteps - 1, 2, 4, 4)
