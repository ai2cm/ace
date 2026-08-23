import cftime
import numpy as np
import pytest
import torch

from fme.core.device import get_device
from fme.core.distributed.non_distributed import DummyWrapper
from fme.core.logging_utils import LoggingConfig
from fme.downscaling.test_video_train import _trainer_config
from fme.downscaling.video_inference import (
    VideoInferenceConfig,
    _bare_module,
    _clip_write_slice,
    _splice_observed_endpoints,
)
from fme.downscaling.video_train import _save_checkpoint


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

    device = get_device()
    loaded_ema = _inference_config(
        trainer_config, ckpt_path, tmp_path, use_ema=True
    ).build_model(device)
    actual_ema_state = _state_dict_cpu(_bare_module(loaded_ema.module))
    assert actual_ema_state.keys() == expected_ema_state.keys()
    for key, expected in expected_ema_state.items():
        torch.testing.assert_close(actual_ema_state[key], expected)

    loaded_raw = _inference_config(
        trainer_config, ckpt_path, tmp_path, use_ema=False
    ).build_model(device)
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
    # (16 would be), so it should write only frames [8, 16) -- not its own
    # trailing endpoint, which the next clip owns as its start frame.
    time = np.arange(19)
    result = _clip_write_slice(8, time, n_timesteps=9)
    assert result == slice(8, 16)


def test_clip_write_slice_last_clip_includes_trailing_endpoint():
    # 17 reference times; a 9-frame clip starting at index 8 ends at 16,
    # i.e. the final reference time, so it must also write that endpoint.
    time = np.arange(17)
    result = _clip_write_slice(8, time, n_timesteps=9)
    assert result == slice(8, 17)


def test_clip_write_slice_raises_on_start_time_mismatch():
    time = np.array(
        [cftime.DatetimeProlepticGregorian(2020, 1, d) for d in range(1, 5)]
    )
    bad_start = cftime.DatetimeProlepticGregorian(2020, 2, 1)
    with pytest.raises(ValueError, match="not found in the reference test time axis"):
        _clip_write_slice(bad_start, time, n_timesteps=2)
