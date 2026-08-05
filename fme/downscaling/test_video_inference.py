import pytest
import torch

from fme.core.device import get_device
from fme.downscaling.test_video_train import _trainer_config
from fme.downscaling.video_inference import load_video_model
from fme.downscaling.video_train import _save_checkpoint


def _state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _bare_module(model) -> torch.nn.Module:
    return getattr(model.module, "module", model.module)


@pytest.mark.medium_duration
def test_load_video_model_round_trip(tmp_path):
    """``video_inference.load_video_model`` is the actual production
    checkpoint->inference path (distinct from, and previously untested by,
    ``VideoTrainer``'s own resume mechanism). It hand-parses the checkpoint's
    ``module``/``ema`` dict and strips a ``module.`` prefix -- this proves
    that round trip preserves weights exactly, for both EMA and raw loading.

    A checkpoint is saved explicitly (rather than reusing ``best.ckpt``) right
    after capturing the expected state, so the comparison isn't confounded by
    ``best.ckpt`` reflecting an earlier epoch than the trainer's final state.
    """
    config = _trainer_config(tmp_path)
    trainer = config.build()
    trainer.train()

    with trainer._ema_context():
        expected_ema_state = _state_dict_cpu(_bare_module(trainer.model))
    expected_raw_state = _state_dict_cpu(_bare_module(trainer.model))

    ckpt_path = str(tmp_path / "round_trip.ckpt")
    _save_checkpoint(trainer, ckpt_path)

    device = get_device()
    loaded_ema = load_video_model(config.model, ckpt_path, device, use_ema=True)
    actual_ema_state = _state_dict_cpu(_bare_module(loaded_ema))
    assert actual_ema_state.keys() == expected_ema_state.keys()
    for key, expected in expected_ema_state.items():
        torch.testing.assert_close(actual_ema_state[key], expected)

    loaded_raw = load_video_model(config.model, ckpt_path, device, use_ema=False)
    actual_raw_state = _state_dict_cpu(_bare_module(loaded_raw))
    assert actual_raw_state.keys() == expected_raw_state.keys()
    for key, expected in expected_raw_state.items():
        torch.testing.assert_close(actual_raw_state[key], expected)
