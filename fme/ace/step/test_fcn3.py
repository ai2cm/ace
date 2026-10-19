import logging
from unittest import mock

import pytest

import fme
from fme.ace.step.fcn3 import FCN3Config, FCN3Selector, FCN3Step, FCN3StepConfig
from fme.core.compile import compile_enabled
from fme.core.testing import get_dataset_info, trivial_network_and_loss_normalization

IMG_SHAPE = (16, 32)
FORCING_NAMES = ["DSWRFtoa"]
ATMOSPHERE_PROGNOSTIC_NAMES = ["air_temperature"]
ATMOSPHERE_DIAGNOSTIC_NAMES = ["radiative_heating"]
ATMOSPHERE_LEVELS = 2
SURFACE_PROGNOSTIC_NAMES = ["PRESsfc"]
SURFACE_DIAGNOSTIC_NAMES = ["PRATEsfc"]


def _get_fcn3_step_config() -> FCN3StepConfig:
    """A minimal FCN3 step config."""
    packed_names = [
        f"{name}_{i}"
        for i in range(ATMOSPHERE_LEVELS)
        for name in ATMOSPHERE_PROGNOSTIC_NAMES + ATMOSPHERE_DIAGNOSTIC_NAMES
    ]
    return FCN3StepConfig(
        builder=FCN3Selector(
            type="FCN3",
            config=FCN3Config(
                scale_factor=1,
                atmo_embed_dim=2,
                surf_embed_dim=2,
                aux_embed_dim=2,
                num_layers=2,
            ),
        ),
        forcing_names=FORCING_NAMES,
        atmosphere_prognostic_names=ATMOSPHERE_PROGNOSTIC_NAMES,
        atmosphere_diagnostic_names=ATMOSPHERE_DIAGNOSTIC_NAMES,
        atmosphere_levels=ATMOSPHERE_LEVELS,
        surface_prognostic_names=SURFACE_PROGNOSTIC_NAMES,
        surface_diagnostic_names=SURFACE_DIAGNOSTIC_NAMES,
        normalization=trivial_network_and_loss_normalization(
            sorted(
                set(FORCING_NAMES)
                .union(packed_names)
                .union(SURFACE_PROGNOSTIC_NAMES)
                .union(SURFACE_DIAGNOSTIC_NAMES)
            )
        ),
    )


def _get_step(config: FCN3StepConfig) -> FCN3Step:
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE, device=fme.get_device())
    return config.get_step(dataset_info, lambda _: None)


def test_fcn3_step_runs_eagerly_and_logs_reason(caplog: pytest.LogCaptureFixture):
    with (
        compile_enabled(True),
        mock.patch("torch.compile") as mock_compile,
        caplog.at_level(logging.INFO),
    ):
        _get_step(_get_fcn3_step_config())
    mock_compile.assert_not_called()
    assert "torch.compile skipped, FCN3Step runs eagerly" in caplog.text
