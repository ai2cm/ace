import dataclasses
import datetime

import pytest
import torch
from torch import nn

import fme
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.step.args import StepArgs
from fme.core.step.infill_prediction import (
    InferenceSchemeConfig,
    InfillPredictionStep,
    InfillPredictionStepConfig,
)
from fme.core.step.step import StepSelector

IMG_SHAPE = (16, 32)
TIMESTEP = datetime.timedelta(hours=6)
STEP_ALL_NAMES = ["a", "b", "c", "forcing_x"]
STEP_FORCING = ["forcing_x"]
STEP_NON_FORCING = ["a", "b", "c"]


def _norm_config(names: list[str]) -> NetworkAndLossNormalizationConfig:
    return NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={n: 0.0 for n in names},
            stds={n: 1.0 for n in names},
        ),
    )


def _make_step_config(
    all_names: list[str] | None = None,
    forcing_names: list[str] | None = None,
    in_names: list[str] | None = None,
    out_names: list[str] | None = None,
) -> InfillPredictionStepConfig:
    if all_names is None:
        all_names = STEP_ALL_NAMES
    if forcing_names is None:
        forcing_names = STEP_FORCING
    if in_names is None:
        in_names = ["a", "b", "forcing_x"]
    if out_names is None:
        out_names = ["a", "b"]
    return InfillPredictionStepConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet",
            config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
        ),
        all_names=all_names,
        forcing_names=forcing_names,
        normalization=_norm_config(all_names),
        inference_scheme=InferenceSchemeConfig(
            in_names=in_names,
            out_names=out_names,
        ),
    )


def _make_dataset_info() -> DatasetInfo:
    device = fme.get_device()
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(IMG_SHAPE[0], device=device),
            lon=torch.zeros(IMG_SHAPE[1], device=device),
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(7, device=device),
            bk=torch.arange(7, device=device),
        ),
        timestep=TIMESTEP,
    )


def _make_step(
    config: InfillPredictionStepConfig | None = None,
) -> InfillPredictionStep:
    if config is None:
        config = _make_step_config()
    dataset_info = _make_dataset_info()
    return config.get_step(dataset_info, lambda _: None)


def _tensor_dict(names: list[str], batch: int = 2) -> dict[str, torch.Tensor]:
    device = fme.get_device()
    return {n: torch.randn(batch, *IMG_SHAPE, device=device) for n in names}


class TestInfillPredictionStepConfigValidation:
    def test_valid_construction(self):
        config = _make_step_config()
        assert config.all_names == STEP_ALL_NAMES

    def test_in_name_not_in_all_names_raises(self):
        with pytest.raises(ValueError, match="inference_scheme.in_names"):
            _make_step_config(in_names=["a", "b", "not_there"])

    def test_out_name_not_in_all_names_raises(self):
        with pytest.raises(ValueError, match="inference_scheme.out_names"):
            _make_step_config(out_names=["a", "not_there"])

    def test_forcing_not_in_all_names_raises(self):
        with pytest.raises(ValueError, match="forcing_names"):
            _make_step_config(forcing_names=["not_there"])

    def test_forcing_in_out_names_raises(self):
        with pytest.raises(ValueError, match="forcing variable"):
            _make_step_config(
                all_names=["a", "b", "c", "f"],
                forcing_names=["f"],
                in_names=["a", "f"],
                out_names=["a", "f"],
            )

    def test_include_channel_mask_inputs_false_raises(self):
        with pytest.raises(ValueError, match="include_channel_mask_inputs"):
            InfillPredictionStepConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet",
                    config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
                ),
                all_names=["a", "b"],
                forcing_names=[],
                normalization=_norm_config(["a", "b"]),
                inference_scheme=InferenceSchemeConfig(in_names=["a"], out_names=["a"]),
                include_channel_mask_inputs=False,
            )


class TestInfillPredictionStepConfigProperties:
    @pytest.fixture()
    def config(self):
        return _make_step_config()

    def test_input_names(self, config):
        assert set(config.input_names) == {"a", "b", "forcing_x"}

    def test_output_names(self, config):
        assert set(config.output_names) == {"a", "b"}

    def test_loss_names(self, config):
        assert set(config.loss_names) == set(STEP_NON_FORCING)

    def test_all_training_names(self, config):
        assert config.all_training_names == STEP_ALL_NAMES

    def test_allow_missing_variables(self, config):
        assert config.allow_missing_variables is True

    def test_n_ic_timesteps(self, config):
        assert config.n_ic_timesteps == 1

    def test_non_forcing_names(self, config):
        assert config.non_forcing_names == STEP_NON_FORCING

    def test_next_step_forcing_names(self, config):
        assert config.get_next_step_forcing_names() == []

    def test_next_step_forcing_names_with_config(self):
        config = InfillPredictionStepConfig(
            builder=ModuleSelector(
                type="SphericalFourierNeuralOperatorNet",
                config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
            ),
            all_names=["a", "b", "forcing_x"],
            forcing_names=["forcing_x"],
            normalization=_norm_config(["a", "b", "forcing_x"]),
            inference_scheme=InferenceSchemeConfig(
                in_names=["a", "b", "forcing_x"],
                out_names=["a", "b"],
                next_step_forcing_names=["forcing_x"],
            ),
        )
        assert config.get_next_step_forcing_names() == ["forcing_x"]

    def test_prognostic_names(self, config):
        assert set(config.prognostic_names) == {"a", "b"}


class TestInfillPredictionStepRegistry:
    def test_selector_construction(self):
        config = _make_step_config()
        selector = StepSelector(
            type="infill_prediction",
            config=dataclasses.asdict(config),
        )
        assert selector.all_training_names == STEP_ALL_NAMES
        assert set(selector.input_names) == {"a", "b", "forcing_x"}
        assert set(selector.output_names) == {"a", "b"}

    def test_selector_get_step(self):
        config = _make_step_config()
        selector = StepSelector(
            type="infill_prediction",
            config=dataclasses.asdict(config),
        )
        dataset_info = _make_dataset_info()
        step = selector.get_step(dataset_info)
        assert isinstance(step, InfillPredictionStep)


class TestInfillPredictionStep:
    def test_forward_with_full_input(self):
        step = _make_step()
        input_data = _tensor_dict(STEP_ALL_NAMES)
        next_step = _tensor_dict(step.next_step_input_names)
        data_mask = {
            n: torch.ones(2, dtype=torch.bool, device=fme.get_device())
            for n in STEP_ALL_NAMES
        }
        output, _ = step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step,
                data_mask=data_mask,
            )
        )
        assert set(output.keys()) == set(STEP_NON_FORCING)
        for v in output.values():
            assert v.shape == (2, *IMG_SHAPE)

    def test_forward_with_partial_input(self):
        step = _make_step()
        inference_names = ["a", "b", "forcing_x"]
        input_data = _tensor_dict(inference_names)
        next_step = _tensor_dict(step.next_step_input_names)
        output, _ = step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step,
            )
        )
        assert set(output.keys()) == set(STEP_NON_FORCING)
        for v in output.values():
            assert v.shape == (2, *IMG_SHAPE)

    def test_output_excludes_forcing(self):
        step = _make_step()
        input_data = _tensor_dict(STEP_ALL_NAMES)
        next_step = _tensor_dict(step.next_step_input_names)
        output, _ = step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step,
            )
        )
        for name in STEP_FORCING:
            assert name not in output

    def test_modules_list(self):
        step = _make_step()
        assert isinstance(step.modules, nn.ModuleList)
        assert len(step.modules) >= 1

    def test_get_and_load_state(self):
        step = _make_step()
        state = step.get_state()
        assert "module" in state
        step.load_state(state)
