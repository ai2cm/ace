import dataclasses

import pytest
import torch

import fme
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry import ModuleSelector, SeparatedModuleSelector
from fme.core.step.args import StepArgs
from fme.core.step.separated_module import SeparatedModuleStepConfig
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector

IMG_SHAPE = (16, 32)
TIMESTEP = __import__("datetime").timedelta(hours=6)


def _get_dataset_info():
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


def _get_normalization(names):
    return NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={name: 0.0 for name in names},
            stds={name: 1.0 for name in names},
        ),
    )


def _get_tensor_dict(names, n_samples=2):
    device = fme.get_device()
    return {name: torch.rand(n_samples, *IMG_SHAPE, device=device) for name in names}


class TestSeparatedModuleStepConfig:
    def test_duplicate_names_raises(self):
        normalization = _get_normalization(["a", "b"])
        with pytest.raises(ValueError, match="appears in both"):
            SeparatedModuleStepConfig(
                builder=SeparatedModuleSelector(
                    type="legacy",
                    config={
                        "legacy_builder": {"type": "MLP", "config": {}},
                    },
                ),
                forcing_names=["a"],
                prognostic_names=["a"],
                diagnostic_names=["b"],
                normalization=normalization,
            )

    def test_prescribed_prognostic_not_in_prognostic_raises(self):
        normalization = _get_normalization(["f", "p", "d"])
        with pytest.raises(ValueError, match="prescribed_prognostic_name"):
            SeparatedModuleStepConfig(
                builder=SeparatedModuleSelector(
                    type="legacy",
                    config={
                        "legacy_builder": {"type": "MLP", "config": {}},
                    },
                ),
                forcing_names=["f"],
                prognostic_names=["p"],
                diagnostic_names=["d"],
                normalization=normalization,
                prescribed_prognostic_names=["d"],
            )

    def test_next_step_forcing_not_in_forcing_raises(self):
        normalization = _get_normalization(["f", "p", "d"])
        with pytest.raises(ValueError, match="next_step_forcing_name"):
            SeparatedModuleStepConfig(
                builder=SeparatedModuleSelector(
                    type="legacy",
                    config={
                        "legacy_builder": {"type": "MLP", "config": {}},
                    },
                ),
                forcing_names=["f"],
                prognostic_names=["p"],
                diagnostic_names=["d"],
                normalization=normalization,
                next_step_forcing_names=["p"],
            )

    def test_empty_prognostic_names_raises(self):
        normalization = _get_normalization(["f", "d"])
        with pytest.raises(ValueError, match="prognostic_names must not be empty"):
            SeparatedModuleStepConfig(
                builder=SeparatedModuleSelector(
                    type="legacy",
                    config={
                        "legacy_builder": {"type": "MLP", "config": {}},
                    },
                ),
                forcing_names=["f"],
                prognostic_names=[],
                diagnostic_names=["d"],
                normalization=normalization,
            )

    def test_input_output_names(self):
        normalization = _get_normalization(["f1", "f2", "p1", "p2", "d1"])
        config = SeparatedModuleStepConfig(
            builder=SeparatedModuleSelector(
                type="legacy",
                config={
                    "legacy_builder": {"type": "MLP", "config": {}},
                },
            ),
            forcing_names=["f1", "f2"],
            prognostic_names=["p1", "p2"],
            diagnostic_names=["d1"],
            normalization=normalization,
        )
        assert set(config.input_names) == {"f1", "f2", "p1", "p2"}
        assert set(config.output_names) == {"p1", "p2", "d1"}
        assert set(config.get_prognostic_names()) == {"p1", "p2"}

    def test_from_state_roundtrip(self):
        normalization = _get_normalization(["f", "p", "d"])
        config = SeparatedModuleStepConfig(
            builder=SeparatedModuleSelector(
                type="legacy",
                config={
                    "legacy_builder": {"type": "MLP", "config": {}},
                },
            ),
            forcing_names=["f"],
            prognostic_names=["p"],
            diagnostic_names=["d"],
            normalization=normalization,
        )
        state = config.get_state()
        config2 = SeparatedModuleStepConfig.from_state(state)
        assert config2.forcing_names == config.forcing_names
        assert config2.prognostic_names == config.prognostic_names
        assert config2.diagnostic_names == config.diagnostic_names


class TestSeparatedModuleStep:
    def test_step_produces_output(self):
        forcing_names = ["f1", "f2"]
        prognostic_names = ["p1", "p2"]
        diagnostic_names = ["d1"]
        all_names = forcing_names + prognostic_names + diagnostic_names
        normalization = _get_normalization(all_names)

        selector = StepSelector(
            type="separated_module",
            config=dataclasses.asdict(
                SeparatedModuleStepConfig(
                    builder=SeparatedModuleSelector(
                        type="legacy",
                        config={
                            "legacy_builder": {
                                "type": "SphericalFourierNeuralOperatorNet",
                                "config": {
                                    "scale_factor": 1,
                                    "embed_dim": 4,
                                    "num_layers": 2,
                                },
                            },
                        },
                    ),
                    forcing_names=forcing_names,
                    prognostic_names=prognostic_names,
                    diagnostic_names=diagnostic_names,
                    normalization=normalization,
                ),
            ),
        )

        step = selector.get_step(_get_dataset_info())
        input_data = _get_tensor_dict(step.input_names)
        next_step_data = _get_tensor_dict(step.next_step_input_names)

        output = step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step_data,
                labels=None,
            )
        )

        for name in prognostic_names + diagnostic_names:
            assert name in output
            assert output[name].shape == (2, *IMG_SHAPE)

    def test_get_state_and_load_state(self):
        forcing_names = ["f1"]
        prognostic_names = ["p1"]
        diagnostic_names = ["d1"]
        all_names = forcing_names + prognostic_names + diagnostic_names
        normalization = _get_normalization(all_names)

        config = SeparatedModuleStepConfig(
            builder=SeparatedModuleSelector(
                type="legacy",
                config={
                    "legacy_builder": {
                        "type": "SphericalFourierNeuralOperatorNet",
                        "config": {
                            "scale_factor": 1,
                            "embed_dim": 4,
                            "num_layers": 2,
                        },
                    },
                },
            ),
            forcing_names=forcing_names,
            prognostic_names=prognostic_names,
            diagnostic_names=diagnostic_names,
            normalization=normalization,
        )

        dataset_info = _get_dataset_info()
        step1 = config.get_step(dataset_info, lambda _: None)
        state = step1.get_state()

        step2 = config.get_step(dataset_info, lambda _: None)
        step2.load_state(state)

        # Verify weights match
        for p1, p2 in zip(
            step1.module.torch_module.parameters(),
            step2.module.torch_module.parameters(),
        ):
            assert torch.equal(p1, p2)

    def test_residual_prediction(self):
        forcing_names = ["f1"]
        prognostic_names = ["p1"]
        diagnostic_names = ["d1"]
        all_names = forcing_names + prognostic_names + diagnostic_names
        normalization = _get_normalization(all_names)

        config = SeparatedModuleStepConfig(
            builder=SeparatedModuleSelector(
                type="legacy",
                config={
                    "legacy_builder": {
                        "type": "SphericalFourierNeuralOperatorNet",
                        "config": {
                            "scale_factor": 1,
                            "embed_dim": 4,
                            "num_layers": 2,
                        },
                    },
                },
            ),
            forcing_names=forcing_names,
            prognostic_names=prognostic_names,
            diagnostic_names=diagnostic_names,
            normalization=normalization,
            residual_prediction=True,
        )

        dataset_info = _get_dataset_info()
        step = config.get_step(dataset_info, lambda _: None)
        input_data = _get_tensor_dict(step.input_names)
        next_step_data = _get_tensor_dict(step.next_step_input_names)

        output = step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step_data,
                labels=None,
            )
        )

        assert "p1" in output
        assert "d1" in output

    def test_step_no_forcings(self):
        prognostic_names = ["p1", "p2"]
        diagnostic_names = ["d1"]
        all_names = prognostic_names + diagnostic_names
        normalization = _get_normalization(all_names)

        config = SeparatedModuleStepConfig(
            builder=SeparatedModuleSelector(
                type="legacy",
                config={
                    "legacy_builder": {
                        "type": "SphericalFourierNeuralOperatorNet",
                        "config": {
                            "scale_factor": 1,
                            "embed_dim": 4,
                            "num_layers": 2,
                        },
                    },
                },
            ),
            forcing_names=[],
            prognostic_names=prognostic_names,
            diagnostic_names=diagnostic_names,
            normalization=normalization,
        )

        dataset_info = _get_dataset_info()
        step = config.get_step(dataset_info, lambda _: None)
        input_data = _get_tensor_dict(step.input_names)
        next_step_data = _get_tensor_dict(step.next_step_input_names)

        output = step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step_data,
                labels=None,
            )
        )

        for name in prognostic_names + diagnostic_names:
            assert name in output
            assert output[name].shape == (2, *IMG_SHAPE)

    def test_step_no_diagnostics(self):
        forcing_names = ["f1"]
        prognostic_names = ["p1", "p2"]
        all_names = forcing_names + prognostic_names
        normalization = _get_normalization(all_names)

        config = SeparatedModuleStepConfig(
            builder=SeparatedModuleSelector(
                type="legacy",
                config={
                    "legacy_builder": {
                        "type": "SphericalFourierNeuralOperatorNet",
                        "config": {
                            "scale_factor": 1,
                            "embed_dim": 4,
                            "num_layers": 2,
                        },
                    },
                },
            ),
            forcing_names=forcing_names,
            prognostic_names=prognostic_names,
            diagnostic_names=[],
            normalization=normalization,
        )

        dataset_info = _get_dataset_info()
        step = config.get_step(dataset_info, lambda _: None)
        input_data = _get_tensor_dict(step.input_names)
        next_step_data = _get_tensor_dict(step.next_step_input_names)

        output = step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step_data,
                labels=None,
            )
        )

        for name in prognostic_names:
            assert name in output
            assert output[name].shape == (2, *IMG_SHAPE)
        assert len(output) == len(prognostic_names)

    def test_step_no_forcings_no_diagnostics(self):
        prognostic_names = ["p1", "p2"]
        normalization = _get_normalization(prognostic_names)

        config = SeparatedModuleStepConfig(
            builder=SeparatedModuleSelector(
                type="legacy",
                config={
                    "legacy_builder": {
                        "type": "SphericalFourierNeuralOperatorNet",
                        "config": {
                            "scale_factor": 1,
                            "embed_dim": 4,
                            "num_layers": 2,
                        },
                    },
                },
            ),
            forcing_names=[],
            prognostic_names=prognostic_names,
            diagnostic_names=[],
            normalization=normalization,
        )

        dataset_info = _get_dataset_info()
        step = config.get_step(dataset_info, lambda _: None)
        input_data = _get_tensor_dict(step.input_names)
        next_step_data = _get_tensor_dict(step.next_step_input_names)

        output = step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step_data,
                labels=None,
            )
        )

        for name in prognostic_names:
            assert name in output
            assert output[name].shape == (2, *IMG_SHAPE)
        assert len(output) == len(prognostic_names)


class TestSeparatedVsSingleModuleEquivalence:
    """Verify that SeparatedModuleStep with legacy adapter produces the same
    output as SingleModuleStep when given the same weights and input."""

    def _build_equivalent_steps(self, residual_prediction=False):
        """Build a SingleModuleStep and SeparatedModuleStep with the same
        architecture and weights."""
        forcing_names = ["f1", "f2"]
        prognostic_names = ["p1", "p2"]
        diagnostic_names = ["d1"]
        in_names = forcing_names + prognostic_names
        out_names = prognostic_names + diagnostic_names
        all_names = forcing_names + prognostic_names + diagnostic_names
        normalization = _get_normalization(all_names)

        sfno_config = {
            "scale_factor": 1,
            "embed_dim": 4,
            "num_layers": 2,
        }

        single_config = SingleModuleStepConfig(
            builder=ModuleSelector(
                type="SphericalFourierNeuralOperatorNet",
                config=sfno_config,
            ),
            in_names=in_names,
            out_names=out_names,
            normalization=normalization,
            residual_prediction=residual_prediction,
        )

        separated_config = SeparatedModuleStepConfig(
            builder=SeparatedModuleSelector(
                type="legacy",
                config={
                    "legacy_builder": {
                        "type": "SphericalFourierNeuralOperatorNet",
                        "config": sfno_config,
                    },
                },
            ),
            forcing_names=forcing_names,
            prognostic_names=prognostic_names,
            diagnostic_names=diagnostic_names,
            normalization=normalization,
            residual_prediction=residual_prediction,
        )

        dataset_info = _get_dataset_info()

        torch.manual_seed(42)
        single_step = single_config.get_step(dataset_info, lambda _: None)

        torch.manual_seed(42)
        separated_step = separated_config.get_step(dataset_info, lambda _: None)

        # Copy weights from single to separated at the inner module level.
        # SingleModuleStep wraps as DummyWrapper(SFNO) while
        # SeparatedModuleStep wraps as DummyWrapper(LegacyWrapper(SFNO)),
        # so we copy at the SFNO level to avoid key prefix mismatches.
        single_sfno = single_step.module.torch_module.module
        separated_sfno = separated_step.module.torch_module.module.inner
        separated_sfno.load_state_dict(single_sfno.state_dict())

        return single_step, separated_step, in_names

    def test_equivalence(self):
        torch.manual_seed(0)
        single_step, separated_step, in_names = self._build_equivalent_steps()

        input_data = _get_tensor_dict(in_names)
        next_step_data_single = _get_tensor_dict(single_step.next_step_input_names)
        next_step_data_separated = _get_tensor_dict(
            separated_step.next_step_input_names
        )

        single_output = single_step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step_data_single,
                labels=None,
            )
        )
        separated_output = separated_step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step_data_separated,
                labels=None,
            )
        )

        for name in single_output:
            assert name in separated_output, f"Missing output: {name}"
            torch.testing.assert_close(
                single_output[name],
                separated_output[name],
                msg=f"Output mismatch for {name}",
            )

    def test_equivalence_with_residual_prediction(self):
        torch.manual_seed(0)
        single_step, separated_step, in_names = self._build_equivalent_steps(
            residual_prediction=True
        )

        input_data = _get_tensor_dict(in_names)
        next_step_data_single = _get_tensor_dict(single_step.next_step_input_names)
        next_step_data_separated = _get_tensor_dict(
            separated_step.next_step_input_names
        )

        single_output = single_step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step_data_single,
                labels=None,
            )
        )
        separated_output = separated_step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step_data_separated,
                labels=None,
            )
        )

        for name in single_output:
            assert name in separated_output, f"Missing output: {name}"
            torch.testing.assert_close(
                single_output[name],
                separated_output[name],
                msg=f"Output mismatch for {name}",
            )
