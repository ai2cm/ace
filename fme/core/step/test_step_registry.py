import dataclasses
import datetime
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from unittest.mock import MagicMock

import dacite
import pytest
import torch
from torch import nn

from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.ocean import OceanConfig
from fme.core.step.args import StepArgs
from fme.core.step.output import StepOutput
from fme.core.typing_ import TensorDict, TensorMapping

from .step import StepABC, StepConfigABC, StepSelector


class MockStep(StepABC):
    def __init__(
        self,
        config: "MockStepConfig",
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None],
    ):
        self.dataset_info = dataset_info
        self._config = config
        self._init_weights = init_weights

    @property
    def config(self) -> "MockStepConfig":
        return self._config

    @property
    def modules(self):
        raise NotImplementedError()

    @property
    def normalizer(self):
        raise NotImplementedError()

    @property
    def surface_temperature_name(self):
        return None

    @property
    def ocean_fraction_name(self):
        return None

    def prescribe_sst(
        self,
        mask_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> TensorDict:
        raise NotImplementedError()

    def get_regularizer_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def step(
        self,
        args: StepArgs,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> StepOutput:
        raise NotImplementedError()

    def get_state(self):
        return {}

    def load_state(self, state):
        pass


@StepSelector.register("mock")
@dataclasses.dataclass
class MockStepConfig(StepConfigABC):
    in_names: list[str] = dataclasses.field(default_factory=list)
    out_names: list[str] = dataclasses.field(default_factory=list)

    def get_step(
        self, dataset_info: DatasetInfo, init_weights: Callable[[list[nn.Module]], None]
    ):
        return MockStep(self, dataset_info, init_weights)

    @property
    def diagnostic_names(self) -> list[str]:
        return list(set(self.out_names).difference(self.in_names))

    def get_next_step_forcing_names(self) -> list[str]:
        return []

    @property
    def input_names(self) -> frozenset[str]:
        return frozenset(self.in_names)

    @property
    def output_names(self) -> frozenset[str]:
        return frozenset(self.out_names)

    @property
    def next_step_input_names(self) -> frozenset[str]:
        raise NotImplementedError()

    @property
    def loss_names(self) -> list[str]:
        return sorted(self.out_names)

    @property
    def n_ic_timesteps(self) -> int:
        raise NotImplementedError()

    def replace_ocean(self, ocean: OceanConfig | None):
        raise NotImplementedError()

    def get_ocean(self) -> OceanConfig | None:
        return None

    def get_loss_normalizer(
        self,
        extra_names: list[str] | None = None,
        extra_residual_scaled_names: list[str] | None = None,
    ):
        raise NotImplementedError()

    def load(self):
        pass

    def replace_prescribed_prognostic_names(self, names: list[str]) -> None:
        pass

    def get_prescribed_prognostic_names(self) -> list[str]:
        return []

    def disable_corrections(self, names: Sequence[str]) -> None:
        raise NotImplementedError("MockStepConfig has no corrector")

    @property
    def allow_missing_variables(self) -> bool:
        return False

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        return dict(state)


class DeprecatingMockStep(StepABC):
    """Step paired with DeprecatingMockStepConfig for hook tests."""

    def __init__(self, config: "DeprecatingMockStepConfig"):
        self._config = config

    @property
    def config(self) -> "DeprecatingMockStepConfig":
        return self._config

    @property
    def modules(self):
        raise NotImplementedError()

    @property
    def normalizer(self):
        raise NotImplementedError()

    @property
    def surface_temperature_name(self):
        return None

    @property
    def ocean_fraction_name(self):
        return None

    def prescribe_sst(
        self,
        mask_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> TensorDict:
        raise NotImplementedError()

    def get_regularizer_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def step(
        self,
        args: StepArgs,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> StepOutput:
        raise NotImplementedError()

    def get_state(self):
        return {}

    def load_state(self, state):
        pass


@StepSelector.register("deprecating_mock")
@dataclasses.dataclass
class DeprecatingMockStepConfig(StepConfigABC):
    """Mock config that drops ``old_key`` and renames ``old_name`` to ``name``."""

    name: str = ""
    in_names: list[str] = dataclasses.field(default_factory=list)
    out_names: list[str] = dataclasses.field(default_factory=list)

    def disable_corrections(self, names: Sequence[str]) -> None:
        pass

    def get_step(
        self, dataset_info: DatasetInfo, init_weights: Callable[[list[nn.Module]], None]
    ):
        return DeprecatingMockStep(self)

    @property
    def diagnostic_names(self) -> list[str]:
        return list(set(self.out_names).difference(self.in_names))

    def get_next_step_forcing_names(self) -> list[str]:
        return []

    @property
    def input_names(self) -> frozenset[str]:
        return frozenset(self.in_names)

    @property
    def output_names(self) -> frozenset[str]:
        return frozenset(self.out_names)

    @property
    def next_step_input_names(self) -> frozenset[str]:
        raise NotImplementedError()

    @property
    def loss_names(self) -> list[str]:
        return sorted(self.out_names)

    @property
    def n_ic_timesteps(self) -> int:
        raise NotImplementedError()

    def replace_ocean(self, ocean: OceanConfig | None):
        raise NotImplementedError()

    def get_ocean(self) -> OceanConfig | None:
        return None

    def get_loss_normalizer(
        self,
        extra_names: list[str] | None = None,
        extra_residual_scaled_names: list[str] | None = None,
    ):
        raise NotImplementedError()

    def load(self):
        pass

    def replace_prescribed_prognostic_names(self, names: list[str]) -> None:
        pass

    def get_prescribed_prognostic_names(self) -> list[str]:
        return []

    @property
    def allow_missing_variables(self) -> bool:
        return False

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(state)
        result.pop("old_key", None)
        if "old_name" in result:
            result["name"] = result.pop("old_name")
        return result


def test_register():
    """Make sure that the registry is working as expected."""
    selector = StepSelector(type="mock", config={})
    img_shape = (16, 32)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.zeros(img_shape[0]), lon=torch.zeros(img_shape[1])
    )
    timestep = datetime.timedelta(hours=6)
    dataset_info = DatasetInfo(
        horizontal_coordinates=horizontal_coordinate,
        vertical_coordinate=vertical_coordinate,
        timestep=timestep,
    )
    init_weights = MagicMock()
    step = selector.get_step(dataset_info, init_weights)
    assert isinstance(step, MockStep)
    assert step.dataset_info == dataset_info
    assert step._init_weights == init_weights


def test_remove_deprecated_keys_drops_and_renames():
    """Hook should drop ``old_key`` and rename ``old_name`` to ``name``."""
    selector = StepSelector(
        type="deprecating_mock",
        config={
            "old_key": "should_be_dropped",
            "old_name": "renamed_value",
            "in_names": ["a"],
            "out_names": ["b"],
        },
    )
    step = selector._step_config_instance
    assert isinstance(step, DeprecatingMockStepConfig)
    assert step.name == "renamed_value"
    assert step.in_names == ["a"]
    assert step.out_names == ["b"]


def test_remove_deprecated_keys_unknown_key_raises():
    """Strict dacite loading should reject genuinely unknown keys."""

    with pytest.raises(dacite.UnexpectedDataError):
        StepSelector(
            type="deprecating_mock",
            config={"totally_unknown_key": 42},
        )


def test_remove_deprecated_keys_does_not_mutate_input():
    """The hook must not mutate the original mapping."""
    original = {"old_key": "x", "old_name": "y", "in_names": [], "out_names": []}
    original_copy = dict(original)
    StepSelector(type="deprecating_mock", config=original)
    assert original == original_copy


def test_single_module_crps_training_deprecated_key():
    """SingleModuleStepConfig.remove_deprecated_keys strips crps_training."""
    from fme.core.step.single_module import SingleModuleStepConfig

    state = {"crps_training": True, "other": "value"}
    result = SingleModuleStepConfig.remove_deprecated_keys(state)
    assert "crps_training" not in result
    assert result["other"] == "value"
    # original not mutated
    assert "crps_training" in state
