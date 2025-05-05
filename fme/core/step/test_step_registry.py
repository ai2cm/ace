import dataclasses
import datetime

import torch

from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import LatLonOperations
from fme.core.ocean import OceanConfig

from .step import StepABC, StepConfigABC, StepSelector


class MockStep(StepABC):
    def __init__(
        self,
        config: "MockStepConfig",
        dataset_info: DatasetInfo,
    ):
        self.dataset_info = dataset_info
        self._config = config

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

    def get_regularizer_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def step(self, input, next_step_input_data, use_activation_checkpointing=False):
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

    def get_step(self, dataset_info: DatasetInfo):
        return MockStep(self, dataset_info)

    @property
    def diagnostic_names(self) -> list[str]:
        return list(set(self.out_names).difference(self.in_names))

    def get_next_step_forcing_names(self) -> list[str]:
        return []

    @property
    def input_names(self) -> list[str]:
        return self.in_names

    @property
    def output_names(self) -> list[str]:
        return self.out_names

    @property
    def next_step_input_names(self):
        raise NotImplementedError()

    @property
    def loss_names(self) -> list[str]:
        return self.out_names

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


def test_register():
    """Make sure that the registry is working as expected."""
    selector = StepSelector(type="mock", config={})
    img_shape = (16, 32)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    gridded_operations = LatLonOperations(area_weights=torch.ones(img_shape))
    timestep = datetime.timedelta(hours=6)
    dataset_info = DatasetInfo(
        img_shape=img_shape,
        gridded_operations=gridded_operations,
        vertical_coordinate=vertical_coordinate,
        timestep=timestep,
    )
    step = selector.get_step(dataset_info)
    assert isinstance(step, MockStep)
    assert step.dataset_info == dataset_info
