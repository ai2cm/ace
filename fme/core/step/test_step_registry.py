import dataclasses
import datetime
from typing import Optional, Tuple

import torch

from fme.core.coordinates import HybridSigmaPressureCoordinate, VerticalCoordinate
from fme.core.gridded_ops import GriddedOperations, LatLonOperations
from fme.core.ocean import OceanConfig

from .step import InferenceDataProtocol, StepABC, StepConfigABC, StepSelector


class MockStep(StepABC):
    def __init__(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: VerticalCoordinate,
        timestep: datetime.timedelta,
    ):
        self.img_shape = img_shape
        self.vertical_coordinate = vertical_coordinate
        self.gridded_operations = gridded_operations
        self.timestep = timestep

    @property
    def prognostic_names(self):
        raise NotImplementedError()

    @property
    def diagnostic_names(self):
        raise NotImplementedError()

    @property
    def forcing_names(self):
        raise NotImplementedError()

    @property
    def output_names(self):
        raise NotImplementedError()

    @property
    def loss_names(self):
        raise NotImplementedError()

    @property
    def next_step_input_names(self):
        raise NotImplementedError()

    @property
    def next_step_forcing_names(self):
        raise NotImplementedError()

    @property
    def n_ic_timesteps(self):
        raise NotImplementedError()

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

    def replace_ocean(self, ocean: Optional[OceanConfig]):
        raise NotImplementedError()

    def validate_inference_data(self, data: InferenceDataProtocol):
        raise NotImplementedError()

    def step(self, input, next_step_input_data, use_activation_checkpointing=False):
        raise NotImplementedError()

    def get_state(self):
        return {}

    def load_state(self, state):
        pass


@StepSelector.register("mock")
@dataclasses.dataclass
class MockStepConfig(StepConfigABC):
    def get_step(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: VerticalCoordinate,
        timestep: datetime.timedelta,
    ):
        return MockStep(img_shape, gridded_operations, vertical_coordinate, timestep)


def test_register():
    """Make sure that the registry is working as expected."""
    selector = StepSelector(type="mock", config={})
    img_shape = (16, 32)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    gridded_operations = LatLonOperations(area_weights=torch.ones(img_shape))
    timestep = datetime.timedelta(hours=6)
    step = selector.get_step(
        img_shape, gridded_operations, vertical_coordinate, timestep
    )
    assert isinstance(step, MockStep)
    assert step.img_shape == img_shape
    assert step.vertical_coordinate == vertical_coordinate
    assert step.gridded_operations == gridded_operations
    assert step.timestep == timestep
