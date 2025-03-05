import dataclasses
import datetime
from typing import Tuple

import torch

from fme.core.coordinates import HybridSigmaPressureCoordinate, VerticalCoordinate
from fme.core.gridded_ops import GriddedOperations, LatLonOperations

from .step import StepABC, StepConfigABC, StepSelector


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
    def input_names(self):
        raise NotImplementedError()

    @property
    def output_names(self):
        raise NotImplementedError()

    @property
    def next_step_forcing_names(self):
        raise NotImplementedError()

    def step(self, input, next_step_forcing_data, use_activation_checkpointing=False):
        raise NotImplementedError()

    def get_state(self):
        raise NotImplementedError()

    @classmethod
    def from_state(cls, state):
        raise NotImplementedError()


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
