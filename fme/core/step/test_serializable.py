import datetime

import torch

from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.gridded_ops import LatLonOperations

from .serializable import SerializableStep
from .step import StepSelector
from .test_step_registry import MockStep


def test_serializable_step_round_trip():
    serializable = SerializableStep(
        selector=StepSelector(type="mock", config={}),
        img_shape=(16, 32),
        gridded_operations=LatLonOperations(area_weights=torch.ones(16, 32)),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(7), bk=torch.arange(7)
        ),
        timestep=datetime.timedelta(hours=6),
    )
    assert isinstance(serializable._instance, MockStep)
    state = serializable.to_state()
    round_tripped = SerializableStep.from_state(state)
    round_tripped_state = round_tripped.to_state()
    assert state == round_tripped_state
