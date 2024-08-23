import pytest
import torch

from fme.core.gridded_ops import GriddedOperations


def test_gridded_operations_from_state():
    class MockGriddedOperations(GriddedOperations):
        def __init__(self, area_weights, scalar):
            self._area_weights = area_weights
            self.scalar = scalar

        def area_weighted_mean(self, data, keepdim: bool = False):
            return data

        def get_initialization_kwargs(self):
            return {"area_weights": self._area_weights, "scalar": self.scalar}

    state = {
        "type": "MockGriddedOperations",
        "state": {"area_weights": torch.tensor([1.0, 2.0]), "scalar": 3.0},
    }
    ops = GriddedOperations.from_state(state)
    assert ops.scalar == state["state"]["scalar"]
    assert isinstance(ops, MockGriddedOperations)

    recovered_state = ops.to_state()
    assert recovered_state == state

    with pytest.raises(RuntimeError):
        MockGriddedOperations.from_state(state["state"])
