from typing import Any, Dict, Type

import pytest
import torch

from fme.core.gridded_ops import GriddedOperations, HEALPixOperations, LatLonOperations


@pytest.mark.parametrize(
    "state, expected_class",
    [
        (
            {
                "type": "LatLonOperations",
                "state": {"area_weights": torch.tensor([1.0, 2.0])},
            },
            LatLonOperations,
        ),
        (
            {
                "type": "HEALPixOperations",
                "state": {},
            },
            HEALPixOperations,
        ),
    ],
)
def test_gridded_operations_from_state(
    state: Dict[str, Any],
    expected_class: Type[GriddedOperations],
):
    ops = GriddedOperations.from_state(state)
    assert isinstance(ops, expected_class)

    recovered_state = ops.to_state()
    assert recovered_state == state

    with pytest.raises(RuntimeError):
        expected_class.from_state(state["state"])
