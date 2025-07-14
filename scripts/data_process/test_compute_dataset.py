import re

import pytest
from compute_dataset import validate_vertical_coarsening_indices


@pytest.mark.parametrize(
    ("size", "indices", "valid"),
    [
        pytest.param(5, [(0, 2), (2, 5)], True, id="valid"),
        pytest.param(5, [(0, 2), (1, 5)], False, id="invalid-overlapping"),
        pytest.param(5, [(0, 2), (2, 4)], False, id="invalid-incomplete"),
        pytest.param(5, [(0, 2), (2, 6)], False, id="invalid-out-of-bounds"),
    ],
)
def test_validate_vertical_coarsening_indices(size, indices, valid):
    component = "atmosphere"
    control_flag = "validate_vertical_coarsening_indices"
    if valid:
        validate_vertical_coarsening_indices(size, indices, component, control_flag)
    else:
        # Check that both the component and control flag appear in the error message.
        match = re.compile(rf"(?=.*\b{component}\b)(?=.*\b{control_flag}\b)")
        with pytest.raises(ValueError, match=match):
            validate_vertical_coarsening_indices(size, indices, component, control_flag)
