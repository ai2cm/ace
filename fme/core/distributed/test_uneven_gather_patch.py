from unittest import mock

import pytest
import torch

from fme.core.distributed._uneven_gather_patch import _padded_gather


@pytest.mark.parametrize("dim", [4, -5])
def test_padded_gather_rejects_out_of_range_dim(dim: int):
    """An out-of-range dim, including a negative one that would otherwise wrap
    around to a valid axis, fails instead of gathering along the wrong axis."""
    with (
        mock.patch("torch.distributed.get_world_size", return_value=2),
        pytest.raises(ValueError, match="out of range"),
    ):
        _padded_gather(torch.zeros(1, 2, 23, 4), dim_=dim, shapes_=[23, 22])
