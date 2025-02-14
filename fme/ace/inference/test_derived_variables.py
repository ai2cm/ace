from unittest.mock import MagicMock

import numpy as np
import torch
import xarray as xr

from fme.ace.stepper import TrainOutput
from fme.core.typing_ import TensorDict, TensorMapping


def test_train_output_compute_derived_quantities():
    derived_data = {
        "a": torch.rand(2, 3, 4, 8),
        "b": torch.rand(2, 3, 4, 8),
    }

    gen_data = MagicMock()
    target_data = MagicMock()

    def derive_func(data: TensorMapping, forcing_data: TensorMapping) -> TensorDict:
        assert data is gen_data or data is target_data
        assert forcing_data is target_data
        return derived_data

    data = TrainOutput(
        metrics={"loss": torch.tensor(0.0)},
        gen_data=gen_data,
        target_data=target_data,
        time=xr.DataArray(np.zeros((2, 3)), dims=["sample", "time"]),
        normalize=lambda x: x,
        derive_func=derive_func,
    )
    out_data = data.compute_derived_variables()
    for name in derived_data:
        assert name in out_data.gen_data
        assert name in out_data.target_data
        assert out_data.gen_data[name].shape == (2, 3, 4, 8)
        assert out_data.target_data[name].shape == (2, 3, 4, 8)
        assert torch.allclose(out_data.gen_data[name], derived_data[name])
        assert torch.allclose(out_data.target_data[name], derived_data[name])
