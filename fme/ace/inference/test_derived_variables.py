import numpy as np
import torch
import xarray as xr

from fme.ace.stepper import TrainOutput
from fme.core.tensors import EnsembleTensorDict
from fme.core.typing_ import TensorDict, TensorMapping


def test_train_output_compute_derived_quantities():
    gen_data = EnsembleTensorDict(
        {
            "a": torch.rand(2, 3, 4, 8),
            "b": torch.rand(2, 3, 4, 8),
        }
    )
    target_data = EnsembleTensorDict(
        {
            "a": torch.rand(2, 1, 4, 8),
            "b": torch.rand(2, 1, 4, 8),
        }
    )

    def derive_func(data: TensorMapping, forcing_data: TensorMapping) -> TensorDict:
        try:
            torch.testing.assert_close(data["a"], gen_data["a"].reshape(2 * 3, 4, 8))
            torch.testing.assert_close(data["b"], gen_data["b"].reshape(2 * 3, 4, 8))
            assert forcing_data["a"].shape == (6, 4, 8)
            # ensemble is fastest dimension between batch and ensemble
            assert (forcing_data["a"][:3] == target_data["a"][0]).all()
            assert (forcing_data["a"][3:] == target_data["a"][1]).all()
        except AssertionError:
            torch.testing.assert_close(data["a"], target_data["a"].reshape(2, 4, 8))
            torch.testing.assert_close(data["b"], target_data["b"].reshape(2, 4, 8))
            assert forcing_data["a"].shape == (2, 4, 8)
            assert (forcing_data["a"] == target_data["a"].reshape(2, 4, 8)).all()
            assert (forcing_data["b"] == target_data["b"].reshape(2, 4, 8)).all()
        return dict(data)

    data = TrainOutput(
        metrics={"loss": torch.tensor(0.0)},
        gen_data=gen_data,
        target_data=target_data,
        time=xr.DataArray(np.zeros((2, 3)), dims=["sample", "time"]),
        normalize=lambda x: x,
        derive_func=derive_func,
    )
    out_data = data.compute_derived_variables()
    for name in gen_data:
        assert name in out_data.gen_data
        assert name in out_data.target_data
        assert out_data.gen_data[name].shape == (2, 3, 4, 8)
        assert out_data.target_data[name].shape == (2, 1, 4, 8)
        assert torch.allclose(out_data.gen_data[name], gen_data[name])
        assert torch.allclose(out_data.target_data[name], target_data[name])
