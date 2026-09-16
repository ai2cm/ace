import numpy as np
import torch
import xarray as xr

from fme.ace.stepper import TrainOutput
from fme.core.tensors import EnsembleTensorDict
from fme.core.typing_ import TensorDict, TensorMapping


def test_train_output_compute_derived_quantities():
    n_samples, n_ensemble = 2, 3
    gen_data = EnsembleTensorDict(
        {
            "a": torch.rand(n_samples, n_ensemble, 4, 8),
            "b": torch.rand(n_samples, n_ensemble, 4, 8),
        }
    )
    # "f" has no generated counterpart, so it is the only name the derive
    # function can only get from the forcing data.
    target_data = EnsembleTensorDict(
        {
            "a": torch.rand(n_samples, 1, 4, 8),
            "b": torch.rand(n_samples, 1, 4, 8),
            "f": torch.rand(n_samples, 1, 4, 8),
        }
    )

    calls: list[tuple[TensorDict, TensorDict]] = []

    def derive_func(data: TensorMapping, forcing_data: TensorMapping) -> TensorDict:
        calls.append((dict(data), dict(forcing_data)))
        return dict(data)

    data = TrainOutput(
        metrics={"loss": torch.tensor(0.0)},
        gen_data=gen_data,
        target_data=target_data,
        time=xr.DataArray(np.zeros((n_samples, 3)), dims=["sample", "time"]),
        normalize=lambda x: x,
        derive_func=derive_func,
    )
    out_data = data.compute_derived_variables()

    assert len(calls) == 2
    gen_call_data, gen_call_forcing = calls[0]
    target_call_data, target_call_forcing = calls[1]

    n_folded = n_samples * n_ensemble
    for name in ("a", "b"):
        torch.testing.assert_close(
            gen_call_data[name], gen_data[name].reshape(n_folded, 4, 8)
        )
    # Names the generated data already carries are not broadcast: the derive
    # function would ignore them, and broadcasting copies.
    assert set(gen_call_forcing) == {"f"}
    assert gen_call_forcing["f"].shape == (n_folded, 4, 8)
    # ensemble is fastest dimension between batch and ensemble
    assert (gen_call_forcing["f"][:n_ensemble] == target_data["f"][0]).all()
    assert (gen_call_forcing["f"][n_ensemble:] == target_data["f"][1]).all()

    # deriving the target is a single-member call, so nothing is broadcast and
    # the whole forcing mapping is passed through
    for name in ("a", "b", "f"):
        torch.testing.assert_close(
            target_call_data[name], target_data[name].reshape(n_samples, 4, 8)
        )
        assert (
            target_call_forcing[name] == target_data[name].reshape(n_samples, 4, 8)
        ).all()

    for name in gen_data:
        assert name in out_data.gen_data
        assert name in out_data.target_data
        assert out_data.gen_data[name].shape == (n_samples, n_ensemble, 4, 8)
        assert out_data.target_data[name].shape == (n_samples, 1, 4, 8)
        assert torch.allclose(out_data.gen_data[name], gen_data[name])
        assert torch.allclose(out_data.target_data[name], target_data[name])
