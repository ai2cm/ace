import os

import pytest
import torch

from fme.core.device import get_device
from fme.core.typing_ import TensorDict


def validate_tensor(x: torch.Tensor, filename: str, **assert_close_kwargs):
    if not os.path.exists(filename):
        torch.save(x.cpu(), filename)
        pytest.fail(f"Regression file {filename} did not exist, so it was created")
    else:
        y = torch.load(filename, map_location=get_device()).to(x.device)
        torch.testing.assert_close(x, y, **assert_close_kwargs)


def validate_tensor_dict(x: TensorDict, filename: str, **assert_close_kwargs):
    if not os.path.exists(filename):
        x_cpu = {k: v.cpu() for k, v in x.items()}
        torch.save(x_cpu, filename)
        pytest.fail(f"Regression file {filename} did not exist, so it was created")
    else:
        y = torch.load(filename, map_location=get_device())
        for k, v in x.items():
            torch.testing.assert_close(v, y[k].to(v.device), **assert_close_kwargs)


NestedTensorDict = dict[str, "torch.Tensor | NestedTensorDict"]


def _to_cpu(x: NestedTensorDict) -> NestedTensorDict:
    result = {}
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.cpu()
        else:
            result[k] = _to_cpu(v)
    return result


def _assert_close(x: NestedTensorDict, y: NestedTensorDict, prefix: str):
    for k, v in x.items():
        key_path = f"{prefix}.{k}" if prefix else k
        assert k in y, f"Key '{key_path}' missing from regression file"
        y_val = y[k]
        if isinstance(v, torch.Tensor):
            assert isinstance(
                y_val, torch.Tensor
            ), f"Expected tensor at '{key_path}' but got dict"
            torch.testing.assert_close(
                v, y_val.to(v.device), msg=f"Mismatch at '{key_path}'"
            )
        else:
            assert isinstance(
                y_val, dict
            ), f"Expected dict at '{key_path}' but got tensor"
            _assert_close(v, y_val, prefix=key_path)


def validate_nested_tensor_dict(
    x: NestedTensorDict,
    filename: str,
):
    if not os.path.exists(filename):
        torch.save(_to_cpu(x), filename)
        pytest.fail(f"Regression file {filename} did not exist, so it was created")
    else:
        y = torch.load(filename, map_location=get_device())
        _assert_close(x, y, prefix="")
