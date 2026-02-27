import os

import pytest
import torch

from fme.core.device import get_device


def validate_tensor(x: torch.Tensor, filename: str, **assert_close_kwargs):
    if not os.path.exists(filename):
        torch.save(x.cpu(), filename)
        pytest.fail(f"Regression file {filename} did not exist, so it was created")
    else:
        y = torch.load(filename, map_location=get_device()).to(x.device)
        torch.testing.assert_close(x, y, **assert_close_kwargs)


NestedTensorDict = dict[str, "torch.Tensor | NestedTensorDict"]


def _to_cpu(x: NestedTensorDict) -> NestedTensorDict:
    result = {}
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.cpu()
        else:
            result[k] = _to_cpu(v)
    return result


def _assert_close(x: NestedTensorDict, y: NestedTensorDict, **assert_close_kwargs):
    for k, v in x.items():
        y_val = y[k]
        if isinstance(v, torch.Tensor):
            assert isinstance(
                y_val, torch.Tensor
            ), f"Expected tensor but got {type(y_val)} at key {k}"
            torch.testing.assert_close(
                v,
                y_val.to(v.device),
                msg=f"Mismatch at key {k}",
                **assert_close_kwargs,
            )
        else:
            assert isinstance(y_val, dict), f"Expected dict at key {k} but got tensor"
            _assert_close(v, y_val, **assert_close_kwargs)


def validate_tensor_dict(
    x: NestedTensorDict,
    filename: str,
    **assert_close_kwargs,
):
    if not os.path.exists(filename):
        torch.save(_to_cpu(x), filename)
        pytest.fail(f"Regression file {filename} did not exist, so it was created")
    else:
        y = torch.load(filename, map_location=get_device())
        _assert_close(x, y, **assert_close_kwargs)
