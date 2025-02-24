import os

import pytest
import torch

from fme.core.typing_ import TensorDict


def validate_tensor(x: torch.Tensor, filename: str):
    if not os.path.exists(filename):
        torch.save(x, filename)
        pytest.fail(f"Regression file {filename} did not exist, so it was created")
    else:
        y = torch.load(filename).to(x.device)
        torch.testing.assert_allclose(x, y)


def validate_tensor_dict(x: TensorDict, filename: str):
    if not os.path.exists(filename):
        torch.save(x, filename)
        pytest.fail(f"Regression file {filename} did not exist, so it was created")
    else:
        y = torch.load(filename)
        for k, v in x.items():
            torch.testing.assert_allclose(v, y[k].to(v.device))
