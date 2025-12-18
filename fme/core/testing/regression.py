import os

import pytest
import torch

from fme.core.device import get_device
from fme.core.typing_ import TensorDict


def validate_tensor(x: torch.Tensor, filename: str):
    if not os.path.exists(filename):
        torch.save(x.cpu(), filename)
        pytest.fail(f"Regression file {filename} did not exist, so it was created")
    else:
        y = torch.load(filename, map_location=get_device()).to(x.device)
        torch.testing.assert_close(x, y)


def validate_tensor_dict(x: TensorDict, filename: str):
    if not os.path.exists(filename):
        x_cpu = {k: v.cpu() for k, v in x.items()}
        torch.save(x_cpu, filename)
        pytest.fail(f"Regression file {filename} did not exist, so it was created")
    else:
        y = torch.load(filename, map_location=get_device())
        for k, v in x.items():
            torch.testing.assert_close(v, y[k].to(v.device))
