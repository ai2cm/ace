import os

import pytest
import torch


def validate_tensor(x: torch.Tensor, filename: str):
    if not os.path.exists(filename):
        torch.save(x, filename)
        pytest.fail(f"Regression file {filename} did not exist, so it was created")
    else:
        y = torch.load(filename).to(x.device)
        torch.testing.assert_allclose(x, y)
