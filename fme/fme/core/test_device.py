import torch

import fme


def test_device_is_defined():
    assert isinstance(fme.get_device(), torch.device)
