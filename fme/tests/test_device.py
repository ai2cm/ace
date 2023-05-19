import fme
import torch


def test_device_is_defined():
    assert isinstance(fme.get_device(), torch.device)
