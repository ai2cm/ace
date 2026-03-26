import pytest
import torch

import fme
from fme.core.device import force_cpu, get_device


def test_device_is_defined():
    assert isinstance(fme.get_device(), torch.device)


def test_force_cpu():
    device_before = get_device()
    if device_before.type == "cpu":
        pytest.skip("Device is already CPU, cannot test force_cpu.")
    with force_cpu():
        device = get_device()
        assert device.type == "cpu"
    device_after = get_device()
    assert device_after.type == device_before.type
