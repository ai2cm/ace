import torch

from fme.core.packer import Packer


def test_pack_singleton_channels():
    data = {
        "u10": torch.randn(2, 4, 8),
        "v10": torch.randn(2, 4, 8),
        "w10": torch.randn(2, 4, 8),
    }
    p = Packer(names=["u10", "v10"])
    packed = p.pack(data, axis=1)
    assert packed.shape == (2, 2, 4, 8)
    assert torch.allclose(packed[:, 0, :, :], data["u10"])
    assert torch.allclose(packed[:, 1, :, :], data["v10"])


def test_unpack():
    tensor = torch.randn(2, 2, 4, 8)
    p = Packer(names=["u10", "v10"])
    unpacked = p.unpack(tensor, axis=1)
    assert len(unpacked) == 2
    assert torch.allclose(unpacked["u10"], tensor[:, 0, :, :])
    assert torch.allclose(unpacked["v10"], tensor[:, 1, :, :])


def test_unpack_first_axis():
    tensor = torch.randn(2, 2, 4, 8)
    p = Packer(names=["u10", "v10"])
    unpacked = p.unpack(tensor, axis=0)
    assert len(unpacked) == 2
    assert torch.allclose(unpacked["u10"], tensor[0, :, :, :])
    assert torch.allclose(unpacked["v10"], tensor[1, :, :, :])
