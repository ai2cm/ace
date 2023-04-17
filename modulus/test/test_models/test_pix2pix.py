import itertools
import torch

from modulus.key import Key
from modulus.models.pix2pix import Pix2PixArch


def test_pix2pix():
    # check 1D
    model = Pix2PixArch(
        input_keys=[Key("x", size=4)],
        output_keys=[Key("y", size=4), Key("z", size=2)],
        dimension=1,
        scaling_factor=2,
    )
    bsize = 4
    x = {"x": torch.randn((bsize, 4, 32))}
    outvar = model.forward(x)
    # Check output size
    assert outvar["y"].shape == (bsize, 4, 64)
    assert outvar["z"].shape == (bsize, 2, 64)

    # check 2D
    model = Pix2PixArch(
        input_keys=[Key("x", size=2)],
        output_keys=[Key("y", size=2), Key("z", size=1)],
        dimension=2,
        n_downsampling=1,
        scaling_factor=4,
    )
    bsize = 4
    x = {"x": torch.randn((bsize, 2, 28, 28))}
    outvar = model.forward(x)
    # Check output size
    assert outvar["y"].shape == (bsize, 2, 112, 112)
    assert outvar["z"].shape == (bsize, 1, 112, 112)

    # check 3D
    model = Pix2PixArch(
        input_keys=[Key("x", size=1)],
        output_keys=[Key("y", size=2), Key("z", size=2)],
        dimension=3,
    )
    bsize = 4
    x = {"x": torch.randn((bsize, 1, 64, 64, 64))}
    outvar = model.forward(x)
    # Check output size
    assert outvar["y"].shape == (bsize, 2, 64, 64, 64)
    assert outvar["z"].shape == (bsize, 2, 64, 64, 64)
