import itertools
import torch

from modulus.key import Key
from modulus.models.fno import FNOArch
from modulus.models.fully_connected import FullyConnectedArch

########################
# load & verify
########################
def test_fno_1d():
    # Construct FNO model
    decoder = FullyConnectedArch(
        input_keys=[Key("z", size=32)],
        output_keys=[Key("u", size=2), Key("p")],
        nr_layers=1,
        layer_size=8,
    )
    model = FNOArch(
        input_keys=[Key("x", size=2)],
        decoder_net=decoder,
        dimension=1,
        fno_modes=4,
        padding=0,
    )
    # Testing JIT
    model.make_node(name="FNO1d", jit=True)

    bsize = 5
    invar = {
        "x": torch.randn(bsize, 2, 64),
    }
    # Model forward
    outvar = model(invar)
    # Check output size
    assert outvar["u"].shape == (bsize, 2, 64)
    assert outvar["p"].shape == (bsize, 1, 64)


def test_fno_2d():
    # Construct FNO model
    decoder = FullyConnectedArch(
        input_keys=[Key("z", size=32)],
        output_keys=[Key("u", size=2), Key("p")],
        nr_layers=2,
        layer_size=16,
    )
    model = FNOArch(
        input_keys=[Key("x"), Key("y"), Key("rho", size=2)],
        decoder_net=decoder,
        dimension=2,
        fno_modes=16,
    )

    # Testing JIT
    model.make_node(name="FNO2d", jit=True)

    bsize = 5
    invar = {
        "x": torch.randn(bsize, 1, 32, 32),
        "y": torch.randn(bsize, 1, 32, 32),
        "rho": torch.randn(bsize, 2, 32, 32),
    }
    # Model forward
    outvar = model(invar)
    # Check output size
    assert outvar["u"].shape == (bsize, 2, 32, 32)
    assert outvar["p"].shape == (bsize, 1, 32, 32)


def test_fno_3d():
    # Construct FNO model
    decoder = FullyConnectedArch(
        input_keys=[Key("z", size=32)],
        output_keys=[Key("u"), Key("v")],
        nr_layers=1,
        layer_size=8,
    )
    model = FNOArch(
        input_keys=[Key("x", size=3), Key("y")],
        decoder_net=decoder,
        dimension=3,
        fno_modes=16,
    )

    # Testing JIT
    model.make_node(name="FNO3d", jit=True)

    bsize = 5
    invar = {
        "x": torch.randn(bsize, 3, 32, 32, 32),
        "y": torch.randn(bsize, 1, 32, 32, 32),
    }
    # Model forward
    outvar = model(invar)
    # Check output size
    assert outvar["u"].shape == (bsize, 1, 32, 32, 32)
    assert outvar["v"].shape == (bsize, 1, 32, 32, 32)


def test_fno():
    test_fno_1d()
    test_fno_2d()
    test_fno_3d()


test_fno()
