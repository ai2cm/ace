import itertools
import torch

from modulus.key import Key
from modulus.models.afno import AFNOArch

########################
# load & verify
########################
def test_afno():
    # Construct FNO model
    model = AFNOArch(
        input_keys=[Key("x", size=2)],
        output_keys=[Key("u", size=2), Key("p")],
        img_shape=(240, 240),
        patch_size=16,
        embed_dim=256,
        depth=4,
        num_blocks=8,
    )
    # Testing JIT
    node = model.make_node(name="AFNO", jit=True)

    bsize = 5
    invar = {
        "x": torch.randn(bsize, 2, 240, 240),
    }
    # Model forward
    outvar = node.evaluate(invar)
    # Check output size
    assert outvar["u"].shape == (bsize, 2, 240, 240)
    assert outvar["p"].shape == (bsize, 1, 240, 240)


test_afno()
