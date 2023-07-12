"""The registry also performs configuration set up so it needs to be tested."""

from fme.fcn_training import registry


def test_sfno_builder():
    """Make sure that the monkey patch is going through. Not that this is note
    testing whether the modulus code runs the code correctly."""

    builder = registry.NET_REGISTRY["SphericalFourierNeuralOperatorNet"]()

    sfno_net = builder.build(1, 1, 32, 16)

    assert (
        sfno_net.trans_down.grid == "legendre-gauss"
        and sfno_net.itrans_up.grid == "legendre-gauss"
    ), "The grid should be set to legendre-gauss"
