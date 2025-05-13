import torch

from fme.ace.models.makani_fcn2.models.networks.fourcastnet2 import (
    AtmoSphericNeuralOperatorNet,
)


def test__AtmoSphericNeuralOperatorNet():
    torch.manual_seed(0)

    n_atmo_channels = 3
    n_atmo_groups = 6
    n_surf_channels = 2
    n_aux_channels = 1
    img_shape = (16, 32)
    n_batch = 2

    net = AtmoSphericNeuralOperatorNet(
        n_atmo_channels=n_atmo_channels,
        n_atmo_diagnostic_channels=0,
        n_atmo_groups=n_atmo_groups,
        n_surf_channels=n_surf_channels,
        n_surf_diagnostic_channels=0,
        n_aux_channels=n_aux_channels,
        inp_shape=img_shape,
        out_shape=img_shape,
        normalization_layer="instance_norm",
    )

    assert net.n_atmo_channels == n_atmo_channels
    assert net.n_atmo_groups == n_atmo_groups
    assert net.n_surf_channels == n_surf_channels
    assert net.n_aux_channels == n_aux_channels

    x_atmo = torch.randn(n_batch, n_atmo_groups * n_atmo_channels, *img_shape)
    x_surf = torch.randn(n_batch, n_surf_channels, *img_shape)
    x_aux = torch.randn(n_batch, n_aux_channels, *img_shape)

    x_atmo_out, x_surf_out = net(x_atmo, x_surf, x_aux)

    assert x_atmo_out.shape == (n_batch, n_atmo_groups * n_atmo_channels, *img_shape)
    assert x_surf_out.shape == (n_batch, n_surf_channels, *img_shape)
