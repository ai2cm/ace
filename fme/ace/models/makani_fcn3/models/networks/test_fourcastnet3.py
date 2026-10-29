import copy
from unittest import mock

import pytest
import torch
import fme.core.sht as th

from fme.ace.models.makani_fcn3.models.networks import fourcastnet3
from fme.ace.models.makani_fcn3.models.networks.fourcastnet3 import (
    AtmoSphericNeuralOperatorNet,
)

N_ATMO_CHANNELS = 3
N_ATMO_GROUPS = 2
N_SURF_CHANNELS = 2
N_AUX_CHANNELS = 1
IMG_SHAPE = (16, 32)


def _get_small_net(
    model_grid_type: str = "legendre-gauss",
) -> AtmoSphericNeuralOperatorNet:
    """A small FCN3 network which uses all three DISCO convolutions (encoder,
    decoder and local processor blocks).

    By default uses the same grid for the model and the SHT, as ``FCN3Config``
    does by default, so no resampling is needed. Pass
    ``model_grid_type="equiangular"`` to resample between the grids.
    """
    return AtmoSphericNeuralOperatorNet(
        n_atmo_channels=N_ATMO_CHANNELS,
        n_atmo_diagnostic_channels=0,
        n_atmo_groups=N_ATMO_GROUPS,
        n_surf_channels=N_SURF_CHANNELS,
        n_surf_diagnostic_channels=0,
        n_aux_channels=N_AUX_CHANNELS,
        inp_shape=IMG_SHAPE,
        out_shape=IMG_SHAPE,
        model_grid_type=model_grid_type,
        sht_grid_type="legendre-gauss",
        scale_factor=1,
        atmo_embed_dim=2,
        surf_embed_dim=2,
        aux_embed_dim=2,
        num_layers=2,
    )


def _get_inputs(n_batch: int = 2):
    x_atmo = torch.randn(n_batch, N_ATMO_GROUPS * N_ATMO_CHANNELS, *IMG_SHAPE)
    x_surf = torch.randn(n_batch, N_SURF_CHANNELS, *IMG_SHAPE)
    x_aux = torch.randn(n_batch, N_AUX_CHANNELS, *IMG_SHAPE)
    return x_atmo, x_surf, x_aux


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


def _get_resamplers(net: AtmoSphericNeuralOperatorNet) -> list[th.ResampleS2]:
    return [m for m in net.modules() if isinstance(m, th.ResampleS2)]


@pytest.mark.parametrize("model_grid_type", ["legendre-gauss", "equiangular"])
def test_compiles_without_graph_breaks(model_grid_type: str):
    """The single-rank network is traceable by torch.compile as one graph."""
    torch.manual_seed(0)
    net = _get_small_net(model_grid_type=model_grid_type)
    inputs = _get_inputs()
    torch._dynamo.reset()
    try:
        explanation = torch._dynamo.explain(net)(*inputs)
    finally:
        torch._dynamo.reset()
    assert explanation.graph_break_count == 0


def test_equiangular_model_grid_resamples_unchanged():
    """With an equiangular model grid the decoders resample to the SHT grid,
    expanding the poles, and give the same outputs as torch-harmonics'
    resampler with its tensor-valued ``expand_poles`` flag.
    """
    torch.manual_seed(0)
    net = _get_small_net(model_grid_type="equiangular")
    resamplers = _get_resamplers(net)
    assert len(resamplers) > 0
    for resampler in resamplers:
        assert not resampler.skip_resampling
        assert resampler.expand_poles is True

    ref_net = copy.deepcopy(net)
    for resampler in _get_resamplers(ref_net):
        resampler.expand_poles = torch.tensor(resampler.expand_poles)

    inputs = _get_inputs()
    with torch.no_grad():
        out = net(*inputs)
        ref_out = ref_net(*inputs)
    for x, x_ref in zip(out, ref_out):
        torch.testing.assert_close(x, x_ref, atol=0, rtol=0)


def test_matches_torch_harmonics_disco_convolution():
    """The network matches one built with torch-harmonics' DISCO convolution.

    Checkpoints must load unchanged (identical state dict keys), and the
    outputs agree to within float32 rounding (the two convolutions contract
    in a different order, so they are not bit-identical).
    """
    torch.manual_seed(0)
    net = _get_small_net()
    with mock.patch.object(
        fourcastnet3, "DiscreteContinuousConvS2", th.DiscreteContinuousConvS2
    ):
        ref_net = _get_small_net()
    assert net.state_dict().keys() == ref_net.state_dict().keys()
    ref_net.load_state_dict(net.state_dict())

    inputs = _get_inputs()
    with torch.no_grad():
        out = net(*inputs)
        ref_out = ref_net(*inputs)
    for x, x_ref in zip(out, ref_out):
        torch.testing.assert_close(x, x_ref, atol=1e-6, rtol=0)
