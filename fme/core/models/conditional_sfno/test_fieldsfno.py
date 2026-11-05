import dataclasses

import pytest
import torch

from fme.core.device import get_device
from fme.core.models.conditional_sfno.fieldsfno import (
    FieldSFNONetConfig,
    get_lat_lon_fieldsfno,
)
from fme.core.models.conditional_sfno.layers import Context, ContextConfig

IMG_SHAPE = (9, 18)
N_SAMPLES = 4


def _context(n_samples: int, embed_dim_noise: int, embed_dim_pos: int) -> Context:
    device = get_device()
    return Context(
        embedding_scalar=None,
        embedding_pos=torch.randn(n_samples, embed_dim_pos, *IMG_SHAPE, device=device),
        labels=None,
        noise=torch.randn(n_samples, embed_dim_noise, *IMG_SHAPE, device=device),
    )


def _build(
    prognostic: int,
    forcing: int,
    diagnostic: int,
    embed_dim_noise: int = 4,
    embed_dim_pos: int = 3,
    **config_kwargs,
):
    config = FieldSFNONetConfig(
        filter_type="makani-linear", num_layers=2, forcing_embed_dim=8, **config_kwargs
    )
    net = get_lat_lon_fieldsfno(
        params=config,
        prognostic_chans=prognostic,
        forcing_chans=forcing,
        diagnostic_chans=diagnostic,
        img_shape=IMG_SHAPE,
        data_grid="legendre-gauss",
        context_config=ContextConfig(
            embed_dim_scalar=0,
            embed_dim_noise=embed_dim_noise,
            embed_dim_labels=0,
            embed_dim_pos=embed_dim_pos,
        ),
    ).to(get_device())
    context = _context(N_SAMPLES, embed_dim_noise, embed_dim_pos)
    x = torch.randn(N_SAMPLES, prognostic + forcing, *IMG_SHAPE, device=get_device())
    return net, x, context


@pytest.mark.parametrize(
    "prognostic, forcing, diagnostic",
    [(5, 3, 2), (5, 0, 2), (5, 3, 0), (1, 0, 0)],
)
@pytest.mark.parametrize("filter_output", [False, True])
@pytest.mark.parametrize("use_mlp", [False, True])
def test_output_shape(prognostic, forcing, diagnostic, filter_output, use_mlp):
    net, x, context = _build(
        prognostic, forcing, diagnostic, filter_output=filter_output, use_mlp=use_mlp
    )
    out = net(x, context)
    assert out.shape == (N_SAMPLES, prognostic + diagnostic, *IMG_SHAPE)
    assert torch.isfinite(out).all()


def test_identity_at_init():
    """With the default gates the net returns the prognostic inputs and zero
    diagnostics."""
    prognostic, forcing, diagnostic = 5, 3, 2
    net, x, context = _build(prognostic, forcing, diagnostic)
    out = net(x, context)
    torch.testing.assert_close(out[:, :prognostic], x[:, :prognostic])
    torch.testing.assert_close(
        out[:, prognostic:], torch.zeros_like(out[:, prognostic:])
    )


def test_gates_open_updates():
    """Opening the gates makes the output depend on the blocks."""
    prognostic, forcing, diagnostic = 5, 3, 2
    net, x, context = _build(prognostic, forcing, diagnostic)
    with torch.no_grad():
        for name, param in net.named_parameters():
            if "gamma" in name:
                param.fill_(1.0)
    out = net(x, context)
    assert not torch.allclose(out[:, :prognostic], x[:, :prognostic])
    assert not torch.allclose(
        out[:, prognostic:], torch.zeros_like(out[:, prognostic:])
    )


def test_forcing_conditions_output():
    prognostic, forcing, diagnostic = 5, 3, 2
    net, x, context = _build(prognostic, forcing, diagnostic)
    with torch.no_grad():
        for name, param in net.named_parameters():
            if "gamma" in name or "W_scale_forcing" in name or "W_bias_forcing" in name:
                param.fill_(0.5)
    out_a = net(x, context)
    x_b = x.clone()
    x_b[:, prognostic:] = torch.randn_like(x_b[:, prognostic:])
    out_b = net(x_b, context)
    assert not torch.allclose(out_a, out_b)


def test_wrong_input_channels_raises():
    net, x, context = _build(5, 3, 2)
    with pytest.raises(ValueError, match="input channels"):
        net(x[:, :-1], context)


def test_forcing_slot_in_context_config_rejected():
    config = FieldSFNONetConfig(filter_type="makani-linear", num_layers=1)
    with pytest.raises(ValueError, match="embed_dim_forcing"):
        get_lat_lon_fieldsfno(
            params=config,
            prognostic_chans=2,
            forcing_chans=1,
            diagnostic_chans=0,
            img_shape=IMG_SHAPE,
            context_config=ContextConfig(
                embed_dim_scalar=0,
                embed_dim_noise=0,
                embed_dim_labels=0,
                embed_dim_pos=0,
                embed_dim_forcing=4,
            ),
        )


def test_gradients_reach_all_parameters():
    prognostic, forcing, diagnostic = 5, 3, 2
    net, x, context = _build(prognostic, forcing, diagnostic, gamma_init=0.5)
    with torch.no_grad():
        for name, param in net.named_parameters():
            if "W_scale_forcing" in name or "W_bias_forcing" in name:
                param.fill_(0.5)
    net(x, context).sum().backward()
    missing = [
        name
        for name, param in net.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert missing == []


def test_context_replace_keeps_other_slots():
    """The net adds the forcing embedding without dropping the other slots."""
    context = _context(N_SAMPLES, 4, 3)
    replaced = dataclasses.replace(context, embedding_forcing=torch.zeros(1))
    assert replaced.noise is context.noise
    assert replaced.embedding_pos is context.embedding_pos
