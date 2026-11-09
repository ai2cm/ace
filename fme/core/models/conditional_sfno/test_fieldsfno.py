import os

import pytest
import torch

from fme.core.device import get_device
from fme.core.models.conditional_sfno.fieldsfno import (
    FieldSFNONetConfig,
    get_lat_lon_fieldsfno,
)
from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.testing.regression import validate_tensor

DIR = os.path.abspath(os.path.dirname(__file__))
IMG_SHAPE = (9, 18)
N_SAMPLES = 4
FILTER_TYPES = ["linear", "makani-linear"]


def _context(
    n_samples: int,
    embed_dim_noise: int,
    embed_dim_pos: int,
    embed_dim_labels: int = 0,
) -> Context:
    device = get_device()
    return Context(
        embedding_scalar=None,
        embedding_pos=torch.randn(n_samples, embed_dim_pos, *IMG_SHAPE, device=device),
        labels=(
            torch.randn(n_samples, embed_dim_labels, device=device)
            if embed_dim_labels > 0
            else None
        ),
        noise=torch.randn(n_samples, embed_dim_noise, *IMG_SHAPE, device=device),
    )


def _build(
    prognostic: int,
    forcing: int,
    diagnostic: int,
    embed_dim_noise: int = 4,
    embed_dim_pos: int = 3,
    embed_dim_labels: int = 0,
    **config_kwargs,
):
    config_kwargs.setdefault("filter_type", "linear")
    config_kwargs.setdefault("embed_dim", 16)
    config_kwargs.setdefault("num_layers", 2)
    config_kwargs.setdefault("forcing_embed_dim", 8)
    config = FieldSFNONetConfig(**config_kwargs)
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
            embed_dim_labels=embed_dim_labels,
            embed_dim_pos=embed_dim_pos,
        ),
    ).to(get_device())
    context = _context(N_SAMPLES, embed_dim_noise, embed_dim_pos, embed_dim_labels)
    x = torch.randn(N_SAMPLES, prognostic + forcing, *IMG_SHAPE, device=get_device())
    return net, x, context


def _open_gates(net: torch.nn.Module, value: float = 0.5):
    """Set every gate and forcing-conditioning weight so the branches act."""
    with torch.no_grad():
        for name, param in net.named_parameters():
            if "gamma" in name or "_forcing" in name:
                param.fill_(value)


@pytest.mark.parametrize(
    "prognostic, forcing, diagnostic",
    [(5, 3, 2), (5, 0, 2), (5, 3, 0), (1, 0, 0)],
)
@pytest.mark.parametrize("filter_type", FILTER_TYPES)
@pytest.mark.parametrize("filter_output", [False, True])
@pytest.mark.parametrize("use_mlp", [False, True])
def test_output_shape(
    prognostic, forcing, diagnostic, filter_type, filter_output, use_mlp
):
    net, x, context = _build(
        prognostic,
        forcing,
        diagnostic,
        filter_type=filter_type,
        filter_output=filter_output,
        use_mlp=use_mlp,
    )
    out = net(x, context)
    assert out.shape == (N_SAMPLES, prognostic + diagnostic, *IMG_SHAPE)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "config_kwargs, embed_dim_labels",
    [
        pytest.param({"checkpointing": 2}, 0, id="checkpoint_mlp"),
        pytest.param({"checkpointing": 3}, 0, id="checkpoint_blocks"),
        pytest.param({"local_blocks": [1]}, 0, id="local_block"),
        pytest.param({"filter_residual": True}, 0, id="filter_residual"),
        pytest.param({"block_inner_skip": False}, 0, id="no_inner_skip"),
        pytest.param({"block_mlp_input": "filter"}, 0, id="mlp_input_filter"),
        pytest.param(
            {"block_mlp_input": "filter", "block_inner_skip": False},
            0,
            id="mlp_input_filter_no_inner_skip",
        ),
        pytest.param(
            {"block_mlp_input": "filter", "use_mlp": False},
            0,
            id="mlp_input_filter_no_mlp",
        ),
        pytest.param({"filter_num_groups": 4}, 0, id="grouped_filter"),
        pytest.param({"spectral_ratio": 0.5}, 0, id="spectral_ratio"),
        pytest.param(
            {"spectral_ratio": 0.5, "filter_num_groups": 2, "local_blocks": [0]},
            0,
            id="spectral_ratio_grouped_local",
        ),
        pytest.param({"mlp_ratio": 0.25}, 0, id="narrow_mlp"),
        pytest.param({"filter_preserves_global_mean": True}, 0, id="preserve_gm"),
        pytest.param({"global_layer_norm": True}, 0, id="global_layer_norm"),
        pytest.param({"affine_norms": True}, 0, id="affine_norms"),
        pytest.param({"drop_path_rate": 0.5, "drop_rate": 0.1}, 0, id="dropout"),
        pytest.param({"activation_function": "silu"}, 0, id="silu"),
        pytest.param({"forcing_encoder_layers": 1}, 0, id="linear_forcing_encoder"),
        pytest.param({}, 3, id="labels"),
    ],
)
def test_options_forward_backward(config_kwargs, embed_dim_labels):
    """Every option runs forward and backward with every parameter trained."""
    prognostic, forcing, diagnostic = 5, 3, 2
    net, x, context = _build(
        prognostic,
        forcing,
        diagnostic,
        embed_dim_labels=embed_dim_labels,
        gamma_init=0.5,
        **config_kwargs,
    )
    _open_gates(net)
    out = net(x, context)
    assert out.shape == (N_SAMPLES, prognostic + diagnostic, *IMG_SHAPE)
    out.sum().backward()
    missing = [
        name
        for name, param in net.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert missing == []


@pytest.mark.parametrize("filter_type", FILTER_TYPES)
@pytest.mark.parametrize("filter_output", [False, True])
def test_identity_at_init(filter_type, filter_output):
    """With the default gates the net returns the prognostic inputs and zero
    diagnostics."""
    prognostic, forcing, diagnostic = 5, 3, 2
    net, x, context = _build(
        prognostic,
        forcing,
        diagnostic,
        filter_type=filter_type,
        filter_output=filter_output,
    )
    out = net(x, context)
    torch.testing.assert_close(out[:, :prognostic], x[:, :prognostic])
    torch.testing.assert_close(
        out[:, prognostic:], torch.zeros_like(out[:, prognostic:])
    )


def test_gates_open_updates():
    """Opening the gates makes the output depend on the blocks."""
    prognostic, forcing, diagnostic = 5, 3, 2
    net, x, context = _build(prognostic, forcing, diagnostic)
    _open_gates(net, 1.0)
    out = net(x, context)
    assert not torch.allclose(out[:, :prognostic], x[:, :prognostic])
    assert not torch.allclose(
        out[:, prognostic:], torch.zeros_like(out[:, prognostic:])
    )


def test_forcing_conditions_output():
    prognostic, forcing, diagnostic = 5, 3, 2
    net, x, context = _build(prognostic, forcing, diagnostic)
    _open_gates(net)
    out_a = net(x, context)
    x_b = x.clone()
    x_b[:, prognostic:] = torch.randn_like(x_b[:, prognostic:])
    out_b = net(x_b, context)
    assert not torch.allclose(out_a, out_b)


def test_forward_does_not_mutate_context():
    net, x, context = _build(5, 3, 2)
    before = context.asdict()
    net(x, context)
    assert context.embedding_forcing is None
    assert context.asdict() == before


def test_wrong_input_channels_raises():
    net, x, context = _build(5, 3, 2)
    with pytest.raises(ValueError, match="input channels"):
        net(x[:, :-1], context)


def test_forcing_slot_in_context_config_rejected():
    config = FieldSFNONetConfig(filter_type="linear", num_layers=1)
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


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"activation_function": "tanh"}, "activation function"),
        ({"forcing_encoder_layers": 0}, "forcing_encoder_layers"),
        ({"forcing_embed_dim": 0}, "forcing_embed_dim"),
        ({"local_blocks": [2], "num_layers": 2}, "local_blocks"),
        ({"embed_dim": 0}, "embed_dim"),
        ({"spectral_ratio": 0.0}, "spectral_ratio"),
        ({"spectral_ratio": 1.5}, "spectral_ratio"),
        ({"embed_dim": 8, "spectral_ratio": 0.01}, "no spectral channels"),
        ({"embed_dim": 8, "spectral_ratio": 0.5, "filter_num_groups": 3}, "divide"),
        ({"embed_dim": 8, "mlp_ratio": 0.1}, "no MLP hidden channels"),
        ({"block_mlp_input": "spectral"}, "block_mlp_input"),
    ],
)
def test_invalid_config_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FieldSFNONetConfig(**kwargs)


def test_block_inner_skip_option():
    net, _, _ = _build(5, 3, 2)
    assert isinstance(net.blocks[0].inner_skip, torch.nn.Conv2d)
    net, _, _ = _build(5, 3, 2, block_inner_skip=False)
    assert net.blocks[0].inner_skip is None
    assert not any("inner_skip" in name for name, _ in net.named_parameters())


@pytest.mark.parametrize("block_mlp_input", ["field", "filter"])
def test_block_mlp_input_filter_branch_reaches_field_only_through_mlp(
    block_mlp_input,
):
    """With block_mlp_input='filter' the block has one gate and no proj_filter_out,
    and once the MLP's output layer is zeroed the filter branch cannot change
    the output; with 'field' the filter branch has its own path to the field."""
    net, x, context = _build(5, 3, 2, num_layers=1, block_mlp_input=block_mlp_input)
    block = net.blocks[0]
    if block_mlp_input == "filter":
        assert block.gate_mlp is None
        assert block.proj_filter_out is None
        assert not any("gate_mlp" in name for name, _ in net.named_parameters())
        assert not any("proj_filter_out" in name for name, _ in net.named_parameters())
    else:
        assert block.gate_mlp is not None
        assert block.proj_filter_out is not None
    with torch.no_grad():
        for name, p in block.named_parameters():
            if name.startswith("gate_filter.") or name.startswith("gate_mlp."):
                p.fill_(1.0)
        last_conv = [m for m in block.mlp.modules() if isinstance(m, torch.nn.Conv2d)]
        last_conv[-1].weight.zero_()
        last_conv[-1].bias.zero_()
        out_before = net(x, context)
        block.proj_filter_in.weight.add_(torch.randn_like(block.proj_filter_in.weight))
        out_after = net(x, context)
    if block_mlp_input == "filter":
        torch.testing.assert_close(out_before, out_after)
    else:
        assert not torch.allclose(out_before, out_after)


def test_branch_widths_follow_embed_dim():
    """The filter and MLP widths come from embed_dim, not the field count."""
    net, _, _ = _build(5, 3, 2, embed_dim=12, spectral_ratio=0.5, mlp_ratio=3.0)
    block = net.blocks[0]
    assert block.proj_filter_in.weight.shape[:2] == (6, 7)
    assert block.proj_filter_out.weight.shape[:2] == (7, 6)
    assert block.inner_skip.weight.shape[:2] == (6, 6)
    assert block.mlp.fwd[0].weight.shape[:2] == (36, 7)
    assert block.mlp.fwd[-1].weight.shape[:2] == (7, 36)


def test_fieldsfno_output_is_unchanged():
    torch.manual_seed(0)
    net, x, context = _build(5, 3, 2, gamma_init=0.5)
    _open_gates(net)
    with torch.no_grad():
        output = net(x, context)
    validate_tensor(
        output, os.path.join(DIR, "testdata/test_fieldsfno_output_is_unchanged.pt")
    )


def test_state_dict_round_trip():
    net, x, context = _build(5, 3, 2, gamma_init=0.5)
    _open_gates(net)
    net2, _, _ = _build(5, 3, 2, gamma_init=0.5)
    net2.load_state_dict(net.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(net(x, context), net2(x, context))


def test_no_forcing_leaves_context_untouched():
    net, x, context = _build(5, 0, 2)
    assert net.forcing_encoder is None
    before = context.asdict()
    net(x, context)
    assert context.asdict() == before
