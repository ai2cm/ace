import pathlib
from typing import Literal

import pytest
import torch

from fme.core.device import get_device
from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.testing.regression import validate_tensor_dict

from .boundary_padding import TensorPaddingConfig
from .swin_layers import (
    ColumnMixer,
    PatchExpanding,
    PatchMerging,
    SwinTransformerBlock,
    WindowAttention2D,
    window_cos_lat,
)
from .swin_transformer import SwinTransformerNet, _padding_index

_EMBED_DIM_NOISE = 8
_CLN_CONTEXT_CONFIG = ContextConfig(
    embed_dim_scalar=0,
    embed_dim_labels=0,
    embed_dim_noise=_EMBED_DIM_NOISE,
    embed_dim_pos=0,
)


def _build_net(
    in_chans: int,
    out_chans: int,
    img_shape: tuple[int, int],
    context_config: ContextConfig | None = None,
    conditioning: Literal["adaln", "cln"] = "adaln",
    use_skip: bool = True,
    skip_projection: bool = False,
    embed_dim: int = 32,
    num_heads: tuple[int, ...] = (2, 4, 4, 2),
    window_size: tuple[int, int] = (4, 4),
    mlp_layer: str = "mlp",
    lat_coords: torch.Tensor | None = None,
    padding_conf: dict | None = None,
    patch_size: tuple[int, int] = (1, 1),
    num_levels: int = 1,
) -> SwinTransformerNet:
    """A small Swin U-Net for tests. ``conditioning="cln"`` without an explicit
    ``context_config`` builds the noise-conditioned variant with
    ``_EMBED_DIM_NOISE`` noise channels."""
    if conditioning == "cln" and context_config is None:
        context_config = _CLN_CONTEXT_CONFIG
    return SwinTransformerNet(
        in_chans=in_chans,
        out_chans=out_chans,
        img_shape=img_shape,
        embed_dim=embed_dim,
        depth_multiplier=1,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        use_skip=use_skip,
        skip_projection=skip_projection,
        context_config=context_config,
        conditioning=conditioning,
        mlp_layer=mlp_layer,
        lat_coords=lat_coords,
        padding_conf=padding_conf,
        patch_size=patch_size,
        num_levels=num_levels,
    )


def _cln_context(n: int, img_shape: tuple[int, int]) -> Context:
    return Context(
        embedding_scalar=None,
        embedding_pos=None,
        labels=None,
        noise=torch.randn(n, _EMBED_DIM_NOISE, *img_shape, device=get_device()),
    )


def test_forward_no_conditioning():
    in_chans, out_chans = 5, 3
    img_shape = (16, 32)
    n = 2
    device = get_device()
    net = _build_net(in_chans, out_chans, img_shape).to(device)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    out = net(x)
    assert out.shape == (n, out_chans, *img_shape)


def test_forward_with_padding():
    in_chans, out_chans = 4, 4
    img_shape = (9, 18)  # not divisible by window_size * 2
    n = 2
    device = get_device()
    net = _build_net(in_chans, out_chans, img_shape).to(device)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    out = net(x)
    assert out.shape == (n, out_chans, *img_shape)


def test_forward_with_conditioning():
    in_chans, out_chans = 5, 3
    img_shape = (16, 32)
    n = 2
    embed_dim_scalar, embed_dim_labels = 8, 4
    device = get_device()
    context_config = ContextConfig(
        embed_dim_scalar=embed_dim_scalar,
        embed_dim_labels=embed_dim_labels,
        embed_dim_noise=0,
        embed_dim_pos=0,
    )
    net = _build_net(in_chans, out_chans, img_shape, context_config=context_config).to(
        device
    )
    x = torch.randn(n, in_chans, *img_shape, device=device)
    context = Context(
        embedding_scalar=torch.randn(n, embed_dim_scalar, device=device),
        embedding_pos=None,
        labels=torch.randn(n, embed_dim_labels, device=device),
        noise=None,
    )
    out = net(x, context)
    assert out.shape == (n, out_chans, *img_shape)


def test_no_skip():
    in_chans, out_chans = 5, 3
    img_shape = (16, 32)
    n = 2
    device = get_device()
    net = _build_net(in_chans, out_chans, img_shape, use_skip=False).to(device)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    out = net(x)
    assert out.shape == (n, out_chans, *img_shape)


def test_skip_projection_runs_decoder_at_embed_dim():
    """With skip_projection the decoder stage has embed_dim channels and fewer
    parameters, the projection gets gradients, and the output shape is
    unchanged."""
    in_chans, out_chans = 5, 3
    img_shape = (16, 32)
    n = 2
    device = get_device()
    net = _build_net(in_chans, out_chans, img_shape, skip_projection=True).to(device)
    assert len(net.skip_projs) == 1
    assert net.layer4.blocks[0].dim == 32
    n_params_concat = sum(
        p.numel() for p in _build_net(in_chans, out_chans, img_shape).parameters()
    )
    assert sum(p.numel() for p in net.parameters()) < n_params_concat
    x = torch.randn(n, in_chans, *img_shape, device=device)
    out = net(x)
    assert out.shape == (n, out_chans, *img_shape)
    out.sum().backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_skip_projection_absent_by_default():
    """The default adds no parameters, so existing checkpoints keep loading."""
    net = _build_net(5, 3, (16, 32))
    assert len(net.skip_projs) == 0
    assert not any("skip_proj" in k for k in net.state_dict())
    assert net.layer4.blocks[0].dim == 64


_EARTH_PADDING_CONF = {
    "activate": True,
    "mode": "earth",
    "pad_lat": [2, 1],
    "pad_lon": [2, 2],
}


@pytest.mark.parametrize(
    "conditioning, img_shape, options",
    [
        ("adaln", (16, 32), dict(patch_size=(2, 2))),
        ("adaln", (16, 32), dict(patch_size=(2, 4))),
        ("adaln", (9, 18), dict(patch_size=(2, 2), padding_conf=_EARTH_PADDING_CONF)),
        ("cln", (9, 18), dict(patch_size=(2, 2), padding_conf=_EARTH_PADDING_CONF)),
        ("adaln", (32, 64), dict(num_levels=2)),
        ("adaln", (32, 64), dict(num_levels=2, use_skip=False)),
        ("adaln", (32, 64), dict(num_levels=2, skip_projection=True)),
        (
            "adaln",
            (9, 18),
            dict(
                num_levels=2,
                lat_coords=torch.linspace(-80.0, 80.0, 9),
                padding_conf=_EARTH_PADDING_CONF,
            ),
        ),
        ("cln", (32, 64), dict(num_levels=2)),
        ("cln", (9, 18), dict(num_levels=2, patch_size=(2, 2))),
    ],
)
def test_token_grid_options_forward_backward(
    conditioning: Literal["adaln", "cln"], img_shape: tuple[int, int], options: dict
):
    """A coarser token grid and/or extra U-Net levels, with or without padding
    up to the token/window multiple, still return the original pixel
    resolution, and every parameter receives a gradient."""
    n, in_chans, out_chans = 2, 4, 2
    device = get_device()
    net = _build_net(
        in_chans, out_chans, img_shape, conditioning=conditioning, **options
    ).to(device)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    context = _cln_context(n, img_shape) if conditioning == "cln" else None
    out = net(x, context)
    assert out.shape == (n, out_chans, *img_shape)
    out.sum().backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_patch_size_sets_token_grid():
    """The U-Net stages run on the padded pixel grid divided by patch_size."""
    net = _build_net(4, 2, (16, 32), patch_size=(2, 2))
    assert net.padded_shape == (16, 32)
    assert net.token_shape == (8, 16)
    assert net.layer1.blocks[0].input_resolution == (8, 16)
    assert net.layer2.blocks[0].input_resolution == (4, 8)


def test_default_options_keep_parameter_shapes():
    """The defaults reproduce the previous network's parameter names and
    shapes, so existing checkpoints keep loading."""
    state = _build_net(5, 3, (16, 32)).state_dict()
    assert state["encoder.weight"].shape == (32, 5, 3, 3)
    assert state["final_linear.weight"].shape == (32, 64)
    assert not any("skip_proj" in k or "extra_" in k for k in state)


@pytest.mark.parametrize(
    "options, match",
    [
        (dict(patch_size=(0, 2)), "patch_size"),
        (dict(num_levels=0), "num_levels"),
        (dict(use_skip=False, skip_projection=True), "requires use_skip"),
    ],
)
def test_option_validation(options: dict, match: str):
    with pytest.raises(ValueError, match=match):
        _build_net(4, 2, (16, 32), **options)


def test_num_levels_keeps_bottleneck_and_halves_its_grid():
    """Extra levels are dim-preserving, so the bottleneck stages and the
    merge/expand around them are unchanged in size; each extra level halves
    the token grid they run on."""
    img_shape = (32, 64)
    one = _build_net(5, 3, img_shape, num_levels=1)
    two = _build_net(5, 3, img_shape, num_levels=2)
    for name in ("layer2", "layer3", "downsample", "upsample"):
        n_one = sum(p.numel() for p in getattr(one, name).parameters())
        n_two = sum(p.numel() for p in getattr(two, name).parameters())
        assert n_one == n_two, name
    assert sum(p.numel() for p in two.parameters()) > sum(
        p.numel() for p in one.parameters()
    )
    assert two.layer1.blocks[0].input_resolution == (32, 64)
    assert len(two.extra_encoders) == len(two.extra_decoders) == 1
    assert two.extra_encoders[0].blocks[0].input_resolution == (16, 32)
    assert two.extra_decoders[0].blocks[0].input_resolution == (16, 32)
    assert two.layer2.blocks[0].input_resolution == (8, 16)
    assert one.layer2.blocks[0].input_resolution == (16, 32)


def test_one_degree_shape_matches_four_degree_bottleneck():
    """patch_size (2, 2) + num_levels 2 at 1 degree gives the same bottleneck
    token grid and bottleneck parameter count as the 4-degree model."""

    def build(
        img_shape: tuple[int, int], patch_size: tuple[int, int], num_levels: int
    ) -> SwinTransformerNet:
        return _build_net(
            3,
            3,
            img_shape,
            embed_dim=16,
            num_heads=(2, 2, 2, 2),
            window_size=(4, 8),
            padding_conf={**_EARTH_PADDING_CONF, "pad_lon": [3, 3]},
            patch_size=patch_size,
            num_levels=num_levels,
            lat_coords=torch.linspace(-89.0, 89.0, img_shape[0]),
        )

    four_degree = build((45, 90), patch_size=(1, 1), num_levels=1)
    one_degree = build((180, 360), patch_size=(2, 2), num_levels=2)
    assert (
        one_degree.layer2.blocks[0].input_resolution
        == four_degree.layer2.blocks[0].input_resolution
    )
    for name in ("layer2", "layer3"):
        assert sum(p.numel() for p in getattr(one_degree, name).parameters()) == sum(
            p.numel() for p in getattr(four_degree, name).parameters()
        ), name
    device = get_device()
    one_degree = one_degree.to(device)
    with torch.no_grad():
        out = one_degree(torch.randn(1, 3, 180, 360, device=device))
    assert out.shape == (1, 3, 180, 360)


@pytest.mark.parametrize(
    "module, default_shapes, preserving_shapes",
    [
        (PatchMerging, {"reduction": (16, 32)}, {"reduction": (8, 32)}),
        (
            PatchExpanding,
            {"expand": (16, 8), "linear": (4, 4)},
            {"expand": (32, 8), "linear": (8, 8)},
        ),
    ],
)
def test_merge_expand_out_dim(module, default_shapes: dict, preserving_shapes: dict):
    """out_dim overrides the default channel doubling / halving; the default
    is unchanged."""
    for out_dim, shapes in [(None, default_shapes), (8, preserving_shapes)]:
        layer = module(8, out_dim=out_dim)
        for name, shape in shapes.items():
            assert getattr(layer, name).weight.shape == shape, (out_dim, name)
    x = torch.randn(2, 4, 8, 8)
    assert PatchMerging(8, out_dim=8)(x).shape == (2, 2, 4, 8)
    assert PatchExpanding(8, out_dim=8)(x).shape == (2, 8, 16, 8)
    with pytest.raises(ValueError, match="must be even"):
        PatchExpanding(7)
    assert PatchExpanding(7, out_dim=4)(torch.randn(2, 4, 8, 7)).shape == (2, 8, 16, 4)


def test_column_mixer():
    """Zeroing the ColumnMixer's Linear makes its output zero, so the folded
    residual ``x + column_mixer(x)`` in a block reduces to ``x``."""
    device = get_device()
    dim = 16
    mixer = ColumnMixer(dim).to(device)
    torch.nn.init.zeros_(mixer.fc.weight)
    torch.nn.init.zeros_(mixer.fc.bias)
    x = torch.randn(2, 4, 8, dim, device=device)
    out = mixer(x)
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_backward():
    in_chans, out_chans = 4, 2
    img_shape = (16, 32)
    n = 2
    device = get_device()
    net = _build_net(in_chans, out_chans, img_shape).to(device)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    out = net(x)
    out.sum().backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_cln_forward_backward():
    """CLN mode forward + backward: all params including CLN noise convs get grads."""
    in_chans, out_chans = 4, 2
    img_shape = (16, 32)
    n = 2
    device = get_device()
    net = _build_net(in_chans, out_chans, img_shape, conditioning="cln").to(device)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    context = _cln_context(n, img_shape)
    out = net(x, context)
    assert out.shape == (n, out_chans, *img_shape)
    out.sum().backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_cln_state_dict_keeps_conv_weight_shapes():
    """The channels-last CLN path reuses the 1x1 conv parameters, so the state
    dict (and therefore checkpoint compatibility) is unchanged."""
    net = _build_net(4, 2, (16, 32), conditioning="cln")
    state = net.state_dict()
    scale_keys = [k for k in state if k.endswith("norm1.W_scale_2d.weight")]
    n_blocks = sum(
        len(layer.blocks) for layer in (net.layer1, net.layer2, net.layer3, net.layer4)
    )
    assert len(scale_keys) == n_blocks
    for key in scale_keys:
        assert state[key].shape[2:] == (1, 1), key


def test_cln_padded_shape():
    """CLN mode with an img_shape that requires padding exercises pad + subsample."""
    in_chans, out_chans = 4, 2
    img_shape = (9, 18)  # not divisible by window_size * 2
    n = 2
    device = get_device()
    net = _build_net(in_chans, out_chans, img_shape, conditioning="cln").to(device)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    context = _cln_context(n, img_shape)
    out = net(x, context)
    assert out.shape == (n, out_chans, *img_shape)


def test_cln_noise_divergence():
    """Two forwards with different noise diverge after one optimizer step.

    CLN's zero-init noise convs (scale=1, bias=0) make the freshly-built model
    noise-independent at init.  After one step the convs move off zero and
    different noise fields produce distinct outputs.
    """
    in_chans, out_chans = 4, 2
    img_shape = (16, 32)
    n = 2
    device = get_device()
    net = _build_net(in_chans, out_chans, img_shape, conditioning="cln").to(device)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=1.0)

    x = torch.randn(n, in_chans, *img_shape, device=device)
    noise_a = torch.randn(n, _EMBED_DIM_NOISE, *img_shape, device=device)
    noise_b = torch.randn(n, _EMBED_DIM_NOISE, *img_shape, device=device)

    ctx_a = Context(
        embedding_scalar=None, embedding_pos=None, labels=None, noise=noise_a
    )
    ctx_b = Context(
        embedding_scalar=None, embedding_pos=None, labels=None, noise=noise_b
    )

    # Verify degenerate at init (zero-init noise convs).
    with torch.no_grad():
        assert torch.allclose(
            net(x, ctx_a), net(x, ctx_b)
        ), "Expected noise-independence at init"

    # Take one optimizer step to push noise convs off zero.
    out = net(x, ctx_a)
    out.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        out_a = net(x, ctx_a)
        out_b = net(x, ctx_b)
    assert not torch.allclose(out_a, out_b), "Expected noise-dependence after step"


def test_adaln_regression():
    """Forward pass produces the correct output shape in AdaLN mode."""
    in_chans, out_chans = 4, 2
    img_shape = (16, 32)
    n = 2
    device = get_device()
    torch.manual_seed(42)
    net_adaln = _build_net(in_chans, out_chans, img_shape).to(device)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    with torch.no_grad():
        out = net_adaln(x)
    assert out.shape == (n, out_chans, *img_shape)


def test_cpb_lat_coords_changes_output():
    """Two nets sharing weights but different lat_coords produce different outputs."""
    in_chans, out_chans = 4, 2
    img_shape = (16, 32)
    n = 2
    device = get_device()
    torch.manual_seed(0)
    lat_low = torch.full((img_shape[0],), 10.0, device=device)
    lat_high = torch.full((img_shape[0],), 60.0, device=device)
    net_low = _build_net(in_chans, out_chans, img_shape).to(device)
    net_low_state = net_low.state_dict()
    net_high = _build_net(in_chans, out_chans, img_shape, lat_coords=lat_high).to(
        device
    )
    net_high.load_state_dict(net_low_state)
    # Push cpb_mlp off zero so the latitude actually changes the bias.
    optimizer = torch.optim.SGD(net_low.parameters(), lr=1.0)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    net_low.train()
    net_low(x).sum().backward()
    optimizer.step()
    # Give net_high the same updated weights.
    net_high.load_state_dict(net_low.state_dict())
    net_low_lat = _build_net(in_chans, out_chans, img_shape, lat_coords=lat_low).to(
        device
    )
    net_low_lat.load_state_dict(net_low.state_dict())
    with torch.no_grad():
        out_low = net_low_lat(x)
        out_high = net_high(x)
    assert not torch.allclose(out_low, out_high), "lat_coords should change output"


@pytest.mark.parametrize("window_size", [(4, 4), (4, 8)])
def test_cpb_backward_with_lat_coords(window_size: tuple[int, int]):
    """Gradients reach cpb_mlp when lat_coords is provided, including with a
    non-square window whose window-row ordering the band index must match."""
    in_chans, out_chans = 4, 2
    img_shape = (16, 32)
    n = 2
    device = get_device()
    lat_coords = torch.linspace(-90.0, 90.0, img_shape[0], device=device)
    net = _build_net(
        in_chans, out_chans, img_shape, window_size=window_size, lat_coords=lat_coords
    ).to(device)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    net(x).sum().backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_cosine_attention_forward():
    """Forward pass completes without NaN; every WindowAttention2D has a tau param."""
    in_chans, out_chans = 5, 3
    img_shape = (16, 32)
    n = 2
    device = get_device()
    net = _build_net(in_chans, out_chans, img_shape).to(device)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    with torch.no_grad():
        out = net(x)
    assert not torch.isnan(out).any(), "NaN in output"
    assert out.shape == (n, out_chans, *img_shape)
    for name, module in net.named_modules():
        if isinstance(module, WindowAttention2D):
            assert hasattr(module, "tau"), f"tau missing on {name}"
            assert isinstance(module.tau, torch.nn.Parameter)


def test_v2_with_conditioning():
    """
    Forward + backward with AdaLN conditioning; all params including tau get grads.
    """
    in_chans, out_chans = 4, 2
    img_shape = (16, 32)
    n = 2
    embed_dim_scalar, embed_dim_labels = 8, 4
    device = get_device()
    context_config = ContextConfig(
        embed_dim_scalar=embed_dim_scalar,
        embed_dim_labels=embed_dim_labels,
        embed_dim_noise=0,
        embed_dim_pos=0,
    )
    net = _build_net(in_chans, out_chans, img_shape, context_config=context_config).to(
        device
    )
    x = torch.randn(n, in_chans, *img_shape, device=device)
    context = Context(
        embedding_scalar=torch.randn(n, embed_dim_scalar, device=device),
        embedding_pos=None,
        labels=torch.randn(n, embed_dim_labels, device=device),
        noise=None,
    )
    out = net(x, context)
    assert out.shape == (n, out_chans, *img_shape)
    out.sum().backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_earth_padding_forward():
    img_shape = (9, 18)
    padding_conf = {
        "activate": True,
        "mode": "earth",
        "pad_lat": [2, 1],
        "pad_lon": [2, 2],
    }
    net = _build_net(3, 3, img_shape, padding_conf=padding_conf).to(get_device())
    x = torch.randn(2, 3, *img_shape, device=get_device())
    assert net(x).shape == (2, 3, *img_shape)


@pytest.mark.parametrize("pad_lat", [(2, 0), (0, 2), (0, 0)])
def test_earth_padding_lat_coords_allow_one_sided_or_zero_padding(
    pad_lat: tuple[int, int],
):
    img_shape = (9, 18)
    lat_coords = torch.arange(img_shape[0], dtype=torch.float32)
    padding_conf = {
        "activate": True,
        "mode": "earth",
        "pad_lat": list(pad_lat),
        "pad_lon": [0, 0],
    }

    net = _build_net(3, 3, img_shape, lat_coords=lat_coords, padding_conf=padding_conf)

    # The south padding is extended to reach the window multiple.
    pad_s = net.padded_shape[0] - img_shape[0] - pad_lat[0]
    expected_pieces = []
    if pad_lat[0] > 0:
        expected_pieces.append(torch.flip(lat_coords[: pad_lat[0]], dims=[0]))
    expected_pieces.append(lat_coords)
    if pad_s > 0:
        expected_pieces.append(torch.flip(lat_coords[-pad_s:], dims=[0]))
    expected = torch.cat(expected_pieces)
    _assert_window_cos_lat(net.layer1.blocks[0], torch.cos(torch.deg2rad(expected)))


def test_earth_padding_cln_forward():
    img_shape = (9, 18)
    padding_conf = {
        "activate": True,
        "mode": "earth",
        "pad_lat": [2, 1],
        "pad_lon": [2, 2],
    }
    net = _build_net(3, 3, img_shape, conditioning="cln", padding_conf=padding_conf).to(
        get_device()
    )
    noise = torch.randn(2, _EMBED_DIM_NOISE, *img_shape, device=get_device())
    ctx = Context(embedding_scalar=None, embedding_pos=None, labels=None, noise=noise)
    assert net(torch.randn(2, 3, *img_shape, device=get_device()), ctx).shape == (
        2,
        3,
        *img_shape,
    )


def _per_window_position_bias(
    attn: WindowAttention2D,
    cos_lat_per_window: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Position bias with the CPB MLP evaluated separately for every window,
    as implemented before the per-latitude-band deduplication. The module
    precomputes its coordinates in float32 at construction; ``dtype`` is the
    dtype they are cast to before the MLP."""
    N = attn.window_size[0] * attn.window_size[1]
    nW = cos_lat_per_window.shape[0]
    base = attn.relative_coords_base.float()
    h_coords = base[:, 0]
    w_coords = base[:, 1].unsqueeze(0) * cos_lat_per_window.float().unsqueeze(1)
    coords = torch.stack([h_coords.unsqueeze(0).expand(nW, -1), w_coords], dim=-1)
    coords_log = (torch.sign(coords) * torch.log(1.0 + coords.abs())).to(dtype)
    bias = 16.0 * torch.sigmoid(attn.cpb_mlp(coords_log))
    return bias.permute(0, 2, 1).reshape(nW, attn.num_heads, N, N)


def _reference_window_attention(
    attn: WindowAttention2D,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    cos_lat: torch.Tensor | None,
) -> torch.Tensor:
    """Explicit-logits cosine attention, as implemented before the switch to
    ``F.scaled_dot_product_attention``. Shares parameters with ``attn``.

    Takes a per-window ``cos_lat`` of shape ``(nW,)``, so it also checks the
    per-band CPB evaluation against the per-window one.
    """
    B_, N, C = x.shape
    qkv = (
        attn.qkv(x)
        .reshape(B_, N, 3, attn.num_heads, C // attn.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    norm_q = torch.norm(q, dim=-1, keepdim=True)
    norm_k = torch.norm(k, dim=-1, keepdim=True).transpose(-2, -1)
    logits = (q @ k.transpose(-2, -1)) / (norm_q * norm_k).clamp(min=1e-6)
    logits = logits / attn.tau.clamp(min=0.01)
    if cos_lat is None:
        bias = 16.0 * torch.sigmoid(attn.cpb_mlp(attn.relative_coords_log))
        bias = bias.permute(1, 0).reshape(attn.num_heads, N, N)
        logits = logits + bias.unsqueeze(0)
    else:
        nW = cos_lat.shape[0]
        bias = _per_window_position_bias(attn, cos_lat, dtype=x.dtype)
        logits = logits.view(B_ // nW, nW, attn.num_heads, N, N) + bias.unsqueeze(0)
        logits = logits.view(B_, attn.num_heads, N, N)
    if mask is not None:
        nW = mask.shape[0]
        logits = logits.view(B_ // nW, nW, attn.num_heads, N, N) + mask.unsqueeze(
            1
        ).unsqueeze(0)
        logits = logits.view(-1, attn.num_heads, N, N)
    probs = torch.softmax(logits, dim=-1)
    out = (probs @ v).transpose(1, 2).reshape(B_, N, C)
    return attn.proj(out)


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("use_lat", [False, True])
def test_window_attention_matches_explicit_logits(use_mask: bool, use_lat: bool):
    """SDPA-based attention equals the explicit softmax(QK^T + bias) V
    formulation for outputs and parameter gradients."""
    device = get_device()
    torch.manual_seed(0)
    dim, num_heads, window_size = 16, 4, (4, 4)
    H, W, B = 8, 16, 2
    nH_win, nW_win = H // window_size[0], W // window_size[1]
    nW = nH_win * nW_win
    cos_lat = (
        torch.cos(
            torch.deg2rad(
                torch.linspace(-60.0, 60.0, nH_win, device=device, dtype=torch.float64)
            )
        )
        if use_lat
        else None
    )
    attn = (
        WindowAttention2D(
            dim, window_size, num_heads, cos_lat=cos_lat, num_windows_w=nW_win
        )
        .to(device)
        .double()
    )
    with torch.no_grad():
        for param in attn.parameters():
            param.normal_()
        attn.tau.abs_().add_(0.1)
    x = torch.randn(B * nW, 16, dim, device=device, dtype=torch.float64)
    x.requires_grad_(True)
    mask = None
    if use_mask:
        mask = torch.zeros(nW, 16, 16, device=device, dtype=torch.float64)
        mask[:, :8, 8:] = -100.0
        mask[:, 8:, :8] = -100.0
    out = attn(x, mask=mask)
    assert out.shape == (B * nW, 16, dim)
    grads = torch.autograd.grad(out.square().sum(), [x, *attn.parameters()])
    cos_lat_per_window = (
        cos_lat.repeat_interleave(nW_win) if cos_lat is not None else None
    )
    ref = _reference_window_attention(attn, x, mask, cos_lat_per_window)
    ref_grads = torch.autograd.grad(ref.square().sum(), [x, *attn.parameters()])
    torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-10)
    for g, g_ref in zip(grads, ref_grads):
        torch.testing.assert_close(g, g_ref, atol=1e-8, rtol=1e-8)


def test_window_cos_lat_matches_rolled_window_means():
    """Precomputed per-window-row ``cos(lat)`` equals the mean of the
    (shifted) row ``cos(lat)`` in each window row, top to bottom."""
    H, W = 8, 16
    ws = (4, 4)
    cos_lat = torch.cos(torch.deg2rad(torch.linspace(-70.0, 70.0, H)))
    for shift in (0, 2):
        band_cos_lat = window_cos_lat(cos_lat, (H, W), ws, shift)
        assert band_cos_lat is not None
        assert band_cos_lat.shape == (H // ws[0],)
        rolled = torch.roll(cos_lat, -shift)
        expected = rolled.reshape(H // ws[0], ws[0]).mean(1)
        torch.testing.assert_close(band_cos_lat, expected)
    assert window_cos_lat(None, (H, W), ws, 0) is None


def test_window_attention_requires_num_windows_w_with_cos_lat():
    """Omitting the windows-per-row count is a construction-time error, not a
    forward-time shape mismatch."""
    with pytest.raises(ValueError, match="num_windows_w"):
        WindowAttention2D(16, (4, 4), 4, cos_lat=torch.ones(3))
    # Without latitude scaling the count is not needed.
    WindowAttention2D(16, (4, 4), 4)


@pytest.mark.parametrize("use_lat", [False, True])
def test_position_bias_matches_per_window_evaluation(use_lat: bool):
    """Evaluating the CPB MLP once per latitude band and gathering to windows
    gives the same bias as evaluating it for every window, and windows in the
    same window-row share identical bias rows. Without latitude scaling the
    bias has no window dimension at all."""
    device = get_device()
    torch.manual_seed(0)
    dim, num_heads, window_size = 16, 4, (4, 4)
    N = window_size[0] * window_size[1]
    nH_win, nW_win = 3, 5
    cos_lat = (
        torch.cos(torch.deg2rad(torch.linspace(-75.0, 75.0, nH_win, device=device)))
        if use_lat
        else None
    )
    attn = WindowAttention2D(
        dim, window_size, num_heads, cos_lat=cos_lat, num_windows_w=nW_win
    ).to(device)
    with torch.no_grad():
        for param in attn.cpb_mlp.parameters():
            param.normal_()
    bias = attn._position_bias()
    if not use_lat:
        assert attn.coords_log is None
        assert attn.band_index is None
        assert bias.shape == (num_heads, N, N)
        expected = 16.0 * torch.sigmoid(attn.cpb_mlp(attn.relative_coords_log))
        expected = expected.permute(1, 0).reshape(num_heads, N, N)
        torch.testing.assert_close(bias, expected)
        return
    assert cos_lat is not None
    assert attn.coords_log.shape == (nH_win, N * N, 2)
    assert attn.band_index.shape == (nH_win * nW_win,)
    assert bias.shape == (nH_win * nW_win, num_heads, N, N)
    expected = _per_window_position_bias(attn, cos_lat.repeat_interleave(nW_win))
    torch.testing.assert_close(bias, expected)
    bias_by_row = bias.view(nH_win, nW_win, num_heads, N, N)
    assert torch.equal(bias_by_row, bias_by_row[:, :1].expand_as(bias_by_row))
    # Distinct latitude bands do give distinct biases.
    assert not torch.equal(bias_by_row[0, 0], bias_by_row[1, 0])


def test_blocks_precompute_cpb_coords_per_shift():
    """Regular and shifted blocks hold distinct precomputed coordinate buffers
    that follow the module across devices and are absent without lat_coords."""
    img_shape = (16, 32)
    lat = torch.linspace(-80.0, 80.0, img_shape[0])
    net = _build_net(4, 2, img_shape, lat_coords=lat).to(get_device())
    regular, shifted = net.layer1.blocks[0].attn, net.layer1.blocks[1].attn
    n_bands = img_shape[0] // 4
    n_windows = n_bands * (img_shape[1] // 4)
    for attn in (regular, shifted):
        assert attn.coords_log is not None
        assert attn.coords_log.shape == (n_bands, 16 * 16, 2)
        assert attn.coords_log.device.type == get_device().type
        assert attn.band_index.shape == (n_windows,)
        assert attn.band_index.device.type == get_device().type
    assert not torch.equal(regular.coords_log, shifted.coords_log)
    state_keys = {k.split(".")[-1] for k in net.state_dict()}
    assert "coords_log" not in state_keys
    assert "band_index" not in state_keys
    net_no_lat = _build_net(4, 2, img_shape)
    assert net_no_lat.layer1.blocks[0].attn.coords_log is None
    assert net_no_lat.layer1.blocks[0].attn.band_index is None


_REGRESSION_DIR = pathlib.Path(__file__).parent / "testdata"
_REGRESSION_PADDING_CONF = {
    "activate": True,
    "mode": "earth",
    "pad_lat": [2, 1],
    "pad_lon": [3, 3],
}


def _build_regression_net(
    conditioning: Literal["adaln", "cln"], context_config: ContextConfig | None
) -> SwinTransformerNet:
    """A tiny net with earth padding on an odd grid so both padding stages are
    exercised; built under a fixed seed so its initial weights are
    reproducible."""
    torch.manual_seed(0)
    net = _build_net(
        4,
        2,
        (9, 18),
        context_config=context_config,
        conditioning=conditioning,
        embed_dim=16,
        num_heads=(2, 2, 2, 2),
        mlp_layer="swiglu",
        lat_coords=torch.linspace(-80.0, 80.0, 9),
        padding_conf=_REGRESSION_PADDING_CONF,
    )
    return net.eval()


def test_regression_adaln():
    """The AdaLN forward pass matches a stored reference output.

    Locks the numerics of the encoder/decoder/level wiring on CPU in float32 so
    refactors of that wiring can be checked to be bit-for-bit unchanged.
    """
    n, embed_dim_scalar, embed_dim_labels = 2, 8, 4
    net = _build_regression_net(
        "adaln",
        ContextConfig(
            embed_dim_scalar=embed_dim_scalar,
            embed_dim_labels=embed_dim_labels,
            embed_dim_noise=0,
            embed_dim_pos=0,
        ),
    )
    torch.manual_seed(0)
    x = torch.randn(n, 4, 9, 18)
    context = Context(
        embedding_scalar=torch.randn(n, embed_dim_scalar),
        embedding_pos=None,
        labels=torch.randn(n, embed_dim_labels),
        noise=None,
    )
    with torch.no_grad():
        out = net(x, context)
    validate_tensor_dict({"output": out}, _REGRESSION_DIR / "swin_regression_adaln.pt")


def test_regression_cln():
    """The CLN (noise-conditioned) forward pass matches a stored reference."""
    n = 2
    net = _build_regression_net("cln", None)
    torch.manual_seed(0)
    x = torch.randn(n, 4, 9, 18)
    context = Context(
        embedding_scalar=None,
        embedding_pos=None,
        labels=None,
        noise=torch.randn(n, _EMBED_DIM_NOISE, 9, 18),
    )
    with torch.no_grad():
        out = net(x, context)
    validate_tensor_dict({"output": out}, _REGRESSION_DIR / "swin_regression_cln.pt")


_PADDED_REGRESSION_IMG_SHAPE = (9, 18)  # not a multiple of 2 * window_size


def _build_padded_net(
    padding_mode: Literal["earth", "mirror"] | None,
    conditioning: Literal["adaln", "cln"],
) -> SwinTransformerNet:
    """Small randomly-initialized net exercising padding and cos-lat CPB.

    All parameters (including the zero-initialized CPB output layer) are
    drawn from a seeded normal on CPU so outputs depend on the latitudes and
    are reproducible across devices.
    """
    torch.manual_seed(0)
    if padding_mode is None:
        padding_conf = None
    else:
        padding_conf = {
            "activate": True,
            "mode": padding_mode,
            "pad_lat": [2, 1],
            "pad_lon": [2, 2],
        }
    context_config = ContextConfig(
        embed_dim_scalar=0,
        embed_dim_labels=0,
        embed_dim_noise=_EMBED_DIM_NOISE if conditioning == "cln" else 0,
        embed_dim_pos=0,
    )
    net = SwinTransformerNet(
        in_chans=3,
        out_chans=2,
        img_shape=_PADDED_REGRESSION_IMG_SHAPE,
        embed_dim=16,
        depth_multiplier=1,
        num_heads=(2, 2, 2, 2),
        window_size=(4, 4),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        context_config=context_config,
        conditioning=conditioning,
        lat_coords=torch.linspace(-80.0, 80.0, _PADDED_REGRESSION_IMG_SHAPE[0]),
        padding_conf=padding_conf,
    )
    # At 0.2 the cln net amplifies float32 rounding to ~1e-3, so its outputs
    # differ across platforms; 0.05 keeps it well conditioned.
    scale = 0.2 if conditioning == "adaln" else 0.05
    with torch.no_grad():
        for param in net.parameters():
            param.copy_(scale * torch.randn_like(param))
    return net


def _padded_regression_inputs(conditioning: str) -> tuple[torch.Tensor, Context]:
    torch.manual_seed(1)
    x = torch.randn(2, 3, *_PADDED_REGRESSION_IMG_SHAPE)
    noise = (
        torch.randn(2, _EMBED_DIM_NOISE, *_PADDED_REGRESSION_IMG_SHAPE)
        if conditioning == "cln"
        else None
    )
    context = Context(
        embedding_scalar=None, embedding_pos=None, labels=None, noise=noise
    )
    return x, context


@pytest.mark.parametrize(
    "padding_mode, conditioning",
    [
        pytest.param(None, "adaln", id="no_padding"),
        pytest.param("earth", "adaln", id="earth_adaln"),
        pytest.param("earth", "cln", id="earth_cln"),
        pytest.param("mirror", "cln", id="mirror_cln"),
    ],
)
def test_padded_output_regression(
    padding_mode: Literal["earth", "mirror"] | None,
    conditioning: Literal["adaln", "cln"],
):
    """Regression test for the padded forward pass, covering window-multiple
    padding and cos-lat CPB scaling."""
    device = get_device()
    net = _build_padded_net(padding_mode, conditioning).to(device)
    x, context = _padded_regression_inputs(conditioning)
    if context.noise is not None:
        context.noise = context.noise.to(device)
    with torch.no_grad():
        out = net(x.to(device), context)
    validate_tensor_dict(
        {"output": out},
        _REGRESSION_DIR
        / f"test_padded_output_regression-{padding_mode}-{conditioning}.pt",
        atol=1e-5,
        rtol=1e-4,
    )


def _cos_deg(lat: torch.Tensor) -> torch.Tensor:
    return torch.cos(torch.deg2rad(lat))


def _band_cos_lat(attn: WindowAttention2D) -> torch.Tensor:
    """The per-window-row ``cos(lat)`` ``attn`` scales its CPB by, recovered
    from its precomputed log coordinates at a unit longitude offset."""
    unit = torch.nonzero(attn.relative_coords_base[:, 1] == 1)[0, 0]
    return torch.expm1(attn.coords_log[:, unit, 1])


def _assert_window_cos_lat(block: SwinTransformerBlock, row_cos_lat: torch.Tensor):
    """``block`` scales its CPB by the mean ``cos`` of each (shifted) window
    row, given the ``cos(lat)`` of each of its (padded) feature-map rows."""
    ws_h = block.window_size[0]
    rows = torch.roll(row_cos_lat, -block.shift_size[0]).reshape(-1, ws_h)
    torch.testing.assert_close(_band_cos_lat(block.attn), rows.mean(1))


def test_shifted_window_cos_lat_is_polar_at_poles():
    """Shifted windows that wrap across both poles must see a polar latitude.

    On a 4-degree grid (-88..88) with earth padding [2, 1] the padded grid has
    48 rows; with window height 4 the last shifted window row holds rows
    from both poles (lats [88, 88, -84, -88]). Its CPB cos-lat scaling must
    reflect a polar latitude, not the near-equatorial signed mean.
    """
    lat_coords = torch.arange(-88.0, 88.0 + 1, 4.0)
    img_shape = (lat_coords.shape[0], 16)
    net = _build_net(
        3,
        3,
        img_shape,
        window_size=(4, 8),
        lat_coords=lat_coords,
        padding_conf={
            "activate": True,
            "mode": "earth",
            "pad_lat": [2, 1],
            "pad_lon": [0, 0],
        },
    )
    padded_lat = torch.cat([lat_coords[:2].flip(0), lat_coords, lat_coords[-1:]])
    for block in (*net.layer1.blocks, *net.layer4.blocks):
        _assert_window_cos_lat(block, _cos_deg(padded_lat))
        if block.shift_size[0] > 0:
            assert _band_cos_lat(block.attn)[-1] < 0.15


def _reference_row(
    mode: Literal["earth", "mirror"], H: int, row: int
) -> tuple[int, bool]:
    """Data row at ``row`` of an ``H``-row grid, walking one pole at a time.

    Returns the row and whether longitude is rotated by 180 degrees.
    """
    rotated = False
    while not 0 <= row < H:
        if mode == "earth":
            row = -1 - row if row < 0 else 2 * H - 1 - row
            rotated = not rotated
        else:
            row = -row if row < 0 else 2 * (H - 1) - row
    return row, rotated


@pytest.mark.parametrize("mode", ["earth", "mirror"])
@pytest.mark.parametrize(
    "img_shape, pad_north, pad_west, padded_shape",
    [
        pytest.param((3, 4), 1, 1, (5, 6), id="both_edges"),
        pytest.param((3, 2), 0, 0, (9, 2), id="lat_past_whole_grid"),
        pytest.param((8, 4), 0, 0, (8, 16), id="lon_past_whole_grid"),
        pytest.param((9, 18), 2, 2, (24, 24), id="window_multiple"),
    ],
)
def test_padding_index_continues_the_sphere(
    mode: Literal["earth", "mirror"],
    img_shape: tuple[int, int],
    pad_north: int,
    pad_west: int,
    padded_shape: tuple[int, int],
):
    H, W = img_shape
    rows, cols = _padding_index(mode, img_shape, pad_north, pad_west, padded_shape)
    for i in range(padded_shape[0]):
        row, rotated = _reference_row(mode, H, i - pad_north)
        assert torch.all(rows[i] == row)
        expected_cols = (torch.arange(padded_shape[1]) - pad_west) % W
        if rotated:
            expected_cols = (expected_cols + W // 2) % W
        assert torch.equal(cols[i], expected_cols)


@pytest.mark.parametrize("mode", ["earth", "mirror", None])
def test_pads_window_multiple_with_data(mode: Literal["earth", "mirror"] | None):
    """Padded rows hold data and its latitude; ``None`` uses earth padding."""
    img_shape = (9, 18)
    lat_coords = torch.linspace(-80.0, 80.0, img_shape[0])
    padding_conf = (
        None
        if mode is None
        else {"activate": True, "mode": mode, "pad_lat": [2, 1], "pad_lon": [2, 2]}
    )
    net = _build_net(3, 3, img_shape, lat_coords=lat_coords, padding_conf=padding_conf)
    x = torch.randn(1, 3, *img_shape)
    expected_rows, expected_cols = _padding_index(
        mode or "earth", img_shape, *net.pad_offset, net.padded_shape
    )
    torch.testing.assert_close(net._pad(x), x[..., expected_rows, expected_cols])
    _assert_window_cos_lat(
        net.layer1.blocks[0], _cos_deg(lat_coords[expected_rows[:, 0]])
    )
    with torch.no_grad():
        assert net(x).shape == x.shape


@pytest.mark.parametrize(
    "patch_size, num_levels", [((2, 2), 2), ((2, 1), 1), ((1, 1), 3)]
)
def test_cos_lat_follows_padded_data_rows_at_every_level(
    patch_size: tuple[int, int], num_levels: int
):
    """With a coarser token grid and extra U-Net levels, every stage scales
    its CPB by the mean ``cos(lat)`` of the padded data rows its token rows
    cover: ``patch_size[0]`` pixel rows per token row, then doubling per
    level."""
    img_shape = (9, 18)
    lat_coords = torch.linspace(-80.0, 80.0, img_shape[0])
    net = _build_net(
        3,
        3,
        img_shape,
        lat_coords=lat_coords,
        padding_conf=_EARTH_PADDING_CONF,
        patch_size=patch_size,
        num_levels=num_levels,
    )
    rows, _ = _padding_index("earth", img_shape, *net.pad_offset, net.padded_shape)
    row_cos_lat = _cos_deg(lat_coords[rows[:, 0]])  # (Hp,)
    stages = [
        (0, net.layer1),
        (0, net.layer4),
        (num_levels, net.layer2),
        (num_levels, net.layer3),
        *((level, stage) for level, stage in enumerate(net.extra_encoders, 1)),
        *((level, stage) for level, stage in enumerate(net.extra_decoders, 1)),
    ]
    for level, stage in stages:
        rows_per_token = patch_size[0] * 2**level
        level_cos_lat = row_cos_lat.reshape(-1, rows_per_token).mean(1)
        for block in stage.blocks:
            _assert_window_cos_lat(block, level_cos_lat)


def test_downsampled_cos_lat_is_polar_when_rows_wrap_both_poles():
    """Padded rows that wrap the whole grid pair rows from opposite poles.

    With 2 rows at -80 and 80 degrees, every row pair merged into the
    half-resolution stages spans both poles; their CPB scaling must stay
    polar instead of the equatorial ``cos`` of the pair's signed mean.
    """
    net = _build_net(
        3, 3, (2, 8), window_size=(4, 8), lat_coords=torch.tensor([-80.0, 80.0])
    )
    cos_80 = torch.cos(torch.deg2rad(torch.tensor(80.0)))
    for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
        for block in layer.blocks:
            band_cos_lat = _band_cos_lat(block.attn)
            torch.testing.assert_close(band_cos_lat, cos_80.expand_as(band_cos_lat))


@pytest.mark.parametrize(
    "pad_lat, pad_lon",
    [([-1, 0], [0, 0]), ([0, 0], [0, -2]), ([3], [0, 0]), ([0, 0], [1, 2, 3])],
)
def test_padding_config_rejects_invalid_sizes(pad_lat: list[int], pad_lon: list[int]):
    with pytest.raises(ValueError, match="pad_l"):
        TensorPaddingConfig(pad_lat=pad_lat, pad_lon=pad_lon)


def test_earth_padding_rejects_odd_longitudes_when_crossing_poles():
    with pytest.raises(ValueError, match="even number of longitudes"):
        _build_net(3, 3, (9, 17))
    _build_net(3, 3, (8, 17), window_size=(4, 8))  # rows need no padding
