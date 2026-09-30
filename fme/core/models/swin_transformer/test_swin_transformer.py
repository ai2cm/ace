import pathlib
from typing import Literal

import pytest
import torch

from fme.core.device import get_device
from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.testing.regression import validate_tensor_dict

from .swin_layers import ColumnMixer, WindowAttention2D
from .swin_transformer import SwinTransformerNet

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
    embed_dim: int = 32,
    num_heads: tuple[int, ...] = (2, 4, 4, 2),
    window_size: tuple[int, int] = (4, 4),
    mlp_layer: str = "mlp",
    lat_coords: torch.Tensor | None = None,
    padding_conf: dict | None = None,
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
        context_config=context_config,
        conditioning=conditioning,
        mlp_layer=mlp_layer,
        lat_coords=lat_coords,
        padding_conf=padding_conf,
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
    # Push cpb_mlp off zero so lat_mean actually changes the bias.
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


def test_cpb_backward_with_lat_coords():
    """Gradients reach cpb_mlp when lat_coords is provided."""
    in_chans, out_chans = 4, 2
    img_shape = (16, 32)
    n = 2
    device = get_device()
    lat_coords = torch.linspace(-90.0, 90.0, img_shape[0], device=device)
    net = _build_net(in_chans, out_chans, img_shape, lat_coords=lat_coords).to(device)
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

    expected_pieces = []
    if pad_lat[0] > 0:
        expected_pieces.append(torch.flip(lat_coords[: pad_lat[0]], dims=[0]))
    expected_pieces.append(lat_coords)
    if pad_lat[1] > 0:
        expected_pieces.append(torch.flip(lat_coords[-pad_lat[1] :], dims=[0]))
    expected = torch.cat(expected_pieces)
    pad_h = net.padded_shape[0] - expected.shape[0]
    if pad_h > 0:
        expected = torch.cat([expected, expected[-1:].expand(pad_h)])
    torch.testing.assert_close(net.layer1.blocks[0].lat_coords, expected)


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
