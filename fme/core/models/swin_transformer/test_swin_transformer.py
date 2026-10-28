import os
from typing import Literal

import pytest
import torch

from fme.core.device import get_device
from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.testing.regression import validate_tensor_dict

from .swin_layers import ColumnMixer, WindowAttention2D
from .swin_transformer import SwinTransformerNet

DIR = os.path.abspath(os.path.dirname(__file__))
_EMBED_DIM_NOISE = 8


def _build_net(
    in_chans: int,
    out_chans: int,
    img_shape: tuple[int, int],
    context_config: ContextConfig | None = None,
    use_skip: bool = True,
) -> SwinTransformerNet:
    return SwinTransformerNet(
        in_chans=in_chans,
        out_chans=out_chans,
        img_shape=img_shape,
        embed_dim=32,
        depth_multiplier=1,
        num_heads=(2, 4, 4, 2),
        window_size=(4, 4),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        use_skip=use_skip,
        context_config=context_config,
    )


def _build_cln_net(
    in_chans: int,
    out_chans: int,
    img_shape: tuple[int, int],
    use_skip: bool = True,
    padding_conf: dict | None = None,
) -> SwinTransformerNet:
    context_config = ContextConfig(
        embed_dim_scalar=0,
        embed_dim_labels=0,
        embed_dim_noise=_EMBED_DIM_NOISE,
        embed_dim_pos=0,
    )
    return SwinTransformerNet(
        in_chans=in_chans,
        out_chans=out_chans,
        img_shape=img_shape,
        embed_dim=32,
        depth_multiplier=1,
        num_heads=(2, 4, 4, 2),
        window_size=(4, 4),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        use_skip=use_skip,
        context_config=context_config,
        conditioning="cln",
        padding_conf=padding_conf,
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
    net = _build_cln_net(in_chans, out_chans, img_shape).to(device)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    context = Context(
        embedding_scalar=None,
        embedding_pos=None,
        labels=None,
        noise=torch.randn(n, _EMBED_DIM_NOISE, *img_shape, device=device),
    )
    out = net(x, context)
    assert out.shape == (n, out_chans, *img_shape)
    out.sum().backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_cln_padded_shape():
    """CLN mode with an img_shape that requires padding exercises pad + subsample."""
    in_chans, out_chans = 4, 2
    img_shape = (9, 18)  # not divisible by window_size * 2
    n = 2
    device = get_device()
    net = _build_cln_net(in_chans, out_chans, img_shape).to(device)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    context = Context(
        embedding_scalar=None,
        embedding_pos=None,
        labels=None,
        noise=torch.randn(n, _EMBED_DIM_NOISE, *img_shape, device=device),
    )
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
    net = _build_cln_net(in_chans, out_chans, img_shape).to(device)
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
    net_high = SwinTransformerNet(
        in_chans=in_chans,
        out_chans=out_chans,
        img_shape=img_shape,
        embed_dim=32,
        depth_multiplier=1,
        num_heads=(2, 4, 4, 2),
        window_size=(4, 4),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        lat_coords=lat_high,
    ).to(device)
    net_high.load_state_dict(net_low_state)
    # Push cpb_mlp off zero so lat_mean actually changes the bias.
    optimizer = torch.optim.SGD(net_low.parameters(), lr=1.0)
    x = torch.randn(n, in_chans, *img_shape, device=device)
    net_low.train()
    net_low(x).sum().backward()
    optimizer.step()
    # Give net_high the same updated weights.
    net_high.load_state_dict(net_low.state_dict())
    net_low_lat = SwinTransformerNet(
        in_chans=in_chans,
        out_chans=out_chans,
        img_shape=img_shape,
        embed_dim=32,
        depth_multiplier=1,
        num_heads=(2, 4, 4, 2),
        window_size=(4, 4),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        lat_coords=lat_low,
    ).to(device)
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
    net = SwinTransformerNet(
        in_chans=in_chans,
        out_chans=out_chans,
        img_shape=img_shape,
        embed_dim=32,
        depth_multiplier=1,
        num_heads=(2, 4, 4, 2),
        window_size=(4, 4),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        lat_coords=lat_coords,
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
    net = SwinTransformerNet(
        3,
        3,
        img_shape,
        embed_dim=32,
        num_heads=(2, 4, 4, 2),
        window_size=(4, 4),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        padding_conf=padding_conf,
    ).to(get_device())
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

    net = SwinTransformerNet(
        3,
        3,
        img_shape,
        embed_dim=32,
        num_heads=(2, 4, 4, 2),
        window_size=(4, 4),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        lat_coords=lat_coords,
        padding_conf=padding_conf,
    )

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
    net = _build_cln_net(3, 3, img_shape, padding_conf=padding_conf).to(get_device())
    noise = torch.randn(2, _EMBED_DIM_NOISE, *img_shape, device=get_device())
    ctx = Context(embedding_scalar=None, embedding_pos=None, labels=None, noise=noise)
    assert net(torch.randn(2, 3, *img_shape, device=get_device()), ctx).shape == (
        2,
        3,
        *img_shape,
    )


_REGRESSION_IMG_SHAPE = (9, 18)  # not a multiple of 2 * window_size


def _build_padded_net(
    padding_mode: Literal["earth", "mirror"] | None,
    conditioning: Literal["adaln", "cln"],
    **kwargs,
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
        img_shape=_REGRESSION_IMG_SHAPE,
        embed_dim=16,
        depth_multiplier=1,
        num_heads=(2, 2, 2, 2),
        window_size=(4, 4),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        context_config=context_config,
        conditioning=conditioning,
        lat_coords=torch.linspace(-80.0, 80.0, _REGRESSION_IMG_SHAPE[0]),
        padding_conf=padding_conf,
        **kwargs,
    )
    with torch.no_grad():
        for param in net.parameters():
            param.copy_(0.2 * torch.randn_like(param))
    return net


def _regression_inputs(conditioning: str) -> tuple[torch.Tensor, Context]:
    torch.manual_seed(1)
    x = torch.randn(2, 3, *_REGRESSION_IMG_SHAPE)
    noise = (
        torch.randn(2, _EMBED_DIM_NOISE, *_REGRESSION_IMG_SHAPE)
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
def test_padded_output_is_unchanged(
    padding_mode: Literal["earth", "mirror"] | None,
    conditioning: Literal["adaln", "cln"],
):
    """Default padding behavior must not change, so trained checkpoints keep
    giving the same outputs."""
    device = get_device()
    net = _build_padded_net(padding_mode, conditioning).to(device)
    x, context = _regression_inputs(conditioning)
    if context.noise is not None:
        context.noise = context.noise.to(device)
    with torch.no_grad():
        out = net(x.to(device), context)
    validate_tensor_dict(
        {"output": out},
        os.path.join(
            DIR,
            "testdata",
            f"test_padded_output_is_unchanged-{padding_mode}-{conditioning}.pt",
        ),
        atol=1e-5,
        rtol=1e-4,
    )


def test_consistent_padding_shifted_window_lat_mean_is_polar_at_poles():
    """Shifted windows that wrap across both poles must see a polar latitude.

    On a 4-degree grid (-88..88) with earth padding [2, 1] the padded grid has
    48 rows; with window height 4 the last shifted window row holds rows
    from both poles (lats [88, 88, -84, -88]). Its CPB cos-lat scaling must
    reflect a polar latitude, not the near-equatorial signed mean.
    """
    lat_coords = torch.arange(-88.0, 88.0 + 1, 4.0)
    img_shape = (lat_coords.shape[0], 16)
    net = SwinTransformerNet(
        3,
        3,
        img_shape,
        embed_dim=16,
        num_heads=(2, 2, 2, 2),
        window_size=(4, 8),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        lat_coords=lat_coords,
        padding_conf={
            "activate": True,
            "mode": "earth",
            "pad_lat": [2, 1],
            "pad_lon": [0, 0],
        },
        consistent_padding=True,
    )
    for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
        for block in layer.blocks:
            ws_h = block.window_size[0]
            sh = block.shift_size[0]
            # Latitudes of each window row, as gathered by the cyclic shift.
            rows = torch.roll(block.lat_coords, -sh).reshape(-1, ws_h)
            abs_rows = rows.abs()
            lat_mean = block.window_lat_mean()  # (nH_win,)
            assert lat_mean is not None
            assert torch.all(lat_mean.abs() >= abs_rows.min(dim=1).values - 1e-4)
            assert torch.all(lat_mean.abs() <= abs_rows.max(dim=1).values + 1e-4)
            if sh > 0:
                # The wrapped window row mixes both poles.
                assert torch.cos(torch.deg2rad(lat_mean[-1])) < 0.15


@pytest.mark.parametrize("mode", ["earth", "mirror"])
def test_consistent_padding_lat_coords_match_padded_data(
    mode: Literal["earth", "mirror"],
):
    """The latitude of each padded row is the latitude of the data in it."""
    img_shape = (9, 18)
    lat_coords = torch.linspace(-80.0, 80.0, img_shape[0])
    net = SwinTransformerNet(
        3,
        3,
        img_shape,
        embed_dim=16,
        num_heads=(2, 2, 2, 2),
        window_size=(4, 4),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        lat_coords=lat_coords,
        padding_conf={
            "activate": True,
            "mode": mode,
            "pad_lat": [2, 1],
            "pad_lon": [2, 2],
        },
        consistent_padding=True,
    )
    lat_field = lat_coords[:, None].expand(img_shape).reshape(1, 1, *img_shape)
    padded_lat_field = net.padding_opt.pad(lat_field)
    expected = padded_lat_field[0, 0, :, 0]
    assert torch.all(padded_lat_field == expected[:, None])
    actual = net.layer1.blocks[0].lat_coords
    torch.testing.assert_close(actual[: expected.shape[0]], expected)


def _capture_padded_input_and_noise(
    net: SwinTransformerNet, x: torch.Tensor, noise: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``net`` and return the padded input and noise seen by the U-Net."""
    captured: dict[str, torch.Tensor] = {}

    def encoder_hook(module, args):
        captured["x"] = args[0]

    def layer1_hook(module, args, kwargs):
        captured["noise"] = kwargs["context"].noise

    handles = [
        net.encoder.register_forward_pre_hook(encoder_hook),
        net.layer1.register_forward_pre_hook(layer1_hook, with_kwargs=True),
    ]
    context = Context(
        embedding_scalar=None, embedding_pos=None, labels=None, noise=noise
    )
    with torch.no_grad():
        net(x, context)
    for handle in handles:
        handle.remove()
    return captured["x"], captured["noise"]


@pytest.mark.parametrize("padding_mode", ["earth", "mirror", None])
def test_consistent_padding_fills_window_multiple_with_data(
    padding_mode: Literal["earth", "mirror"] | None,
):
    """Rows/columns added to reach a window multiple continue the sphere.

    Longitude wraps circularly. Past the south/last latitude edge, earth
    padding crosses the pole (rows reversed, longitude rotated by 180 deg),
    mirror padding reflects about the edge row, and without padding the edge
    row is replicated (matching the latitudes used for CPB scaling).
    """
    net = _build_padded_net(padding_mode, "cln", consistent_padding=True)
    H, W = _REGRESSION_IMG_SHAPE
    Hp, Wp = net.padded_shape
    pl0, pw0 = (2, 2) if padding_mode is not None else (0, 0)
    x, context = _regression_inputs("cln")
    assert context.noise is not None
    padded_x, padded_noise = _capture_padded_input_and_noise(net, x, context.noise)
    for data, padded in ((x, padded_x), (context.noise, padded_noise)):
        assert padded.shape[-2:] == (Hp, Wp)
        cols = (torch.arange(Wp) - pw0) % W
        rows = []
        for i in range(Hp):
            k = i - pl0 - H  # index past the south/last edge
            if k < 0:
                rows.append(data[..., i - pl0, cols])
            elif padding_mode == "earth":
                rows.append(torch.roll(data, W // 2, dims=-1)[..., H - 1 - k, cols])
            elif padding_mode == "mirror":
                rows.append(data[..., H - 2 - k, cols])
            else:
                rows.append(data[..., H - 1, cols])
        expected = torch.stack(rows, dim=-2)
        # rows above the north edge are covered by the TensorPadding tests
        torch.testing.assert_close(padded[..., pl0:, :], expected[..., pl0:, :])
    lat = net.layer1.blocks[0].lat_coords
    lat_field = torch.linspace(-80.0, 80.0, H)[:, None].expand(H, W)
    _, padded_lat = _capture_padded_input_and_noise(
        net, x, lat_field.expand_as(context.noise).contiguous()
    )
    torch.testing.assert_close(lat, padded_lat[0, 0, :, 0])


@pytest.mark.parametrize("mode", ["earth", "mirror"])
def test_consistent_padding_raises_if_padding_exceeds_grid(
    mode: Literal["earth", "mirror"],
):
    # 9 + 2 + 9 = 20 rows -> 24, so the south padding grows to 13 > 9 rows.
    with pytest.raises(ValueError, match="pad_lat"):
        SwinTransformerNet(
            3,
            3,
            (9, 18),
            window_size=(4, 4),
            padding_conf={
                "activate": True,
                "mode": mode,
                "pad_lat": [2, 9],
                "pad_lon": [2, 2],
            },
            consistent_padding=True,
        )


@pytest.mark.parametrize(
    "padding_mode, conditioning",
    [(None, "adaln"), ("earth", "cln"), ("mirror", "cln")],
)
def test_consistent_padding_changes_output(
    padding_mode: Literal["earth", "mirror"] | None,
    conditioning: Literal["adaln", "cln"],
):
    """The fixes are opt-in: the same weights give different outputs."""
    default_net = _build_padded_net(padding_mode, conditioning)
    fixed_net = _build_padded_net(padding_mode, conditioning, consistent_padding=True)
    fixed_net.load_state_dict(default_net.state_dict())
    x, context = _regression_inputs(conditioning)
    with torch.no_grad():
        assert not torch.allclose(default_net(x, context), fixed_net(x, context))
