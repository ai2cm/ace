import pytest
import torch

from fme.core.device import get_device
from fme.core.models.conditional_sfno.layers import Context, ContextConfig

from .swin_layers import ColumnMixer, WindowAttention2D, window_partition_2d
from .swin_transformer import SwinTransformerNet

_EMBED_DIM_NOISE = 8


def _build_net(
    in_chans: int,
    out_chans: int,
    img_shape: tuple[int, int],
    context_config: ContextConfig | None = None,
    use_skip: bool = True,
    skip_projection: bool = False,
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
        skip_projection=skip_projection,
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


def test_skip_projection_runs_decoder_at_embed_dim():
    """With skip_projection the decoder stage has embed_dim channels, the
    projection gets gradients, and the output shape is unchanged."""
    in_chans, out_chans = 5, 3
    img_shape = (16, 32)
    n = 2
    device = get_device()
    net = _build_net(in_chans, out_chans, img_shape, skip_projection=True).to(device)
    assert net.skip_proj is not None
    assert net.layer4.blocks[0].dim == 32
    x = torch.randn(n, in_chans, *img_shape, device=device)
    out = net(x)
    assert out.shape == (n, out_chans, *img_shape)
    out.sum().backward()
    assert net.skip_proj.weight.grad is not None
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_skip_projection_has_fewer_parameters_than_concat_decoder():
    in_chans, out_chans = 5, 3
    img_shape = (16, 32)
    n_params_concat = sum(
        p.numel() for p in _build_net(in_chans, out_chans, img_shape).parameters()
    )
    n_params_proj = sum(
        p.numel()
        for p in _build_net(
            in_chans, out_chans, img_shape, skip_projection=True
        ).parameters()
    )
    assert n_params_proj < n_params_concat


def test_skip_projection_default_keeps_state_dict_keys():
    """Default skip_projection=False must not add parameters, so existing
    checkpoints keep loading."""
    net = _build_net(5, 3, (16, 32))
    assert net.skip_proj is None
    assert not any("skip_proj" in k for k in net.state_dict())


def test_skip_projection_ignored_without_skip():
    net = _build_net(5, 3, (16, 32), use_skip=False, skip_projection=True)
    assert net.skip_proj is None
    assert net.layer4.blocks[0].dim == 32


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


def test_cln_state_dict_keeps_conv_weight_shapes():
    """The channels-last CLN path reuses the 1x1 conv parameters, so the state
    dict (and therefore checkpoint compatibility) is unchanged."""
    net = _build_cln_net(4, 2, (16, 32))
    state = net.state_dict()
    scale_keys = [k for k in state if k.endswith("norm1.W_scale_2d.weight")]
    assert len(scale_keys) == sum(len(layer.blocks) for layer in _layers(net))
    for key in scale_keys:
        assert state[key].shape[2:] == (1, 1), key


def _layers(net: SwinTransformerNet):
    return [net.layer1, net.layer2, net.layer3, net.layer4]


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


def _reference_window_attention(
    attn: WindowAttention2D,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    lat_mean: torch.Tensor | None,
) -> torch.Tensor:
    """Explicit-logits cosine attention, as implemented before the switch to
    ``F.scaled_dot_product_attention``. Shares parameters with ``attn``."""
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
    if lat_mean is None:
        bias = 16.0 * torch.sigmoid(attn.cpb_mlp(attn.relative_coords_log))
        bias = bias.permute(1, 0).reshape(attn.num_heads, N, N)
        logits = logits + bias.unsqueeze(0)
    else:
        nW = lat_mean.shape[0]
        lat_rad = lat_mean * (torch.pi / 180.0)
        h_coords = attn.relative_coords_base[:, 0]
        w_coords = attn.relative_coords_base[:, 1].unsqueeze(0) * torch.cos(
            lat_rad
        ).unsqueeze(1)
        coords = torch.stack([h_coords.unsqueeze(0).expand(nW, -1), w_coords], dim=-1)
        coords_log = torch.sign(coords) * torch.log(1.0 + coords.abs())
        bias = 16.0 * torch.sigmoid(attn.cpb_mlp(coords_log))
        bias = bias.permute(0, 2, 1).reshape(nW, attn.num_heads, N, N)
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
    nW = (H // window_size[0]) * (W // window_size[1])
    attn = WindowAttention2D(dim, window_size, num_heads).to(device).double()
    with torch.no_grad():
        for param in attn.parameters():
            param.normal_()
        attn.tau.abs_().add_(0.1)
    x = torch.randn(B, H, W, dim, device=device, dtype=torch.float64)
    x = window_partition_2d(x, *window_size).view(-1, 16, dim).requires_grad_(True)
    mask = None
    if use_mask:
        mask = torch.zeros(nW, 16, 16, device=device, dtype=torch.float64)
        mask[:, :8, 8:] = -100.0
        mask[:, 8:, :8] = -100.0
    lat_mean = (
        torch.linspace(-60.0, 60.0, nW, device=device, dtype=torch.float64)
        if use_lat
        else None
    )
    out = attn(x, mask=mask, lat_mean=lat_mean)
    grads = torch.autograd.grad(out.square().sum(), [x, *attn.parameters()])
    ref = _reference_window_attention(attn, x, mask, lat_mean)
    ref_grads = torch.autograd.grad(ref.square().sum(), [x, *attn.parameters()])
    torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-10)
    for g, g_ref in zip(grads, ref_grads):
        torch.testing.assert_close(g, g_ref, atol=1e-8, rtol=1e-8)
