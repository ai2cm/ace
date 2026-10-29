import pathlib
from typing import Literal

import pytest
import torch

from fme.core.device import get_device
from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.testing.regression import validate_tensor_dict

from .swin_layers import (
    ColumnMixer,
    WindowAttention2D,
    _cos_lat_scaled_coords_log,
    window_lat_mean,
)
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
    # The block precomputes its CPB coordinates from the padded lat rows.
    attn = net.layer1.blocks[0].attn
    expected_lat_mean = window_lat_mean(expected, net.padded_shape, (4, 4), shift=0)
    assert expected_lat_mean is not None
    torch.testing.assert_close(
        attn.coords_log,
        _cos_lat_scaled_coords_log(attn.relative_coords_base, expected_lat_mean),
    )


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
    lat_mean_per_window: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Position bias with the CPB MLP evaluated separately for every window,
    as implemented before the per-latitude-band deduplication. The module
    precomputes its coordinates in float32 at construction; ``dtype`` is the
    dtype they are cast to before the MLP."""
    N = attn.window_size[0] * attn.window_size[1]
    nW = lat_mean_per_window.shape[0]
    base = attn.relative_coords_base.float()
    lat_rad = lat_mean_per_window.float() * (torch.pi / 180.0)
    h_coords = base[:, 0]
    w_coords = base[:, 1].unsqueeze(0) * torch.cos(lat_rad).unsqueeze(1)
    coords = torch.stack([h_coords.unsqueeze(0).expand(nW, -1), w_coords], dim=-1)
    coords_log = (torch.sign(coords) * torch.log(1.0 + coords.abs())).to(dtype)
    bias = 16.0 * torch.sigmoid(attn.cpb_mlp(coords_log))
    return bias.permute(0, 2, 1).reshape(nW, attn.num_heads, N, N)


def _reference_window_attention(
    attn: WindowAttention2D,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    lat_mean: torch.Tensor | None,
) -> torch.Tensor:
    """Explicit-logits cosine attention, as implemented before the switch to
    ``F.scaled_dot_product_attention``. Shares parameters with ``attn``.

    Takes a per-window ``lat_mean`` of shape ``(nW,)``, so it also checks the
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
    if lat_mean is None:
        bias = 16.0 * torch.sigmoid(attn.cpb_mlp(attn.relative_coords_log))
        bias = bias.permute(1, 0).reshape(attn.num_heads, N, N)
        logits = logits + bias.unsqueeze(0)
    else:
        nW = lat_mean.shape[0]
        bias = _per_window_position_bias(attn, lat_mean, dtype=x.dtype)
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
    lat_mean = (
        torch.linspace(-60.0, 60.0, nH_win, device=device, dtype=torch.float64)
        if use_lat
        else None
    )
    attn = (
        WindowAttention2D(
            dim, window_size, num_heads, lat_mean=lat_mean, num_windows_w=nW_win
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
    lat_mean_per_window = (
        lat_mean.repeat_interleave(nW_win) if lat_mean is not None else None
    )
    ref = _reference_window_attention(attn, x, mask, lat_mean_per_window)
    ref_grads = torch.autograd.grad(ref.square().sum(), [x, *attn.parameters()])
    torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-10)
    for g, g_ref in zip(grads, ref_grads):
        torch.testing.assert_close(g, g_ref, atol=1e-8, rtol=1e-8)


def test_window_lat_mean_matches_rolled_window_means():
    """Precomputed per-window-row latitudes equal the mean of the (shifted)
    latitude rows in each window row, top to bottom."""
    H, W = 8, 16
    ws = (4, 4)
    lat = torch.linspace(-70.0, 70.0, H)
    for shift in (0, 2):
        lat_mean = window_lat_mean(lat, (H, W), ws, shift)
        assert lat_mean is not None
        assert lat_mean.shape == (H // ws[0],)
        rolled = torch.roll(lat, -shift)
        expected = rolled.reshape(H // ws[0], ws[0]).mean(1)
        torch.testing.assert_close(lat_mean, expected)
    assert window_lat_mean(None, (H, W), ws, 0) is None


def test_window_attention_requires_num_windows_w_with_lat_mean():
    """Omitting the windows-per-row count is a construction-time error, not a
    forward-time shape mismatch."""
    with pytest.raises(ValueError, match="num_windows_w"):
        WindowAttention2D(16, (4, 4), 4, lat_mean=torch.zeros(3))
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
    lat_mean = torch.linspace(-75.0, 75.0, nH_win, device=device) if use_lat else None
    attn = WindowAttention2D(
        dim, window_size, num_heads, lat_mean=lat_mean, num_windows_w=nW_win
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
    assert lat_mean is not None
    assert attn.coords_log.shape == (nH_win, N * N, 2)
    assert attn.band_index.shape == (nH_win * nW_win,)
    assert bias.shape == (nH_win * nW_win, num_heads, N, N)
    expected = _per_window_position_bias(attn, lat_mean.repeat_interleave(nW_win))
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
