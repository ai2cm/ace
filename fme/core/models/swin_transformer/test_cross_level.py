import pytest
import torch

from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.stacker import infer_column_layout

from .swin_layers import CrossLevelAttention
from .swin_transformer import SwinTransformerNet

_COLUMN = [f"{p}_{lvl}" for p in ["t", "q"] for lvl in range(4)]
_NAMES_IN = ["land_fraction", "DSWRFtoa"] + _COLUMN + ["co2"]
_NAMES_OUT = ["PRESsfc"] + _COLUMN + ["PRATEsfc", "h500"]


def _build(**kwargs) -> SwinTransformerNet:
    defaults = dict(
        in_chans=len(_NAMES_IN),
        out_chans=len(_NAMES_OUT),
        img_shape=(16, 32),
        embed_dim=16,
        depth_multiplier=1,
        num_heads=(2, 2, 2, 2),
        window_size=(4, 8),
        in_layout=infer_column_layout(_NAMES_IN),
        out_layout=infer_column_layout(_NAMES_OUT),
        column_num_heads=2,
    )
    defaults.update(kwargs)
    return SwinTransformerNet(**defaults)  # type: ignore[arg-type]


class _PadToWidth(torch.nn.Module):
    """Zero-pad the channel axis, preserving leading channel values."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.width - x.shape[1]
        return torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))


class _TakeFirst(torch.nn.Module):
    """Slice the leading channels, the inverse of ``_PadToWidth``."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, : self.width]


def test_column_encode_decode_roundtrip_preserves_channel_positions():
    """Channels must land back in the positions the packer expects."""
    net = _build()
    layout_in = net.in_layout
    layout_out = net.out_layout
    assert layout_in is not None and layout_out is not None
    # Transparent stems let us follow a channel through the fold and back.
    embed_dim = 16
    net.level_encoder = _PadToWidth(embed_dim)
    net.surface_encoder = _PadToWidth(embed_dim)
    net.level_decoder = _TakeFirst(layout_out.n_vars)
    net.surface_decoder = _TakeFirst(len(layout_out.surface_indices))

    # Mark each input channel with its own index so we can see where it goes.
    x = torch.arange(len(_NAMES_IN), dtype=torch.float32)
    x = x.view(1, -1, 1, 1).expand(1, len(_NAMES_IN), 16, 32).contiguous()
    tokens = net._encode_columns(x)
    assert tokens.shape[0] == net.n_tokens  # batch of 1

    # Token 0 is the surface projection, tokens 1.. are the levels in order.
    surface = tokens[0, : len(layout_in.surface_indices), 0, 0]
    assert surface.tolist() == [float(i) for i in layout_in.surface_indices]
    for level in range(layout_in.n_levels):
        expected = [
            float(layout_in.start + var * layout_in.n_levels + level)
            for var in range(layout_in.n_vars)
        ]
        assert tokens[1 + level, : layout_in.n_vars, 0, 0].tolist() == expected


def test_column_decode_places_channels_at_output_indices():
    net = _build()
    layout_out = net.out_layout
    assert layout_out is not None
    net.level_decoder = _TakeFirst(layout_out.n_vars)
    net.surface_decoder = _TakeFirst(len(layout_out.surface_indices))

    # Give each token a distinct constant value.
    tokens = torch.zeros(net.n_tokens, 16, 4, 8)
    for token in range(net.n_tokens):
        tokens[token] = float(token)
    out = net._decode_columns(tokens, batch_size=1)

    assert out.shape == (1, len(_NAMES_OUT), 4, 8)
    # Surface channels come from token 0.
    for index in layout_out.surface_indices:
        assert torch.all(out[0, index] == 0.0)
    # Level channels come from token 1 + level, regardless of variable.
    for var in range(layout_out.n_vars):
        for level in range(layout_out.n_levels):
            index = layout_out.start + var * layout_out.n_levels + level
            assert torch.all(out[0, index] == float(1 + level))


def test_cross_level_attention_is_spatially_pointwise():
    """Columns must not exchange information across space."""
    torch.manual_seed(0)
    attn = CrossLevelAttention(dim=8, n_levels=5, num_heads=2).eval()
    x = torch.randn(2 * 5, 4, 6, 8)
    with torch.no_grad():
        base = attn(x)
        perturbed = x.clone()
        perturbed[:, 2:] += 10.0
        after = attn(perturbed)
    assert torch.allclose(base[:, :2], after[:, :2], atol=1e-6)


def test_cross_level_attention_mixes_levels():
    """A change at one level must reach the others in the same column."""
    torch.manual_seed(0)
    attn = CrossLevelAttention(dim=8, n_levels=5, num_heads=2).eval()
    x = torch.randn(5, 2, 2, 8)  # batch of 1, 5 levels
    with torch.no_grad():
        base = attn(x)
        perturbed = x.clone()
        perturbed[0] += 10.0  # perturb level 0 only
        after = attn(perturbed)
    other_levels_changed = (after[1:] - base[1:]).abs().max()
    assert other_levels_changed > 1e-3


def test_layout_must_be_given_for_both_or_neither():
    with pytest.raises(ValueError, match="both be provided or both be None"):
        _build(out_layout=None)


def test_mismatched_level_counts_raise():
    with pytest.raises(ValueError, match="same number of levels"):
        _build(out_layout=infer_column_layout([f"t_{lvl}" for lvl in range(3)]))


def test_forward_without_layout_keeps_flat_encoder():
    """Absent a column layout the network stays a plain 2D model."""
    net = SwinTransformerNet(
        in_chans=len(_NAMES_IN),
        out_chans=len(_NAMES_OUT),
        img_shape=(16, 32),
        embed_dim=16,
        depth_multiplier=1,
        num_heads=(2, 2, 2, 2),
        window_size=(4, 8),
    ).eval()
    assert net.n_tokens == 1
    with torch.no_grad():
        out = net(torch.randn(2, len(_NAMES_IN), 16, 32))
    assert out.shape == (2, len(_NAMES_OUT), 16, 32)


def test_noise_conditioned_forward_shapes():
    """Per-sample noise must broadcast over the folded vertical axis."""
    net = _build(
        conditioning="cln",
        context_config=ContextConfig(
            embed_dim_scalar=0,
            embed_dim_labels=0,
            embed_dim_noise=4,
            embed_dim_pos=0,
        ),
    ).eval()
    context = Context(
        embedding_scalar=None,
        embedding_pos=None,
        labels=None,
        noise=torch.randn(3, 4, 16, 32),
    )
    with torch.no_grad():
        out = net(torch.randn(3, len(_NAMES_IN), 16, 32), context)
    assert out.shape == (3, len(_NAMES_OUT), 16, 32)
