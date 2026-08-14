import torch

from fme.downscaling.modules.video_modules import TemporalAttention, VideoUNet


def _count_temporal_attention(model: VideoUNet) -> int:
    return sum(1 for m in model.modules() if isinstance(m, TemporalAttention))


def _build(temporal_attention_levels):
    return VideoUNet(
        in_channels=3,
        out_channels=2,
        seq_length=5,
        model_channels=8,
        channel_mult=(1, 2),
        num_blocks=1,
        n_heads=2,
        attention_levels=(),
        temporal_attention_levels=temporal_attention_levels,
    )


def test_default_temporal_attention_levels_include_mid_block():
    # Default (None) preserves existing behavior: every level, including the
    # mid/bottleneck block, gets temporal attention.
    model = _build(None)
    assert _count_temporal_attention(model) > 0
    assert model.mid_attn.temporal is not None


def test_empty_temporal_attention_levels_disables_mid_block_too():
    # An explicit empty tuple must disable temporal attention EVERYWHERE,
    # including the mid/bottleneck block -- the mid block used to hardcode
    # temporal=True regardless of temporal_attention_levels, which silently
    # left one TemporalAttention layer (cross-frame mixing) active even when
    # the caller asked for a fully per-frame-independent (HiRO-style) model.
    model = _build(())
    assert _count_temporal_attention(model) == 0
    assert model.mid_attn.temporal is None


def test_no_temporal_attention_model_is_frame_permutation_invariant():
    # With zero TemporalAttention layers anywhere, output at each frame must
    # depend only on that frame's own input -- permuting frames in the input
    # must permute the output identically (up to floating point tolerance).
    torch.manual_seed(0)
    model = _build(())
    model.eval()
    B, C, T, H, W = 2, 3, 5, 8, 8
    x = torch.randn(B, C, T, H, W)
    c_noise = torch.randn(B)
    day_of_year = torch.zeros(B, T)
    second_of_day = torch.zeros(B, T)
    lon = torch.linspace(0, 360, W)

    with torch.no_grad():
        out = model(x, c_noise, day_of_year, second_of_day, lon)

    perm = torch.tensor([2, 0, 4, 1, 3])
    with torch.no_grad():
        out_permuted_input = model(
            x[:, :, perm], c_noise, day_of_year[:, perm], second_of_day[:, perm], lon
        )
    torch.testing.assert_close(out_permuted_input, out[:, :, perm])
