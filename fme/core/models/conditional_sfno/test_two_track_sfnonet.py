import pytest
import torch

from fme.core.device import get_device

from .latents import Latents
from .layers import Context, ContextConfig
from .sfnonet import SFNONetConfig, get_lat_lon_sfnonet
from .two_track_sfnonet import TwoTrackSFNONetConfig, get_lat_lon_two_track_sfnonet

IMG_SHAPE = (9, 18)
NOISE_DIM = 8


def _noise_context(n_samples: int, img_shape=IMG_SHAPE) -> Context:
    device = get_device()
    return Context(
        embedding_scalar=None,
        labels=None,
        noise=torch.randn(n_samples, NOISE_DIM, *img_shape, device=device),
        embedding_pos=None,
    )


def _context_config() -> ContextConfig:
    return ContextConfig(
        embed_dim_scalar=0,
        embed_dim_labels=0,
        embed_dim_noise=NOISE_DIM,
        embed_dim_pos=0,
    )


def _build_two_track(
    embed_dim=16,
    local_embed_dim=6,
    global_in=2,
    local_in=3,
    global_out=3,
    local_out=1,
    **kwargs,
):
    config = TwoTrackSFNONetConfig(
        embed_dim=embed_dim,
        local_embed_dim=local_embed_dim,
        num_layers=2,
        filter_type="linear",
        **kwargs,
    )
    return get_lat_lon_two_track_sfnonet(
        params=config,
        global_in_channels=global_in,
        local_in_channels=local_in,
        global_out_channels=global_out,
        local_out_channels=local_out,
        img_shape=IMG_SHAPE,
        data_grid="legendre-gauss",
        context_config=_context_config(),
    ).to(get_device())


# ---------------------------------------------------------------------------
# Latents
# ---------------------------------------------------------------------------
def test_latents_new_from_all_splits_at_global_channels():
    t = torch.randn(2, 7, 4, 5)
    latents = Latents.new_from_all(t, global_channels=3)
    assert latents.global_channels.shape[-3] == 3
    assert latents.local_channels.shape[-3] == 4
    torch.testing.assert_close(latents.global_channels, t[:, :3])
    torch.testing.assert_close(latents.local_channels, t[:, 3:])
    torch.testing.assert_close(latents.all, t)


def test_latents_new_from_global_pads_local_with_zeros():
    g = torch.randn(2, 3, 4, 5)
    latents = Latents.new_from_global(g, local_channels=4)
    assert latents.local_channels.shape[-3] == 4
    assert torch.count_nonzero(latents.local_channels) == 0
    torch.testing.assert_close(latents.global_channels, g)


def test_latents_new_from_global_zero_local_is_global_only():
    g = torch.randn(2, 3, 4, 5)
    latents = Latents.new_from_global(g, local_channels=0)
    assert latents.local_channels.shape[-3] == 0
    torch.testing.assert_close(latents.all, g)


def test_latents_addition_is_per_track():
    a = Latents.new_from_all(torch.randn(2, 5, 4, 4), global_channels=3)
    b = Latents.new_from_all(torch.randn(2, 5, 4, 4), global_channels=3)
    summed = a + b
    torch.testing.assert_close(
        summed.global_channels, a.global_channels + b.global_channels
    )
    torch.testing.assert_close(
        summed.local_channels, a.local_channels + b.local_channels
    )


def test_latents_option2_wiring():
    # Option 2 sums a global-only spectral output with a dense conv1x1 all->all
    # map: global = spectral + conv1x1[global], local = conv1x1[local].
    spectral = torch.randn(2, 3, 4, 4)
    dense = torch.randn(2, 5, 4, 4)
    out = Latents.new_from_global(spectral, local_channels=2) + Latents.new_from_all(
        dense, global_channels=3
    )
    torch.testing.assert_close(out.global_channels, spectral + dense[:, :3])
    torch.testing.assert_close(out.local_channels, dense[:, 3:])


# ---------------------------------------------------------------------------
# L=0 equivalence (regression): the two-track net with zero local channels and
# all options off is byte-for-byte the single-track conditional SFNO.
# ---------------------------------------------------------------------------
def test_two_track_zero_local_matches_single_track():
    # data_grid="legendre-gauss" makes all four transforms share a grid, so the
    # spectral filter never round-trips its residual and the two-track block's
    # (full-normed-input) skip equals the single-track block's (filter-residual)
    # skip. Byte-for-byte output equivalence holds ONLY in this regime; with
    # data_grid="equiangular" the first/last blocks round-trip and the outputs
    # diverge (the checkpoint still loads -- module tree and names are identical
    # -- it just does not reproduce the output). See the module docstring.
    torch.manual_seed(0)
    in_chans, out_chans, n = 2, 3, 4
    context_config = _context_config()
    single = get_lat_lon_sfnonet(
        params=SFNONetConfig(embed_dim=16, num_layers=2, filter_type="linear"),
        in_chans=in_chans,
        out_chans=out_chans,
        img_shape=IMG_SHAPE,
        data_grid="legendre-gauss",
        context_config=context_config,
    ).to(get_device())
    two_track = get_lat_lon_two_track_sfnonet(
        params=TwoTrackSFNONetConfig(
            embed_dim=16, local_embed_dim=0, num_layers=2, filter_type="linear"
        ),
        global_in_channels=in_chans,
        local_in_channels=0,
        global_out_channels=out_chans,
        local_out_channels=0,
        img_shape=IMG_SHAPE,
        data_grid="legendre-gauss",
        context_config=context_config,
    ).to(get_device())

    # Parameter names match exactly: the old checkpoint loads with no remap.
    missing, unexpected = two_track.load_state_dict(single.state_dict(), strict=True)
    assert missing == []
    assert unexpected == []

    x = torch.randn(n, in_chans, *IMG_SHAPE, device=get_device())
    context = _noise_context(n)
    single.eval()
    two_track.eval()
    with torch.no_grad():
        expected = single(x, context)
        actual = two_track(x, context)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# L>0 network behavior and option wiring
# ---------------------------------------------------------------------------
def test_two_track_forward_shape_and_grad():
    torch.manual_seed(0)
    net = _build_two_track()
    n = 2
    x = torch.randn(n, 5, *IMG_SHAPE, device=get_device())  # 2 global + 3 local
    out = net(x, _noise_context(n))
    assert out.shape == (n, 4, *IMG_SHAPE)  # 3 global out + 1 local out
    out.sum().backward()
    assert net.encoder[0].weight.grad is not None
    assert net.local_encoder[0].weight.grad is not None


def _local_encoder_output_changes_with_global_input(net) -> bool:
    """Whether perturbing a global input changes the local encoder's output.

    Captures the local-encoder output via a forward hook so we probe exactly
    the graph edge option 1 adds, isolated from the downstream pointwise
    global<->local mixing that happens in every block regardless.
    """
    net.eval()
    captured = []
    handle = net.local_encoder.register_forward_hook(
        lambda _m, _in, out: captured.append(out.detach().clone())
    )
    n = 2
    x = torch.randn(n, 5, *IMG_SHAPE, device=get_device())
    x2 = x.clone()
    x2[:, 0] += 5.0  # perturb a global input channel
    context = _noise_context(n)
    with torch.no_grad():
        net(x, context)
        net(x2, context)
    handle.remove()
    return not torch.allclose(captured[0], captured[1])


def test_option1_feeds_global_into_local_encoder():
    net_off = _build_two_track(feed_global_to_local=False)
    net_on = _build_two_track(feed_global_to_local=True)
    # local encoder input width = local_in (off) vs local_in + global_in (on)
    assert net_off.local_encoder[0].in_channels == 3
    assert net_on.local_encoder[0].in_channels == 5


def test_option1_off_local_encoder_ignores_global_input():
    torch.manual_seed(0)
    assert not _local_encoder_output_changes_with_global_input(
        _build_two_track(feed_global_to_local=False)
    )


def test_option1_on_local_encoder_reads_global_input():
    torch.manual_seed(0)
    assert _local_encoder_output_changes_with_global_input(
        _build_two_track(feed_global_to_local=True)
    )


def test_option2_off_local_filter_stage_output_is_zero():
    torch.manual_seed(0)
    net = _build_two_track(parallel_conv1x1=False)
    block = net.blocks[0]
    assert block.conv1x1 is None
    n = 2
    x_norm = torch.randn(n, net.embed_dim, *IMG_SHAPE, device=get_device())
    latents = Latents.new_from_all(x_norm, block.global_channels)
    spectral_global, _ = block.filter(latents.global_channels)
    filter_out = Latents.new_from_global(spectral_global, block.local_channels)
    assert torch.count_nonzero(filter_out.local_channels) == 0


def test_option2_on_adds_dense_conv1x1():
    net = _build_two_track(parallel_conv1x1=True)
    block = net.blocks[0]
    assert isinstance(block.conv1x1, torch.nn.Conv2d)
    assert block.conv1x1.in_channels == net.embed_dim
    assert block.conv1x1.out_channels == net.embed_dim


def test_option3_off_uses_joint_layer_norm():
    net = _build_two_track(per_track_layer_norm=False)
    block = net.blocks[0]
    assert block.norm0 is not None
    assert block.norm0_global is None and block.norm0_local is None
    # joint norm covers the full concatenation
    assert block.norm0.n_channels == net.embed_dim


def test_option3_on_uses_per_track_layer_norm():
    net = _build_two_track(per_track_layer_norm=True)
    block = net.blocks[0]
    assert block.norm0 is None
    assert block.norm0_global is not None and block.norm0_local is not None
    assert block.norm0_global.n_channels == net.global_embed_dim
    assert block.norm0_local.n_channels == net.local_embed_dim


def test_config_rejects_local_wider_than_embed_dim():
    with pytest.raises(ValueError, match="local_embed_dim must be < embed_dim"):
        TwoTrackSFNONetConfig(embed_dim=8, local_embed_dim=8)


def test_config_rejects_non_linear_filter():
    with pytest.raises(NotImplementedError, match="filter_type='linear'"):
        TwoTrackSFNONetConfig(
            embed_dim=8, local_embed_dim=2, filter_type="makani-linear"
        )
