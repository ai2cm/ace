import os
from typing import Literal

import pytest
import torch
from torch import nn

from fme.ace.models.modulus.sfnonet import SphericalFourierNeuralOperatorNet
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.models.conditional_sfno.benchmark import get_block_benchmark
from fme.core.testing.regression import validate_tensor

from .layers import Context, ContextConfig
from .s2convolutions import SpectralConvS2
from .sfnonet import SFNONetConfig, get_lat_lon_sfnonet

DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize(
    "conditional_embed_dim_scalar, conditional_embed_dim_labels, "
    "conditional_embed_dim_noise, "
    "conditional_embed_dim_pos, residual_filter_factor",
    [
        (0, 0, 0, 0, 1),
        (16, 8, 0, 0, 1),
        (16, 0, 16, 0, 1),
        (16, 15, 14, 13, 1),
        (0, 0, 0, 16, 1),
        (0, 0, 16, 0, 1),
        (16, 0, 0, 0, 4),
    ],
)
def test_can_call_sfnonet(
    conditional_embed_dim_scalar: int,
    conditional_embed_dim_labels: int,
    conditional_embed_dim_noise: int,
    conditional_embed_dim_pos: int,
    residual_filter_factor: int,
):
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    params = SFNONetConfig(
        embed_dim=16,
        num_layers=2,
        filter_type="makani-linear",
    )
    model = get_lat_lon_sfnonet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
        context_config=ContextConfig(
            embed_dim_scalar=conditional_embed_dim_scalar,
            embed_dim_labels=conditional_embed_dim_labels,
            embed_dim_noise=conditional_embed_dim_noise,
            embed_dim_pos=conditional_embed_dim_pos,
        ),
    ).to(device)
    x = torch.randn(n_samples, input_channels, *img_shape, device=device)
    context_embedding = torch.randn(
        n_samples, conditional_embed_dim_scalar, device=device
    )
    context_embedding_labels = torch.randn(
        n_samples, conditional_embed_dim_labels, device=device
    )
    context_embedding_noise = torch.randn(
        n_samples, conditional_embed_dim_noise, *img_shape, device=device
    )
    context_embedding_pos = torch.randn(
        n_samples, conditional_embed_dim_pos, *img_shape, device=device
    )
    context = Context(
        embedding_scalar=context_embedding,
        labels=context_embedding_labels,
        noise=context_embedding_noise,
        embedding_pos=context_embedding_pos,
    )
    output = model(x, context)
    assert output.shape == (n_samples, output_channels, *img_shape)


def test_scale_factor_not_implemented():
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    device = get_device()
    params = SFNONetConfig(embed_dim=16, num_layers=2, scale_factor=2)
    with pytest.raises(NotImplementedError):
        # if this ever gets implemented, we need to instead test that the scale factor
        # is used to determine the nlat/nlon of the image in the network
        get_lat_lon_sfnonet(
            params=params,
            img_shape=img_shape,
            in_chans=input_channels,
            out_chans=output_channels,
            context_config=ContextConfig(
                embed_dim_scalar=0,
                embed_dim_noise=0,
                embed_dim_labels=0,
                embed_dim_pos=0,
            ),
        ).to(device)


def setup_sfnonet():
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    conditional_embed_dim_scalar = 8
    conditional_embed_dim_labels = 4
    conditional_embed_dim_noise = 16
    conditional_embed_dim_pos = 0
    device = get_device()
    params = SFNONetConfig(embed_dim=16, num_layers=2, filter_type="linear")
    model = get_lat_lon_sfnonet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
        context_config=ContextConfig(
            embed_dim_scalar=conditional_embed_dim_scalar,
            embed_dim_labels=conditional_embed_dim_labels,
            embed_dim_noise=conditional_embed_dim_noise,
            embed_dim_pos=conditional_embed_dim_pos,
        ),
    ).to(device)
    # must initialize on CPU to get the same results on GPU
    x = torch.randn(n_samples, input_channels, *img_shape).to(device)
    context_embedding = torch.randn(n_samples, conditional_embed_dim_scalar).to(device)
    context_embedding_labels = torch.randn(
        n_samples, conditional_embed_dim_labels, device=device
    )
    context_embedding_noise = torch.randn(
        n_samples, conditional_embed_dim_noise, *img_shape, device=device
    ).to(device)
    context_embedding_pos = None
    context = Context(
        embedding_scalar=context_embedding,
        labels=context_embedding_labels,
        noise=context_embedding_noise,
        embedding_pos=context_embedding_pos,
    )
    return model, x, context


def test_sfnonet_output_is_unchanged():
    torch.manual_seed(0)
    model, x, context = setup_sfnonet()
    with torch.no_grad():
        output = model(x, context)
    validate_tensor(
        output,
        os.path.join(DIR, "testdata/test_sfnonet_output_is_unchanged.pt"),
    )


def load_or_cache_model_state(
    model: SphericalFourierNeuralOperatorNet,
    x: torch.Tensor,
    context: Context,
    path: str,
):
    if os.path.exists(path):
        data = torch.load(path, map_location=get_device())
        x = data.pop("x")
        context = Context.from_dict(data.pop("context"))
        model.load_state_dict(data)
    else:
        data = model.state_dict()
        data["x"] = x
        data["context"] = context.asdict()
        torch.save(data, path)
    return model, x, context


def test_sfnonet_output_from_checkpoint_is_unchanged():
    torch.manual_seed(0)
    model, x, context = setup_sfnonet()
    checkpoint_path = os.path.join(DIR, "testdata/test_sfnonet_checkpoint_input.pt")
    model, x, context = load_or_cache_model_state(model, x, context, checkpoint_path)
    with torch.no_grad():
        output = model(x, context)
    validate_tensor(
        output,
        os.path.join(DIR, "testdata/test_sfnonet_checkpoint_output.pt"),
    )


@pytest.mark.parametrize("normalize_big_skip", [True, False])
def test_all_inputs_get_layer_normed(normalize_big_skip: bool):
    torch.manual_seed(0)
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    conditional_embed_dim_scalar = 8
    conditional_embed_dim_noise = 16
    conditional_embed_dim_labels = 3
    conditional_embed_dim_pos = 12
    device = get_device()

    class SetToZero(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return torch.zeros_like(x)

    original_layer_norm = nn.LayerNorm
    try:
        nn.LayerNorm = SetToZero
        params = SFNONetConfig(
            embed_dim=16,
            num_layers=2,
            normalize_big_skip=normalize_big_skip,
            global_layer_norm=True,  # so it uses nn.LayerNorm
        )
        model = get_lat_lon_sfnonet(
            params=params,
            img_shape=img_shape,
            in_chans=input_channels,
            out_chans=output_channels,
            context_config=ContextConfig(
                embed_dim_scalar=conditional_embed_dim_scalar,
                embed_dim_noise=conditional_embed_dim_noise,
                embed_dim_labels=conditional_embed_dim_labels,
                embed_dim_pos=conditional_embed_dim_pos,
            ),
        ).to(device)
    finally:
        nn.LayerNorm = original_layer_norm
    x = torch.full((n_samples, input_channels, *img_shape), torch.nan).to(device)
    context_embedding = torch.randn(n_samples, conditional_embed_dim_scalar).to(device)
    context_embedding_noise = torch.randn(
        n_samples, conditional_embed_dim_noise, *img_shape
    ).to(device)
    context_embedding_labels = torch.randn(n_samples, conditional_embed_dim_labels).to(
        device
    )
    context_embedding_pos = torch.randn(
        n_samples, conditional_embed_dim_pos, *img_shape
    ).to(device)
    context = Context(
        embedding_scalar=context_embedding,
        embedding_pos=context_embedding_pos,
        noise=context_embedding_noise,
        labels=context_embedding_labels,
    )
    with torch.no_grad():
        output = model(x, context)
    if normalize_big_skip:
        assert not torch.isnan(output).any()
    else:
        assert torch.isnan(output).any()


@pytest.mark.skipif(
    get_device().type != "cuda",
    reason=(
        "This test is only relevant for CUDA since "
        "it's testing speed of SFNO blocks on GPU."
    ),
)  # noqa: E501
@pytest.mark.serial
def test_block_speed():
    ungrouped = get_block_benchmark(filter_num_groups=1).run_benchmark(
        iters=5, warmup=1
    )
    grouped = get_block_benchmark(filter_num_groups=8).run_benchmark(iters=5, warmup=1)
    assert grouped.timer.avg_time < ungrouped.timer.avg_time, (
        "Expected grouped DHConv to be faster than ungrouped, but got "
        f"{grouped.timer.avg_time:.6f} ms for grouped and "
        f"{ungrouped.timer.avg_time:.6f} ms for ungrouped."
    )
    assert grouped.memory.max_alloc < ungrouped.memory.max_alloc, (
        "Expected grouped DHConv to use less memory than ungrouped, but got "
        f"{grouped.memory.max_alloc / 1e6:.2f} MB for grouped and "
        f"{ungrouped.memory.max_alloc / 1e6:.2f} MB for ungrouped."
    )


def test_remove_latent_global_mean():
    torch.manual_seed(0)
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    params = SFNONetConfig(
        embed_dim=16,
        num_layers=2,
        remove_latent_global_mean=True,
    )
    model = get_lat_lon_sfnonet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
    ).to(device)
    x = torch.randn(n_samples, input_channels, *img_shape, device=device)
    context = Context(
        embedding_scalar=torch.zeros(n_samples, 0, device=device),
        labels=torch.zeros(n_samples, 0, device=device),
        noise=None,
        embedding_pos=None,
    )
    output = model(x, context)
    assert output.shape == (n_samples, output_channels, *img_shape)
    output.sum().backward()


def test_remove_latent_global_mean_zeroes_spatial_mean():
    torch.manual_seed(0)
    input_channels = 2
    img_shape = (9, 18)
    n_samples = 2
    device = get_device()
    params = SFNONetConfig(
        embed_dim=16,
        num_layers=1,
        remove_latent_global_mean=True,
        pos_embed=False,
        drop_rate=0.0,
    )
    model = get_lat_lon_sfnonet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=input_channels,
    ).to(device)
    x = torch.randn(n_samples, input_channels, *img_shape, device=device)

    with torch.no_grad():
        encoded = model.encoder(x)
    latent_mean = encoded.mean(dim=(-2, -1))
    assert not torch.allclose(
        latent_mean, torch.zeros_like(latent_mean), atol=1e-5
    ), "encoder output should have non-zero spatial mean for this test to be meaningful"

    with torch.no_grad():
        encoded_demeaned = encoded - encoded.mean(dim=(-2, -1), keepdim=True)
    demeaned_mean = encoded_demeaned.mean(dim=(-2, -1))
    torch.testing.assert_close(
        demeaned_mean, torch.zeros_like(demeaned_mean), atol=1e-6, rtol=0.0
    )


def _make_latent_mean_model(
    device,
    remove_latent_global_mean: bool = False,
    concat_latent_global_mean: Literal["none", "first", "every"] = "none",
    add_latent_global_mean_to_output: bool = False,
    global_mean_noise: float = 0.0,
    clip_latent_global_means: bool = False,
):
    params = SFNONetConfig(
        embed_dim=16,
        num_layers=2,
        remove_latent_global_mean=remove_latent_global_mean,
        concat_latent_global_mean=concat_latent_global_mean,
        add_latent_global_mean_to_output=add_latent_global_mean_to_output,
        global_mean_noise=global_mean_noise,
        clip_latent_global_means=clip_latent_global_means,
    )
    model = get_lat_lon_sfnonet(
        params=params,
        img_shape=(9, 18),
        in_chans=2,
        out_chans=3,
    ).to(device)
    return model


def _empty_context(n_samples, device):
    return Context(
        embedding_scalar=torch.zeros(n_samples, 0, device=device),
        labels=torch.zeros(n_samples, 0, device=device),
        noise=None,
        embedding_pos=None,
    )


@pytest.mark.parametrize("concat_mode", ["first", "every"])
def test_concat_latent_global_mean(concat_mode):
    torch.manual_seed(0)
    n_samples = 4
    device = get_device()
    model = _make_latent_mean_model(
        device,
        remove_latent_global_mean=True,
        concat_latent_global_mean=concat_mode,
    )
    x = torch.randn(n_samples, 2, 9, 18, device=device)
    output = model(x, _empty_context(n_samples, device))
    assert output.shape == (n_samples, 3, 9, 18)
    output.sum().backward()


def test_add_latent_global_mean_to_output():
    torch.manual_seed(0)
    n_samples = 4
    device = get_device()
    model = _make_latent_mean_model(
        device,
        remove_latent_global_mean=True,
        add_latent_global_mean_to_output=True,
    )
    x = torch.randn(n_samples, 2, 9, 18, device=device)
    output = model(x, _empty_context(n_samples, device))
    assert output.shape == (n_samples, 3, 9, 18)
    output.sum().backward()


def test_all_latent_mean_flags():
    torch.manual_seed(0)
    n_samples = 4
    device = get_device()
    model = _make_latent_mean_model(
        device,
        remove_latent_global_mean=True,
        concat_latent_global_mean="every",
        add_latent_global_mean_to_output=True,
    )
    x = torch.randn(n_samples, 2, 9, 18, device=device)
    output = model(x, _empty_context(n_samples, device))
    assert output.shape == (n_samples, 3, 9, 18)
    output.sum().backward()


def test_concat_without_remove():
    torch.manual_seed(0)
    n_samples = 4
    device = get_device()
    model = _make_latent_mean_model(
        device,
        remove_latent_global_mean=False,
        concat_latent_global_mean="first",
    )
    x = torch.randn(n_samples, 2, 9, 18, device=device)
    output = model(x, _empty_context(n_samples, device))
    assert output.shape == (n_samples, 3, 9, 18)
    output.sum().backward()


def test_add_output_without_remove():
    torch.manual_seed(0)
    n_samples = 4
    device = get_device()
    model = _make_latent_mean_model(
        device,
        remove_latent_global_mean=False,
        add_latent_global_mean_to_output=True,
    )
    x = torch.randn(n_samples, 2, 9, 18, device=device)
    output = model(x, _empty_context(n_samples, device))
    assert output.shape == (n_samples, 3, 9, 18)
    output.sum().backward()


def test_global_mean_noise_only_during_training():
    torch.manual_seed(0)
    n_samples = 4
    device = get_device()
    model = _make_latent_mean_model(
        device,
        remove_latent_global_mean=True,
        global_mean_noise=1.0,
    )
    x = torch.randn(n_samples, 2, 9, 18, device=device)
    ctx = _empty_context(n_samples, device)

    model.eval()
    with torch.no_grad():
        out_eval_1 = model(x, ctx)
        out_eval_2 = model(x, ctx)
    torch.testing.assert_close(out_eval_1, out_eval_2)

    model.train()
    with torch.no_grad():
        out_train_1 = model(x, ctx)
        out_train_2 = model(x, ctx)
    assert not torch.allclose(
        out_train_1, out_train_2
    ), "training outputs should differ due to global mean noise"


def test_clip_latent_global_means_requires_latent_mean_enabled():
    device = get_device()
    with pytest.raises(ValueError, match="clip_latent_global_means"):
        _make_latent_mean_model(device, clip_latent_global_means=True)


def test_clip_latent_global_means_envelope_tracks_and_clamps():
    torch.manual_seed(0)
    device = get_device()
    model = _make_latent_mean_model(
        device,
        remove_latent_global_mean=True,
        add_latent_global_mean_to_output=True,
        clip_latent_global_means=True,
    )
    n_samples = 4
    ctx = _empty_context(n_samples, device)

    # Envelope starts at sentinels.
    assert torch.isinf(model._gm_min).all() and (model._gm_min > 0).all()
    assert torch.isinf(model._gm_max).all() and (model._gm_max < 0).all()

    # Eval with no training run yet: sentinel envelope, clamp is a no-op.
    model.eval()
    x = torch.randn(n_samples, 2, 9, 18, device=device)
    with torch.no_grad():
        out_uninit = model(x, ctx)
    assert torch.isfinite(out_uninit).all()
    assert torch.isinf(model._gm_min).all()
    assert torch.isinf(model._gm_max).all()

    # One training forward populates the envelope.
    model.train()
    with torch.no_grad():
        model(x, ctx)
    assert torch.isfinite(model._gm_min).all()
    assert torch.isfinite(model._gm_max).all()
    assert (model._gm_max >= model._gm_min).all()

    # A larger-magnitude training batch widens the envelope.
    gm_min_before = model._gm_min.clone()
    gm_max_before = model._gm_max.clone()
    x_big = 10.0 * torch.randn(n_samples, 2, 9, 18, device=device)
    with torch.no_grad():
        model(x_big, ctx)
    assert (model._gm_min <= gm_min_before).all()
    assert (model._gm_max >= gm_max_before).all()

    # Eval clamps means outside the envelope. To check clamping, we set
    # the envelope to a tight known range and verify the encoder
    # output's spatial mean cannot exceed it.
    model._gm_min.fill_(-0.01)
    model._gm_max.fill_(0.01)
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(x_big)
        post_norm = model.latent_norm(
            encoded
            + (
                model.pos_embed[..., model._spatial_h_slice, model._spatial_w_slice]
                if model.pos_embed is not None
                else 0.0
            ),
            context=ctx,
        )
        raw_means = post_norm.mean(dim=(-2, -1), keepdim=True)
    # Without clamping, |raw_means| would generally exceed the tight
    # envelope at some channels.
    assert (raw_means.abs() > 0.01).any()
    # With clamping, the model's behavior should use clamped means; we
    # can't observe the post-clip means directly, but we can verify the
    # eval forward runs without producing NaNs/Infs.
    with torch.no_grad():
        out_clamped = model(x_big, ctx)
    assert torch.isfinite(out_clamped).all()


def test_clip_latent_global_means_lazy_reset():
    torch.manual_seed(0)
    device = get_device()
    model = _make_latent_mean_model(
        device,
        remove_latent_global_mean=True,
        clip_latent_global_means=True,
    )
    n_samples = 4
    ctx = _empty_context(n_samples, device)
    x = torch.randn(n_samples, 2, 9, 18, device=device)

    model.train()
    with torch.no_grad():
        model(x, ctx)
    gm_min_after_epoch1 = model._gm_min.clone()
    gm_max_after_epoch1 = model._gm_max.clone()
    assert torch.isfinite(gm_min_after_epoch1).all()

    # Requesting a reset should NOT immediately wipe the envelope.
    model.request_latent_global_mean_envelope_reset()
    torch.testing.assert_close(model._gm_min, gm_min_after_epoch1)
    torch.testing.assert_close(model._gm_max, gm_max_after_epoch1)

    # Eval between request and next training forward sees the prior envelope.
    model.eval()
    with torch.no_grad():
        model(x, ctx)
    torch.testing.assert_close(model._gm_min, gm_min_after_epoch1)
    torch.testing.assert_close(model._gm_max, gm_max_after_epoch1)

    # The first subsequent training forward performs the reset, then
    # records new statistics from that batch alone.
    model.train()
    x_new = 0.5 * torch.randn(n_samples, 2, 9, 18, device=device)
    with torch.no_grad():
        model(x_new, ctx)
    # Envelope should now reflect only x_new, not the union with x.
    # A weak check: at least one channel's bounds must have changed.
    assert not torch.allclose(model._gm_min, gm_min_after_epoch1)
    assert not torch.allclose(model._gm_max, gm_max_after_epoch1)


def _make_spectral_conv(nlat, nlon, embed_dim, preserve_global_mean, bias=False):
    dist = Distributed.get_instance()
    modes_lat = nlat
    modes_lon = nlon // 2 + 1
    sht = dist.get_sht(
        nlat, nlon, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"
    )
    isht = dist.get_isht(
        nlat, nlon, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"
    )
    conv = SpectralConvS2(
        sht,
        isht,
        embed_dim,
        embed_dim,
        bias=bias,
        preserve_global_mean=preserve_global_mean,
    )
    return conv, sht


def test_filter_preserves_global_mean():
    torch.manual_seed(0)
    nlat, nlon, embed_dim = 16, 32, 8
    device = get_device()
    conv, sht = _make_spectral_conv(nlat, nlon, embed_dim, preserve_global_mean=True)
    conv = conv.to(device)
    sht = sht.to(device)

    x = torch.randn(2, embed_dim, nlat, nlon, device=device)
    with torch.no_grad():
        output, _ = conv(x)

    x_spectral = sht(x.float())
    out_spectral = sht(output.float())
    torch.testing.assert_close(
        out_spectral[:, :, 0, :], x_spectral[:, :, 0, :], atol=1e-5, rtol=1e-5
    )


def test_filter_preserves_global_mean_allows_grad():
    torch.manual_seed(0)
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    device = get_device()
    params = SFNONetConfig(
        embed_dim=16,
        num_layers=2,
        filter_type="linear",
        filter_preserves_global_mean=True,
    )
    model = get_lat_lon_sfnonet(
        params=params,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
    ).to(device)
    x = torch.randn(2, input_channels, *img_shape, device=device)
    context = Context(
        embedding_scalar=torch.zeros(2, 0, device=device),
        labels=torch.zeros(2, 0, device=device),
        noise=None,
        embedding_pos=None,
    )
    output = model(x, context)
    assert output.shape == (2, output_channels, *img_shape)
    output.sum().backward()
    for block in model.blocks:
        weight = block.filter.filter.weight
        assert weight.grad is not None
        assert not torch.all(weight.grad == 0)
