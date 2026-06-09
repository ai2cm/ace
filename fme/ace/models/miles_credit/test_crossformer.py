import pytest
import torch

from fme.ace.models.miles_credit.crossformer import CrossFormer
from fme.core.models.conditional_sfno.layers import ContextConfig


def make_crossformer(img_shape, context_config=None):
    return CrossFormer(
        image_height=img_shape[0],
        image_width=img_shape[1],
        frames=1,
        channels=2,
        surface_channels=2,
        input_only_channels=1,
        output_only_channels=0,
        levels=2,
        dim=[16, 32, 64, 128],
        depth=[1, 1, 1, 1],
        dim_head=8,
        global_window_size=[4, 4, 2, 1],
        local_window_size=3,
        cross_embed_kernel_sizes=[[4, 8, 16, 32], [2, 4], [2, 4], [2, 4]],
        cross_embed_strides=[2, 2, 2, 2],
        use_spectral_norm=False,
        interp=True,
        context_config=context_config,
    )


def make_padded_crossformer(img_shape, context_config=None):
    # pad_lat=[2,1]: 45->48, pad_lon=[3,3]: 90->96; stages: 24x48,12x24,6x12,3x6
    return CrossFormer(
        image_height=img_shape[0],
        image_width=img_shape[1],
        frames=1,
        channels=2,
        surface_channels=2,
        input_only_channels=1,
        output_only_channels=0,
        levels=2,
        dim=[16, 32, 64, 128],
        depth=[1, 1, 1, 1],
        dim_head=8,
        global_window_size=[4, 4, 2, 1],
        local_window_size=3,
        cross_embed_kernel_sizes=[[4, 8, 16, 32], [2, 4], [2, 4], [2, 4]],
        cross_embed_strides=[2, 2, 2, 2],
        use_spectral_norm=False,
        interp=True,
        padding_conf={
            "activate": True,
            "mode": "earth",
            "pad_lat": [2, 1],
            "pad_lon": [3, 3],
        },
        context_config=context_config,
    )


@pytest.mark.parametrize("img_shape", [(48, 96)])
def test_crossformer_forward_shape(img_shape):
    model = make_crossformer(img_shape)
    batch_size = 2
    n_channels = 2 * 2 + 2 + 1  # channels*levels + surface + input_only
    x = torch.randn(batch_size, n_channels, 1, *img_shape)
    out = model(x)
    out = out.squeeze(2)
    assert out.shape == (batch_size, 2 * 2 + 2, *img_shape)


@pytest.mark.parametrize("img_shape", [(45, 90)])
def test_crossformer_with_padding(img_shape):
    model = make_padded_crossformer(img_shape)
    batch_size = 2
    n_channels = 2 * 2 + 2 + 1
    x = torch.randn(batch_size, n_channels, 1, *img_shape)
    out = model(x)
    out = out.squeeze(2)
    assert out.shape == (batch_size, 2 * 2 + 2, *img_shape)


@pytest.mark.parametrize("img_shape", [(48, 96)])
def test_crossformer_noise_conditioned_shape(img_shape):
    noise_embed_dim = 8
    context_config = ContextConfig(
        embed_dim_scalar=0,
        embed_dim_labels=0,
        embed_dim_noise=noise_embed_dim,
        embed_dim_pos=0,
    )
    model = make_crossformer(img_shape, context_config=context_config)
    batch_size = 2
    n_channels = 2 * 2 + 2 + 1
    x = torch.randn(batch_size, n_channels, 1, *img_shape)
    # noise shape: (batch, embed_dim_noise, H, W)
    from fme.core.models.conditional_sfno.layers import Context

    noise = torch.randn(batch_size, noise_embed_dim, *img_shape)
    ctx = Context(embedding_scalar=None, embedding_pos=None, labels=None, noise=noise)
    out = model(x, context=ctx)
    out = out.squeeze(2)
    assert out.shape == (batch_size, 2 * 2 + 2, *img_shape)


@pytest.mark.parametrize("img_shape", [(48, 96)])
def test_crossformer_noise_stochastic(img_shape):
    """
    Different noise tensors produce different outputs when CLN weights are non-zero.
    """
    noise_embed_dim = 8
    context_config = ContextConfig(
        embed_dim_scalar=0,
        embed_dim_labels=0,
        embed_dim_noise=noise_embed_dim,
        embed_dim_pos=0,
    )
    model = make_crossformer(img_shape, context_config=context_config)
    # CLN weight initializes to zero (no-op); set non-zero to enable noise effect.
    for m in model.modules():
        from fme.core.models.conditional_sfno.layers import ConditionalLayerNorm

        if isinstance(m, ConditionalLayerNorm):
            if m.W_scale_2d is not None:
                torch.nn.init.normal_(m.W_scale_2d.weight)
    model.eval()
    batch_size = 1
    n_channels = 2 * 2 + 2 + 1
    x = torch.randn(batch_size, n_channels, 1, *img_shape)
    from fme.core.models.conditional_sfno.layers import Context

    noise1 = torch.randn(batch_size, noise_embed_dim, *img_shape)
    noise2 = torch.randn(batch_size, noise_embed_dim, *img_shape)
    ctx1 = Context(embedding_scalar=None, embedding_pos=None, labels=None, noise=noise1)
    ctx2 = Context(embedding_scalar=None, embedding_pos=None, labels=None, noise=noise2)
    with torch.no_grad():
        out1 = model(x, context=ctx1)
        out2 = model(x, context=ctx2)
    assert not torch.allclose(out1, out2)


@pytest.mark.parametrize("img_shape", [(48, 96)])
def test_crossformer_deterministic_no_context(img_shape):
    """Without context, two forward passes produce identical outputs."""
    model = make_crossformer(img_shape)
    model.eval()
    batch_size = 1
    n_channels = 2 * 2 + 2 + 1
    x = torch.randn(batch_size, n_channels, 1, *img_shape)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)


@pytest.mark.parametrize("img_shape", [(48, 96)])
def test_noise_conditioned_crossformer_spectral_norm_no_nan(img_shape):
    """
    CrossFormer with spectral norm and CLN noise conditioning must not produce NaN.
    CLN conditioning layers are zero-initialized; spectral norm must skip them to
    avoid sigma=0 → weight=0/0=NaN.
    """
    from fme.core.models.conditional_sfno.layers import Context

    noise_embed_dim = 8
    context_config = ContextConfig(
        embed_dim_scalar=0,
        embed_dim_labels=0,
        embed_dim_noise=noise_embed_dim,
        embed_dim_pos=0,
    )
    model = CrossFormer(
        image_height=img_shape[0],
        image_width=img_shape[1],
        frames=1,
        channels=2,
        surface_channels=2,
        input_only_channels=1,
        output_only_channels=0,
        levels=2,
        dim=[16, 32, 64, 128],
        depth=[1, 1, 1, 1],
        dim_head=8,
        global_window_size=[4, 4, 2, 1],
        local_window_size=3,
        cross_embed_kernel_sizes=[[4, 8, 16, 32], [2, 4], [2, 4], [2, 4]],
        cross_embed_strides=[2, 2, 2, 2],
        use_spectral_norm=True,
        interp=True,
        context_config=context_config,
    )
    model.eval()
    batch_size = 1
    n_channels = 2 * 2 + 2 + 1
    x = torch.randn(batch_size, n_channels, 1, *img_shape)
    noise = torch.randn(batch_size, noise_embed_dim, *img_shape)
    ctx = Context(embedding_scalar=None, embedding_pos=None, labels=None, noise=noise)
    with torch.no_grad():
        out = model(x, context=ctx)
    assert not torch.isnan(out).any(), "NaN in output with spectral norm + CLN"
