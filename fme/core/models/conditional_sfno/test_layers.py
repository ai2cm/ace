import pytest
import torch

from fme.core.device import get_device

from .layers import ConditionalLayerNorm, Context, ContextConfig


def get_context(
    img_shape,
    embed_dim_scalar: int,
    embed_dim_labels: int,
    embed_dim_noise: int,
    embed_dim_pos: int,
) -> Context:
    device = get_device()
    context_embedding_scalar = (
        torch.randn(1, embed_dim_scalar, device=device)
        if embed_dim_scalar > 0
        else None
    )
    context_embedding_labels = (
        torch.randn(1, embed_dim_labels, device=device)
        if embed_dim_labels > 0
        else None
    )
    context_embedding_noise = (
        torch.randn(1, embed_dim_noise, *img_shape, device=device)
        if embed_dim_noise > 0
        else None
    )
    context_embedding_pos = (
        torch.randn(1, embed_dim_pos, *img_shape, device=device)
        if embed_dim_pos > 0
        else None
    )
    return Context(
        embedding_scalar=context_embedding_scalar,
        noise=context_embedding_noise,
        labels=context_embedding_labels,
        embedding_pos=context_embedding_pos,
    )


@pytest.mark.parametrize("global_layer_norm", [True, False])
@pytest.mark.parametrize("n_channels", [32])
@pytest.mark.parametrize("embed_dim_scalar", [9, 0])
@pytest.mark.parametrize("embed_dim_noise", [10, 0])
@pytest.mark.parametrize("embed_dim_labels", [11, 0])
@pytest.mark.parametrize("embed_dim_pos", [18, 0])
@pytest.mark.parametrize("img_shape", [(8, 16)])
def test_conditional_layer_norm(
    n_channels: int,
    img_shape: tuple[int, int],
    global_layer_norm: bool,
    embed_dim_scalar: int,
    embed_dim_labels: int,
    embed_dim_noise: int,
    embed_dim_pos: int,
):
    epsilon = 1e-6
    device = get_device()
    conditional_layer_norm = ConditionalLayerNorm(
        n_channels,
        img_shape,
        context_config=ContextConfig(
            embed_dim_scalar=embed_dim_scalar,
            embed_dim_labels=embed_dim_labels,
            embed_dim_noise=embed_dim_noise,
            embed_dim_pos=embed_dim_pos,
        ),
        global_layer_norm=global_layer_norm,
        epsilon=epsilon,
    ).to(device)
    x = torch.randn(1, n_channels, *img_shape, device=device) * 5 + 2
    context = get_context(
        img_shape,
        embed_dim_scalar=embed_dim_scalar,
        embed_dim_labels=embed_dim_labels,
        embed_dim_noise=embed_dim_noise,
        embed_dim_pos=embed_dim_pos,
    )
    output = conditional_layer_norm(x, context)
    assert output.shape == x.shape
    torch.testing.assert_close(
        output.mean(), torch.tensor(0.0, device=device), atol=1e-3, rtol=0
    )
    torch.testing.assert_close(
        output.std(), torch.tensor(1.0, device=device), atol=1e-3, rtol=0
    )
    if not global_layer_norm:
        zero = torch.zeros(1, *img_shape, device=device)
        torch.testing.assert_close(output.mean(dim=1), zero, atol=1e-3, rtol=0)
        torch.testing.assert_close(
            (((n_channels - 1) / n_channels) ** 0.5 * output.std(dim=1) - 1),
            zero,
            atol=1e-3,
            rtol=0,
        )


def _mask_cln(n_channels: int, img_shape: tuple[int, int], embed_dim_mask: int):
    return ConditionalLayerNorm(
        n_channels,
        img_shape,
        context_config=ContextConfig(
            embed_dim_scalar=0,
            embed_dim_labels=0,
            embed_dim_noise=0,
            embed_dim_pos=0,
            embed_dim_mask=embed_dim_mask,
        ),
    ).to(get_device())


def test_channel_mask_zero_init_is_identity_across_masks():
    """At init the mask linears are zero, so different masks give same output."""
    n_channels, img_shape, embed_dim_mask = 32, (8, 16), 5
    device = get_device()
    cln = _mask_cln(n_channels, img_shape, embed_dim_mask)
    x = torch.randn(2, n_channels, *img_shape, device=device)
    mask_a = torch.ones(2, embed_dim_mask, device=device)
    mask_b = torch.zeros(2, embed_dim_mask, device=device)
    out_a = cln(x, _mask_context(mask_a))
    out_b = cln(x, _mask_context(mask_b))
    torch.testing.assert_close(out_a, out_b)


def test_channel_mask_changes_output_after_weights_set():
    n_channels, img_shape, embed_dim_mask = 32, (8, 16), 5
    device = get_device()
    cln = _mask_cln(n_channels, img_shape, embed_dim_mask)
    torch.nn.init.normal_(cln.W_scale_mask.weight)
    torch.nn.init.normal_(cln.W_bias_mask.weight)
    x = torch.randn(2, n_channels, *img_shape, device=device)
    mask_a = torch.ones(2, embed_dim_mask, device=device)
    mask_b = torch.zeros(2, embed_dim_mask, device=device)
    out_a = cln(x, _mask_context(mask_a))
    out_b = cln(x, _mask_context(mask_b))
    assert not torch.allclose(out_a, out_b)


def test_channel_mask_missing_raises():
    cln = _mask_cln(32, (8, 16), 5)
    x = torch.randn(2, 32, 8, 16, device=get_device())
    with pytest.raises(ValueError, match="channel_mask must be provided"):
        cln(x, _mask_context(None))


def test_context_channel_mask_ndim_guard():
    with pytest.raises(ValueError, match="channel_mask must have 2 dimensions"):
        _mask_context(torch.ones(2, 5, 1, device=get_device()))


def _mask_context(channel_mask: torch.Tensor | None) -> Context:
    return Context(
        embedding_scalar=None,
        embedding_pos=None,
        labels=None,
        noise=None,
        channel_mask=channel_mask,
    )


@pytest.mark.parametrize("embed_dim_scalar", [9, 0])
@pytest.mark.parametrize("embed_dim_noise", [10, 0])
@pytest.mark.parametrize("embed_dim_labels", [11, 0])
@pytest.mark.parametrize("embed_dim_pos", [18, 0])
def test_context_round_trip(
    embed_dim_scalar: int,
    embed_dim_labels: int,
    embed_dim_noise: int,
    embed_dim_pos: int,
):
    img_shape = (8, 16)
    context = get_context(
        img_shape,
        embed_dim_scalar=embed_dim_scalar,
        embed_dim_labels=embed_dim_labels,
        embed_dim_noise=embed_dim_noise,
        embed_dim_pos=embed_dim_pos,
    )
    round_trip = Context.from_dict(context.asdict())
    for attr_name in dir(context):
        # this ensures the test fails if we ever add additional attributes to
        # Context and forget to update the test, which would be a problem since
        # those attributes would not be tested for correct round-tripping
        if not attr_name.startswith("_") and not callable(getattr(context, attr_name)):
            original = getattr(context, attr_name)
            round_tripped = getattr(round_trip, attr_name)
            if original is None:
                assert round_tripped is None
            elif isinstance(original, torch.Tensor):
                torch.testing.assert_close(original, round_tripped)
            else:
                raise NotImplementedError(
                    f"Unsupported attribute type for {attr_name}: {type(original)}, "
                    "update the test if this is expected"
                )
