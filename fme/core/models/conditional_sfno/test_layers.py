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
