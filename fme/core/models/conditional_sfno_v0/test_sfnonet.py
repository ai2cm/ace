import os
from types import SimpleNamespace

import torch

from fme.core.device import get_device
from fme.core.testing.regression import validate_tensor

from .layers import Context, ContextConfig
from .sfnonet import get_lat_lon_sfnonet

DIR = os.path.abspath(os.path.dirname(__file__))


def test_sfnonet_output_is_unchanged():
    torch.manual_seed(0)
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    conditional_embed_dim_scalar = 8
    conditional_embed_dim_labels = 4
    conditional_embed_dim_noise = 16
    conditional_embed_dim_pos = 0
    device = get_device()
    params = SimpleNamespace(
        embed_dim=16, num_layers=2, filter_type="linear", operator_type="dhconv"
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
    with torch.no_grad():
        output = model(x, context)
    validate_tensor(
        output,
        os.path.join(DIR, "testdata/test_sfnonet_output_is_unchanged.pt"),
    )
