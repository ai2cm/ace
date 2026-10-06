import unittest.mock

import pytest
import torch
from torch_harmonics import InverseRealSHT

from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNO, isotropic_noise
from fme.core.device import get_device
from fme.core.models.conditional_sfno.layers import Context


@pytest.mark.parametrize("nlat, nlon", [(8, 16), (64, 128)])
def test_isotropic_noise(nlat: int, nlon: int):
    torch.manual_seed(0)
    n_batch = 1000
    embed_dim = 4
    leading_shape = (n_batch, embed_dim)
    isht = InverseRealSHT(nlat, nlon, grid="legendre-gauss")
    lmax = isht.lmax
    mmax = isht.mmax
    noise = isotropic_noise(leading_shape, lmax, mmax, isht, device=get_device())
    assert noise.shape == (n_batch, embed_dim, nlat, nlon)
    assert noise.dtype == torch.float32
    torch.testing.assert_close(
        noise.mean(), torch.tensor(0.0, device=noise.device), atol=2e-3, rtol=0.0
    )
    torch.testing.assert_close(
        noise.std(), torch.tensor(1.0, device=noise.device), atol=5e-3, rtol=0.0
    )


def test_noise_conditioned_sfno_conditioning():
    mock_sfno = unittest.mock.MagicMock()
    img_shape = (32, 64)
    n_noise = 16
    n_pos = 8
    n_labels = 4
    label_embed_dim = 3
    model = NoiseConditionedSFNO(
        conditional_model=mock_sfno,
        img_shape=img_shape,
        embed_dim_noise=n_noise,
        embed_dim_pos=n_pos,
        n_labels=n_labels,
        label_embed_dim=label_embed_dim,
    )
    batch_size = 2
    x = torch.randn(batch_size, 3, img_shape[0], img_shape[1])
    labels = torch.randn(batch_size, n_labels)
    _ = model(x, labels=labels)
    mock_sfno.assert_called()
    args, _ = mock_sfno.call_args
    conditioned_x = args[0]
    assert conditioned_x.shape == (batch_size, 3, img_shape[0], img_shape[1])
    context = args[1]
    assert isinstance(context, Context)
    assert context.embedding_scalar is None
    assert context.embedding_pos is not None
    assert context.labels is not None
    assert context.noise is not None
    assert context.embedding_pos.shape == (
        batch_size,
        n_pos,
        img_shape[0],
        img_shape[1],
    )
    assert context.labels.shape == (batch_size, label_embed_dim)
    assert context.noise.shape == (batch_size, n_noise, img_shape[0], img_shape[1])


def test_noise_conditioned_sfno_onehot_labels():
    """When label_embed_dim=0, one-hot labels pass through directly."""
    mock_sfno = unittest.mock.MagicMock()
    img_shape = (32, 64)
    n_labels = 4
    model = NoiseConditionedSFNO(
        conditional_model=mock_sfno,
        img_shape=img_shape,
        embed_dim_noise=8,
        embed_dim_pos=4,
        n_labels=n_labels,
        label_embed_dim=0,
    )
    batch_size = 2
    x = torch.randn(batch_size, 3, img_shape[0], img_shape[1])
    labels = torch.randn(batch_size, n_labels)
    _ = model(x, labels=labels)
    args, _ = mock_sfno.call_args
    context = args[1]
    assert context.labels.shape == (batch_size, n_labels)
