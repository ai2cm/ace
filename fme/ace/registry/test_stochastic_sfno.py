import datetime
import unittest.mock
from typing import Literal

import pytest
import torch
from torch_harmonics import InverseRealSHT

from fme.ace.registry.stochastic_sfno import (
    NoiseConditionedSFNO,
    NoiseConditionedSFNOBuilder,
    isotropic_noise,
)
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.models.conditional_sfno.layers import Context
from fme.core.rand import set_seed


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


def _build_dataset_info(img_shape: tuple[int, int] = (9, 18)) -> DatasetInfo:
    device = get_device()
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(img_shape[0], device=device),
            lon=torch.zeros(img_shape[1], device=device),
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(7, device=device), bk=torch.arange(7, device=device)
        ),
        timestep=datetime.timedelta(hours=6),
    )


def _build_ncsfno(
    input_noise_channels: int,
    input_noise_type: Literal["isotropic", "gaussian"],
    noise_type: Literal["isotropic", "gaussian"],
    n_in_channels: int = 5,
    n_out_channels: int = 6,
) -> torch.nn.Module:
    builder = NoiseConditionedSFNOBuilder(
        embed_dim=8,
        noise_embed_dim=4,
        noise_type=noise_type,
        input_noise_channels=input_noise_channels,
        input_noise_type=input_noise_type,
        num_layers=2,
        use_mlp=True,
        affine_norms=True,
    )
    return builder.build(
        n_in_channels=n_in_channels,
        n_out_channels=n_out_channels,
        dataset_info=_build_dataset_info(),
    )


@pytest.mark.parametrize("noise_type", ["isotropic", "gaussian"])
@pytest.mark.parametrize("input_noise_type", ["isotropic", "gaussian"])
def test_input_noise_channels_preserve_output_shape(
    noise_type: Literal["isotropic", "gaussian"],
    input_noise_type: Literal["isotropic", "gaussian"],
):
    """A model with input-noise channels forward-passes to the same output
    shape as one without, for every conditioning/input noise combination."""
    set_seed(0)
    n_in, n_out = 5, 6
    img_shape = (9, 18)
    baseline = _build_ncsfno(0, "gaussian", noise_type, n_in, n_out)
    with_noise = _build_ncsfno(8, input_noise_type, noise_type, n_in, n_out)
    x = torch.randn(2, n_in, *img_shape, device=get_device())
    out_baseline = baseline(x)
    out_with_noise = with_noise(x)
    assert out_baseline.shape == (2, n_out, *img_shape)
    assert out_with_noise.shape == out_baseline.shape


def test_default_config_adds_no_input_noise_channels():
    """Backward compat: the default builder wires zero input-noise channels,
    so the wrapped network sees exactly n_in_channels."""
    set_seed(0)
    n_in, n_out = 5, 6
    model = _build_ncsfno(0, "gaussian", "gaussian", n_in, n_out)
    assert model._input_noise_channels == 0
    x = torch.randn(2, n_in, 9, 18, device=get_device())
    assert model(x).shape == (2, n_out, 9, 18)


@pytest.mark.parametrize("input_noise_type", ["isotropic", "gaussian"])
def test_input_noise_channels_are_concatenated_to_input(
    input_noise_type: Literal["isotropic", "gaussian"],
):
    """The input-noise channels are appended to x before the wrapped module,
    independently of the (here gaussian) conditioning noise."""
    mock_sfno = unittest.mock.MagicMock()
    img_shape = (8, 16)
    n_in = 3
    input_noise_channels = 5
    isht = InverseRealSHT(*img_shape, grid="legendre-gauss")
    model = NoiseConditionedSFNO(
        conditional_model=mock_sfno,
        img_shape=img_shape,
        embed_dim_noise=4,
        inverse_sht=isht,
        lmax=isht.lmax,
        mmax=isht.mmax,
        conditioning_noise_isotropic=False,
        input_noise_channels=input_noise_channels,
        input_noise_type=input_noise_type,
    )
    x = torch.randn(2, n_in, *img_shape)
    _ = model(x)
    args, _ = mock_sfno.call_args
    conditioned_x = args[0]
    assert conditioned_x.shape == (2, n_in + input_noise_channels, *img_shape)
    context = args[1]
    # conditioning noise is gaussian despite inverse_sht being provided
    assert context.noise.shape == (2, 4, *img_shape)


@pytest.mark.parametrize("input_noise_type", ["isotropic", "gaussian"])
def test_input_noise_channels_backward(
    input_noise_type: Literal["isotropic", "gaussian"],
):
    """Gradients flow through a model with input-noise channels."""
    set_seed(0)
    n_in, n_out = 5, 6
    model = _build_ncsfno(8, input_noise_type, "gaussian", n_in, n_out)
    x = torch.randn(2, n_in, 9, 18, device=get_device())
    loss = model(x).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert len(grads) > 0
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
