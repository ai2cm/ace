import pytest
import torch
from torch_harmonics import InverseRealSHT

from fme.ace.registry.stochastic_sfno import isotropic_noise
from fme.core.device import get_device


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
