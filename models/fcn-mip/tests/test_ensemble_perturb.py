from fcn_mip.ensemble_utils import generate_noise_correlated
from modulus.distributed.manager import DistributedManager
import torch


def test_generate_noise_correlated(dist: DistributedManager):
    torch.manual_seed(0)
    shape = (2, 34, 32, 64)
    noise = generate_noise_correlated(
        shape=shape, reddening=2.0, noise_amplitude=0.1, device=dist.device
    )
    assert tuple(noise.shape) == tuple(shape)
    assert torch.mean(noise) < torch.tensor(1e-09).to(dist.device)
