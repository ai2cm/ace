import torch
import numpy as np
from einops import rearrange
from fcn_mip import schema


def generate_noise_correlated(shape, *, reddening, device, noise_amplitude):
    return noise_amplitude * brown_noise(shape, reddening).to(device)


def brown_noise(shape, reddening=2):

    noise = torch.normal(torch.zeros(shape), torch.ones(shape))

    x_white = torch.fft.fft2(noise)
    S = (
        torch.abs(torch.fft.fftfreq(noise.shape[-2]).reshape(-1, 1)) ** reddening
        + torch.abs(torch.fft.fftfreq(noise.shape[-1])) ** reddening
    )

    S = torch.where(S == 0, 0, 1 / S)
    S = S / torch.sqrt(torch.mean(S**2))

    x_shaped = x_white * S
    noise_shaped = torch.fft.ifft2(x_shaped).real

    return noise_shaped


def gp_sample(length_scale, num_features=1000, coefficient=1.0):

    x = rearrange(np.mgrid[0:720, 0:1440], "d x y -> (x y) d")
    x = torch.tensor(x).float().cuda()

    omega_shape = (1, num_features, 2)
    omega = torch.normal(
        mean=torch.zeros(omega_shape), std=torch.ones(omega_shape)
    ).cuda()
    omega /= length_scale

    weight_shape = (1, num_features)
    weights = torch.normal(
        mean=torch.zeros(weight_shape), std=torch.ones(weight_shape)
    ).cuda()

    phi = torch.rand((1, num_features, 1)) * (2 * np.pi)
    phi = phi.cuda()

    features = torch.cos(torch.einsum("sfd, nd -> sfn", omega, x) + phi)
    features = (2 / num_features) ** 0.5 * features * coefficient

    functions = torch.einsum("sf, sfn -> sn", weights, features)

    return functions.reshape(1, 720, 1440)


def draw_noise(corr, spreads, length_scales, device):

    z = [gp_sample(l) for l in length_scales]
    z = torch.stack(z, dim=1)

    if spreads is not None:
        sigma = spreads.permute(1, 2, 0)
        A = corr * (sigma[..., None] * sigma[..., None, :])

    else:
        sigma = torch.ones((720, 1440, 34))
        A = corr

    L = torch.linalg.cholesky_ex(A.cpu())[0].to(device)
    z = z[0].permute(1, 2, 0)[..., None]

    return torch.matmul(L, z).permute(3, 2, 0, 1)


def get_skill_spread(ens, obs):
    ensemble_mean = np.mean(ens, axis=0)
    MSE = np.power(obs - ensemble_mean, 2.0)
    RMSE = np.sqrt(MSE)
    spread = np.std(ens, axis=0)
    return RMSE, spread


def load_correlation(grid=schema.Grid.grid_721x1440):
    correlation = torch.load("/lustre/fsw/sw_climate_fno/ensemble_init_stats/corr.pth")
    if grid == schema.Grid.grid_721x1440:
        correlation = torch.cat((correlation, correlation[-1, :, :, :].unsqueeze(0)), 0)
    return correlation
