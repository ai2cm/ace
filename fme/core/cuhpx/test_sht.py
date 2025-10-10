import random

import torch

from fme.core.cuhpx.sht import SHT, iSHT
from fme.core.device import get_device


def generate_xyv(n, xmax, xmin):
    v, x, y = [], [], []
    for _ in range(n):
        vi = random.random()  # Generate a random float number between 0 and 1
        xi = random.randint(
            xmin, xmax - 1
        )  # Generate a random int number, 0 <= x < xmax
        yi = random.randint(xmin, xi)  # Generate a random int number, 0 <= y <= x

        v.append(vi)
        x.append(xi)
        y.append(yi)

    return x, y, v


def fill_matrix(x, y, v, matrix):
    n = len(x)
    for i in range(n):
        matrix[x[i], y[i]] = v[i]  # Fill the matrix at position (x, y) with the value v
    return matrix


def test_sht_round_trip():
    device = get_device()
    torch.manual_seed(0)
    nside = 32
    npix = 12 * nside**2

    quad_weights = "ring"
    lmax = 2 * nside - 1
    mmax = lmax

    sht = SHT(nside, lmax=lmax, mmax=mmax, quad_weights=quad_weights)
    isht = iSHT(nside, lmax=lmax, mmax=mmax)

    signal_ori = torch.randn(npix, dtype=torch.float32).to(device)

    signal_back = isht(sht(signal_ori))
    signal_back_again = isht(sht(signal_back))
    diff_round_trip = signal_back - signal_back_again
    rms_after_round_trip = torch.sqrt((diff_round_trip.abs().pow(2)).mean())

    assert rms_after_round_trip < 0.001
