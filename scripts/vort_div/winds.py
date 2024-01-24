from typing import Tuple

import numpy as np
import torch
from torch_harmonics.sht import (
    InverseRealSHT,
    InverseRealVectorSHT,
    RealSHT,
    RealVectorSHT,
)


def u_v_to_vort_div(
    u: np.ndarray,
    v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_lat = u.shape[-2]
    n_lon = u.shape[-1]
    x = torch.stack([torch.asarray(v), torch.asarray(u)], dim=-3).double()
    vector_sht = RealVectorSHT(
        nlat=n_lat,
        nlon=n_lon,
        grid="legendre-gauss",
    ).double()
    # x = torch.stack([torch.asarray(u), torch.asarray(v)], dim=-3)
    x_sht = vector_sht.forward(x)
    # x_sht are the poloidal and toroidal components of the vector field
    # the poloidal component is the divergence-free component
    # the toroidal component is the curl-free component
    #
    # we want to convert these to vorticity and divergence by taking the
    # laplacian of each component
    # x_sht_lap = -l * (l + 1) * x_sht
    l = torch.arange(0, vector_sht.lmax)  # noqa: E741
    # the axis for l is -3, ensure correct broadcasting
    x_sht_lap = torch.einsum("l,...lm->...lm", -l * (l + 1), x_sht)
    inverse_sht = InverseRealSHT(
        nlat=n_lat,
        nlon=n_lon,
        grid="legendre-gauss",
    ).double()
    div_vort = inverse_sht.forward(x_sht_lap)
    div = div_vort[..., 0, :, :]
    vort = div_vort[..., 1, :, :]
    return vort.cpu().numpy(), div.cpu().numpy()


def vort_div_to_u_v(
    vort: np.ndarray,
    div: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_lat = vort.shape[-2]
    n_lon = vort.shape[-1]
    x = torch.stack([torch.asarray(div), torch.asarray(vort)], dim=-3).double()
    sht = RealSHT(
        nlat=n_lat,
        nlon=n_lon,
        grid="legendre-gauss",
    ).double()
    # here we do the opposite operations as in u_v_to_vort_div, see that function
    # for further helpful comments about what's going on.
    x_sht = sht.forward(x)
    l = torch.arange(0, sht.lmax)  # noqa: E741
    # take the inverse laplacian of x_sht
    # the axis for l is -3, ensure correct broadcasting
    # x_sht_lap = x_sht / (-l * (l + 1))
    inverse_lap = 1.0 / (-l * (l + 1))
    inverse_lap[0] = 0.0
    x_sht_inverse_lap = torch.einsum("l,...lm->...lm", inverse_lap, x_sht)
    inverse_vector_sht = InverseRealVectorSHT(
        nlat=n_lat,
        nlon=n_lon,
        grid="legendre-gauss",
    ).double()
    v_u = inverse_vector_sht.forward(x_sht_inverse_lap)
    v = v_u[..., 0, :, :]
    u = v_u[..., 1, :, :]
    return u.cpu().numpy(), v.cpu().numpy()
