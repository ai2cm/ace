"""
This code is NVIDIA Confidential: SWIPAT underway

Spherical Harmonic transforms copied over from
https://gitlab-master.nvidia.com/bbonev/torch_sht

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


"""
These are the helper routines for computing the 
"""
def leggauss_scaled(nlg, a=-1.0, b=1.0):
    """
    Returns the Legendre Gauss points on the interval [a, b]
    """

    xlg, wlg = np.polynomial.legendre.leggauss(nlg)
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return xlg, wlg

def precompute_legendre_nodes(m_max, n_max, x):
    """
    Computes the values of P^m_n(\cos \theta) at the positions specified by x (theta)
    """

    # compute the tensor P^m_n:
    pct = np.zeros((m_max, n_max, len(x)), dtype=np.float64)

    sinx = np.sin(x)
    cosx = np.cos(x)

    a = lambda m, n: np.sqrt((4*n**2 - 1) / (n**2 - m**2))
    b = lambda m, n: -1 * np.sqrt((2*n+1)/(2*n-3)) * np.sqrt(((n-1)**2 - m**2)/(n**2 - m**2))

    # start by populating the diagonal and the second higher diagonal
    amm = np.sqrt( 1. / (4 * np.pi) )
    pct[0,0,:] = amm
    pct[0,1,:] = a(0, 1) * cosx * amm
    for m in range(1, min(m_max, n_max)):
        pct[m,m,:] = -1*np.sqrt( (2*m+1) / (2*m) ) * pct[m-1,m-1,:] * sinx
        if m + 1 < n_max:
            pct[m,m+1,:] = a(m, m+1) * cosx * pct[m,m,:]

    # fill the remaining values on the upper triangle
    for m in range(0, m_max):
        for n in range(m+2, n_max):
            pct[m,n,:] = a(m,n) * cosx * pct[m,n-1,:] + b(m,n) * pct[m,n-2,:]

    return pct

class RealSphericalHarmonicTransform(nn.Module):
    """
    Defines a module for computing the forward (real) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input
    """

    def __init__(self, n_theta, n_lambda, grid="legendre-gauss"):
        """
        Initializes the SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        n_theta: input grid resolution in the latitudinal direction
        n_lambda: input grid resolution in the longitudinal direction
        grid: type of grid the data lives on
        """

        super().__init__()

        self.n_theta = n_theta
        self.n_lambda = n_lambda
        self.grid = grid

        cost_lg, wlg = leggauss_scaled(n_theta, -1, 1)
        tq = np.flip(np.arccos(cost_lg))

        if grid == "equiangular":
            teq = np.linspace(0, np.pi, n_theta, dtype=np.float64)
            j = np.searchsorted(teq, tq) - 1
            # this is the one for lerp
            d = torch.from_numpy( (tq - teq[j]) / np.diff(teq)[j] )
            d = d.unsqueeze(-1) # append two dimensions, so it properly broadcasts
            j = torch.as_tensor(j)
            # for interpolation, do:
            # x = torch.lerp(x[..., self.j, :, :], x[..., self.j+1, :, :], self.d)

            # alternative way without lerp
            # d = torch.from_numpy((tq - teq[j]) / (teq[j+1] - teq[j]))
            # d = d.unsqueeze(-1).unsqueeze(-1)
            # j = torch.as_tensor(j)
            # for interpolation do:
            # x = (1 - self.d) * x[..., self.j, :, :] + self.d * x[..., self.j+1, :, :]

            self.register_buffer('d', d)
            self.register_buffer('j', j)


        weights = torch.from_numpy(wlg)
        pct = torch.from_numpy( precompute_legendre_nodes(n_lambda//2+1, n_theta, tq) )
        weights = torch.einsum('mnk,k->mnk', pct, weights)

        # remember quadrature weights
        self.register_buffer('weights', weights)

    def extra_repr(self):
        """
        This is mainly for properly printing module info
        """
        return f'n_theta={self.n_theta}, n_lambda={self.n_lambda},\n grid={self.grid}'

    def forward(self, x):

        # first do the interpolation in case the data is on the wrong grid
        if self.grid == "equiangular":
            x = torch.lerp(x[..., self.j, :], x[..., self.j+1, :], self.d)

        # apply real fft in the longitudinal direction
        x = 2.0 * torch.pi * torch.fft.rfft(x, axis=-1, norm="forward")
        
        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)
        x[..., 0] = torch.einsum('...km,mnk->...nm', x[..., 0], self.weights )
        x[..., 1] = torch.einsum('...km,mnk->...nm', x[..., 1], self.weights )
        x = torch.view_as_complex(x)

        return x

class InverseRealSphericalHarmonicTransform(nn.Module):
    """
    Defines a module for computing the inverse (real) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    n_theta, n_lambda: resolution of the input (spherical coefficients)
    h_out, w_out = output signal
    """

    def __init__(self, n_theta, n_lambda, h_out=None, w_out=None, grid="legendre-gauss"):

        super().__init__()

        self.n_theta = n_theta
        self.n_lambda = n_lambda
        self.h_out = h_out or n_theta
        self.w_out = w_out or n_lambda
        self.grid = grid

        if grid == "legendre-gauss":
            cost_lg, _ = leggauss_scaled(self.h_out, -1, 1)
            t = np.flip(np.arccos(cost_lg))
        elif grid == "equiangular":
            t = np.linspace(0, np.pi, self.h_out)
        else:
            raise("Unknown grid type")

        pct = torch.from_numpy( precompute_legendre_nodes(n_lambda//2+1, n_theta, t) )

        self.register_buffer('pct', pct)

    def extra_repr(self):
        """
        This is mainly for properly printing module info
        """
        return f'n_theta={self.n_theta}, n_lambda={self.n_lambda},\n h_out={self.h_out}, w_out={self.w_out},\n grid={self.grid}'

    def forward(self, x):

        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)
        # x[..., 0] = torch.einsum('...nm, mnk->...km', x[..., 0], self.pct )
        # x[..., 1] = torch.einsum('...nm, mnk->...km', x[..., 1], self.pct )
        # x = torch.view_as_complex(x)
        rl = torch.einsum('...nm, mnk->...km', x[..., 0], self.pct )
        im = torch.einsum('...nm, mnk->...km', x[..., 1], self.pct )
        x = torch.view_as_complex(torch.stack((rl, im), -1))

        # apply the inverse (real) FFT
        x = torch.fft.irfft(x, n=self.w_out, axis=-1, norm="forward")

        return x