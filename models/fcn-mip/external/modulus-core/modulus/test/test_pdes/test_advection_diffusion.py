import numpy as np
import torch
import os
from modulus.eq.pdes.advection_diffusion import AdvectionDiffusion


def test_advection_diffusion():
    # test data for advection diffusion equation
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    t = np.random.rand(1024, 1)

    T = np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u = np.exp(2 * x + y + z)
    v = np.exp(x + 2 * y + z)
    w = np.exp(x + y + 2 * z)

    rho = 1.0

    D = 0.1

    T__t = -np.sin(x) * np.sin(y) * np.sin(z) * np.sin(t)
    T__x = np.cos(x) * np.sin(y) * np.sin(z) * np.cos(t)
    T__y = np.sin(x) * np.cos(y) * np.sin(z) * np.cos(t)
    T__z = np.sin(x) * np.sin(y) * np.cos(z) * np.cos(t)

    T__x__x = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    T__y__y = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    T__z__z = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)

    advection = u * T__x + v * T__y + w * T__z
    diffusion = D * T__x__x + D * T__y__y + D * T__z__z
    curl = 0
    advection_diffusion_equation_true = T__t + advection + T * curl - diffusion

    # evaluate the equation
    eq = AdvectionDiffusion(T="T", D=D, rho=float(rho), dim=3, time=True)
    evaluations = eq.make_nodes()[0].evaluate(
        {
            "T__t": torch.tensor(T__t, dtype=torch.float32),
            "T__x": torch.tensor(T__x, dtype=torch.float32),
            "T__y": torch.tensor(T__y, dtype=torch.float32),
            "T__z": torch.tensor(T__z, dtype=torch.float32),
            "T__x__x": torch.tensor(T__x__x, dtype=torch.float32),
            "T__y__y": torch.tensor(T__y__y, dtype=torch.float32),
            "T__z__z": torch.tensor(T__z__z, dtype=torch.float32),
            "u": torch.tensor(u, dtype=torch.float32),
            "v": torch.tensor(v, dtype=torch.float32),
            "w": torch.tensor(w, dtype=torch.float32),
        }
    )
    eq_eval = evaluations["advection_diffusion_T"].numpy()

    # verify PDE computation
    assert np.allclose(eq_eval, advection_diffusion_equation_true), "Test Failed!"


if __name__ == "__main__":
    test_advection_diffusion()
