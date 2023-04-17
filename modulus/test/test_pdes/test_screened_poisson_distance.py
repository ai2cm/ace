import numpy as np
import torch
import os
from modulus.eq.pdes.signed_distance_function import ScreenedPoissonDistance


def test_screened_poisson_distance_equation():
    # test data for screened poisson distance
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)

    distance = np.exp(x + y + z)
    distance__x = np.exp(x + y + z)
    distance__y = np.exp(x + y + z)
    distance__z = np.exp(x + y + z)
    distance__x__x = np.exp(x + y + z)
    distance__y__y = np.exp(x + y + z)
    distance__z__z = np.exp(x + y + z)

    tau = 0.1

    sdf_grad = 1 - distance__x**2 - distance__y**2 - distance__z**2
    poisson = np.sqrt(tau) * (distance__x__x + distance__y__y + distance__z__z)
    screened_poisson_distance_true = sdf_grad + poisson

    # evaluate the equation
    screened_poisson_distance_eq = ScreenedPoissonDistance(
        distance="distance", tau=tau, dim=3
    )
    evaluations = screened_poisson_distance_eq.make_nodes()[0].evaluate(
        {
            "distance__x": torch.tensor(distance__x, dtype=torch.float32),
            "distance__y": torch.tensor(distance__y, dtype=torch.float32),
            "distance__z": torch.tensor(distance__z, dtype=torch.float32),
            "distance__x__x": torch.tensor(distance__x__x, dtype=torch.float32),
            "distance__y__y": torch.tensor(distance__y__y, dtype=torch.float32),
            "distance__z__z": torch.tensor(distance__z__z, dtype=torch.float32),
        }
    )
    screened_poisson_distance_eq_eval_pred = evaluations[
        "screened_poisson_distance"
    ].numpy()

    # verify PDE computation
    assert np.allclose(
        screened_poisson_distance_eq_eval_pred, screened_poisson_distance_true
    ), "Test Failed!"


if __name__ == "__main__":
    test_screened_poisson_distance_equation()
