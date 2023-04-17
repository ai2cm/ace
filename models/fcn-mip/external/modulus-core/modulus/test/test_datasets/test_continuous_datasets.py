import torch
import numpy as np
from sympy import Symbol, sin

from modulus.geometry.primitives_2d import Rectangle
from modulus.dataset import (
    DictImportanceSampledPointwiseIterableDataset,
)
from modulus.domain.constraint.utils import _compute_outvar
from modulus.geometry.parameterization import Bounds


def test_DictImportanceSampledPointwiseIterableDataset():
    "sample sin function on a rectangle with importance measure sqrt(x**2 + y**2) and check its integral is zero"

    torch.manual_seed(123)
    np.random.seed(123)

    # make rectangle
    rec = Rectangle((-0.5, -0.5), (0.5, 0.5))

    # sample interior
    invar = rec.sample_interior(
        100000,
        bounds=Bounds({Symbol("x"): (-0.5, 0.5), Symbol("y"): (-0.5, 0.5)}),
    )

    # compute outvar
    outvar = _compute_outvar(invar, {"u": sin(2 * np.pi * Symbol("x") / 0.5)})

    # create importance measure
    def importance_measure(invar):
        return ((invar["x"] ** 2 + invar["y"] ** 2) ** (0.5)) + 0.01

    # make importance dataset
    dataset = DictImportanceSampledPointwiseIterableDataset(
        invar=invar,
        outvar=outvar,
        batch_size=10000,
        importance_measure=importance_measure,
    )

    # sample importance dataset
    invar, outvar, lambda_weighting = next(iter(dataset))

    # check integral calculation
    assert np.isclose(torch.sum(outvar["u"] * invar["area"]), 0.0, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":

    test_DictImportanceSampledPointwiseIterableDataset()
