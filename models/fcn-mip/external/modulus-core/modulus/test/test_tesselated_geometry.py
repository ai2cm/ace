from sympy import Symbol
import numpy as np

from modulus.geometry.tessellation import Tessellation
from modulus.geometry import Parameterization


def test_tesselated_geometry():
    # read in cube file
    cube = Tessellation.from_stl("stls/cube.stl")

    # sample boundary
    boundary = cube.sample_boundary(
        1000, parameterization=Parameterization({Symbol("fake_param"): 1})
    )

    # sample interior
    interior = cube.sample_interior(
        1000, parameterization=Parameterization({Symbol("fake_param"): 1})
    )

    # check if surface area is right for boundary
    assert np.isclose(np.sum(boundary["area"]), 6.0)

    # check if volume is right for interior
    assert np.isclose(np.sum(interior["area"]), 1.0)
