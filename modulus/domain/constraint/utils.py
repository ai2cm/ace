import numpy as np

from modulus.utils.sympy import np_lambdify


def _compute_outvar(invar, outvar_sympy):
    outvar = {}
    for key in outvar_sympy.keys():
        outvar[key] = np_lambdify(outvar_sympy[key], {**invar})(**invar)
    return outvar


def _compute_lambda_weighting(invar, outvar, lambda_weighting_sympy):
    lambda_weighting = {}
    if lambda_weighting_sympy is None:
        for key in outvar.keys():
            lambda_weighting[key] = np.ones_like(next(iter(invar.values())))
    else:
        for key in outvar.keys():
            lambda_weighting[key] = np_lambdify(
                lambda_weighting_sympy[key], {**invar, **outvar}
            )(**invar, **outvar)
    return lambda_weighting
