import numpy as np
import torch
from modulus.utils.sympy import SympyToTorch
import sympy


def test_sympy_node():
    # Define SymPy symbol and expression
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    expr = sympy.Max(sympy.sin(x), sympy.cos(y))

    # Get numpy reference
    x_np = np.random.random(10)
    y_np = np.random.random(10)
    expr_np = np.maximum(np.sin(x_np), np.cos(y_np))

    sn = SympyToTorch(expr, "node")

    # Choose device to run on and copy data from numpy
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_th = torch.tensor(x_np, dtype=torch.float32, device=device)
    y_th = torch.tensor(y_np, dtype=torch.float32, device=device)
    assert np.allclose(x_th.cpu().detach().numpy(), x_np)
    assert np.allclose(y_th.cpu().detach().numpy(), y_np)

    # Run the compiled function on input tensors
    var = {"x": x_th, "y": y_th}
    expr_th = sn(var)
    expr_th_out = expr_th["node"].cpu().detach().numpy()

    assert np.allclose(expr_th_out, expr_np, rtol=1.0e-3), "SymPy printer test failed!"


if __name__ == "__main__":
    test_sympy_node()
