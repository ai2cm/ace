import torch

from fme.core.typing_ import TensorMapping


def raise_if_any_variable_all_nan(data: TensorMapping, context: str = "") -> None:
    """Raise ``ValueError`` if any tensor in *data* is entirely NaN."""
    for name, tensor in data.items():
        if tensor.numel() > 0 and torch.isnan(tensor).all():
            prefix = f"{context}: " if context else ""
            raise ValueError(f"{prefix}Variable '{name}' is entirely NaN.")
