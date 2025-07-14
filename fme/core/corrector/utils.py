import torch

from fme.core.typing_ import TensorDict, TensorMapping


def force_positive(data: TensorMapping, names: list[str]) -> TensorDict:
    """Clamp all tensors defined by `names` to be greater than or equal to zero."""
    out = {**data}
    for name in names:
        out[name] = torch.clamp(data[name], min=0.0)
    return out
