import torch

from fme.core.typing_ import TensorDict, TensorMapping


def force_positive(data: TensorMapping, names: list[str]) -> TensorDict:
    """Clamp all tensors defined by `names` to be greater than or equal to zero."""
    out = {**data}
    for name in names:
        out[name] = torch.clamp(data[name], min=0.0)
    return out


def captured_before(original: TensorMapping, corrected: TensorMapping) -> TensorDict:
    """Return the pre-correction values of variables a corrector modified.

    A variable is considered modified if and only if its tensor in ``corrected``
    is a different object than in ``original``. This relies on the invariant that
    correctors apply their changes out-of-place (required for autograd): an
    unmodified variable retains its original tensor object, and a modified one is
    replaced by a freshly constructed tensor. In-place mutation of an input tensor
    would both corrupt the returned ``original`` values and evade detection here.
    """
    return {
        name: original[name]
        for name in corrected
        if name in original and corrected[name] is not original[name]
    }
