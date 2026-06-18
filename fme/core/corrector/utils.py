import torch

from fme.core.typing_ import TensorDict, TensorMapping


def replace_value_keep_gradient(
    x: torch.Tensor, new_value: torch.Tensor
) -> torch.Tensor:
    """Use `new_value` in the forward pass but keep x's gradient in the backward pass.

    Lets us apply a hard correction (clamp, rebalance, etc.) to the output a
    network sees, while the network still gets a gradient as if the correction
    hadn't happened — so clamped/out-of-range cells still get a learning signal
    instead of a zero gradient.

    Forward: ``x + (new_value - x) = new_value`` (the exact projected value is
    preserved). Backward: the detached term contributes zero, so the gradient is
    the identity ``d(out)/dx = 1``.
    """
    return x + (new_value - x).detach()


def force_positive(
    data: TensorMapping, names: list[str], keep_gradient: bool = False
) -> TensorDict:
    """Clamp all tensors defined by `names` to be greater than or equal to zero.

    If ``keep_gradient`` is True, the clamp is applied with
    :func:`replace_value_keep_gradient` so the forward value is still clamped to
    zero but the gradient flows as if the clamp had not happened.
    """
    out = {**data}
    for name in names:
        clamped = torch.clamp(data[name], min=0.0)
        if keep_gradient:
            clamped = replace_value_keep_gradient(data[name], clamped)
        out[name] = clamped
    return out
