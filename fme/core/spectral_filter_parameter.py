import copy
from typing import Any

import torch


class SpectralFilterParameter(torch.nn.Parameter):
    """A parameter holding one complex matrix per spectral mode.

    The data is either a complex tensor or a real tensor whose last dimension
    (of size 2) holds real and imaginary parts. In the complex view (the
    shape without that trailing dimension), axis ``out_dim`` indexes the
    matrices' output channels and axis ``in_dim`` their input channels; every
    other axis (e.g. groups or spherical harmonic modes) indexes independent
    matrices.

    The Muon optimizer routes these parameters to per-mode complex
    orthogonalization, which needs this layout: it cannot be inferred from
    the shape, since spectral filters order their channel and mode axes
    differently across SFNO implementations.

    The layout survives ``copy.deepcopy`` and pickling. Like any parameter,
    tensor operations on it return plain tensors.

    Args:
        data: The parameter's initial value.
        in_dim: Input-channel axis of the complex view.
        out_dim: Output-channel axis of the complex view.
        requires_grad: Whether the parameter requires a gradient.
    """

    in_dim: int
    out_dim: int

    def __new__(
        cls,
        data: torch.Tensor,
        *,
        in_dim: int,
        out_dim: int,
        requires_grad: bool = True,
    ):
        if data.is_complex():
            complex_ndim = data.ndim
        else:
            if data.ndim < 3 or data.shape[-1] != 2:
                raise ValueError(
                    "A real SpectralFilterParameter must have at least 3 "
                    "dimensions, the last of size 2 (real/imaginary), got shape "
                    f"{tuple(data.shape)}."
                )
            complex_ndim = data.ndim - 1
        if complex_ndim < 2:
            raise ValueError(
                "A complex SpectralFilterParameter must have at least 2 "
                f"dimensions, got shape {tuple(data.shape)}."
            )
        in_dim_normalized = _normalize_dim(in_dim, complex_ndim, "in_dim")
        out_dim_normalized = _normalize_dim(out_dim, complex_ndim, "out_dim")
        if in_dim_normalized == out_dim_normalized:
            raise ValueError(
                f"in_dim and out_dim must differ, got in_dim={in_dim}, "
                f"out_dim={out_dim}."
            )
        param = super().__new__(cls, data, requires_grad)
        param.in_dim = in_dim_normalized
        param.out_dim = out_dim_normalized
        return param

    def __deepcopy__(self, memo: dict[int, Any]) -> "SpectralFilterParameter":
        if id(self) in memo:
            return memo[id(self)]
        result = type(self)(
            self.data.clone(memory_format=torch.preserve_format),
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            requires_grad=self.requires_grad,
        )
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, copy.deepcopy(value, memo))
        return result

    def __reduce_ex__(self, protocol):
        return (
            _rebuild_spectral_filter_parameter,
            (self.data, self.requires_grad, self.__dict__),
        )

    def __repr__(self) -> str:
        return (
            f"SpectralFilterParameter(in_dim={self.in_dim}, "
            f"out_dim={self.out_dim}) containing:\n{self.data!r}"
        )


def _normalize_dim(dim: int, ndim: int, name: str) -> int:
    if not -ndim <= dim < ndim:
        raise ValueError(
            f"{name}={dim} is out of range for a complex view with {ndim} "
            "dimensions."
        )
    return dim % ndim


def _rebuild_spectral_filter_parameter(
    data: torch.Tensor, requires_grad: bool, state: dict[str, Any]
) -> SpectralFilterParameter:
    param = SpectralFilterParameter(
        data,
        in_dim=state["in_dim"],
        out_dim=state["out_dim"],
        requires_grad=requires_grad,
    )
    param.__dict__.update(state)
    return param
