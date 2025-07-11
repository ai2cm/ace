import torch
import torch.jit

from fme.core.typing_ import TensorDict


class DataShapesNotUniform(ValueError):
    """Indicates that a set of tensors do not all have the same shape."""

    pass


class Packer:
    """
    Responsible for packing tensors into a single tensor.
    """

    def __init__(self, names: list[str]):
        self.names = names

    def pack(self, tensors: TensorDict, axis=0) -> torch.Tensor:
        """
        Packs tensors into a single tensor, concatenated along a new axis.

        Args:
            tensors: Dict from names to tensors.
            axis: index for new concatenation axis.

        Raises:
            DataShapesNotUniform: when packed tensors do not all have the same shape.
        """
        shape = next(iter(tensors.values())).shape
        for name in tensors:
            if tensors[name].shape != shape:
                raise DataShapesNotUniform(
                    f"Cannot pack tensors of different shapes. "
                    f'Expected "{shape}" got "{tensors[name].shape}"'
                )
        return _pack(tensors, self.names, axis=axis)

    def unpack(self, tensor: torch.Tensor, axis=0) -> TensorDict:
        return _unpack(tensor, self.names, axis=axis)


@torch.jit.script
def _pack(tensors: TensorDict, names: list[str], axis: int = 0) -> torch.Tensor:
    return torch.cat([tensors[n].unsqueeze(axis) for n in names], dim=axis)


@torch.jit.script
def _unpack(tensor: torch.Tensor, names: list[str], axis: int = 0) -> TensorDict:
    return {n: tensor.select(axis, index=i) for i, n in enumerate(names)}
