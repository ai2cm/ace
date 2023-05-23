import torch
from typing import Dict, List
import torch.jit


class Packer:
    """
    Responsible for packing tensors into a single tensor.
    """

    def __init__(self, names: List[str]):
        self.names = names

    def pack(self, tensors: Dict[str, torch.Tensor], axis=0) -> torch.Tensor:
        """
        Packs tensors into a single tensor, concatenated along a new axis

        Args:
            tensors: Dict from names to tensors.
            axis: index for new concatenation axis.
        """
        return _pack(tensors, self.names, axis=axis)

    def unpack(self, tensor: torch.Tensor, axis=0) -> Dict[str, torch.Tensor]:
        return _unpack(tensor, self.names, axis=axis)

    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        return {"names": self.names}

    @classmethod
    def from_state(self, state) -> "Packer":
        """
        Loads state from a serializable data structure.
        """
        return Packer(state["names"])


@torch.jit.script
def _pack(
    tensors: Dict[str, torch.Tensor], names: List[str], axis: int = 0
) -> torch.Tensor:
    return torch.cat([tensors[n].unsqueeze(axis) for n in names], dim=axis)


@torch.jit.script
def _unpack(
    tensor: torch.Tensor, names: List[str], axis: int = 0
) -> Dict[str, torch.Tensor]:
    return {n: tensor.select(axis, index=i) for i, n in enumerate(names)}
