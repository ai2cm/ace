from typing import List, Dict
import torch
import netCDF4
import numpy as np
from fme.core.device import get_device
import torch.jit


class StandardNormalizer:
    """
    Responsible for normalizing tensors.
    """

    def __init__(self, means: Dict[str, torch.Tensor], stds: Dict[str, torch.Tensor]):
        self.means = means
        self.stds = stds

    def normalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return _normalize(tensors, means=self.means, stds=self.stds)

    def denormalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return _denormalize(tensors, means=self.means, stds=self.stds)

    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        return {
            "means": {k: float(v.cpu().numpy()) for k, v in self.means.items()},
            "stds": {k: float(v.cpu().numpy()) for k, v in self.stds.items()},
        }

    @classmethod
    def from_state(self, state) -> "StandardNormalizer":
        """
        Loads state from a serializable data structure.
        """
        means = {
            k: torch.tensor(v, device=get_device(), dtype=torch.float)
            for k, v in state["means"].items()
        }
        stds = {
            k: torch.tensor(v, device=get_device(), dtype=torch.float)
            for k, v in state["stds"].items()
        }
        return StandardNormalizer(means=means, stds=stds)


@torch.jit.script
def _normalize(
    tensors: Dict[str, torch.Tensor],
    means: Dict[str, torch.Tensor],
    stds: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: (t - means[k]) / stds[k] for k, t in tensors.items()}


@torch.jit.script
def _denormalize(
    tensors: Dict[str, torch.Tensor],
    means: Dict[str, torch.Tensor],
    stds: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: t * stds[k] + means[k] for k, t in tensors.items()}


def get_normalizer(
    global_means_path, global_stds_path, names: List[str]
) -> StandardNormalizer:
    means = load_Dict_from_netcdf(global_means_path, names)
    means = {k: torch.as_tensor(v, dtype=torch.float) for k, v in means.items()}
    stds = load_Dict_from_netcdf(global_stds_path, names)
    stds = {k: torch.as_tensor(v, dtype=torch.float) for k, v in stds.items()}
    return StandardNormalizer(means=means, stds=stds)


def load_Dict_from_netcdf(path, names) -> Dict[str, np.ndarray]:
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    Dict = {c: ds.variables[c][:] for c in names}
    ds.close()
    return Dict
