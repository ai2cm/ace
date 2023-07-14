from typing import List, Dict, Mapping, Optional
import torch
import netCDF4
import numpy as np
from fme.core.device import get_device
import torch.jit
import dataclasses


@dataclasses.dataclass
class NormalizationConfig:
    global_means_path: Optional[str] = None
    global_stds_path: Optional[str] = None
    exclude_names: Optional[List[str]] = None
    means: Mapping[str, float] = dataclasses.field(default_factory=dict)
    stds: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        using_path = (
            self.global_means_path is not None and self.global_stds_path is not None
        )
        using_explicit = len(self.means) > 0 and len(self.stds) > 0
        if using_path and using_explicit:
            raise ValueError(
                "Cannot use both global_means_path and global_stds_path "
                "and explicit means and stds."
            )
        if not (using_path or using_explicit):
            raise ValueError(
                "Must use either global_means_path and global_stds_path "
                "or explicit means and stds."
            )

    def build(self, names: List[str]):
        if self.exclude_names is not None:
            names = list(set(names) - set(self.exclude_names))
        using_path = (
            self.global_means_path is not None and self.global_stds_path is not None
        )
        if using_path:
            return get_normalizer(
                global_means_path=self.global_means_path,
                global_stds_path=self.global_stds_path,
                names=names,
            )
        else:
            means = {k: torch.tensor(self.means[k]) for k in names}
            stds = {k: torch.tensor(self.stds[k]) for k in names}
            return StandardNormalizer(means=means, stds=stds)


class StandardNormalizer:
    """
    Responsible for normalizing tensors.
    """

    def __init__(
        self,
        means: Dict[str, torch.Tensor],
        stds: Dict[str, torch.Tensor],
    ):
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
    return {
        k: (t - means[k]) / stds[k] if k in means.keys() else t
        for k, t in tensors.items()
    }


@torch.jit.script
def _denormalize(
    tensors: Dict[str, torch.Tensor],
    means: Dict[str, torch.Tensor],
    stds: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        k: t * stds[k] + means[k] if k in means.keys() else t
        for k, t in tensors.items()
    }


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
