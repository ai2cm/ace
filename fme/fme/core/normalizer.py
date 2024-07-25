import dataclasses
from typing import Dict, Iterable, List, Mapping, Optional, Union

import netCDF4
import numpy as np
import torch
import torch.jit

from fme.core.device import get_device
from fme.core.typing_ import TensorDict


@dataclasses.dataclass
class NormalizationConfig:
    """
    Configuration for normalizing data.

    Either global_means_path and global_stds_path or explicit means and stds
    must be provided.

    Attributes:
        global_means_path: Path to a netCDF file containing global means.
        global_stds_path: Path to a netCDF file containing global stds.
        exclude_names: Names to exclude from normalization.
        means: Mapping from variable names to means.
        stds: Mapping from variable names to stds.
    """

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


@dataclasses.dataclass
class FromStateNormalizer:
    """
    An alternative to NormalizationConfig which provides a normalizer
    initialized from a serializable state. This is not a public configuration
    class, but instead allows for loading trained models that have been
    serialized to disk, using the pre-existing normalization state.

    Attributes:
        state: State dict of a normalizer.
    """

    state: Dict[str, Dict[str, float]]

    def build(self, names: List[str]):
        return StandardNormalizer.from_state(self.state)


class StandardNormalizer:
    """
    Responsible for normalizing tensors.
    """

    def __init__(
        self,
        means: TensorDict,
        stds: TensorDict,
    ):
        self.means = means
        self.stds = stds

    def normalize(self, tensors: TensorDict) -> TensorDict:
        return _normalize(tensors, means=self.means, stds=self.stds)

    def denormalize(self, tensors: TensorDict) -> TensorDict:
        return _denormalize(tensors, means=self.means, stds=self.stds)

    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        return {
            "means": {k: float(v.cpu().numpy().item()) for k, v in self.means.items()},
            "stds": {k: float(v.cpu().numpy().item()) for k, v in self.stds.items()},
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
    tensors: TensorDict,
    means: TensorDict,
    stds: TensorDict,
) -> TensorDict:
    return {
        k: (t - means[k]) / stds[k] if k in means.keys() else t
        for k, t in tensors.items()
    }


@torch.jit.script
def _denormalize(
    tensors: TensorDict,
    means: TensorDict,
    stds: TensorDict,
) -> TensorDict:
    return {
        k: t * stds[k] + means[k] if k in means.keys() else t
        for k, t in tensors.items()
    }


def get_normalizer(
    global_means_path, global_stds_path, names: List[str]
) -> StandardNormalizer:
    means = load_dict_from_netcdf(
        global_means_path, names, defaults={"x": 0.0, "y": 0.0, "z": 0.0}
    )
    means = {k: torch.as_tensor(v, dtype=torch.float) for k, v in means.items()}
    stds = load_dict_from_netcdf(
        global_stds_path, names, defaults={"x": 1.0, "y": 1.0, "z": 1.0}
    )
    stds = {k: torch.as_tensor(v, dtype=torch.float) for k, v in stds.items()}
    return StandardNormalizer(means=means, stds=stds)


def load_dict_from_netcdf(
    path: str, names: Iterable[str], defaults: Mapping[str, Union[float, np.ndarray]]
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Load a dictionary of variables from a netCDF file.

    Args:
        path: Path to the netCDF file.
        names: List of variable names to load.
        defaults: Dictionary of default values for each variable, if not found
            in the netCDF file.
    """
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    result = {}
    for c in names:
        if c in ds.variables:
            result[c] = ds.variables[c][:]
        elif c in defaults:
            result[c] = defaults[c]
        else:
            raise ValueError(f"Variable {c} not found in {path}")
    ds.close()
    return result
