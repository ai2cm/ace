import dataclasses
import pathlib
from collections.abc import Iterable, Mapping
from copy import copy

import fsspec
import numpy as np
import torch
import torch.jit
import xarray as xr

from fme.core.device import move_tensordict_to_device
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class NormalizationConfig:
    """
    Configuration for normalizing data.

    Either global_means_path and global_stds_path or explicit means and stds
    must be provided.

    Parameters:
        global_means_path: Path to a netCDF file containing global means.
        global_stds_path: Path to a netCDF file containing global stds.
        means: Mapping from variable names to means.
        stds: Mapping from variable names to stds.
        fill_nans_on_normalize: Whether to fill NaNs during normalization. If
            true, on normalization NaNs in the denormalized input become zeros in
            the normalized output.
        fill_nans_on_denormalize: Whether to fill NaNs during denormalization. If
            true, on denormalization NaNs in the normalized input become global means in
            the denormalized output.
    """

    global_means_path: str | pathlib.Path | None = None
    global_stds_path: str | pathlib.Path | None = None
    means: Mapping[str, float] = dataclasses.field(default_factory=dict)
    stds: Mapping[str, float] = dataclasses.field(default_factory=dict)
    fill_nans_on_normalize: bool = False
    fill_nans_on_denormalize: bool = False

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

    def load(self):
        """
        Load the normalization configuration from the netCDF files.

        Updates the configuration so it no longer requires external files.
        """
        if self.global_means_path is not None and self.global_stds_path is not None:
            # convert to explicit means and stds so if the object is stored
            # and reloaded, we no longer need the netCDF files
            means = load_dict_from_netcdf(
                self.global_means_path,
                names=None,
                defaults={"x": 0.0, "y": 0.0, "z": 0.0},
            )
            stds = load_dict_from_netcdf(
                self.global_stds_path,
                names=None,
                defaults={"x": 1.0, "y": 1.0, "z": 1.0},
            )
            self.means = means
            self.stds = stds
            self.global_means_path = None
            self.global_stds_path = None

    def build(self, names: list[str]):
        using_path = (
            self.global_means_path is not None and self.global_stds_path is not None
        )
        if using_path:
            return get_normalizer(
                global_means_path=self.global_means_path,
                global_stds_path=self.global_stds_path,
                names=names,
                fill_nans_on_normalize=self.fill_nans_on_normalize,
                fill_nans_on_denormalize=self.fill_nans_on_denormalize,
            )
        else:
            means = {k: torch.tensor(self.means[k]) for k in names}
            stds = {k: torch.tensor(self.stds[k]) for k in names}
            return StandardNormalizer(
                means=means,
                stds=stds,
                fill_nans_on_normalize=self.fill_nans_on_normalize,
                fill_nans_on_denormalize=self.fill_nans_on_denormalize,
            )


class StandardNormalizer:
    """
    Responsible for normalizing tensors.
    """

    def __init__(
        self,
        means: TensorDict,
        stds: TensorDict,
        fill_nans_on_normalize: bool = False,
        fill_nans_on_denormalize: bool = False,
    ):
        self.means = move_tensordict_to_device(means)
        self.stds = move_tensordict_to_device(stds)
        self._names = set(means).intersection(stds)
        self._fill_nans_on_normalize = fill_nans_on_normalize
        self._fill_nans_on_denormalize = fill_nans_on_denormalize

    @property
    def fill_nans_on_normalize(self):
        return self._fill_nans_on_normalize

    @property
    def fill_nans_on_denormalize(self):
        return self._fill_nans_on_denormalize

    def normalize(self, tensors: TensorMapping) -> TensorDict:
        filtered_tensors = {k: v for k, v in tensors.items() if k in self._names}
        return _normalize(
            filtered_tensors,
            means=self.means,
            stds=self.stds,
            fill_nans=self._fill_nans_on_normalize,
        )

    def denormalize(self, tensors: TensorMapping) -> TensorDict:
        filtered_tensors = {k: v for k, v in tensors.items() if k in self._names}
        return _denormalize(
            filtered_tensors,
            means=self.means,
            stds=self.stds,
            fill_nans=self._fill_nans_on_denormalize,
        )

    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        return {
            "means": {k: float(v.cpu().numpy().item()) for k, v in self.means.items()},
            "stds": {k: float(v.cpu().numpy().item()) for k, v in self.stds.items()},
            "fill_nans_on_normalize": self._fill_nans_on_normalize,
            "fill_nans_on_denormalize": self._fill_nans_on_denormalize,
        }

    @classmethod
    def from_state(cls, state) -> "StandardNormalizer":
        """
        Loads state from a serializable data structure.
        """
        means = {
            k: torch.tensor(v, dtype=torch.float) for k, v in state["means"].items()
        }
        stds = {k: torch.tensor(v, dtype=torch.float) for k, v in state["stds"].items()}
        return cls(
            means=means,
            stds=stds,
            fill_nans_on_normalize=state.get("fill_nans_on_normalize", False),
            fill_nans_on_denormalize=state.get("fill_nans_on_denormalize", False),
        )

    def get_normalization_config(self) -> NormalizationConfig:
        return NormalizationConfig(
            means={k: float(v.cpu().numpy().item()) for k, v in self.means.items()},
            stds={k: float(v.cpu().numpy().item()) for k, v in self.stds.items()},
            fill_nans_on_normalize=self.fill_nans_on_normalize,
            fill_nans_on_denormalize=self.fill_nans_on_denormalize,
        )


@torch.jit.script
def _normalize(
    tensors: TensorDict,
    means: TensorDict,
    stds: TensorDict,
    fill_nans: bool,
) -> TensorDict:
    normalized = {k: (t - means[k]) / stds[k] for k, t in tensors.items()}
    if fill_nans:
        for k, v in normalized.items():
            normalized[k] = torch.where(torch.isnan(v), torch.zeros_like(v), v)
    return normalized


@torch.jit.script
def _denormalize(
    tensors: TensorDict,
    means: TensorDict,
    stds: TensorDict,
    fill_nans: bool,
) -> TensorDict:
    denormalized = {k: t * stds[k] + means[k] for k, t in tensors.items()}
    if fill_nans:
        for k, v in denormalized.items():
            denormalized[k] = torch.where(
                torch.isnan(v), torch.full_like(v, fill_value=means[k]), v
            )
    return denormalized


def get_normalizer(
    global_means_path, global_stds_path, names: list[str], **normalizer_kwargs
) -> StandardNormalizer:
    means = load_dict_from_netcdf(
        global_means_path, names, defaults={"x": 0.0, "y": 0.0, "z": 0.0}
    )
    means = {k: torch.as_tensor(v, dtype=torch.float) for k, v in means.items()}
    stds = load_dict_from_netcdf(
        global_stds_path, names, defaults={"x": 1.0, "y": 1.0, "z": 1.0}
    )
    stds = {k: torch.as_tensor(v, dtype=torch.float) for k, v in stds.items()}
    return StandardNormalizer(means=means, stds=stds, **normalizer_kwargs)


def load_dict_from_netcdf(
    path: str | pathlib.Path,
    names: Iterable[str] | None,
    defaults: Mapping[str, float | np.ndarray],
) -> dict[str, float]:
    """
    Load a dictionary of scalar variables from a netCDF file.

    Args:
        path: Path to the netCDF file.
        names: List of variable names to load. If None, all variables in the netCDF
            file are loaded.
        defaults: Dictionary of default values for each variable, if not found
            in the netCDF file.
    """
    with fsspec.open(path, "rb") as f:
        ds = xr.load_dataset(f, mask_and_scale=False)

    result = {}
    if names is None:
        names = set(ds.variables.keys()).union(defaults.keys())
        skip_non_scalar = True
    else:
        skip_non_scalar = False
    for c in names:
        if c in ds.variables:
            if skip_non_scalar and ds.variables[c].ndim > 0:
                continue
            result[c] = float(ds.variables[c].values.item())
        elif c in defaults:
            result[c] = float(defaults[c])
        else:
            raise ValueError(f"Variable {c} not found in {path}")
    ds.close()
    return result


def _combine_normalizers(
    base_normalizer: StandardNormalizer,
    override_normalizer: StandardNormalizer,
) -> StandardNormalizer:
    """
    Combine two normalizers by overwriting the base normalizer values that are
    present in the override normalizer.

    NaN-filling behavior is inherited from the base normalizer.
    """
    means, stds = copy(base_normalizer.means), copy(base_normalizer.stds)
    means.update(override_normalizer.means)
    stds.update(override_normalizer.stds)
    return StandardNormalizer(
        means=means,
        stds=stds,
        fill_nans_on_normalize=base_normalizer.fill_nans_on_normalize,
        fill_nans_on_denormalize=base_normalizer.fill_nans_on_denormalize,
    )


@dataclasses.dataclass
class NetworkAndLossNormalizationConfig:
    """
    Combined configuration for network and loss normalization.

    Allows loss normalization to be defined as equal to the network
    normalization, apart from a set of residual-scaled variables.

    Parameters:
        network: The normalization configuration for the network.
        loss: The normalization configuration for the loss. Default is to
            use the network configuration, except for residual-scaled variables
            which instead use the residual configuration if given.
        residual: The normalization configuration for residuals. Cannot be
            provided if loss normalization is also provided.
    """

    network: NormalizationConfig
    loss: NormalizationConfig | None = None
    residual: NormalizationConfig | None = None

    def __post_init__(self):
        if self.loss is not None and self.residual is not None:
            raise ValueError("Cannot provide both loss and residual normalization.")

    def get_network_normalizer(self, names: list[str]) -> StandardNormalizer:
        return self.network.build(names=names)

    def get_loss_normalizer(
        self,
        names: list[str],
        residual_scaled_names: list[str],
    ) -> StandardNormalizer:
        if self.loss is not None:
            return self.loss.build(names=names)
        elif self.residual is not None:
            return _combine_normalizers(
                base_normalizer=self.network.build(names=names),
                override_normalizer=self.residual.build(names=residual_scaled_names),
            )
        else:
            return self.network.build(names=names)

    def load(self):
        self.network.load()
        if self.loss is not None:
            self.loss.load()
        if self.residual is not None:
            self.residual.load()
