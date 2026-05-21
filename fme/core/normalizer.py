import dataclasses
import pathlib
from collections.abc import Iterable, Mapping
from copy import copy

import fsspec
import numpy as np
import torch
import xarray as xr

from fme.core.co2_temperature_offset import CO2TemperatureProfileConfig
from fme.core.device import move_tensordict_to_device
from fme.core.typing_ import TensorDict, TensorMapping

# Experimental: when NormalizationConfig.experimental_use_shared_temperature_offset
# is True, every field named here gets a per-sample additive offset applied
# before normalization (and removed after denormalization). The offset for a
# sample is the difference between the surface_temperature normalization mean
# and the sample's cell-wise mean surface_temperature. Hard-coded for now.
_TEMPERATURE_FIELD_NAMES = frozenset(
    [
        "surface_temperature",
        "TMP2m",
        "TMP850",
        "air_temperature_0",
        "air_temperature_1",
        "air_temperature_2",
        "air_temperature_3",
        "air_temperature_4",
        "air_temperature_5",
        "air_temperature_6",
        "air_temperature_7",
    ]
)
_SHARED_TEMPERATURE_OFFSET_REFERENCE = "surface_temperature"

# Experimental: name of the CO2 forcing input read by the CO2 temperature
# offset feature. Must match the data variable name.
_CO2_INPUT_NAME = "global_mean_co2"

# Idealized layer mid-pressures (Pa) for the dataset's 8-layer hybrid-sigma
# grid, evaluated at p_surface = 1e5 Pa. Derived from the ak/bk interface
# values in the era5-8layer zarr (layer 0 = upper stratosphere ~25 hPa,
# layer 7 = near surface ~952 hPa). Used only by the CO2 temperature offset.
_TEMPERATURE_FIELD_PRESSURES_PA: dict[str, float] = {
    "surface_temperature": 1.00e5,
    "TMP2m": 1.00e5,
    "TMP850": 8.50e4,
    "air_temperature_0": 2.5605e3,
    "air_temperature_1": 9.7696e3,
    "air_temperature_2": 1.98678e4,
    "air_temperature_3": 3.28763e4,
    "air_temperature_4": 4.99361e4,
    "air_temperature_5": 6.81903e4,
    "air_temperature_6": 8.36975e4,
    "air_temperature_7": 9.52251e4,
}


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
        experimental_use_shared_temperature_offset: If true, the built
            normalizer computes a per-sample offset from the input
            surface_temperature (surface_temperature normalization mean minus
            the sample's cell-wise mean) and adds it to every temperature
            field on normalize / subtracts it on denormalize. Intended for the
            network (full-field) normalizer only -- the loss normalizer skips
            this even if the flag is set, since offsets are unnecessary there.
            surface_temperature must be in the means. Temperature field names
            are hard-coded; experimental knob, not a stable feature.
        experimental_co2_temperature_offset: If set, the built normalizer
            subtracts a physically-motivated ΔT(p, CO2) profile from every
            temperature field on normalize and adds it back on denormalize.
            The CO2 input is read from ``global_mean_co2`` in the input
            tensors (which must be provided even if not in ``in_names``).
            Network-normalizer only; loss normalizer always strips it.
    """

    global_means_path: str | pathlib.Path | None = None
    global_stds_path: str | pathlib.Path | None = None
    means: Mapping[str, float] = dataclasses.field(default_factory=dict)
    stds: Mapping[str, float] = dataclasses.field(default_factory=dict)
    fill_nans_on_normalize: bool = False
    fill_nans_on_denormalize: bool = False
    experimental_use_shared_temperature_offset: bool = False
    experimental_co2_temperature_offset: CO2TemperatureProfileConfig | None = None

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
                use_shared_temperature_offset=(
                    self.experimental_use_shared_temperature_offset
                ),
                co2_temperature_offset=self.experimental_co2_temperature_offset,
            )
        else:
            means = {k: torch.tensor(self.means[k]) for k in names}
            stds = {k: torch.tensor(self.stds[k]) for k in names}
            return StandardNormalizer(
                means=means,
                stds=stds,
                fill_nans_on_normalize=self.fill_nans_on_normalize,
                fill_nans_on_denormalize=self.fill_nans_on_denormalize,
                use_shared_temperature_offset=(
                    self.experimental_use_shared_temperature_offset
                ),
                co2_temperature_offset=self.experimental_co2_temperature_offset,
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
        use_shared_temperature_offset: bool = False,
        co2_temperature_offset: CO2TemperatureProfileConfig | None = None,
    ):
        self.means = move_tensordict_to_device(means)
        self.stds = move_tensordict_to_device(stds)
        self._names = set(means).intersection(stds)
        self._fill_nans_on_normalize = fill_nans_on_normalize
        self._fill_nans_on_denormalize = fill_nans_on_denormalize
        self._use_shared_temperature_offset = use_shared_temperature_offset
        self._co2_temperature_offset = co2_temperature_offset
        # Per-sample offsets cached by the last normalize() call so the
        # matching denormalize() can invert them. None until first call.
        self._cached_temperature_offset: torch.Tensor | None = None
        self._cached_co2_offsets: dict[str, torch.Tensor] | None = None
        if use_shared_temperature_offset and (
            _SHARED_TEMPERATURE_OFFSET_REFERENCE not in self.means
        ):
            raise ValueError(
                "use_shared_temperature_offset is set but "
                f"{_SHARED_TEMPERATURE_OFFSET_REFERENCE!r} is not present in "
                "the normalization means."
            )

    @property
    def fill_nans_on_normalize(self):
        return self._fill_nans_on_normalize

    @property
    def fill_nans_on_denormalize(self):
        return self._fill_nans_on_denormalize

    @property
    def use_shared_temperature_offset(self) -> bool:
        return self._use_shared_temperature_offset

    @property
    def co2_temperature_offset(self) -> CO2TemperatureProfileConfig | None:
        return self._co2_temperature_offset

    def normalize(self, tensors: TensorMapping) -> TensorDict:
        filtered_tensors = {k: v for k, v in tensors.items() if k in self._names}
        # Order matters: subtract the secular CO2 forcing first so that the
        # shared-temperature offset (which depends on surface_temperature's
        # spatial mean) sees CO2-detrended surface_temperature.
        if self._co2_temperature_offset is not None:
            co2_offsets = _compute_co2_temperature_offsets(
                tensors, self._co2_temperature_offset
            )
            self._cached_co2_offsets = co2_offsets
            filtered_tensors = _shift_temperature_fields_per_field(
                filtered_tensors, co2_offsets, add=False
            )
        if self._use_shared_temperature_offset:
            offset = self._compute_temperature_offset(filtered_tensors)
            self._cached_temperature_offset = offset
            filtered_tensors = _shift_temperature_fields(
                filtered_tensors, offset, add=True
            )
        return _normalize(
            filtered_tensors,
            means=self.means,
            stds=self.stds,
            fill_nans=self._fill_nans_on_normalize,
        )

    def denormalize(self, tensors: TensorMapping) -> TensorDict:
        filtered_tensors = {k: v for k, v in tensors.items() if k in self._names}
        denormalized = _denormalize(
            filtered_tensors,
            means=self.means,
            stds=self.stds,
            fill_nans=self._fill_nans_on_denormalize,
        )
        if self._use_shared_temperature_offset:
            if self._cached_temperature_offset is None:
                raise RuntimeError(
                    "denormalize() was called before normalize() on a "
                    "StandardNormalizer with use_shared_temperature_offset "
                    "enabled; the offset is computed during normalize()."
                )
            denormalized = _shift_temperature_fields(
                denormalized, self._cached_temperature_offset, add=False
            )
        if self._co2_temperature_offset is not None:
            if self._cached_co2_offsets is None:
                raise RuntimeError(
                    "denormalize() was called before normalize() on a "
                    "StandardNormalizer with co2_temperature_offset enabled; "
                    "the per-field offsets are computed during normalize()."
                )
            denormalized = _shift_temperature_fields_per_field(
                denormalized, self._cached_co2_offsets, add=True
            )
        return denormalized

    def _compute_temperature_offset(self, tensors: TensorMapping) -> torch.Tensor:
        ref_name = _SHARED_TEMPERATURE_OFFSET_REFERENCE
        if ref_name not in tensors:
            raise ValueError(
                "use_shared_temperature_offset is set but "
                f"{ref_name!r} is not in the input tensors to normalize()."
            )
        ref = tensors[ref_name]
        if ref.ndim < 2:
            raise ValueError(
                f"{ref_name!r} must have a sample dim and at least one "
                f"non-sample dim to compute a per-sample offset; got shape "
                f"{tuple(ref.shape)}."
            )
        sample_mean = ref.mean(dim=tuple(range(1, ref.ndim)))
        return self.means[ref_name] - sample_mean

    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        return {
            "means": {k: float(v.cpu().numpy().item()) for k, v in self.means.items()},
            "stds": {k: float(v.cpu().numpy().item()) for k, v in self.stds.items()},
            "fill_nans_on_normalize": self._fill_nans_on_normalize,
            "fill_nans_on_denormalize": self._fill_nans_on_denormalize,
            "use_shared_temperature_offset": self._use_shared_temperature_offset,
            "co2_temperature_offset": (
                dataclasses.asdict(self._co2_temperature_offset)
                if self._co2_temperature_offset is not None
                else None
            ),
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
        co2_state = state.get("co2_temperature_offset")
        co2_offset = (
            CO2TemperatureProfileConfig(**co2_state) if co2_state is not None else None
        )
        return cls(
            means=means,
            stds=stds,
            fill_nans_on_normalize=state.get("fill_nans_on_normalize", False),
            fill_nans_on_denormalize=state.get("fill_nans_on_denormalize", False),
            use_shared_temperature_offset=state.get(
                "use_shared_temperature_offset", False
            ),
            co2_temperature_offset=co2_offset,
        )

    def get_normalization_config(self) -> NormalizationConfig:
        return NormalizationConfig(
            means={k: float(v.cpu().numpy().item()) for k, v in self.means.items()},
            stds={k: float(v.cpu().numpy().item()) for k, v in self.stds.items()},
            fill_nans_on_normalize=self.fill_nans_on_normalize,
            fill_nans_on_denormalize=self.fill_nans_on_denormalize,
            experimental_use_shared_temperature_offset=(
                self._use_shared_temperature_offset
            ),
            experimental_co2_temperature_offset=self._co2_temperature_offset,
        )


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


def _shift_temperature_fields(
    tensors: TensorDict,
    offset: torch.Tensor,
    add: bool,
) -> TensorDict:
    """Add (or subtract) a per-sample offset to every temperature field.

    ``offset`` has shape ``[batch]``; it is broadcast over the non-sample
    dimensions of each temperature tensor.
    """
    result = dict(tensors)
    for name in _TEMPERATURE_FIELD_NAMES:
        if name not in result:
            continue
        t = result[name]
        broadcast = offset.view(offset.shape[0], *(1,) * (t.ndim - 1))
        result[name] = t + broadcast if add else t - broadcast
    return result


def _shift_temperature_fields_per_field(
    tensors: TensorDict,
    per_field_offsets: dict[str, torch.Tensor],
    add: bool,
) -> TensorDict:
    """Add (or subtract) a per-field, per-sample offset to temperature fields.

    Each entry of ``per_field_offsets`` is a tensor of shape ``[batch]`` that
    is broadcast over the non-sample dimensions of its corresponding field.
    """
    result = dict(tensors)
    for name, offset in per_field_offsets.items():
        if name not in result:
            continue
        t = result[name]
        broadcast = offset.view(offset.shape[0], *(1,) * (t.ndim - 1))
        result[name] = t + broadcast if add else t - broadcast
    return result


def _compute_co2_temperature_offsets(
    tensors: TensorMapping,
    profile: CO2TemperatureProfileConfig,
) -> dict[str, torch.Tensor]:
    """Compute per-sample ΔT for each known temperature field given CO2 input."""
    if _CO2_INPUT_NAME not in tensors:
        raise ValueError(
            "co2_temperature_offset is set but "
            f"{_CO2_INPUT_NAME!r} is not in the input tensors to normalize()."
        )
    co2 = tensors[_CO2_INPUT_NAME]
    if co2.ndim < 2:
        # Already per-sample scalar; broadcasting will handle the rest.
        sample_co2 = co2
    else:
        sample_co2 = co2.mean(dim=tuple(range(1, co2.ndim)))
    return {
        name: profile.delta_t(sample_co2, pressure_pa=p)
        for name, p in _TEMPERATURE_FIELD_PRESSURES_PA.items()
    }


def _disable_experimental_offsets_for_loss(
    normalizer: StandardNormalizer,
) -> StandardNormalizer:
    """Return a copy of ``normalizer`` with the experimental
    network-only offsets (shared-temperature and CO2) turned off, since
    they are only meaningful for the paired-normalize/denormalize cycle
    of the network normalizer.
    """
    if (
        not normalizer.use_shared_temperature_offset
        and normalizer.co2_temperature_offset is None
    ):
        return normalizer
    return StandardNormalizer(
        means=normalizer.means,
        stds=normalizer.stds,
        fill_nans_on_normalize=normalizer.fill_nans_on_normalize,
        fill_nans_on_denormalize=normalizer.fill_nans_on_denormalize,
        use_shared_temperature_offset=False,
        co2_temperature_offset=None,
    )


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
            built = self.loss.build(names=names)
        elif self.residual is not None:
            built = _combine_normalizers(
                base_normalizer=self.network.build(names=names),
                override_normalizer=self.residual.build(names=residual_scaled_names),
            )
        else:
            built = self.network.build(names=names)
        # Experimental network-only offsets (shared-temperature, CO2) don't
        # belong on the loss normalizer: there's no denormalize on the loss
        # path, and the offsets would either cancel anyway (CO2) or fail to
        # cancel cleanly across target vs prediction (shared-temperature).
        return _disable_experimental_offsets_for_loss(built)

    @property
    def required_forcing_names(self) -> list[str]:
        """Variables that must be present in the input batch for normalize()
        to function, but are not necessarily inputs to the network itself.
        Used by the stepper to extend ``input_names`` for the data loader.
        """
        names: list[str] = []
        if self.network.experimental_co2_temperature_offset is not None:
            names.append(_CO2_INPUT_NAME)
        return names

    def load(self):
        self.network.load()
        if self.loss is not None:
            self.loss.load()
        if self.residual is not None:
            self.residual.load()
