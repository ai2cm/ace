import dataclasses
import logging
import pathlib
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from copy import copy
from typing import Protocol, TypeAlias

import numpy as np
import torch
import xarray as xr

from fme.core.cloud import open_dataset_via_inter_filesystem_copy
from fme.core.device import get_device, move_tensordict_to_device
from fme.core.distributed import Distributed
from fme.core.labels import BatchLabels
from fme.core.typing_ import TensorDict, TensorMapping

# A float, or a tensor of per-grid-cell values.
NormalizationConstant: TypeAlias = float | torch.Tensor

# Mapping from dimension name to coordinate values, in dimension order.
GridCoords: TypeAlias = dict[str, list[float]]


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
        per_grid_cell: Whether variables in the netCDF files may have the
            data's horizontal dimensions (in the same order), giving a mean
            and std per grid cell. Each dimension needs numeric coordinates
            matching the data's (on index grids such as HEALPix only sizes
            are effectively checked). Means must be finite and stds positive
            in every cell, so missing or fill values must be filled in the
            file. Packed variables are unpacked. If false, non-scalar
            variables are skipped, or rejected if requested by name. Only
            supported by the single-module step.
        per_grid_cell_min_std_fraction: Floor for per-grid-cell stds read
            from file, as a fraction of each variable's mean std over grid
            cells, to avoid blowing up near-constant cells. Must be in (0, 1).
        per_grid_cell_coords: Coordinates of each per-grid-cell variable's
            dimensions, recorded on load from file so they can be checked
            against the data's. Explicit tensor ``means`` and ``stds`` (e.g.
            restored from a checkpoint) without them are only shape-checked.
    """

    global_means_path: str | pathlib.Path | None = None
    global_stds_path: str | pathlib.Path | None = None
    means: Mapping[str, NormalizationConstant] = dataclasses.field(default_factory=dict)
    stds: Mapping[str, NormalizationConstant] = dataclasses.field(default_factory=dict)
    fill_nans_on_normalize: bool = False
    fill_nans_on_denormalize: bool = False
    per_grid_cell: bool = False
    per_grid_cell_min_std_fraction: float = 0.01
    per_grid_cell_coords: dict[str, GridCoords] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self):
        if not 0.0 < self.per_grid_cell_min_std_fraction < 1.0:
            raise ValueError(
                "per_grid_cell_min_std_fraction must be in (0, 1), got "
                f"{self.per_grid_cell_min_std_fraction}"
            )
        if not self.per_grid_cell:
            non_scalar = sorted(
                {
                    name
                    for constants in (self.means, self.stds)
                    for name, value in constants.items()
                    if _as_float_tensor(value).numel() > 1
                }
            )
            if non_scalar:
                raise ValueError(
                    f"Normalization constants for {non_scalar} are not scalars; "
                    "set per_grid_cell to use per-grid-cell constants."
                )
            if self.per_grid_cell_coords:
                raise ValueError(
                    "per_grid_cell_coords is set, but per_grid_cell is not; "
                    "set per_grid_cell to use per-grid-cell constants."
                )
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
        if using_path and self.per_grid_cell_coords:
            raise ValueError(
                "per_grid_cell_coords cannot be given with global_means_path "
                "and global_stds_path; the coordinates are read from the files."
            )
        self._validate_per_grid_cell_coords()

    def _validate_per_grid_cell_coords(self) -> None:
        """Check per_grid_cell_coords against the constants' names and shapes."""
        for name, coords in sorted(self.per_grid_cell_coords.items()):
            if name not in self.means and name not in self.stds:
                raise ValueError(
                    f"per_grid_cell_coords has coordinates for '{name}', which "
                    "has neither means nor stds."
                )
            coords_shape = tuple(len(values) for values in coords.values())
            for description, constants in (("means", self.means), ("stds", self.stds)):
                if name not in constants:
                    continue
                shape = tuple(_as_float_tensor(constants[name]).shape)
                if shape != () and shape != coords_shape:
                    raise ValueError(
                        f"Per-grid-cell normalization {description} for "
                        f"'{name}' have shape {shape}, which does not match "
                        f"the lengths {coords_shape} of its coordinates "
                        f"{list(coords)} in per_grid_cell_coords."
                    )

    @property
    def fills_nans(self) -> bool:
        """Whether NaN filling is enabled on normalize or denormalize."""
        return self.fill_nans_on_normalize or self.fill_nans_on_denormalize

    def load(self, names: Iterable[str] | None = None):
        """
        Load the normalization configuration from the netCDF files.

        Updates the configuration so it no longer requires external files.

        Args:
            names: Names of the per-grid-cell variables to load, or None for
                all with numeric coordinates. Scalars are always all loaded.
        """
        if self.global_means_path is not None and self.global_stds_path is not None:
            # convert to explicit means and stds so if the object is stored
            # and reloaded, we no longer need the netCDF files
            self.means, self.stds, self.per_grid_cell_coords = _load_means_and_stds(
                self.global_means_path,
                self.global_stds_path,
                names=None,
                per_grid_cell=self.per_grid_cell,
                min_std_fraction=self.per_grid_cell_min_std_fraction,
                per_grid_cell_names=None if names is None else set(names),
            )
            self.global_means_path = None
            self.global_stds_path = None

    def raise_if_per_grid_cell(self, context: str) -> None:
        """Raise if per_grid_cell is set; ``context`` names the caller."""
        if self.per_grid_cell:
            raise ValueError(
                f"{context} does not support per-grid-cell normalization; "
                "unset per_grid_cell in its normalization config."
            )

    def build(self, names: list[str]) -> "StandardNormalizer":
        using_path = (
            self.global_means_path is not None and self.global_stds_path is not None
        )
        if using_path:
            normalizer = get_normalizer(
                global_means_path=self.global_means_path,
                global_stds_path=self.global_stds_path,
                names=names,
                per_grid_cell=self.per_grid_cell,
                per_grid_cell_min_std_fraction=self.per_grid_cell_min_std_fraction,
                fill_nans_on_normalize=self.fill_nans_on_normalize,
                fill_nans_on_denormalize=self.fill_nans_on_denormalize,
            )
        else:
            self._raise_on_missing_constants(names)
            means = {k: _as_float_tensor(self.means[k]) for k in names}
            stds = {k: _as_float_tensor(self.stds[k]) for k in names}
            normalizer = StandardNormalizer(
                means=means,
                stds=stds,
                fill_nans_on_normalize=self.fill_nans_on_normalize,
                fill_nans_on_denormalize=self.fill_nans_on_denormalize,
                grid_coords={
                    k: v for k, v in self.per_grid_cell_coords.items() if k in names
                },
            )
        _validate_per_grid_cell_values(normalizer.means, normalizer.stds)
        return normalizer

    def _raise_on_missing_constants(self, names: Iterable[str]) -> None:
        missing = sorted(
            name for name in names if name not in self.means or name not in self.stds
        )
        if not missing:
            return
        message = f"No normalization constants for variables {missing}."
        if self.per_grid_cell:
            message += (
                " Per-grid-cell variables are only loaded if requested by name "
                "or if they have numeric coordinates."
            )
        raise _MissingConstantsError(missing, message)


class _MissingConstantsError(ValueError):
    """A normalization config lacks constants for requested variables."""

    def __init__(self, names: list[str], message: str):
        super().__init__(message)
        self.names = names


class NormalizeFn(Protocol):
    """
    A callable that normalizes a mapping of tensors, with an option to skip
    the mean subtraction (see :meth:`StandardNormalizer.normalize`).
    """

    def __call__(
        self, tensors: TensorMapping, /, apply_mean: bool = True
    ) -> TensorDict:
        # NOTE: ``tensors`` is positional-only so implementations may name their
        # first parameter freely (e.g. test lambdas); a positional-or-keyword
        # parameter would require every implementation to use the same name.
        ...


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
        grid_coords: Mapping[str, GridCoords] | None = None,
    ):
        self.means = move_tensordict_to_device(means)
        self.stds = move_tensordict_to_device(stds)
        self._names = set(means).intersection(stds)
        self._fill_nans_on_normalize = fill_nans_on_normalize
        self._fill_nans_on_denormalize = fill_nans_on_denormalize
        self.grid_coords: dict[str, GridCoords] = dict(grid_coords or {})

    @property
    def fill_nans_on_normalize(self):
        return self._fill_nans_on_normalize

    @property
    def fill_nans_on_denormalize(self):
        return self._fill_nans_on_denormalize

    def normalize(self, tensors: TensorMapping, apply_mean: bool = True) -> TensorDict:
        """
        Normalize the tensors.

        Args:
            tensors: Mapping from variable names to tensors; names without
                normalization constants are dropped from the output.
            apply_mean: If False, skip the mean subtraction and divide by the
                standard deviation only, e.g. to normalize a difference of
                fields without centering it.
        """
        filtered_tensors = {k: v for k, v in tensors.items() if k in self._names}
        return _normalize(
            filtered_tensors,
            means=self.means,
            stds=self.stds,
            fill_nans=self._fill_nans_on_normalize,
            apply_mean=apply_mean,
        )

    def denormalize(self, tensors: TensorMapping) -> TensorDict:
        filtered_tensors = {k: v for k, v in tensors.items() if k in self._names}
        return _denormalize(
            filtered_tensors,
            means=self.means,
            stds=self.stds,
            fill_nans=self._fill_nans_on_denormalize,
        )

    @property
    def is_per_grid_cell(self) -> bool:
        """Whether any constant varies over grid cells."""
        return any(v.numel() > 1 for v in (*self.means.values(), *self.stds.values()))

    def validate_spatial_shape(
        self,
        spatial_shape: tuple[int, ...],
        spatial_coords: Sequence[np.ndarray] | None = None,
    ) -> None:
        """Check that per-grid-cell constants match the data's grid.

        Per-grid-cell constants are rejected under spatial parallelism.

        Args:
            spatial_shape: Shape of the data's trailing spatial dimensions.
            spatial_coords: The data's coordinates along those dimensions, if
                known, to compare with those recorded for the constants.
        """
        _validate_constants_spatial_shape(
            {"means": self.means, "stds": self.stds}, spatial_shape
        )
        if spatial_coords is not None:
            for name, coords in sorted(self.grid_coords.items()):
                _validate_grid_coords(name, coords, spatial_coords)

    def _serializable_constants(
        self,
    ) -> tuple[dict[str, NormalizationConstant], dict[str, NormalizationConstant]]:
        """Means and stds as floats, or CPU tensors if per-grid-cell."""
        return (
            {k: _to_serializable(v) for k, v in self.means.items()},
            {k: _to_serializable(v) for k, v in self.stds.items()},
        )

    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        means, stds = self._serializable_constants()
        state = {
            "means": means,
            "stds": stds,
            "fill_nans_on_normalize": self._fill_nans_on_normalize,
            "fill_nans_on_denormalize": self._fill_nans_on_denormalize,
        }
        if self.grid_coords:
            state["grid_coords"] = _copy_grid_coords(self.grid_coords)
        return state

    @classmethod
    def from_state(cls, state) -> "StandardNormalizer":
        """
        Loads state from a serializable data structure.
        """
        means = {k: _as_float_tensor(v) for k, v in state["means"].items()}
        stds = {k: _as_float_tensor(v) for k, v in state["stds"].items()}
        return cls(
            means=means,
            stds=stds,
            fill_nans_on_normalize=state.get("fill_nans_on_normalize", False),
            fill_nans_on_denormalize=state.get("fill_nans_on_denormalize", False),
            grid_coords=state.get("grid_coords"),
        )

    def get_normalization_config(self) -> NormalizationConfig:
        means, stds = self._serializable_constants()
        return NormalizationConfig(
            means=means,
            stds=stds,
            fill_nans_on_normalize=self.fill_nans_on_normalize,
            fill_nans_on_denormalize=self.fill_nans_on_denormalize,
            per_grid_cell=self.is_per_grid_cell or bool(self.grid_coords),
            per_grid_cell_coords=_copy_grid_coords(self.grid_coords),
        )


class _PerSampleNormalizer(StandardNormalizer):
    """A normalizer bound to one batch by ``GroupedNormalizer.bind``, whose
    constants have a leading sample dimension and so cannot be serialized.
    """

    def _serializable_constants(self):
        raise ValueError(
            "Cannot serialize per-sample normalization means and stds; "
            "serialize the grouped normalization configuration instead."
        )

    @property
    def is_per_grid_cell(self) -> bool:
        # Ambiguous here: per-sample scalars also have more than one element.
        raise ValueError(
            "Ask the GroupedNormalizer, not a normalizer bound to a batch, "
            "whether it is per-grid-cell."
        )

    def validate_spatial_shape(
        self,
        spatial_shape: tuple[int, ...],
        spatial_coords: Sequence[np.ndarray] | None = None,
    ) -> None:
        raise ValueError(
            "Validate the GroupedNormalizer, not a normalizer bound to a batch."
        )


class _HasSpatialGrid(Protocol):
    @property
    def spatial_shape(self) -> tuple[int, ...]: ...

    @property
    def spatial_coords(self) -> tuple[np.ndarray, ...] | None: ...


def validate_normalizer_spatial_shape(
    normalizer: "StandardNormalizer | GroupedNormalizer",
    dataset_info: _HasSpatialGrid,
) -> None:
    """Check a normalizer's per-grid-cell constants, if any, against the
    dataset's grid.
    """
    if normalizer.is_per_grid_cell:
        normalizer.validate_spatial_shape(
            dataset_info.spatial_shape, dataset_info.spatial_coords
        )


def _as_float_tensor(value: NormalizationConstant) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float)


def _to_serializable(value: torch.Tensor) -> NormalizationConstant:
    if value.numel() == 1:
        return float(value.item())
    return value.detach().cpu().clone()


def _copy_grid_coords(grid_coords: Mapping[str, GridCoords]) -> dict[str, GridCoords]:
    return {
        name: {dim: list(values) for dim, values in coords.items()}
        for name, coords in grid_coords.items()
    }


def _validate_constants_spatial_shape(
    constants: Mapping[str, TensorMapping], spatial_shape: tuple[int, ...]
) -> None:
    """Check per-grid-cell constants, keyed by a description such as
    "means", against the data's spatial shape.
    """
    per_grid_cell = [
        (description, name, value)
        for description, tensors in constants.items()
        for name, value in tensors.items()
        if value.numel() > 1
    ]
    if per_grid_cell:
        # Checked first, as the local shape would give a misleading error.
        Distributed.get_instance().require_no_spatial_parallelism(
            "Per-grid-cell normalization is not supported with spatial parallelism."
        )
    for description, name, value in per_grid_cell:
        if tuple(value.shape) != tuple(spatial_shape):
            raise ValueError(
                f"Per-grid-cell normalization {description} for '{name}' "
                f"have shape {tuple(value.shape)}, which does not match the "
                f"data's spatial shape {tuple(spatial_shape)}."
            )


def _coords_match(a: np.ndarray, b: np.ndarray) -> bool:
    # Tolerant of the float32 rounding of the data's coordinates.
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-5, atol=1e-4))


def _validate_grid_coords(
    name: str, coords: GridCoords, spatial_coords: Sequence[np.ndarray]
) -> None:
    """Check a per-grid-cell constant's coordinates against the data's."""
    expected = [np.asarray(values, dtype=np.float64) for values in spatial_coords]
    if len(coords) != len(expected):
        raise ValueError(
            f"Per-grid-cell normalization constants for '{name}' have dimensions "
            f"{list(coords)}, but the data has {len(expected)} spatial "
            "dimensions."
        )
    for i, (dim, values) in enumerate(coords.items()):
        actual = np.asarray(values, dtype=np.float64)
        if _coords_match(actual, expected[i]):
            continue
        other_dims = [
            j
            for j, other in enumerate(expected)
            if j != i and _coords_match(actual, other)
        ]
        if _coords_match(actual[::-1], expected[i]):
            problem = "are reversed relative to the data's"
        elif other_dims:
            problem = (
                "do not match the data's, but match those along spatial "
                f"dimension {other_dims[0]}, so the dimensions are transposed"
            )
        elif _coords_match(
            np.sort(np.mod(actual, 360.0)), np.sort(np.mod(expected[i], 360.0))
        ):
            problem = (
                "do not match the data's, but are the same values modulo 360, "
                "so a longitude convention may differ (e.g. 0..360 vs -180..180)"
            )
        else:
            problem = "do not match the data's"
        raise ValueError(
            f"Coordinates of dimension '{dim}' (spatial dimension {i}) of the "
            f"per-grid-cell normalization constants for '{name}' {problem}. "
            "Reorder or regrid the constants to match the data's grid."
        )


def _validate_per_grid_cell_values(means: TensorMapping, stds: TensorMapping) -> None:
    """Require per-grid-cell means to be finite and stds positive."""
    for description, constants in (("means", means), ("stds", stds)):
        for name, constant in sorted(constants.items()):
            if constant.numel() == 1:
                continue
            n_non_finite = int((~torch.isfinite(constant)).sum())
            if n_non_finite > 0:
                raise ValueError(
                    f"Per-grid-cell normalization {description} for '{name}' "
                    f"are non-finite in {n_non_finite} of {constant.numel()} "
                    "grid cells; fill them in the statistics files."
                )
    for name, std in sorted(stds.items()):
        if std.numel() == 1:
            continue
        if not bool((std != 0).any()):
            raise ValueError(
                f"Per-grid-cell normalization stds for '{name}' are zero in "
                "every grid cell, so the variable cannot be normalized."
            )
        if not bool((std > 0).all()):
            raise ValueError(
                f"Per-grid-cell normalization stds for '{name}' are not positive "
                f"in every grid cell (minimum {float(std.min())}); fix them in "
                "the statistics files."
            )


def _normalize(
    tensors: TensorDict,
    means: TensorDict,
    stds: TensorDict,
    fill_nans: bool,
    apply_mean: bool = True,
) -> TensorDict:
    if apply_mean:
        normalized = {k: (t - means[k]) / stds[k] for k, t in tensors.items()}
    else:
        normalized = {k: t / stds[k] for k, t in tensors.items()}
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
            # means may be per-sample or per-grid-cell tensors
            fill = torch.broadcast_to(means[k].to(v.dtype), v.shape)
            denormalized[k] = torch.where(torch.isnan(v), fill, v)
    return denormalized


def get_normalizer(
    global_means_path,
    global_stds_path,
    names: list[str],
    per_grid_cell: bool = False,
    per_grid_cell_min_std_fraction: float = 0.01,
    **normalizer_kwargs,
) -> StandardNormalizer:
    """Build a normalizer from means and stds netCDF files.

    See ``NormalizationConfig`` for the per-grid-cell arguments.
    """
    means, stds, grid_coords = _load_means_and_stds(
        global_means_path,
        global_stds_path,
        names,
        per_grid_cell=per_grid_cell,
        min_std_fraction=per_grid_cell_min_std_fraction,
    )
    return StandardNormalizer(
        means={k: _as_float_tensor(v) for k, v in means.items()},
        stds={k: _as_float_tensor(v) for k, v in stds.items()},
        grid_coords=grid_coords,
        **normalizer_kwargs,
    )


def _load_means_and_stds(
    means_path: str | pathlib.Path,
    stds_path: str | pathlib.Path,
    names: Iterable[str] | None,
    per_grid_cell: bool,
    min_std_fraction: float,
    per_grid_cell_names: Iterable[str] | None = None,
) -> tuple[
    dict[str, NormalizationConstant],
    dict[str, NormalizationConstant],
    dict[str, GridCoords],
]:
    """Load means, floored stds, and per-grid-cell coordinates, which must
    agree between the two files.
    """
    means, means_coords = load_dict_from_netcdf(
        means_path,
        names,
        defaults={"x": 0.0, "y": 0.0, "z": 0.0},
        per_grid_cell=per_grid_cell,
        per_grid_cell_names=per_grid_cell_names,
    )
    stds, stds_coords = load_dict_from_netcdf(
        stds_path,
        names,
        defaults={"x": 1.0, "y": 1.0, "z": 1.0},
        per_grid_cell=per_grid_cell,
        per_grid_cell_names=per_grid_cell_names,
    )
    grid_coords = dict(means_coords)
    for name, coords in stds_coords.items():
        if name not in grid_coords:
            grid_coords[name] = coords
        elif list(grid_coords[name]) != list(coords) or not all(
            _coords_match(np.asarray(grid_coords[name][dim]), np.asarray(values))
            for dim, values in coords.items()
        ):
            raise ValueError(
                f"Per-grid-cell means for '{name}' in {means_path} and stds in "
                f"{stds_path} have different dimensions or coordinates."
            )
    return means, _floor_per_grid_cell_stds(stds, min_std_fraction), grid_coords


def _floor_per_grid_cell_stds(
    stds: dict[str, NormalizationConstant], min_fraction: float
) -> dict[str, NormalizationConstant]:
    """Clamp each non-negative per-grid-cell std to at least ``min_fraction``
    of its mean over valid cells. Invalid values are left for
    ``_validate_per_grid_cell_values`` to reject.
    """
    result: dict[str, NormalizationConstant] = {}
    for name, std in stds.items():
        values = _as_float_tensor(std)
        if values.numel() > 1:
            is_valid = torch.isfinite(values) & (values >= 0)
            if bool(is_valid.any()):
                floor = min_fraction * values[is_valid].mean()
                if floor > 0:
                    is_floored = (values >= 0) & (values < floor)
                    n_floored = int(is_floored.sum())
                    if n_floored > 0:
                        logging.info(
                            f"Flooring per-grid-cell normalization stds for "
                            f"'{name}' to {float(floor):.6g} in {n_floored} of "
                            f"{values.numel()} grid cells."
                        )
                    std = torch.where(is_floored, floor, values)
        result[name] = std
    return result


def _dimension_coords(ds: xr.Dataset, dim: Hashable) -> np.ndarray | None:
    """Decoded numeric coordinates of a dimension, or None if it has none."""
    if dim not in ds.coords or ds.coords[dim].dims != (dim,):
        return None
    if not np.issubdtype(ds.coords[dim].dtype, np.number):
        return None
    return _decoded_values(ds, str(dim))


def _is_per_grid_cell_candidate(ds: xr.Dataset, name: Hashable) -> bool:
    """Whether ``name`` is a numeric data variable with numeric coordinates."""
    variable = ds.variables[name]
    return (
        name in ds.data_vars
        and np.issubdtype(variable.dtype, np.number)
        and all(_dimension_coords(ds, dim) is not None for dim in variable.dims)
    )


def _decoded_values(ds: xr.Dataset, name: str) -> np.ndarray:
    """Values of a variable with fill values masked to NaN and unpacked."""
    decoded = xr.decode_cf(
        ds[[name]],
        mask_and_scale=True,
        decode_times=False,
        decode_coords=False,
        decode_timedelta=False,
        concat_characters=False,
    )
    return np.asarray(decoded[name].values)


def load_dict_from_netcdf(
    path: str | pathlib.Path,
    names: Iterable[str] | None,
    defaults: Mapping[str, float | np.ndarray],
    per_grid_cell: bool = False,
    per_grid_cell_names: Iterable[str] | None = None,
) -> tuple[dict[str, NormalizationConstant], dict[str, GridCoords]]:
    """
    Load a dictionary of variables from a netCDF file.

    Args:
        path: Path to the netCDF file.
        names: List of variable names to load. If None, all scalar variables
            in the netCDF file are loaded, plus non-scalar ones if
            ``per_grid_cell`` (see ``per_grid_cell_names``).
        defaults: Dictionary of default values for each variable, if not found
            in the netCDF file.
        per_grid_cell: Whether to load non-scalar variables as tensors.
        per_grid_cell_names: If ``names`` is None, the non-scalar variables
            to load. If None, those with numeric coordinates are loaded.

    Returns:
        The loaded constants, and the coordinates of per-grid-cell ones.
    """
    # Per-grid-cell variables are decoded from the raw dataset below.
    raw = open_dataset_via_inter_filesystem_copy(
        path, mask_and_scale=False, decode_times=False, decode_timedelta=False
    )
    ds = xr.decode_cf(raw, mask_and_scale=False)

    result: dict[str, NormalizationConstant] = {}
    grid_coords: dict[str, GridCoords] = {}
    if names is None:
        if per_grid_cell:
            scalars = {
                str(name)
                for name in ds.variables
                if ds.variables[name].ndim == 0
                and np.issubdtype(ds.variables[name].dtype, np.number)
            }
            if per_grid_cell_names is None:
                non_scalars = {
                    str(name)
                    for name in ds.variables
                    if _is_per_grid_cell_candidate(ds, name)
                }
            else:
                non_scalars = set(per_grid_cell_names).intersection(
                    str(name) for name in ds.data_vars
                )
            names = scalars.union(non_scalars, defaults.keys())
        else:
            names = {
                str(name) for name in ds.variables if ds.variables[name].ndim == 0
            }.union(defaults.keys())
    for c in names:
        if c in ds.variables:
            variable = ds.variables[c]
            if variable.ndim == 0:
                result[c] = float(variable.values.item())
            elif per_grid_cell:
                coords: GridCoords = {}
                for dim in variable.dims:
                    values = _dimension_coords(ds, dim)
                    if values is None:
                        raise ValueError(
                            f"Per-grid-cell variable '{c}' in {path} has "
                            f"dimension '{dim}' without numeric coordinates; "
                            "add them."
                        )
                    coords[str(dim)] = values.astype(np.float64).tolist()
                result[c] = _as_float_tensor(_decoded_values(raw, c))
                grid_coords[c] = coords
            else:
                raise ValueError(
                    f"Variable '{c}' in {path} has dimensions {variable.dims}; "
                    "set per_grid_cell to use per-grid-cell normalization "
                    "constants."
                )
        elif c in defaults:
            result[c] = float(defaults[c])
        else:
            raise ValueError(f"Variable {c} not found in {path}")
    ds.close()
    raw.close()
    return result, grid_coords


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
    overridden = set(override_normalizer.means) | set(override_normalizer.stds)
    grid_coords = {
        name: coords
        for name, coords in base_normalizer.grid_coords.items()
        if name not in overridden
    }
    grid_coords.update(override_normalizer.grid_coords)
    return StandardNormalizer(
        means=means,
        stds=stds,
        fill_nans_on_normalize=base_normalizer.fill_nans_on_normalize,
        fill_nans_on_denormalize=base_normalizer.fill_nans_on_denormalize,
        grid_coords=grid_coords,
    )


# Per-group std below this fraction of pooled std is degenerate (near-constant).
_MIN_GROUP_TO_POOLED_STD_RATIO = 1e-4


class GroupedNormalizer:
    """
    Normalizer which selects normalization constants per sample, based on the
    sample's labels.

    Each group is named by a dataset label, and each sample must carry exactly
    one label, which selects its group. Variables listed in ``pinned_names``
    use the pooled constants regardless of group; this is required for
    variables which are near-constant within a group (whose per-group standard
    deviation would be ~0) and for variables whose per-group normalization
    would place different data sources in disjoint input spaces.

    The per-group constants are applied to the network's inputs and outputs;
    to global mean removal, which shifts each sample's fields to the
    climatological mean of the constants that then normalize them, so their
    spatial means still normalize to approximately zero in every group; and
    to a "mean" input masking fill, so masked cells normalize to zero. The
    aggregators' normalized metrics continue to use the pooled constants, as
    does the loss unless it has its own config, so those quantities remain
    comparable across models which share pooled constants but are trained
    with different grouping strategies. Any constants may be per-grid-cell.
    """

    def __init__(
        self,
        pooled: StandardNormalizer,
        groups: Mapping[str, StandardNormalizer],
        default_label: str,
        n_spatial_dims: int,
        pinned_names: Iterable[str] = (),
    ):
        """
        Args:
            pooled: Normalizer holding constants pooled over all groups. Used
                for pinned variables.
            groups: Mapping from dataset label to that group's normalizer,
                which must hold constants for every non-pinned variable.
            default_label: Label whose group to use when a batch carries no
                labels, e.g. during inference on an unlabeled dataset.
            n_spatial_dims: Number of trailing spatial dimensions on the
                tensors this normalizer is applied to (2 for lat/lon, 3 for
                HEALPix). Per-sample constants are reshaped to broadcast
                against ``[n_samples, *spatial]``.
            pinned_names: Variables which always use the pooled constants.
        """
        self._pooled = pooled
        self._groups = dict(groups)
        self._group_names = sorted(groups)
        self._default_label = default_label
        self._pinned_names = set(pinned_names)
        self._n_spatial_dims = n_spatial_dims
        self._per_group_names = set(pooled.means).intersection(pooled.stds) - set(
            pinned_names
        )
        self._raise_on_mismatched_cell_shapes()
        self._stacked_means = self._stack("means")
        self._stacked_stds = self._stack("stds")
        self._raise_on_degenerate_group_stds()
        default_index = self._group_names.index(default_label)
        self._default_normalizer = self._select(lambda t: t[default_index])
        # Caches bind() per labels object, since resolving groups forces a device sync.
        self._bind_cache: tuple[BatchLabels, StandardNormalizer] | None = None
        self._warned_default_label = False

    def _stack(self, attr: str) -> TensorDict:
        """Stack each per-group variable's constants into a [n_groups] tensor,
        or [n_groups, *spatial] if any are per-grid-cell.

        The stack is ordered by ``self._group_names``, so a group index
        computed against that ordering indexes it directly.
        """
        return {
            name: torch.stack(
                torch.broadcast_tensors(
                    *[
                        getattr(self._groups[group], attr)[name]
                        for group in self._group_names
                    ]
                )
            ).to(get_device())
            for name in self._per_group_names
        }

    @property
    def is_per_grid_cell(self) -> bool:
        """Whether any pooled or per-group constant varies over grid cells."""
        return self._pooled.is_per_grid_cell or any(
            group.is_per_grid_cell for group in self._groups.values()
        )

    def validate_spatial_shape(
        self,
        spatial_shape: tuple[int, ...],
        spatial_coords: Sequence[np.ndarray] | None = None,
    ) -> None:
        """See ``StandardNormalizer.validate_spatial_shape``."""
        self._pooled.validate_spatial_shape(spatial_shape, spatial_coords)
        for group in self._groups.values():
            group.validate_spatial_shape(spatial_shape, spatial_coords)

    def _raise_on_mismatched_cell_shapes(self) -> None:
        """Reject per-grid-cell constants whose shapes differ between groups."""
        sources = {"pooled": self._pooled} | {
            f"group '{label}'": self._groups[label] for label in self._group_names
        }
        for name in sorted(self._per_group_names):
            shapes = {
                f"{source} {attr}": tuple(constant.shape)
                for source, normalizer in sources.items()
                for attr, constant in (
                    ("means", normalizer.means[name]),
                    ("stds", normalizer.stds[name]),
                )
                if constant.numel() > 1
            }
            if len(set(shapes.values())) > 1:
                raise ValueError(
                    f"Per-grid-cell normalization constants for '{name}' have "
                    f"different shapes: {shapes}. Every group's constants "
                    "must be on the same grid."
                )

    def _raise_on_degenerate_group_stds(self) -> None:
        """Reject a non-pinned variable that is ~constant within some group.

        ``validate_pinned_variables`` only catches a typo in a pinned name;
        this catches a variable that should have been pinned but was not, e.g.
        a global-mean CO2 which is fixed within each dataset. Grid cells where
        the pooled std is not positive are skipped, as pinning can't fix them.
        """
        for name in sorted(self._per_group_names):
            pooled_std = self._pooled.stds[name].to(get_device())
            # align scalar group stds with per-grid-cell pooled ones
            stacked = self._stacked_stds[name]
            n_missing_dims = max(pooled_std.ndim - (stacked.ndim - 1), 0)
            stacked = stacked.reshape(
                stacked.shape[0], *(1,) * n_missing_dims, *stacked.shape[1:]
            )
            ratio = stacked / pooled_std
            is_degenerate_per_group = (
                (pooled_std > 0) & ~(ratio >= _MIN_GROUP_TO_POOLED_STD_RATIO)
            ).reshape(len(self._group_names), -1)
            degenerate = [
                group_name
                for group_name, is_degenerate in zip(
                    self._group_names,
                    is_degenerate_per_group.any(dim=1).tolist(),
                )
                if is_degenerate
            ]
            if degenerate:
                raise ValueError(
                    f"Variable '{name}' has a standard deviation in groups "
                    f"{degenerate} below {_MIN_GROUP_TO_POOLED_STD_RATIO} of its "
                    "pooled standard deviation, so normalizing it per group "
                    "would blow up the network's input. Add it to "
                    "pinned_variables to normalize it with the pooled constants."
                )

    def _select(
        self,
        select: Callable[[torch.Tensor], torch.Tensor],
        normalizer_type: type[StandardNormalizer] = StandardNormalizer,
    ) -> StandardNormalizer:
        """Normalizer with ``select`` applied to each stacked per-group
        constant, and the pooled constants for pinned variables.
        """
        means = dict(self._pooled.means)
        stds = dict(self._pooled.stds)
        for name in self._per_group_names:
            means[name] = select(self._stacked_means[name])
            stds[name] = select(self._stacked_stds[name])
        return normalizer_type(
            means=means,
            stds=stds,
            fill_nans_on_normalize=self._pooled.fill_nans_on_normalize,
            fill_nans_on_denormalize=self._pooled.fill_nans_on_denormalize,
        )

    def bind(self, labels: BatchLabels | None) -> StandardNormalizer:
        """
        Resolve per-sample constants into a normalizer for a single batch.

        The returned normalizer holds means and stds of shape
        ``[n_samples, *(1,) * n_spatial_dims]`` for non-pinned variables, or
        ``[n_samples, *spatial]`` for per-grid-cell ones, which broadcast
        against the ``[n_batch, *spatial]`` tensors the step operates on.
        Pinned variables keep their pooled constants.

        Args:
            labels: Labels for each sample in the batch. If None or empty,
                every sample uses the default group's constants.
        """
        if labels is None or len(labels.names) == 0:
            if not self._warned_default_label:
                # Warn once, not per step; fallback is expected on unlabeled data.
                logging.warning(
                    "Batch carries no labels; normalizing every sample with "
                    f"default_label '{self._default_label}'. Set labels on the "
                    "dataset or inference config to select groups per sample."
                )
                self._warned_default_label = True
            return self._default_normalizer
        if self._bind_cache is not None and self._bind_cache[0] is labels:
            return self._bind_cache[1]
        group_index = self._resolve_group_index(labels)
        normalizer = self._select(
            lambda t: self._per_sample(t, group_index), _PerSampleNormalizer
        )
        self._bind_cache = (labels, normalizer)
        return normalizer

    def _per_sample(
        self, stacked: torch.Tensor, group_index: torch.Tensor
    ) -> torch.Tensor:
        """Select each sample's constants, padded to n_spatial_dims."""
        cell_shape = stacked.shape[1:]
        n_missing_dims = self._n_spatial_dims - len(cell_shape)
        return stacked[group_index].reshape(-1, *(1,) * n_missing_dims, *cell_shape)

    def _resolve_group_index(self, labels: BatchLabels) -> torch.Tensor:
        """Map each sample's label to its group index.

        A sample carrying zero or several labels is an error rather than a
        silent pick, since that would quietly normalize data against the wrong
        distribution.

        Called once per batch rather than once per forward step: the
        ``n_labels == 1`` check forces a device sync, which is too expensive to
        repeat inside a rollout. ``bind`` handles the caching.
        """
        unknown = set(labels.names) - set(self._group_names)
        if unknown:
            raise ValueError(
                f"Labels {sorted(unknown)} have no normalization group. "
                f"Known labels: {self._group_names}."
            )
        is_set = labels.tensor > 0
        n_labels = is_set.sum(dim=1)
        if not bool((n_labels == 1).all()):
            bad = torch.nonzero(n_labels != 1).flatten().tolist()
            raise ValueError(
                f"Samples at batch indices {bad} carry a number of labels other "
                "than exactly one. Each dataset must carry a single label, "
                "which selects its normalization group."
            )
        column_to_group = torch.tensor(
            [self._group_names.index(name) for name in labels.names],
            device=labels.tensor.device,
        )
        return column_to_group[is_set.int().argmax(dim=1)]


@dataclasses.dataclass
class GroupedNormalizationConfig:
    """
    Configuration for per-group network normalization.

    Layers per-group constants on top of the pooled ``network`` constants of
    the enclosing :class:`NetworkAndLossNormalizationConfig`. The pooled
    constants remain in use for pinned variables and for every consumer other
    than the network's inputs and outputs and global mean removal.

    Parameters:
        groups: Mapping from dataset label to that label's normalization
            constants. Their NaN-filling options must be left unset: they are
            taken from the pooled ``network`` config, which applies to every
            group.
        default_label: Label whose group to use for batches which carry no
            labels, such as inference on an unlabeled dataset. Required, since
            an implicit choice here would silently normalize against the wrong
            distribution.
        pinned_variables: Variables which always use the pooled constants.

    Every label the training dataset carries must have a group, and each
    dataset must carry exactly one label. The same labels also drive module
    conditioning when the module is conditional, so the conditioning is
    exactly as fine-grained as the grouping.
    """

    groups: dict[str, NormalizationConfig]
    default_label: str
    pinned_variables: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if len(self.groups) == 0:
            raise ValueError("At least one normalization group must be provided.")
        if self.default_label not in self.groups:
            raise ValueError(
                f"default_label '{self.default_label}' is not one of the "
                f"configured groups: {sorted(self.groups)}"
            )
        nan_filling = sorted(
            label for label, group in self.groups.items() if group.fills_nans
        )
        if nan_filling:
            raise ValueError(
                f"NaN filling is not supported in normalization groups "
                f"{nan_filling}; set it on the pooled network config instead."
            )

    def validate_pinned_variables(self, names: Iterable[str]) -> None:
        """Reject a pinned name which is not a variable being normalized.

        Called by the step config, which owns the variable list.
        """
        unknown = sorted(set(self.pinned_variables) - set(names))
        if unknown:
            raise ValueError(
                f"pinned_variables {unknown} are not normalized variables, so "
                "pinning them has no effect. Check for a typo; the normalized "
                f"variables are {sorted(names)}."
            )

    def get_pinned_variables(self) -> frozenset[str]:
        """Variables which always use the pooled constants."""
        return frozenset(self.pinned_variables)

    def validate_dataset_labels(self, dataset_labels: Iterable[str]) -> None:
        """Reject training labels which cannot select a group.

        Without this, an unlabeled training dataset would train entirely on
        ``default_label`` behind a single warning, and a label missing from
        every group would only fail at the first batch.
        """
        dataset_labels = set(dataset_labels)
        if len(dataset_labels) == 0:
            raise ValueError(
                "Grouped network normalization requires a labeled dataset, but "
                "the dataset carries no labels. Set labels on each dataset."
            )
        unknown = sorted(dataset_labels - set(self.groups))
        if unknown:
            raise ValueError(
                f"Dataset labels {unknown} have no normalization group. "
                f"Known labels: {sorted(self.groups)}."
            )

    def build(
        self,
        pooled: StandardNormalizer,
        names: list[str],
        n_spatial_dims: int,
        dataset_labels: Iterable[str],
    ) -> GroupedNormalizer:
        """
        Args:
            pooled: The pooled network normalizer.
            names: Names of the variables to normalize.
            n_spatial_dims: Number of trailing spatial dimensions on the
                tensors being normalized.
            dataset_labels: All labels carried by the training dataset.
        """
        self.validate_dataset_labels(dataset_labels)
        per_group_names = [name for name in names if name not in self.pinned_variables]
        groups = {}
        for group_name, group in self.groups.items():
            try:
                groups[group_name] = group.build(names=per_group_names)
            except _MissingConstantsError as err:
                raise ValueError(
                    f"Normalization group '{group_name}' has no constants for "
                    f"variables {err.names}. Every group must provide constants "
                    "for every non-pinned variable."
                ) from err
        return GroupedNormalizer(
            pooled=pooled,
            groups=groups,
            default_label=self.default_label,
            pinned_names=self.pinned_variables,
            n_spatial_dims=n_spatial_dims,
        )

    def load(self, names: Iterable[str] | None = None):
        """See ``NormalizationConfig.load``; pinned variables are skipped."""
        if names is not None:
            names = [name for name in names if name not in self.pinned_variables]
        for group in self.groups.values():
            group.load(names)


@dataclasses.dataclass
class NetworkAndLossNormalizationConfig:
    """
    Combined configuration for network and loss normalization.

    Allows loss normalization to be defined as equal to the network
    normalization, apart from a set of residual-scaled variables.

    Parameters:
        network: The normalization configuration for the network. Also used
            for the aggregators' normalized metrics, which are therefore
            per-grid-cell normalized if it is.
        loss: The normalization configuration for the loss. Default is to
            use the network configuration. Required if the network's is
            per-grid-cell, so per-grid-cell loss weighting is deliberate.
        residual: The normalization configuration for residual-scaled
            variables in the loss, overriding ``loss`` (or ``network``) for
            those variables only.
        grouped: Optional per-group network normalization. When provided, the
            network's inputs and outputs are normalized using constants
            selected per sample from the sample's labels, as is global mean
            removal, while ``network`` supplies the pooled constants used for
            pinned variables and for every other consumer of normalization
            constants.
    """

    network: NormalizationConfig
    loss: NormalizationConfig | None = None
    residual: NormalizationConfig | None = None
    grouped: GroupedNormalizationConfig | None = None

    def raise_if_per_grid_cell_loss_is_implicit(self) -> None:
        """Require an explicit ``loss`` block with a per-grid-cell network."""
        if self.network.per_grid_cell and self.loss is None:
            raise ValueError(
                "Per-grid-cell network normalization requires an explicit loss "
                "normalization config: set normalization.loss to scalar stats, "
                "or to the same per-grid-cell config for per-cell loss weighting."
            )

    def raise_if_unsupported(self, step_type: str) -> None:
        """Reject grouped and per-grid-cell normalization, for a step which
        supports neither.

        This config is shared by several step types, but only the
        single-module step supports them. Without this, setting them on one
        of the others would be silently ignored or misapplied.
        """
        if self.grouped is not None:
            raise ValueError(
                f"{step_type} does not support grouped network normalization; "
                "remove the 'grouped' block from its normalization config."
            )
        configs = {
            "network": self.network,
            "loss": self.loss,
            "residual": self.residual,
        }
        for key, config in configs.items():
            if config is not None:
                config.raise_if_per_grid_cell(f"{step_type} ({key} normalization)")

    @property
    def is_grouped(self) -> bool:
        """Whether the network normalizes per label group."""
        return self.grouped is not None

    def validate_pinned_variables(self, names: Iterable[str]) -> None:
        """Check pinned variable names against the variables being normalized."""
        if self.grouped is not None:
            self.grouped.validate_pinned_variables(names)

    @property
    def pinned_variables(self) -> frozenset[str]:
        """Variables which the network normalizes with the pooled constants
        under grouped normalization; empty without it.
        """
        if self.grouped is None:
            return frozenset()
        return self.grouped.get_pinned_variables()

    def get_network_normalizer(self, names: list[str]) -> StandardNormalizer:
        return self.network.build(names=names)

    def get_grouped_network_normalizer(
        self,
        pooled: StandardNormalizer,
        names: list[str],
        n_spatial_dims: int,
        dataset_labels: Iterable[str],
    ) -> GroupedNormalizer | None:
        """
        Args:
            pooled: The network normalizer from ``get_network_normalizer``,
                which supplies the pooled constants.
            names: Names of the variables to normalize.
            n_spatial_dims: Number of trailing spatial dimensions on the
                tensors being normalized.
            dataset_labels: All labels carried by the training dataset.
        """
        if self.grouped is None:
            return None
        return self.grouped.build(
            pooled=pooled,
            names=names,
            n_spatial_dims=n_spatial_dims,
            dataset_labels=dataset_labels,
        )

    def get_loss_normalizer(
        self,
        names: list[str],
        residual_scaled_names: list[str],
    ) -> StandardNormalizer:
        base = self.loss if self.loss is not None else self.network
        if self.residual is None:
            return base.build(names=names)
        return _combine_normalizers(
            base_normalizer=base.build(names=names),
            override_normalizer=self.residual.build(names=residual_scaled_names),
        )

    def load(
        self, names: Iterable[str] | None = None, extra_loss_names: Iterable[str] = ()
    ):
        """
        Args:
            names: Names of the variables to load (see
                ``NormalizationConfig.load``), or None for all of them.
            extra_loss_names: Further names to load for the loss only.
        """
        if names is None:
            loss_names = None
        else:
            loss_names = sorted(set(names).union(extra_loss_names))
            names = sorted(set(names))
        # Loss configs first: a config shared with the network loads only once.
        if self.loss is not None:
            self.loss.load(loss_names)
        if self.residual is not None:
            self.residual.load(loss_names)
        self.network.load(names if self.loss is not None else loss_names)
        if self.grouped is not None:
            self.grouped.load(names)
