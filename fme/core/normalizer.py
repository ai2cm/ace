import dataclasses
import pathlib
from collections.abc import Iterable, Mapping
from copy import copy
from typing import Protocol

import fsspec
import numpy as np
import torch
import xarray as xr

from fme.core.device import get_device, move_tensordict_to_device
from fme.core.labels import BatchLabels
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


class GroupedNormalizer:
    """
    Normalizer which selects normalization constants per sample, based on the
    sample's labels.

    Each sample is assigned to exactly one group, determined by its labels.
    Variables listed in ``pinned_names`` use the pooled constants regardless of
    group; this is required for variables which are near-constant within a group
    (whose per-group standard deviation would be ~0) and for variables whose
    per-group normalization would place different data sources in disjoint
    input spaces.

    The per-group constants are only applied to the network's inputs and
    outputs. Everything else which reads normalization constants -- the loss,
    global mean removal, spatial masking fill values, and the aggregators'
    normalized metrics -- continues to use the pooled constants, so those
    quantities remain comparable across models trained with different grouping
    strategies.

    Note the interaction with global mean removal, which runs before
    normalization and shifts its fields to their *pooled* climatological mean.
    Normalizing those fields against a group's mean therefore leaves a constant
    ``(pooled_mean - group_mean) / group_std`` in the network's input, which
    differs by group. This is by design -- global mean removal is one of the
    consumers deliberately kept on the pooled constants -- but it does mean the
    fields it covers are not fully aligned across groups.
    """

    def __init__(
        self,
        pooled: StandardNormalizer,
        groups: Mapping[str, StandardNormalizer],
        label_to_group: Mapping[str, str],
        default_group: str,
        pinned_names: Iterable[str] = (),
        n_spatial_dims: int = 2,
    ):
        """
        Args:
            pooled: Normalizer holding constants pooled over all groups. Used
                for pinned variables.
            groups: Mapping from group name to that group's normalizer.
            label_to_group: Mapping from dataset label to group name.
            default_group: Group to use when a batch carries no labels, e.g.
                during inference on an unlabeled dataset.
            pinned_names: Variables which always use the pooled constants.
            n_spatial_dims: Number of trailing spatial dimensions on the
                tensors this normalizer is applied to (2 for lat/lon, 3 for
                HEALPix). Per-sample constants are reshaped to broadcast
                against ``[n_samples, *spatial]``.
        """
        if default_group not in groups:
            raise ValueError(
                f"default_group '{default_group}' is not one of the "
                f"configured groups: {sorted(groups)}"
            )
        if n_spatial_dims < 1:
            raise ValueError(f"n_spatial_dims must be positive, got {n_spatial_dims}")
        self._pooled = pooled
        self._groups = dict(groups)
        self._group_names = sorted(groups)
        self._label_to_group = dict(label_to_group)
        self._default_group = default_group
        self._pinned_names = set(pinned_names)
        self._n_spatial_dims = n_spatial_dims
        self._per_group_names = set(pooled.means).intersection(pooled.stds) - set(
            pinned_names
        )
        self._stacked_means = self._stack("means")
        self._stacked_stds = self._stack("stds")
        self._default_normalizer = self._build_default_normalizer()
        # Single-entry cache for ``bind``. The same BatchLabels instance is
        # threaded through every forward step of a window, and resolving it
        # costs a device sync (see _resolve_group_index), so the resolve is
        # done once per batch rather than once per step. The key holds a
        # reference to the labels object, which keeps its id() from being
        # recycled onto a different object while the entry is live.
        self._bind_cache: tuple[BatchLabels, StandardNormalizer] | None = None

    def _stack(self, attr: str) -> TensorDict:
        """Stack each per-group variable's constants into a [n_groups] tensor.

        The stack is ordered by ``self._group_names``, so a group index
        computed against that ordering indexes it directly.
        """
        stacked = {}
        for name in self._per_group_names:
            values = []
            for group_name in self._group_names:
                group = self._groups[group_name]
                constants = getattr(group, attr)
                if name not in constants:
                    raise ValueError(
                        f"Variable '{name}' has pooled normalization constants "
                        f"but is missing from group '{group_name}'. Every group "
                        "must provide constants for every non-pinned variable."
                    )
                values.append(constants[name])
            stacked[name] = torch.stack(values).to(get_device())
        return stacked

    def _build_default_normalizer(self) -> StandardNormalizer:
        """The normalizer used for a batch which carries no labels.

        Holds the default group's constants for the non-pinned variables and
        the pooled constants for the pinned ones. They are scalars, not
        per-sample, since every sample resolves to the same group.
        """
        default = self._groups[self._default_group]
        means = dict(self._pooled.means)
        stds = dict(self._pooled.stds)
        for name in self._per_group_names:
            means[name] = default.means[name]
            stds[name] = default.stds[name]
        return self._make_normalizer(means, stds)

    def _make_normalizer(self, means: TensorDict, stds: TensorDict):
        return StandardNormalizer(
            means=means,
            stds=stds,
            fill_nans_on_normalize=self._pooled.fill_nans_on_normalize,
            fill_nans_on_denormalize=self._pooled.fill_nans_on_denormalize,
        )

    @property
    def pooled(self) -> StandardNormalizer:
        """The pooled normalizer, for consumers which must not vary by group."""
        return self._pooled

    def bind(self, labels: BatchLabels | None) -> StandardNormalizer:
        """
        Resolve per-sample constants into a normalizer for a single batch.

        The returned normalizer holds means and stds of shape
        ``[n_samples, *(1,) * n_spatial_dims]`` for non-pinned variables, which
        broadcast against the ``[n_batch, *spatial]`` tensors the step operates
        on. Pinned variables keep their scalar pooled constants.

        Args:
            labels: Labels for each sample in the batch, or None for a batch
                which carries none, in which case every variable uses the
                default group's constants. A batch is never normalized with
                the pooled constants: no model trained on this normalizer ever
                saw its network inputs on the pooled scale.
        """
        if labels is None or len(labels.names) == 0:
            return self._default_normalizer
        if self._bind_cache is not None and self._bind_cache[0] is labels:
            return self._bind_cache[1]
        group_index = self._resolve_group_index(labels)
        per_sample_shape = (-1, *(1,) * self._n_spatial_dims)
        means = dict(self._pooled.means)
        stds = dict(self._pooled.stds)
        for name in self._per_group_names:
            means[name] = self._stacked_means[name][group_index].reshape(
                per_sample_shape
            )
            stds[name] = self._stacked_stds[name][group_index].reshape(per_sample_shape)
        normalizer = self._make_normalizer(means, stds)
        self._bind_cache = (labels, normalizer)
        return normalizer

    def _resolve_group_index(self, labels: BatchLabels) -> torch.Tensor:
        """Map each sample's labels to exactly one group index.

        Labels are multi-hot, so a sample may carry several labels; they must
        all resolve to the same group. A sample resolving to zero or to more
        than one group is an error rather than a silent pick, since that would
        quietly normalize data against the wrong distribution.

        Called once per batch rather than once per forward step: the
        ``(counts > 0).sum(...) == 1`` check forces a device sync, which is too
        expensive to repeat inside a rollout. ``bind`` handles the caching.
        """
        unknown = set(labels.names) - set(self._label_to_group)
        if unknown:
            raise ValueError(
                f"Labels {sorted(unknown)} are not assigned to any normalization "
                f"group. Known labels: {sorted(self._label_to_group)}."
            )
        # membership[i, g] is 1 when label i belongs to group g, so
        # labels @ membership counts each sample's labels per group.
        membership = torch.zeros(
            (len(labels.names), len(self._group_names)),
            dtype=labels.tensor.dtype,
            device=labels.tensor.device,
        )
        for i, label in enumerate(labels.names):
            group = self._label_to_group[label]
            membership[i, self._group_names.index(group)] = 1.0
        counts = labels.tensor @ membership
        n_groups_per_sample = (counts > 0).sum(dim=1)
        if not bool((n_groups_per_sample == 1).all()):
            bad = torch.nonzero(n_groups_per_sample != 1).flatten().tolist()
            raise ValueError(
                f"Samples at batch indices {bad} resolve to a number of "
                "normalization groups other than exactly one. Each dataset must "
                "carry labels belonging to a single group."
            )
        return counts.argmax(dim=1)


@dataclasses.dataclass
class NormalizationGroupConfig:
    """
    Configuration for one group's normalization constants.

    Parameters:
        labels: Dataset labels which belong to this group.
        normalization: Normalization constants for this group.
    """

    labels: list[str]
    normalization: NormalizationConfig

    def __post_init__(self):
        if len(self.labels) == 0:
            raise ValueError("A normalization group must list at least one label.")


@dataclasses.dataclass
class GroupedNormalizationConfig:
    """
    Configuration for per-group network normalization.

    Layers per-group constants on top of the pooled ``network`` constants of
    the enclosing :class:`NetworkAndLossNormalizationConfig`. The pooled
    constants remain in use for pinned variables and for every consumer other
    than the network's inputs and outputs.

    Parameters:
        groups: Mapping from group name to that group's configuration.
        default_group: Group to use for batches which carry no labels, such as
            inference on an unlabeled dataset. Required, since an implicit
            choice here would silently normalize against the wrong
            distribution.
        pinned_variables: Variables which always use the pooled constants.
    """

    groups: dict[str, NormalizationGroupConfig]
    default_group: str
    pinned_variables: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if len(self.groups) == 0:
            raise ValueError("At least one normalization group must be provided.")
        if self.default_group not in self.groups:
            raise ValueError(
                f"default_group '{self.default_group}' is not one of the "
                f"configured groups: {sorted(self.groups)}"
            )
        seen: dict[str, str] = {}
        for group_name, group in self.groups.items():
            for label in group.labels:
                if label in seen:
                    raise ValueError(
                        f"Label '{label}' is assigned to both group "
                        f"'{seen[label]}' and group '{group_name}'. Each label "
                        "must belong to exactly one group."
                    )
                seen[label] = group_name

    @property
    def label_to_group(self) -> dict[str, str]:
        return {
            label: group_name
            for group_name, group in self.groups.items()
            for label in group.labels
        }

    def validate_pinned_variables(self, names: Iterable[str]) -> None:
        """Reject a pinned name which is not a variable being normalized.

        Cannot live in ``__post_init__``: the variable list belongs to the
        enclosing step config, not to this one. Pinning is load-bearing rather
        than cosmetic -- a near-constant variable like ``global_mean_co2`` has a
        per-group standard deviation of ~0, so a typo here would silently
        normalize it per group and blow up the network's input.
        """
        unknown = sorted(set(self.pinned_variables) - set(names))
        if unknown:
            raise ValueError(
                f"pinned_variables {unknown} are not normalized variables, so "
                "pinning them has no effect. Check for a typo; the normalized "
                f"variables are {sorted(names)}."
            )

    def build(
        self, pooled: StandardNormalizer, names: list[str], n_spatial_dims: int = 2
    ) -> "GroupedNormalizer":
        return GroupedNormalizer(
            pooled=pooled,
            groups={
                group_name: group.normalization.build(names=names)
                for group_name, group in self.groups.items()
            },
            label_to_group=self.label_to_group,
            default_group=self.default_group,
            pinned_names=self.pinned_variables,
            n_spatial_dims=n_spatial_dims,
        )

    def load(self):
        for group in self.groups.values():
            group.normalization.load()


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
        grouped: Optional per-group network normalization. When provided, the
            network's inputs and outputs are normalized using constants
            selected per sample from the sample's labels, while ``network``
            supplies the pooled constants used for pinned variables and for
            every other consumer of normalization constants.
    """

    network: NormalizationConfig
    loss: NormalizationConfig | None = None
    residual: NormalizationConfig | None = None
    grouped: GroupedNormalizationConfig | None = None

    def __post_init__(self):
        if self.loss is not None and self.residual is not None:
            raise ValueError("Cannot provide both loss and residual normalization.")

    def validate_pinned_variables(self, names: Iterable[str]) -> None:
        """Check pinned variable names against the variables being normalized."""
        if self.grouped is not None:
            self.grouped.validate_pinned_variables(names)

    def raise_if_grouped(self, step_type: str) -> None:
        """Reject ``grouped`` for a step which does not apply it.

        This config is shared by several step types, but only those which bind
        the grouped normalizer at their network call honor it. Without this,
        setting ``grouped`` on one of the others parses cleanly and silently
        trains with pooled constants.
        """
        if self.grouped is not None:
            raise ValueError(
                f"{step_type} does not support grouped network normalization; "
                "remove the 'grouped' block from its normalization config."
            )

    def get_network_normalizer(self, names: list[str]) -> StandardNormalizer:
        return self.network.build(names=names)

    def get_grouped_network_normalizer(
        self, names: list[str], n_spatial_dims: int = 2
    ) -> "GroupedNormalizer | None":
        if self.grouped is None:
            return None
        return self.grouped.build(
            pooled=self.network.build(names=names),
            names=names,
            n_spatial_dims=n_spatial_dims,
        )

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
        if self.grouped is not None:
            self.grouped.load()
