import dataclasses
import logging
import pathlib
from collections.abc import Callable, Iterable, Mapping
from copy import copy
from typing import Protocol

import numpy as np
import torch

from fme.core.cloud import open_dataset_via_inter_filesystem_copy
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

    @property
    def fills_nans(self) -> bool:
        """Whether NaN filling is enabled on normalize or denormalize."""
        return self.fill_nans_on_normalize or self.fill_nans_on_denormalize

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

    def _scalar_constants(self) -> tuple[dict[str, float], dict[str, float]]:
        """Means and stds as floats, for serialization.

        Raises if a constant is per-sample, as in a normalizer bound by
        ``GroupedNormalizer.bind``: that is batch state, and the grouped
        configuration is what gets serialized instead.
        """
        for name, constants in (("means", self.means), ("stds", self.stds)):
            per_sample = sorted(k for k, v in constants.items() if v.numel() != 1)
            if per_sample:
                raise ValueError(
                    f"Cannot serialize per-sample normalization {name} for "
                    f"{per_sample}; serialize the grouped normalization "
                    "configuration instead."
                )
        means = {k: float(v.cpu().numpy().item()) for k, v in self.means.items()}
        stds = {k: float(v.cpu().numpy().item()) for k, v in self.stds.items()}
        return means, stds

    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        means, stds = self._scalar_constants()
        return {
            "means": means,
            "stds": stds,
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
        means, stds = self._scalar_constants()
        return NormalizationConfig(
            means=means,
            stds=stds,
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
            # broadcast_to, not full_like: a grouped normalizer's means are
            # per-sample tensors, not scalars.
            fill = torch.broadcast_to(means[k].to(v.dtype), v.shape)
            denormalized[k] = torch.where(torch.isnan(v), fill, v)
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
    ds = open_dataset_via_inter_filesystem_copy(path, mask_and_scale=False)

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
    loss and the aggregators' normalized metrics continue to use the pooled
    constants, so those quantities remain comparable across models trained
    with different grouping strategies.
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
        self._stacked_means = self._stack("means")
        self._stacked_stds = self._stack("stds")
        self._raise_on_degenerate_group_stds()
        default_index = self._group_names.index(default_label)
        self._default_normalizer = self._select(lambda t: t[default_index])
        # Caches bind() per labels object, since resolving groups forces a device sync.
        self._bind_cache: tuple[BatchLabels, StandardNormalizer] | None = None
        self._warned_default_label = False

    def _stack(self, attr: str) -> TensorDict:
        """Stack each per-group variable's constants into a [n_groups] tensor.

        The stack is ordered by ``self._group_names``, so a group index
        computed against that ordering indexes it directly.
        """
        return {
            name: torch.stack(
                [
                    getattr(self._groups[group], attr)[name]
                    for group in self._group_names
                ]
            ).to(get_device())
            for name in self._per_group_names
        }

    def _raise_on_degenerate_group_stds(self) -> None:
        """Reject a non-pinned variable that is ~constant within some group.

        ``validate_pinned_variables`` only catches a typo in a pinned name;
        this catches a variable that should have been pinned but was not, e.g.
        a global-mean CO2 which is fixed within each dataset.
        """
        for name in sorted(self._per_group_names):
            pooled_std = self._pooled.stds[name].to(get_device())
            if not (pooled_std > 0):
                # Constant in the pooled stats too; pinning can't fix it.
                continue
            ratio = self._stacked_stds[name] / pooled_std
            degenerate = [
                group_name
                for group_name, is_degenerate in zip(
                    self._group_names,
                    (~(ratio >= _MIN_GROUP_TO_POOLED_STD_RATIO)).tolist(),
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
        self, select: Callable[[torch.Tensor], torch.Tensor]
    ) -> StandardNormalizer:
        """Normalizer with ``select`` applied to each stacked per-group
        constant, and the pooled constants for pinned variables.
        """
        means = dict(self._pooled.means)
        stds = dict(self._pooled.stds)
        for name in self._per_group_names:
            means[name] = select(self._stacked_means[name])
            stds[name] = select(self._stacked_stds[name])
        return StandardNormalizer(
            means=means,
            stds=stds,
            fill_nans_on_normalize=self._pooled.fill_nans_on_normalize,
            fill_nans_on_denormalize=self._pooled.fill_nans_on_denormalize,
        )

    def bind(self, labels: BatchLabels | None) -> StandardNormalizer:
        """
        Resolve per-sample constants into a normalizer for a single batch.

        The returned normalizer holds means and stds of shape
        ``[n_samples, *(1,) * n_spatial_dims]`` for non-pinned variables, which
        broadcast against the ``[n_batch, *spatial]`` tensors the step operates
        on. Pinned variables keep their scalar pooled constants.

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
        per_sample_shape = (-1, *(1,) * self._n_spatial_dims)
        normalizer = self._select(lambda t: t[group_index].reshape(per_sample_shape))
        self._bind_cache = (labels, normalizer)
        return normalizer

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
            except KeyError as err:
                raise ValueError(
                    f"Normalization group '{group_name}' has no constants for "
                    f"variable {err}. Every group must provide constants for "
                    "every non-pinned variable."
                ) from err
        return GroupedNormalizer(
            pooled=pooled,
            groups=groups,
            default_label=self.default_label,
            pinned_names=self.pinned_variables,
            n_spatial_dims=n_spatial_dims,
        )

    def load(self):
        for group in self.groups.values():
            group.load()


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
            selected per sample from the sample's labels, as is global mean
            removal, while ``network`` supplies the pooled constants used for
            pinned variables and for every other consumer of normalization
            constants.
    """

    network: NormalizationConfig
    loss: NormalizationConfig | None = None
    residual: NormalizationConfig | None = None
    grouped: GroupedNormalizationConfig | None = None

    def __post_init__(self):
        if self.loss is not None and self.residual is not None:
            raise ValueError("Cannot provide both loss and residual normalization.")

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
