import abc
import dataclasses
import logging
from typing import Literal

import torch

from fme.core.normalizer import StandardNormalizer
from fme.core.typing_ import TensorDict, TensorMapping

DEFAULT_TEMPERATURE_FIELD_NAMES: list[str] = [
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


logger = logging.getLogger(__name__)


_EXTRA_CHANNEL_PREFIX = "__gmr_extra__"


def _extra_channel_name(field: str) -> str:
    return f"{_EXTRA_CHANNEL_PREFIX}{field}"


def extra_channel_source_field(name: str) -> str | None:
    """Return the source field for a GMR sentinel name, or None if not a sentinel.

    GMR synthetic input channels are named ``__gmr_extra__<source_field>``;
    they share their source field's data mask because their values are
    derived from that field and are zeroed when it is masked.
    """
    if name.startswith(_EXTRA_CHANNEL_PREFIX):
        return name[len(_EXTRA_CHANNEL_PREFIX) :]
    return None


@dataclasses.dataclass
class GlobalMeanRemovalState:
    """Opaque token produced by ``forward_transform`` and consumed by the
    same ``GlobalMeanRemoval`` instance via ``inverse_transform`` and
    ``extras_normalized``.

    Callers should thread this value from ``forward_transform`` through
    to those methods, but not read its fields directly — layout is an
    implementation detail.
    """

    shifts: dict[str, torch.Tensor]
    extras: TensorDict


class GlobalMeanRemoval(abc.ABC):
    """Removes global means from fields before normalization and restores
    them after denormalization.

    ``forward_transform`` shifts input fields toward their climatological
    means and returns a ``GlobalMeanRemovalState`` that must be passed
    back to ``inverse_transform`` to reverse the shift on output fields.
    Threading the state explicitly (rather than caching it on the
    instance) makes the transform stateless and order-independent.

    Optional synthetic input channels (when ``append_as_input=True``) are
    produced by ``forward_transform`` already in *normalized* space and
    exposed via ``extras_normalized(state)`` as a name -> tensor mapping
    keyed by ``extra_channel_names``.  Callers append the sentinel names
    to their input packer's name list and merge the dict into the
    normalized-input dict, so the extras flow through packing and the
    channel-mask machinery uniformly with real input channels.

    Fields listed in ``field_names`` that appear only in the output (e.g.
    diagnostic temperatures) are not shifted by ``forward_transform``
    (they are not inputs), but *are* un-shifted by ``inverse_transform``.
    The network learns to compensate for this through end-to-end training,
    so all listed fields — whether input, output, or both — share the
    same global-mean offset.

    Note:
        "Global mean" here refers to a *cellwise* (unweighted) spatial
        mean of each sample, not the area-weighted global mean used
        elsewhere in ACE for stats and metrics.  The two differ slightly
        on a non-uniform grid; this transform uses the cellwise mean
        for simplicity, since the network learns to compensate during
        end-to-end training.
    """

    @property
    @abc.abstractmethod
    def extra_channel_names(self) -> list[str]:
        """Sentinel names for the synthetic input channels this transform
        contributes.  Empty if no extras are configured.

        Names are prefixed with ``__gmr_extra__`` and are intended to be
        appended to the stepper's input packer so the extras flow through
        packing and channel-mask routines as ordinary inputs.
        """

    @property
    def n_extra_input_channels(self) -> int:
        return len(self.extra_channel_names)

    @abc.abstractmethod
    def forward_transform(
        self,
        input: TensorMapping,
        data_mask: TensorMapping | None,
    ) -> tuple[TensorDict, GlobalMeanRemovalState]:
        """Remove global means from denormalized input fields.

        Returns:
            ``(shifted_input, state)``.  ``state`` is an opaque value
            that must be passed back to ``inverse_transform`` to reverse
            the shift, and to ``extras_normalized`` to obtain the
            synthetic input channels.
        """

    @abc.abstractmethod
    def inverse_transform(
        self,
        output: TensorDict,
        state: GlobalMeanRemovalState,
    ) -> TensorDict:
        """Restore global means on denormalized output fields using
        ``state`` produced by ``forward_transform``.
        """

    def extras_normalized(
        self,
        state: GlobalMeanRemovalState,
    ) -> TensorDict:
        """Return the synthetic input channels in *normalized* space,
        keyed by the names in ``extra_channel_names``.

        Each value has shape ``[batch, *spatial]`` (no channel dim) so
        it can be fed to the same packer that handles real inputs.
        Returns an empty dict if no extra channels are configured.
        """
        return dict(state.extras)


class NoGlobalMeanRemoval(GlobalMeanRemoval):
    """No-op implementation used when global mean removal is disabled."""

    @property
    def extra_channel_names(self) -> list[str]:
        return []

    def forward_transform(
        self,
        input: TensorMapping,
        data_mask: TensorMapping | None,
    ) -> tuple[TensorDict, GlobalMeanRemovalState]:
        return dict(input), GlobalMeanRemovalState(shifts={}, extras={})

    def inverse_transform(
        self,
        output: TensorDict,
        state: GlobalMeanRemovalState,
    ) -> TensorDict:
        return output


def _broadcast_to_spatial(
    scalar_per_sample: torch.Tensor, spatial_shape: tuple[int, ...]
) -> torch.Tensor:
    """Broadcast a ``[batch]`` tensor to ``[batch, *spatial]``."""
    return scalar_per_sample.view(-1, *(1,) * len(spatial_shape)).expand(
        -1, *spatial_shape
    )


class SharedGlobalMeanRemoval(GlobalMeanRemoval):
    """Remove a single reference field's cellwise global mean from a set
    of fields.

    All listed fields share the same offset (derived from the reference
    field's cellwise spatial mean), regardless of whether they appear in
    the input, the output, or both.  See ``GlobalMeanRemoval`` for
    details on the asymmetric forward/inverse behavior for output-only
    fields, and on the choice of cellwise (vs. area-weighted) mean.
    """

    def __init__(
        self,
        reference_field: str,
        field_names: frozenset[str],
        append_as_input: bool,
        reference_mean: torch.Tensor,
        reference_std: torch.Tensor,
    ):
        self._reference_field = reference_field
        self._field_names = field_names
        self._append_as_input = append_as_input
        self._reference_mean = reference_mean
        self._reference_std = reference_std

    @property
    def extra_channel_names(self) -> list[str]:
        if not self._append_as_input:
            return []
        return [_extra_channel_name(self._reference_field)]

    def forward_transform(
        self,
        input: TensorMapping,
        data_mask: TensorMapping | None,
    ) -> tuple[TensorDict, GlobalMeanRemovalState]:
        ref_name = self._reference_field
        if ref_name not in input:
            raise ValueError(
                f"Reference field '{ref_name}' is not present in the input."
            )
        if data_mask is not None and ref_name in data_mask:
            if not data_mask[ref_name].all():
                raise ValueError(
                    f"Reference field '{ref_name}' is masked for some samples, "
                    "which is not supported for shared global mean removal."
                )
        ref = input[ref_name]
        sample_mean = ref.mean(dim=tuple(range(1, ref.ndim)))
        offset = self._reference_mean - sample_mean
        spatial_shape = tuple(ref.shape[1:])

        extras: TensorDict = {}
        if self._append_as_input:
            normalized_mean = -offset / self._reference_std
            extras[_extra_channel_name(ref_name)] = _broadcast_to_spatial(
                normalized_mean, spatial_shape
            )

        result = dict(input)
        for name in self._field_names:
            if name not in result:
                continue
            t = result[name]
            broadcast = offset.view(offset.shape[0], *(1,) * (t.ndim - 1))
            result[name] = t + broadcast

        # All listed fields share the same offset; inverse will un-shift
        # those that appear in the output (including output-only fields).
        shifts = {name: offset for name in self._field_names}
        return result, GlobalMeanRemovalState(shifts=shifts, extras=extras)

    def inverse_transform(
        self,
        output: TensorDict,
        state: GlobalMeanRemovalState,
    ) -> TensorDict:
        result = dict(output)
        for name, shift in state.shifts.items():
            if name not in result:
                continue
            t = result[name]
            broadcast = shift.view(shift.shape[0], *(1,) * (t.ndim - 1))
            result[name] = t - broadcast
        return result


class PerChannelGlobalMeanRemoval(GlobalMeanRemoval):
    """Shift each field's per-sample cellwise spatial mean to its climatology.

    For each field, ``forward_transform`` adds ``clim_mean - sample_mean``
    (where ``sample_mean`` is the cellwise, unweighted spatial mean) so
    the field's spatial mean equals its climatology mean in physical
    space; after normalization the field's spatial mean is approximately
    zero.  This mirrors ``SharedGlobalMeanRemoval`` but computes the
    offset per field rather than from a single reference field.  See
    ``GlobalMeanRemoval`` for the choice of cellwise (vs. area-weighted)
    mean.
    """

    def __init__(
        self,
        field_names: list[str],
        append_as_input: bool,
        means: TensorDict,
        stds: TensorDict,
    ):
        self._field_names = field_names
        self._append_as_input = append_as_input
        self._means = means
        self._stds = stds

    @property
    def extra_channel_names(self) -> list[str]:
        if not self._append_as_input:
            return []
        return [_extra_channel_name(name) for name in self._field_names]

    def forward_transform(
        self,
        input: TensorMapping,
        data_mask: TensorMapping | None,
    ) -> tuple[TensorDict, GlobalMeanRemovalState]:
        result = dict(input)
        shifts: dict[str, torch.Tensor] = {}
        spatial_shape: tuple[int, ...] | None = None

        for name in self._field_names:
            if name not in result:
                continue
            t = result[name]
            if spatial_shape is None:
                spatial_shape = tuple(t.shape[1:])
            sample_mean = t.mean(dim=tuple(range(1, t.ndim)))
            shift = self._means[name] - sample_mean
            if data_mask is not None and name in data_mask:
                mask = data_mask[name]
                shift = torch.where(mask, shift, torch.zeros_like(shift))
            shifts[name] = shift
            broadcast = shift.view(shift.shape[0], *(1,) * (t.ndim - 1))
            result[name] = t + broadcast

        extras: TensorDict = {}
        if self._append_as_input and spatial_shape is not None:
            for name in self._field_names:
                if name not in shifts:
                    continue
                normalized = -shifts[name] / self._stds[name]
                extras[_extra_channel_name(name)] = _broadcast_to_spatial(
                    normalized, spatial_shape
                )

        return result, GlobalMeanRemovalState(shifts=shifts, extras=extras)

    def inverse_transform(
        self,
        output: TensorDict,
        state: GlobalMeanRemovalState,
    ) -> TensorDict:
        result = dict(output)
        for name, shift in state.shifts.items():
            if name not in result:
                continue
            t = result[name]
            broadcast = shift.view(shift.shape[0], *(1,) * (t.ndim - 1))
            result[name] = t - broadcast
        return result


@dataclasses.dataclass
class SharedGlobalMeanRemovalConfig:
    """Remove a shared reference field's cellwise global mean from specified
    fields.

    The offset is derived from the reference field's *cellwise* (unweighted)
    spatial mean, not the area-weighted mean used elsewhere in ACE — see
    ``GlobalMeanRemoval`` for the rationale.

    Fields in ``field_names`` that appear only in the output (not the
    input) are still un-shifted by ``inverse_transform``, so the network
    learns to produce them in the offset-shifted space.  Fields that
    appear in neither input nor output are silently ignored (a warning
    is logged).

    Parameters:
        kind: Must be ``"shared"``.
        reference_field: Name of the field whose per-sample cellwise
            spatial mean determines the offset applied to all
            ``field_names``.
        field_names: Names of fields to shift by the shared offset.
            Fields may be inputs, outputs, or both.
        append_as_input: If true, the removed sample mean (normalized) is
            appended as an extra network input channel.
    """

    kind: Literal["shared"] = "shared"
    reference_field: str = "surface_temperature"
    field_names: list[str] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_TEMPERATURE_FIELD_NAMES)
    )
    append_as_input: bool = False

    def validate_names(self, in_names: list[str], out_names: list[str]) -> None:
        if self.reference_field not in in_names:
            raise ValueError(
                f"reference_field '{self.reference_field}' not in in_names: "
                f"{in_names}"
            )
        all_names = set(in_names) | set(out_names)
        for name in self.field_names:
            if name not in all_names:
                logger.warning(
                    "global_mean_removal field_name '%s' is not in "
                    "in_names or out_names and will have no effect",
                    name,
                )

    def build(
        self,
        normalizer: StandardNormalizer,
        in_names: list[str],
    ) -> SharedGlobalMeanRemoval:
        return SharedGlobalMeanRemoval(
            reference_field=self.reference_field,
            field_names=frozenset(self.field_names),
            append_as_input=self.append_as_input,
            reference_mean=normalizer.means[self.reference_field],
            reference_std=normalizer.stds[self.reference_field],
        )


@dataclasses.dataclass
class PerChannelGlobalMeanRemovalConfig:
    """Shift each field's per-sample cellwise spatial mean to its climatology.

    For each listed field, ``forward_transform`` adds
    ``clim_mean - sample_mean`` (where ``sample_mean`` is the cellwise,
    unweighted spatial mean) so that after normalization the field's
    spatial mean is approximately zero (avoiding large constant biases
    on the network input).  See ``GlobalMeanRemoval`` for the choice of
    cellwise (vs. area-weighted) mean.  Unlike
    ``SharedGlobalMeanRemovalConfig``, per-channel removal requires each
    field to be present in the input so its individual sample mean can
    be computed.

    Parameters:
        kind: Must be ``"per_channel"``.
        field_names: Names of fields to process. ``None`` means all
            input fields.  Explicit names must be input fields.
        append_as_input: If true, each field's normalized sample-mean
            anomaly ``(sample_mean - clim_mean) / clim_std`` is appended
            as an extra network input channel.
    """

    kind: Literal["per_channel"] = "per_channel"
    field_names: list[str] | None = None
    append_as_input: bool = False

    def _resolve_names(self, in_names: list[str]) -> list[str]:
        return self.field_names if self.field_names is not None else list(in_names)

    def validate_names(self, in_names: list[str], out_names: list[str]) -> None:
        if self.field_names is not None:
            all_names = set(in_names) | set(out_names)
            for name in self.field_names:
                if name not in all_names:
                    logger.warning(
                        "global_mean_removal field_name '%s' is not in "
                        "in_names or out_names and will have no effect",
                        name,
                    )
                elif name not in in_names:
                    raise ValueError(f"field_name '{name}' not in in_names: {in_names}")

    def build(
        self,
        normalizer: StandardNormalizer,
        in_names: list[str],
    ) -> PerChannelGlobalMeanRemoval:
        names = self._resolve_names(in_names)
        return PerChannelGlobalMeanRemoval(
            field_names=names,
            append_as_input=self.append_as_input,
            means={n: normalizer.means[n] for n in names},
            stds={n: normalizer.stds[n] for n in names},
        )


GlobalMeanRemovalConfigUnion = (
    SharedGlobalMeanRemovalConfig | PerChannelGlobalMeanRemovalConfig
)
