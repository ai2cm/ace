import abc
import dataclasses
import logging
from typing import Literal

import torch

from fme.core.humidity import bolton_saturation_vapor_pressure
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


class GlobalMeanRemoval(abc.ABC):
    """Removes global means from fields before normalization and restores
    them after denormalization.

    ``forward_transform`` shifts input fields toward their climatological
    means.  ``inverse_transform`` reverses that shift on output fields.
    Fields listed in ``field_names`` that appear only in the output (e.g.
    diagnostic temperatures) are not shifted by ``forward_transform``
    (they are not inputs), but *are* un-shifted by ``inverse_transform``.
    The network learns to compensate for this through end-to-end training,
    so all listed fields — whether input, output, or both — share the
    same global-mean offset.

    Optional synthetic input channels (when ``append_as_input=True``) are
    produced by ``forward_transform`` already in *normalized* space and
    exposed via ``extras_normalized()`` as a name -> tensor mapping
    keyed by ``extra_channel_names``.  Callers append the sentinel names
    to their input packer's name list and merge the dict into the
    normalized-input dict, so the extras flow through packing and the
    channel-mask machinery uniformly with real input channels.

    Note:
        "Global mean" here refers to a *cellwise* (unweighted) spatial
        mean of each sample, not the area-weighted global mean used
        elsewhere in ACE for stats and metrics.  The two differ slightly
        on a non-uniform grid; this transform uses the cellwise mean
        for simplicity, since the network learns to compensate during
        end-to-end training.

    Call sequence per step: ``forward_transform`` -> ``extras_normalized``
    -> ``inverse_transform``.
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
    ) -> TensorDict:
        """Remove global means from denormalized input fields.

        Caches internal state needed by ``inverse_transform`` and
        ``extras_normalized``.
        """

    @abc.abstractmethod
    def inverse_transform(self, output: TensorDict) -> TensorDict:
        """Restore global means on denormalized output fields."""

    @abc.abstractmethod
    def extras_normalized(self) -> TensorDict:
        """Return the synthetic input channels in *normalized* space,
        keyed by the names in ``extra_channel_names``.

        Each value has shape ``[batch, *spatial]`` (no channel dim).
        Returns an empty dict when no extras are configured.  Must be
        called after ``forward_transform``.
        """


class NoGlobalMeanRemoval(GlobalMeanRemoval):
    """No-op implementation used when global mean removal is disabled."""

    @property
    def extra_channel_names(self) -> list[str]:
        return []

    def forward_transform(
        self,
        input: TensorMapping,
        data_mask: TensorMapping | None,
    ) -> TensorDict:
        return dict(input)

    def inverse_transform(self, output: TensorDict) -> TensorDict:
        return output

    def extras_normalized(self) -> TensorDict:
        return {}


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
        qsat_scaled_names: frozenset[str] = frozenset(),
    ):
        self._reference_field = reference_field
        self._field_names = field_names
        self._qsat_scaled_names = qsat_scaled_names
        self._append_as_input = append_as_input
        self._reference_mean = reference_mean
        self._reference_std = reference_std
        self._cached_offset: torch.Tensor | None = None
        self._cached_qsat_factor: torch.Tensor | None = None
        self._cached_extras: TensorDict = {}

    @property
    def extra_channel_names(self) -> list[str]:
        if not self._append_as_input:
            return []
        return [_extra_channel_name(self._reference_field)]

    def forward_transform(
        self,
        input: TensorMapping,
        data_mask: TensorMapping | None,
    ) -> TensorDict:
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
        self._cached_offset = offset
        spatial_shape = tuple(ref.shape[1:])

        if self._qsat_scaled_names:
            qsat_clim = bolton_saturation_vapor_pressure(self._reference_mean)
            qsat_sample = bolton_saturation_vapor_pressure(sample_mean)
            qsat_factor = qsat_clim / qsat_sample
        else:
            qsat_factor = None
        self._cached_qsat_factor = qsat_factor

        if self._append_as_input:
            normalized_mean = -offset / self._reference_std
            self._cached_extras = {
                _extra_channel_name(ref_name): _broadcast_to_spatial(
                    normalized_mean, spatial_shape
                )
            }
        else:
            self._cached_extras = {}

        result = dict(input)
        for name in self._field_names:
            if name not in result:
                continue
            t = result[name]
            broadcast = offset.view(offset.shape[0], *(1,) * (t.ndim - 1))
            result[name] = t + broadcast
        if qsat_factor is not None:
            for name in self._qsat_scaled_names:
                if name not in result:
                    continue
                t = result[name]
                broadcast = qsat_factor.view(qsat_factor.shape[0], *(1,) * (t.ndim - 1))
                result[name] = t * broadcast
        return result

    def inverse_transform(self, output: TensorDict) -> TensorDict:
        if self._cached_offset is None:
            raise RuntimeError("inverse_transform() called before forward_transform().")
        offset = self._cached_offset
        qsat_factor = self._cached_qsat_factor
        result = dict(output)
        if qsat_factor is not None:
            for name in self._qsat_scaled_names:
                if name not in result:
                    continue
                t = result[name]
                broadcast = qsat_factor.view(qsat_factor.shape[0], *(1,) * (t.ndim - 1))
                result[name] = t / broadcast
        for name in self._field_names:
            if name not in result:
                continue
            t = result[name]
            broadcast = offset.view(offset.shape[0], *(1,) * (t.ndim - 1))
            result[name] = t - broadcast
        return result

    def extras_normalized(self) -> TensorDict:
        return dict(self._cached_extras)


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
        self._cached_shifts: dict[str, torch.Tensor] | None = None
        self._cached_extras: TensorDict = {}

    @property
    def extra_channel_names(self) -> list[str]:
        if not self._append_as_input:
            return []
        return [_extra_channel_name(name) for name in self._field_names]

    def forward_transform(
        self,
        input: TensorMapping,
        data_mask: TensorMapping | None,
    ) -> TensorDict:
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

        self._cached_shifts = shifts

        extras: TensorDict = {}
        if self._append_as_input and spatial_shape is not None:
            for name in self._field_names:
                if name not in shifts:
                    continue
                normalized = -shifts[name] / self._stds[name]
                extras[_extra_channel_name(name)] = _broadcast_to_spatial(
                    normalized, spatial_shape
                )
        self._cached_extras = extras

        return result

    def inverse_transform(self, output: TensorDict) -> TensorDict:
        if self._cached_shifts is None:
            raise RuntimeError("inverse_transform() called before forward_transform().")
        result = dict(output)
        for name in self._field_names:
            if name not in result or name not in self._cached_shifts:
                continue
            t = result[name]
            sh = self._cached_shifts[name]
            broadcast = sh.view(sh.shape[0], *(1,) * (t.ndim - 1))
            result[name] = t - broadcast
        return result

    def extras_normalized(self) -> TensorDict:
        return dict(self._cached_extras)


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

    Fields in ``qsat_scaled_names`` are multiplied by
    ``qsat(reference_mean) / qsat(sample_mean_of_reference_field)`` in
    ``forward_transform`` and divided back in ``inverse_transform``,
    where ``qsat`` is the Bolton (1980) saturation vapor pressure as a
    function of temperature in Kelvin.  This is intended for humidity-
    like fields that scale with saturation vapor pressure; the reference
    field is expected to be a temperature.

    Parameters:
        kind: Must be ``"shared"``.
        reference_field: Name of the field whose per-sample cellwise
            spatial mean determines the offset applied to all
            ``field_names`` and the qsat ratio applied to all
            ``qsat_scaled_names``.
        field_names: Names of fields to shift by the shared offset.
            Fields may be inputs, outputs, or both.
        append_as_input: If true, the removed sample mean (normalized) is
            appended as an extra network input channel.
        qsat_scaled_names: Names of fields to scale by the qsat ratio.
            Fields may be inputs, outputs, or both.
    """

    kind: Literal["shared"] = "shared"
    reference_field: str = "surface_temperature"
    field_names: list[str] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_TEMPERATURE_FIELD_NAMES)
    )
    append_as_input: bool = False
    qsat_scaled_names: list[str] = dataclasses.field(default_factory=list)

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
        for name in self.qsat_scaled_names:
            if name not in all_names:
                logger.warning(
                    "global_mean_removal qsat_scaled_name '%s' is not in "
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
            qsat_scaled_names=frozenset(self.qsat_scaled_names),
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
