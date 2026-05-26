import abc
import dataclasses
from collections.abc import Mapping
from typing import Any, Literal, Self

import dacite
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


class GlobalMeanRemoval(abc.ABC):
    """Removes global means from fields before normalization and restores
    them after denormalization.

    Call sequence per step: ``forward_transform`` -> ``get_extra_channels``
    -> ``inverse_transform``.
    """

    @property
    @abc.abstractmethod
    def n_extra_input_channels(self) -> int:
        """Number of extra channels appended to the network input."""

    @abc.abstractmethod
    def forward_transform(
        self,
        input: TensorMapping,
        data_mask: TensorMapping | None,
    ) -> TensorDict:
        """Remove global means from denormalized input fields.

        Caches internal state needed by ``inverse_transform`` and
        ``get_extra_channels``.
        """

    @abc.abstractmethod
    def inverse_transform(self, output: TensorDict) -> TensorDict:
        """Restore global means on denormalized output fields."""

    @abc.abstractmethod
    def get_extra_channels(self) -> torch.Tensor | None:
        """Return ``[batch, n_extra, *spatial]`` normalized extra channels.

        Returns ``None`` when ``n_extra_input_channels == 0``.  Must be
        called after ``forward_transform``.
        """


class SharedGlobalMeanRemoval(GlobalMeanRemoval):
    """Remove a single reference field's global mean from a set of fields."""

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
        self._cached_offset: torch.Tensor | None = None
        self._cached_extra: torch.Tensor | None = None

    @property
    def n_extra_input_channels(self) -> int:
        return 1 if self._append_as_input else 0

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

        if self._append_as_input:
            normalized_mean = -offset / self._reference_std
            spatial_shape = ref.shape[1:]
            self._cached_extra = normalized_mean.view(
                -1, 1, *(1,) * len(spatial_shape)
            ).expand(-1, 1, *spatial_shape)
        else:
            self._cached_extra = None

        result = dict(input)
        for name in self._field_names:
            if name not in result:
                continue
            t = result[name]
            broadcast = offset.view(offset.shape[0], *(1,) * (t.ndim - 1))
            result[name] = t + broadcast
        return result

    def inverse_transform(self, output: TensorDict) -> TensorDict:
        if self._cached_offset is None:
            raise RuntimeError("inverse_transform() called before forward_transform().")
        offset = self._cached_offset
        result = dict(output)
        for name in self._field_names:
            if name not in result:
                continue
            t = result[name]
            broadcast = offset.view(offset.shape[0], *(1,) * (t.ndim - 1))
            result[name] = t - broadcast
        return result

    def get_extra_channels(self) -> torch.Tensor | None:
        return self._cached_extra


class PerChannelGlobalMeanRemoval(GlobalMeanRemoval):
    """Remove each field's own per-sample global mean."""

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
        self._cached_sample_means: dict[str, torch.Tensor] | None = None
        self._cached_extra: torch.Tensor | None = None

    @property
    def n_extra_input_channels(self) -> int:
        return len(self._field_names) if self._append_as_input else 0

    def forward_transform(
        self,
        input: TensorMapping,
        data_mask: TensorMapping | None,
    ) -> TensorDict:
        result = dict(input)
        sample_means: dict[str, torch.Tensor] = {}
        spatial_shape: tuple[int, ...] | None = None

        for name in self._field_names:
            if name not in result:
                continue
            t = result[name]
            if spatial_shape is None:
                spatial_shape = t.shape[1:]
            raw_mean = t.mean(dim=tuple(range(1, t.ndim)))
            if data_mask is not None and name in data_mask:
                mask = data_mask[name]
                raw_mean = torch.where(mask, raw_mean, torch.zeros_like(raw_mean))
            sample_means[name] = raw_mean
            broadcast = raw_mean.view(raw_mean.shape[0], *(1,) * (t.ndim - 1))
            result[name] = t - broadcast

        self._cached_sample_means = sample_means

        if self._append_as_input and spatial_shape is not None:
            channels: list[torch.Tensor] = []
            for name in self._field_names:
                if name not in sample_means:
                    continue
                sm = sample_means[name]
                normalized = (sm - self._means[name]) / self._stds[name]
                ch = normalized.view(-1, 1, *(1,) * len(spatial_shape)).expand(
                    -1, 1, *spatial_shape
                )
                channels.append(ch)
            self._cached_extra = torch.cat(channels, dim=1) if channels else None
        else:
            self._cached_extra = None

        return result

    def inverse_transform(self, output: TensorDict) -> TensorDict:
        if self._cached_sample_means is None:
            raise RuntimeError("inverse_transform() called before forward_transform().")
        result = dict(output)
        for name in self._field_names:
            if name not in result or name not in self._cached_sample_means:
                continue
            t = result[name]
            sm = self._cached_sample_means[name]
            broadcast = sm.view(sm.shape[0], *(1,) * (t.ndim - 1))
            result[name] = t + broadcast
        return result

    def get_extra_channels(self) -> torch.Tensor | None:
        return self._cached_extra


@dataclasses.dataclass
class GlobalMeanRemovalConfig(abc.ABC):
    """Base configuration for global mean removal."""

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        return dacite.from_dict(cls, state, dacite.Config(strict=True))

    @abc.abstractmethod
    def build(
        self,
        normalizer: StandardNormalizer,
        in_names: list[str],
    ) -> GlobalMeanRemoval: ...

    @abc.abstractmethod
    def get_n_extra_input_channels(self, in_names: list[str]) -> int: ...

    @abc.abstractmethod
    def validate_names(self, in_names: list[str], out_names: list[str]) -> None: ...


@dataclasses.dataclass
class SharedGlobalMeanRemovalConfig(GlobalMeanRemovalConfig):
    """Remove a shared reference field's global mean from specified fields.

    Parameters:
        kind: Must be ``"shared"``.
        reference_field: Name of the field whose per-sample spatial mean
            determines the offset applied to all ``field_names``.
        field_names: Names of fields to shift by the shared offset.
        append_as_input: If true, the removed sample mean (normalized) is
            appended as an extra network input channel.
    """

    kind: Literal["shared"] = "shared"
    reference_field: str = "surface_temperature"
    field_names: list[str] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_TEMPERATURE_FIELD_NAMES)
    )
    append_as_input: bool = False

    def get_n_extra_input_channels(self, in_names: list[str]) -> int:
        return 1 if self.append_as_input else 0

    def validate_names(self, in_names: list[str], out_names: list[str]) -> None:
        if self.reference_field not in in_names:
            raise ValueError(
                f"reference_field '{self.reference_field}' not in in_names: "
                f"{in_names}"
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
class PerChannelGlobalMeanRemovalConfig(GlobalMeanRemovalConfig):
    """Remove each field's own per-sample global mean.

    Parameters:
        kind: Must be ``"per_channel"``.
        field_names: Names of fields to process. ``None`` means all
            input fields.
        append_as_input: If true, each field's removed sample mean
            (normalized) is appended as an extra network input channel.
    """

    kind: Literal["per_channel"] = "per_channel"
    field_names: list[str] | None = None
    append_as_input: bool = False

    def _resolve_names(self, in_names: list[str]) -> list[str]:
        return self.field_names if self.field_names is not None else list(in_names)

    def get_n_extra_input_channels(self, in_names: list[str]) -> int:
        return len(self._resolve_names(in_names)) if self.append_as_input else 0

    def validate_names(self, in_names: list[str], out_names: list[str]) -> None:
        if self.field_names is not None:
            for name in self.field_names:
                if name not in in_names:
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
