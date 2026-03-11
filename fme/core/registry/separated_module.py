import abc
import dataclasses
from collections.abc import Callable, Mapping

# we use Type to distinguish from type attr of SeparatedModuleSelector
from typing import Any, ClassVar, Type  # noqa: UP035

import dacite
import torch
from torch import nn

from fme.core.dataset_info import DatasetInfo
from fme.core.labels import BatchLabels, LabelEncoding

from .registry import Registry


@dataclasses.dataclass
class SeparatedModuleConfig(abc.ABC):
    """
    Builds an nn.Module that takes separate forcing and prognostic tensors
    as input, and returns separate prognostic and diagnostic tensors as output.

    The built nn.Module must have the forward signature::

        forward(
            forcing: Tensor,
            prognostic: Tensor,
            labels: Tensor | None = None,
        ) -> tuple[Tensor, Tensor]

    where the return value is (prognostic_out, diagnostic_out).
    """

    @abc.abstractmethod
    def build(
        self,
        n_forcing_channels: int,
        n_prognostic_channels: int,
        n_diagnostic_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        """
        Build a nn.Module with separated forcing/prognostic/diagnostic channels.

        Args:
            n_forcing_channels: number of input-only (forcing) channels
            n_prognostic_channels: number of input-output (prognostic) channels
            n_diagnostic_channels: number of output-only (diagnostic) channels
            dataset_info: Information about the dataset, including img_shape,
                horizontal coordinates, vertical coordinate, etc.

        Returns:
            An nn.Module whose forward method takes
            (forcing, prognostic, labels=None) and returns
            (prognostic_out, diagnostic_out) tensors.
        """
        ...

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "SeparatedModuleConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )


class SeparatedModule:
    """
    Wrapper around an nn.Module with separated channel interface.

    The wrapped module takes (forcing, prognostic) tensors and returns
    (prognostic_out, diagnostic_out) tensors. This wrapper handles
    optional label encoding for conditional models.
    """

    def __init__(self, module: nn.Module, label_encoding: LabelEncoding | None):
        self._module = module
        self._label_encoding = label_encoding

    def __call__(
        self,
        forcing: torch.Tensor,
        prognostic: torch.Tensor,
        labels: BatchLabels | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if labels is not None and self._label_encoding is not None:
            encoded_labels = labels.conform_to_encoding(self._label_encoding)
            return self._module(forcing, prognostic, labels=encoded_labels.tensor)
        else:
            return self._module(forcing, prognostic, labels=None)

    @property
    def torch_module(self) -> nn.Module:
        return self._module

    def get_state(self) -> dict[str, Any]:
        if self._label_encoding is not None:
            label_encoder_state = self._label_encoding.get_state()
        else:
            label_encoder_state = None
        return {
            **self._module.state_dict(),
            "label_encoding": label_encoder_state,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        state = state.copy()
        if state.get("label_encoding") is not None:
            if self._label_encoding is None:
                self._label_encoding = LabelEncoding.from_state(
                    state.pop("label_encoding")
                )
            else:
                self._label_encoding.conform_to_state(state.pop("label_encoding"))
        state.pop("label_encoding", None)
        self._module.load_state_dict(state)

    def wrap_module(
        self, callable: Callable[[nn.Module], nn.Module]
    ) -> "SeparatedModule":
        return SeparatedModule(callable(self._module), self._label_encoding)

    def to(self, device: torch.device) -> "SeparatedModule":
        return SeparatedModule(self._module.to(device), self._label_encoding)


@dataclasses.dataclass
class SeparatedModuleSelector:
    """
    A dataclass for building SeparatedModuleConfig instances from config dicts.

    Mirrors ModuleSelector but for the separated channel interface.

    Parameters:
        type: the type of the SeparatedModuleConfig
        config: data for a SeparatedModuleConfig instance of the indicated type
    """

    type: str
    config: Mapping[str, Any]
    registry: ClassVar[Registry[SeparatedModuleConfig]] = Registry[
        SeparatedModuleConfig
    ]()

    def __post_init__(self):
        if not isinstance(self.registry, Registry):
            raise ValueError(
                "SeparatedModuleSelector.registry should not be set manually"
            )
        self._instance = self.registry.get(self.type, self.config)

    @property
    def module_config(self) -> SeparatedModuleConfig:
        return self._instance

    @classmethod
    def register(
        cls, type_name: str
    ) -> Callable[[Type[SeparatedModuleConfig]], Type[SeparatedModuleConfig]]:  # noqa: UP006
        return cls.registry.register(type_name)

    def build(
        self,
        n_forcing_channels: int,
        n_prognostic_channels: int,
        n_diagnostic_channels: int,
        dataset_info: DatasetInfo,
    ) -> SeparatedModule:
        """
        Build a SeparatedModule with separated forcing/prognostic/diagnostic
        channels.

        Args:
            n_forcing_channels: number of input-only (forcing) channels
            n_prognostic_channels: number of input-output (prognostic) channels
            n_diagnostic_channels: number of output-only (diagnostic) channels
            dataset_info: Information about the dataset, including img_shape.

        Returns:
            a SeparatedModule object
        """
        if len(dataset_info.all_labels) > 0:
            label_encoding = LabelEncoding(sorted(list(dataset_info.all_labels)))
        else:
            label_encoding = None
        module = self._instance.build(
            n_forcing_channels=n_forcing_channels,
            n_prognostic_channels=n_prognostic_channels,
            n_diagnostic_channels=n_diagnostic_channels,
            dataset_info=dataset_info,
        )
        return SeparatedModule(module, label_encoding)

    @classmethod
    def get_available_types(cls):
        return cls.registry._types.keys()


class _SimpleSeparatedModule(nn.Module):
    """A trivial module for testing the separated interface."""

    def __init__(self, n_forcing: int, n_prognostic: int, n_diagnostic: int):
        super().__init__()
        n_in = n_forcing + n_prognostic
        self.prog_linear = nn.Linear(n_in, n_prognostic, bias=False)
        self.diag_linear = nn.Linear(n_in, n_diagnostic, bias=False)

    def forward(
        self,
        forcing: torch.Tensor,
        prognostic: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([forcing, prognostic], dim=-3)
        b, c, h, w = combined.shape
        flat = combined.permute(0, 2, 3, 1).reshape(-1, c)
        prog_out = self.prog_linear(flat).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        diag_out = self.diag_linear(flat).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return prog_out, diag_out


def register_test_types() -> None:
    """Register test-only module types. Call from tests before use."""

    @SeparatedModuleSelector.register("test_simple")
    @dataclasses.dataclass
    class _SimpleSeparatedBuilder(SeparatedModuleConfig):
        def build(
            self,
            n_forcing_channels,
            n_prognostic_channels,
            n_diagnostic_channels,
            dataset_info,
        ):
            return _SimpleSeparatedModule(
                n_forcing_channels, n_prognostic_channels, n_diagnostic_channels
            )
