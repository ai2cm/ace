import abc
import dataclasses
from collections.abc import Callable, Mapping

# we use Type to distinguish from type attr of ModuleSelector
from typing import Any, ClassVar, Self, Type, final  # noqa: UP035

import dacite
import torch
from torch import nn

from fme.core.dataset_info import DatasetInfo
from fme.core.labels import BatchLabels, LabelEncoding

from .registry import Registry


@dataclasses.dataclass
class ModuleConfig(abc.ABC):
    """
    Builds a nn.Module given information about the input and output channels
    and dataset information.

    This is a "Config" as in practice it is a dataclass loaded directly from yaml,
    allowing us to specify details of the network architecture in a config file.
    """

    compile_unsupported_reason: ClassVar[str | None] = None
    """Why ``torch.compile`` cannot be used with the modules this builder builds.

    Set to a non-None string in a builder to declare that its modules cannot be
    compiled, e.g. because the forward pass has data-dependent shapes.
    :meth:`Module.compile` then raises ``NotImplementedError`` with this reason
    instead of letting dynamo silently fall back to eager after a long tracing
    attempt.

    This must be a ``ClassVar`` (or a plain class-level assignment): annotating
    it without ``ClassVar`` would make it a dataclass field, which would leak
    into ``ModuleSelector.config`` and into serialized checkpoints.
    """

    @abc.abstractmethod
    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        """
        Build a nn.Module given information about the input and output channels
        and the dataset.

        Args:
            n_in_channels: number of input channels
            n_out_channels: number of output channels
            dataset_info: Information about the dataset, including img_shape,
                horizontal coordinates, vertical coordinate, etc.

        Returns:
            a nn.Module
        """
        ...

    @classmethod
    @abc.abstractmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        """Remove or transform deprecated keys from a serialized config.

        Called by ``from_state`` before the config dict is loaded into a
        dataclass instance.  Must return a new dict and never mutate the
        input.  When there is nothing to remove, implement as
        ``return dict(state)``.
        """
        ...

    @classmethod
    @final
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        """Create a ModuleConfig from a serialized config dict."""
        state = cls.remove_deprecated_keys(state)
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )


CONDITIONAL_BUILDERS = [
    "NoiseConditionedSFNO",
    "LocalNet",
    "SwinTransformer",
    "NoiseConditionedSwinTransformer",
]


def compile_torch_module(module: nn.Module, **kwargs: Any) -> nn.Module:
    """Compile ``module`` with ``torch.compile``, failing loudly on recompiles.

    Sets the process-global ``torch._dynamo.config.fail_on_recompile_limit_hit``
    flag: by default, when a compiled region exceeds dynamo's recompile limit,
    dynamo silently falls back to running it in eager mode, so a configuration
    that asked for compilation quietly stops being compiled. Since compilation
    is requested explicitly, a clear error is more useful than that silent
    fallback. Note the flag is process-global, so it applies to every
    ``torch.compile``d region in the process, not just this module.

    Varying the batch size does not trip the limit: dynamo recompiles once with
    dynamic shapes and then reuses that graph, so only genuinely static
    recompiles count toward the limit.

    Args:
        module: The module to compile.
        kwargs: Forwarded to ``torch.compile``.

    Returns:
        The compiled module.
    """
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    return torch.compile(module, **kwargs)


class Module:
    """A built network together with its label encoding.

    ``module`` owns the parameters and state; ``forward_module``, when given,
    is what is actually called on the forward pass (e.g. a ``torch.compile``d
    view of ``module``). Keeping the two separate means compilation never
    changes the state dict, so checkpoints are unaffected.
    """

    def __init__(
        self,
        module: nn.Module,
        label_encoding: LabelEncoding | None,
        forward_module: nn.Module | None = None,
        compile_unsupported_reason: str | None = None,
    ):
        self._module = module
        self._label_encoding = label_encoding
        self._forward_module = forward_module
        self._compile_unsupported_reason = compile_unsupported_reason

    @property
    def forward_module(self) -> nn.Module:
        """The module actually called on the forward pass.

        This is the compiled view when :meth:`compile` has been applied, and
        the parameter-owning module otherwise.
        """
        if self._forward_module is None:
            return self._module
        return self._forward_module

    def __call__(
        self, input: torch.Tensor, labels: BatchLabels | None = None
    ) -> torch.Tensor:
        if labels is not None and self._label_encoding is None:
            raise TypeError("Labels are not allowed for unconditional models")

        if self._label_encoding is not None:
            if labels is None:
                raise TypeError("Labels are required for conditional models")
            encoded_labels = labels.conform_to_encoding(self._label_encoding)
            return self.forward_module(input, labels=encoded_labels.tensor)
        else:
            return self.forward_module(input)

    def compile(self, **kwargs: Any) -> "Module":
        """Return a Module whose forward pass runs through ``torch.compile``.

        The underlying module (parameters, state dict, ``torch_module``) is
        unchanged; only the callable used in ``__call__`` is compiled. Call
        after any distributed wrapping so the compiled graph includes it.

        Args:
            kwargs: Forwarded to ``torch.compile``.

        Raises:
            NotImplementedError: If the builder declared compilation
                unsupported via ``ModuleConfig.compile_unsupported_reason``.
        """
        if self._compile_unsupported_reason is not None:
            raise NotImplementedError(
                "torch.compile is not supported for this module: "
                f"{self._compile_unsupported_reason}"
            )
        return Module(
            self._module,
            self._label_encoding,
            forward_module=compile_torch_module(self._module, **kwargs),
            compile_unsupported_reason=self._compile_unsupported_reason,
        )

    @property
    def is_compiled(self) -> bool:
        return self._forward_module is not None

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

    def wrap_module(self, callable: Callable[[nn.Module], nn.Module]) -> "Module":
        """Wrap the underlying module (and the forward callable, if it differs).

        On a compiled Module, ``callable`` is invoked twice, once on each, and
        the forward callable's wrapper sits outside the compiled graph. That
        suits per-call wrappers such as activation checkpointing, but a
        wrapper that must be inside the graph or must wrap the parameters
        exactly once (e.g. distributed data parallel) has to be applied before
        :meth:`compile`.
        """
        forward_module = (
            callable(self._forward_module) if self._forward_module is not None else None
        )
        return Module(
            callable(self._module),
            self._label_encoding,
            forward_module,
            compile_unsupported_reason=self._compile_unsupported_reason,
        )

    def to(self, device: torch.device) -> "Module":
        if self._forward_module is not None:
            raise RuntimeError(
                "Module.to must be called before Module.compile; moving a "
                "compiled module between devices is not supported."
            )
        return Module(
            self._module.to(device),
            self._label_encoding,
            compile_unsupported_reason=self._compile_unsupported_reason,
        )


@dataclasses.dataclass
class ModuleSelector:
    """
    A dataclass containing all the information needed to build a ModuleConfig,
    including the type of the ModuleConfig and the data needed to build it.

    This is helpful as ModuleSelector can be serialized and deserialized
    without any additional information, whereas to load a ModuleConfig you
    would need to know the type of the ModuleConfig being loaded.

    It is also convenient because ModuleSelector is a single class that can be
    used to represent any ModuleConfig, whereas ModuleConfig is a protocol
    that can be implemented by many different classes.

    Parameters:
        type: the type of the ModuleConfig
        config: data for a ModuleConfig instance of the indicated type
        conditional: whether to condition the predictions on batch labels.
        allow_missing_variables: whether the data pipeline is allowed to
            produce variable masks (for incomplete datasets). When False
            (default), missing required variables cause an error.
    """

    type: str
    config: Mapping[str, Any]
    conditional: bool = False
    allow_missing_variables: bool = False
    registry: ClassVar[Registry[ModuleConfig]] = Registry[ModuleConfig]()

    def __post_init__(self):
        if not isinstance(self.registry, Registry):
            raise ValueError("ModuleSelector.registry should not be set manually")
        if self.conditional and self.type not in CONDITIONAL_BUILDERS:
            raise ValueError(
                "Conditional predictions require a conditional builder, "
                f"got {self.type} (available: {CONDITIONAL_BUILDERS})"
            )
        self._instance = self.registry.get(self.type, self.config)
        # Normalize config to include the built ModuleConfig's default values,
        # so that defaults are captured when the config is serialized (e.g.
        # logged to Weights & Biases). See issue #596.
        self.config = dataclasses.asdict(self._instance)

    @property
    def module_config(self) -> ModuleConfig:
        return self._instance

    @classmethod
    def register(
        cls, type_name: str
    ) -> Callable[[Type[ModuleConfig]], Type[ModuleConfig]]:  # noqa: UP006
        return cls.registry.register(type_name)

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> Module:
        """
        Build a nn.Module given information about the input and output channels
        and the dataset.

        Args:
            n_in_channels: number of input channels
            n_out_channels: number of output channels
            dataset_info: Information about the dataset, including img_shape
                (shape of last two dimensions of data, e.g. latitude and
                longitude), horizontal coordinates, vertical coordinate, etc.

        Returns:
            a Module object
        """
        if self.conditional and len(dataset_info.all_labels) == 0:
            raise ValueError("Conditional predictions require labels")
        if self.conditional:
            label_encoding = LabelEncoding(sorted(list(dataset_info.all_labels)))
        else:
            label_encoding = None
        module = self._instance.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            dataset_info=dataset_info,
        )
        return Module(
            module,
            label_encoding,
            compile_unsupported_reason=type(self._instance).compile_unsupported_reason,
        )

    @classmethod
    def get_available_types(cls):
        """This class method is used to expose all available types of Modules."""
        return cls.registry._types.keys()
