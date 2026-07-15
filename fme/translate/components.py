"""Named pool of translate components and its built composite container.

The translate abstraction is a *pool of named components* wrapped around a
backbone stepper. Two kinds of component live in the pool:

- **Transforms** (encoders / decoders / translation maps): trainable
  ``nn.Module``s built via :class:`ModuleSelector`, each declaring its input
  and output variable (or latent-channel) name sets. The number of input and
  output channels is ``len(in_names)`` and ``len(out_names)``.
- **Backbones**: full ace :class:`Stepper`s (normalization, ocean, corrector,
  derived variables intact), either built fresh from a :class:`StepperConfig`
  or sourced from a checkpoint via :class:`CheckpointStepperConfig`.

The built composite (:class:`ComponentPool`) holds the components, exposes a
flattened ``.modules`` (mirroring
:class:`fme.coupled.stepper.CoupledStepper`), fans out train/eval/epoch
control, and implements ``get_state``/``from_state``/``load_state``.

Per-component freezing (:class:`FrozenParameterConfig`) and name-matched
partial checkpoint initialization (:class:`ParameterInitializationConfig` ->
``overwrite_weights``) are supported. Freezing *must* happen before DDP
wrapping, because ``DistributedDataParallel`` registers gradient hooks at wrap
time based on ``requires_grad``. This ordering is enforced by threading a
:class:`ParameterInitializer` into ``StepperConfig.get_stepper`` for backbones
(freeze happens inside step construction, pre-wrap) and by freezing the raw
module before ``wrap_module`` for transforms.
"""

import dataclasses
from typing import Any

import dacite
import torch
from torch import nn

from fme.ace.stepper.parameter_init import (
    ParameterInitializationConfig,
    StepperWeightsAndHistory,
    WeightsAndHistoryLoader,
    null_weights_and_history,
)
from fme.ace.stepper.single_module import (
    CheckpointStepperConfig,
    Stepper,
    StepperConfig,
    load_stepper,
    load_weights_and_history,
)
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed
from fme.core.registry.module import Module, ModuleSelector
from fme.core.training_history import TrainingHistory


def load_module_weights(path: str | None) -> StepperWeightsAndHistory:
    """Load a single module's ``state_dict`` for name-matched initialization.

    The file at ``path`` must be a ``torch.save``-d module ``state_dict``
    (a mapping of parameter name to tensor). Its keys must be a subset of the
    destination module's ``state_dict`` keys; ``overwrite_weights`` matches by
    name and, where a destination axis is larger, overwrites its initial slice.

    Returns ``(None, TrainingHistory())`` when ``path`` is None so an unset
    ``weights_path`` is a no-op.
    """
    if path is None:
        return null_weights_and_history()
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    return [state_dict], TrainingHistory()


def _stepper_weights_loader(stepper: Stepper) -> WeightsAndHistoryLoader:
    """A weights loader that pulls each module's ``state_dict`` from a stepper.

    Mirrors ``fme.coupled.stepper``'s checkpoint-sourced loader: it lets a
    donor stepper's weights flow through the parameter-init path so that
    freezing can be threaded through ``get_stepper`` (pre-wrap), rather than
    freezing after ``load_stepper`` has already DDP-wrapped an unfrozen model.
    """

    def load(*_: Any) -> StepperWeightsAndHistory:
        return [module.state_dict() for module in stepper.modules], (
            stepper.training_history
        )

    return load


@dataclasses.dataclass
class TransformConfig:
    """Configuration for a trainable transform / encoder / decoder component.

    Parameters:
        module: The module builder. Built with ``n_in_channels=len(in_names)``
            and ``n_out_channels=len(out_names)``.
        in_names: Input variable (or latent-channel) names.
        out_names: Output variable (or latent-channel) names.
        parameter_init: Freezing and name-matched weight initialization for this
            transform. ``weights_path`` (if set) points to a module
            ``state_dict`` file loaded by :func:`load_module_weights`; the
            ``parameters[0].frozen`` config selects which parameters to freeze.
    """

    module: ModuleSelector
    in_names: list[str]
    out_names: list[str]
    parameter_init: ParameterInitializationConfig = dataclasses.field(
        default_factory=ParameterInitializationConfig
    )

    def _build_raw(self, dataset_info: DatasetInfo) -> Module:
        return self.module.build(
            n_in_channels=len(self.in_names),
            n_out_channels=len(self.out_names),
            dataset_info=dataset_info,
        )

    def build(self, dataset_info: DatasetInfo) -> Module:
        """Build the transform, applying weight init and freezing before wrap.

        Order: build the raw module, overwrite a subset of its weights from the
        configured checkpoint (name-matched), freeze the configured parameters,
        then DDP-wrap. Freezing precedes wrapping so a fully-frozen module is
        wrapped as a ``DummyWrapper`` with no live-gradient expectations.
        """
        module = self._build_raw(dataset_info)
        initializer = self.parameter_init.build(load_module_weights)
        # Both operate on the raw (pre-wrap) module, so state-dict keys carry
        # no "module." prefix and match an unwrapped donor's names directly.
        initializer.apply_weights([module.torch_module])
        initializer.freeze_weights([module.torch_module])
        return module.wrap_module(Distributed.get_instance().wrap_module)

    def build_for_load(self, dataset_info: DatasetInfo) -> Module:
        """Build a wrapped transform without weight init or freezing.

        Used when reconstructing from a saved state: the weights are restored by
        a subsequent ``load_state``, so external initialization is skipped and,
        as with ``Stepper.from_state``, the reloaded module is left unfrozen.
        """
        module = self._build_raw(dataset_info)
        return module.wrap_module(Distributed.get_instance().wrap_module)


@dataclasses.dataclass
class BackboneConfig:
    """Configuration for a backbone stepper component.

    The stepper *configuration* is either built fresh (``stepper``) or read from
    a checkpoint (``checkpoint``); exactly one must be provided. Freezing and
    name-matched weight initialization are threaded through ``get_stepper`` via
    ``parameter_init`` so that freezing happens before DDP wrapping.

    When ``checkpoint`` is set and ``init_from_checkpoint`` is True (the default,
    and the transfer-learning arm's central case of a frozen SHiELD+ donor), the
    donor's weights are sourced through the parameter-init path: the config comes
    from ``CheckpointStepperConfig.to_stepper_config()`` and the weights from the
    loaded stepper's modules. This lets a frozen donor be built pre-wrap, so a
    fully-frozen backbone is wrapped as a ``DummyWrapper`` rather than a
    ``DistributedDataParallel`` expecting gradients.

    Set ``init_from_checkpoint`` False to build fresh (random) weights over a
    checkpoint-sourced config. For a non-frozen inference/fine-tune backbone,
    ``load_stepper`` (used directly elsewhere) remains the convenience path.

    Parameters:
        stepper: A fresh stepper configuration. Mutually exclusive with
            ``checkpoint``.
        checkpoint: A checkpoint to source the stepper configuration from.
            Mutually exclusive with ``stepper``.
        parameter_init: Freezing and (for the fresh-config case) weight
            initialization. When ``checkpoint`` is set and
            ``init_from_checkpoint`` is True, ``weights_path`` must be None
            because the weights come from the checkpoint.
        init_from_checkpoint: Whether to source weights from ``checkpoint`` via
            the parameter-init path. Only meaningful when ``checkpoint`` is set.
    """

    stepper: StepperConfig | None = None
    checkpoint: CheckpointStepperConfig | None = None
    parameter_init: ParameterInitializationConfig = dataclasses.field(
        default_factory=ParameterInitializationConfig
    )
    init_from_checkpoint: bool = True

    def __post_init__(self):
        if (self.stepper is None) == (self.checkpoint is None):
            raise ValueError(
                "Exactly one of 'stepper' or 'checkpoint' must be provided "
                "for a BackboneConfig."
            )
        if self.checkpoint is not None and self.init_from_checkpoint:
            if self.parameter_init.weights_path is not None:
                raise ValueError(
                    "When sourcing backbone weights from 'checkpoint', "
                    "parameter_init.weights_path must be None (the weights come "
                    "from the checkpoint)."
                )

    def _config_and_loader(self) -> tuple[StepperConfig, WeightsAndHistoryLoader]:
        if self.stepper is not None:
            return self.stepper, load_weights_and_history
        assert self.checkpoint is not None  # guaranteed by __post_init__
        config = self.checkpoint.to_stepper_config()
        if self.init_from_checkpoint:
            donor = load_stepper(self.checkpoint.checkpoint_path)
            return config, _stepper_weights_loader(donor)
        return config, load_weights_and_history

    def build(self, dataset_info: DatasetInfo) -> Stepper:
        """Build the backbone stepper with freezing applied before DDP wrapping.

        Freezing and weight initialization are carried by the
        :class:`ParameterInitializer` threaded into ``get_stepper``; the stepper
        applies the freeze inside step construction (pre-wrap).
        """
        config, loader = self._config_and_loader()
        initializer = self.parameter_init.build(loader)
        return config.get_stepper(
            dataset_info=dataset_info,
            parameter_initializer=initializer,
        )


@dataclasses.dataclass
class ComponentPoolConfig:
    """Configuration for a named pool of translate components.

    Parameters:
        transforms: Named trainable transform / encoder / decoder components.
        backbones: Named backbone steppers.
    """

    transforms: dict[str, TransformConfig] = dataclasses.field(default_factory=dict)
    backbones: dict[str, BackboneConfig] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        overlap = set(self.transforms).intersection(self.backbones)
        if overlap:
            raise ValueError(
                f"Component names must be unique across transforms and "
                f"backbones; found duplicates: {sorted(overlap)}."
            )

    def build(self, dataset_info: DatasetInfo) -> "ComponentPool":
        """Build every component against ``dataset_info``.

        Note: a single ``dataset_info`` (grid) is shared by all components in
        this version; per-component grids for resolution-changing operators are
        a later extension.
        """
        transforms = {
            name: config.build(dataset_info) for name, config in self.transforms.items()
        }
        backbones = {
            name: config.build(dataset_info) for name, config in self.backbones.items()
        }
        return ComponentPool(
            config=self,
            transforms=transforms,
            backbones=backbones,
            dataset_info=dataset_info,
        )

    def get_state(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "ComponentPoolConfig":
        return dacite.from_dict(
            data_class=cls,
            data=state,
            config=dacite.Config(
                strict=True,
                type_hooks={
                    StepperConfig: StepperConfig.from_state,
                },
            ),
        )


class ComponentPool:
    """The built composite of a translate component pool.

    Holds the built transforms and backbones, exposes them for training via a
    flattened ``.modules`` list, fans out train/eval/epoch control, and
    round-trips its full state (config + weights) via
    ``get_state``/``from_state``/``load_state``.

    This container is deliberately minimal: it builds, holds, exposes modules,
    and saves/loads state. It has no forward/predict/objective logic and is not
    yet ``Stepper``-compatible.
    """

    def __init__(
        self,
        config: ComponentPoolConfig,
        transforms: dict[str, Module],
        backbones: dict[str, Stepper],
        dataset_info: DatasetInfo,
    ):
        self._config = config
        self._transforms = transforms
        self._backbones = backbones
        self._dataset_info = dataset_info

    @property
    def transforms(self) -> dict[str, Module]:
        return self._transforms

    @property
    def backbones(self) -> dict[str, Stepper]:
        return self._backbones

    @property
    def modules(self) -> nn.ModuleList:
        """All trainable modules in the pool, as a flattened ``nn.ModuleList``.

        Transforms contribute their wrapped torch module; backbones contribute
        every module in their own ``.modules``. Ordering is transforms first
        (in insertion order), then backbones.
        """
        result: list[nn.Module] = [
            transform.torch_module for transform in self._transforms.values()
        ]
        for backbone in self._backbones.values():
            result.extend(backbone.modules)
        return nn.ModuleList(result)

    def set_train(self) -> None:
        for transform in self._transforms.values():
            transform.torch_module.train()
        for backbone in self._backbones.values():
            backbone.set_train()

    def set_eval(self) -> None:
        for transform in self._transforms.values():
            transform.torch_module.eval()
        for backbone in self._backbones.values():
            backbone.set_eval()

    def set_epoch(self, epoch: int) -> None:
        # Transforms have no epoch-dependent state; only backbones react.
        for backbone in self._backbones.values():
            backbone.set_epoch(epoch)

    def get_state(self) -> dict[str, Any]:
        return {
            "config": self._config.get_state(),
            "dataset_info": self._dataset_info.get_state(),
            "transforms": {
                name: transform.get_state()
                for name, transform in self._transforms.items()
            },
            "backbones": {
                name: backbone.get_state() for name, backbone in self._backbones.items()
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        for name, transform in self._transforms.items():
            transform.load_state(state["transforms"][name])
        for name, backbone in self._backbones.items():
            backbone.load_state(state["backbones"][name])

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "ComponentPool":
        """Rebuild a pool from saved state.

        Transforms are rebuilt from the pool config against the saved
        ``dataset_info`` (without external init or freezing) and their weights
        restored; backbones are rebuilt via the self-contained
        ``Stepper.from_state`` (its checkpoint path need not still exist).
        """
        config = ComponentPoolConfig.from_state(state["config"])
        dataset_info = DatasetInfo.from_state(state["dataset_info"])
        transforms = {
            name: transform_config.build_for_load(dataset_info)
            for name, transform_config in config.transforms.items()
        }
        for name, transform in transforms.items():
            transform.load_state(state["transforms"][name])
        backbones = {
            name: Stepper.from_state(backbone_state)
            for name, backbone_state in state["backbones"].items()
        }
        return cls(
            config=config,
            transforms=transforms,
            backbones=backbones,
            dataset_info=dataset_info,
        )
