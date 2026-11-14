import contextlib
import dataclasses
import itertools
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import torch
from torch import nn

from fme.core.device import get_device
from fme.core.generics.optimization import OptimizationABC, OptimizationModules
from fme.core.muon import ADJUST_LR_FNS as MUON_ADJUST_LR_FNS
from fme.core.muon import Muon, check_muon_state_dict_compatible
from fme.core.scheduler import LRScheduler, SchedulerConfig, SequentialSchedulerConfig
from fme.core.typing_ import TensorDict, TensorMapping


class Checkpoint:
    def __init__(self, kwargs: Mapping[str, Any]):
        self._kwargs = kwargs

    def __call__(self, module: nn.Module):
        def wrapped(*args):
            return torch.utils.checkpoint.checkpoint(
                module,
                *args,
                use_reentrant=False,
                **self._kwargs,
            )

        return wrapped


class NoCheckpoint:
    def __call__(self, module: nn.Module):
        return module


@dataclasses.dataclass
class CheckpointConfig:
    """
    Configuration for activation checkpointing.

    Trades increased computation in exchange for lowered memory consumption during
    training by recomputing activations in the backward pass.

    Parameters:
        after_n_forward_steps: Number of forward steps to generate before activation
            checkpointing is applied. Activation checkpointing is not used unless this
            number is less than the number of forward steps in the optimization.
        kwargs: Keyword arguments to pass to torch.utils.checkpoint.checkpoint.
            Note that use_reentrant=False is always explicitly passed
            as is recommended by the docs.
    """

    after_n_forward_steps: float = np.inf
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def build(self, step: int) -> Checkpoint | NoCheckpoint:
        """
        Builds a checkpoint function.

        Args:
            step: The current zero-indexed step number.

        Returns:
            A checkpoint function.
        """
        if step >= self.after_n_forward_steps:
            return Checkpoint(self.kwargs)
        else:
            return NoCheckpoint()


_MUON_KWARGS = frozenset(
    {
        "momentum",
        "nesterov",
        "weight_decay",
        "ns_steps",
        "adjust_lr_fn",
        "adamw_lr",
        "adamw_betas",
        "adamw_eps",
        "adamw_weight_decay",
    }
)


def _split_muon_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    adamw_names: Sequence[str],
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split parameters into (Muon, AdamW) lists.

    Parameters with ``ndim >= 2`` go to Muon unless their name contains one of
    ``adamw_names``; all others go to AdamW.

    Raises:
        ValueError: If an entry of ``adamw_names`` matches no parameter name.
    """
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    unmatched = set(adamw_names)
    for name, param in named_parameters:
        matches = {pattern for pattern in adamw_names if pattern in name}
        unmatched -= matches
        if param.ndim >= 2 and not matches:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    if unmatched:
        raise ValueError(
            "OptimizationConfig.adamw_names entries matched no parameter "
            f"name: {sorted(unmatched)}"
        )
    return muon_params, adamw_params


def _build_optimizer(
    optimizer_type: Literal["Adam", "FusedAdam", "AdamW", "Muon"],
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    lr: float,
    kwargs: Mapping[str, Any],
    adamw_names: Sequence[str] = (),
) -> torch.optim.Optimizer:
    if optimizer_type == "Muon":
        muon_params, adamw_params = _split_muon_parameters(
            named_parameters, adamw_names
        )
        return Muon(muon_params, adamw_params, lr=lr, **kwargs)
    parameters = [param for _, param in named_parameters]
    if optimizer_type == "FusedAdam":
        return torch.optim.AdamW(parameters, lr=lr, fused=True, **kwargs)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(parameters, lr=lr, **kwargs)
    elif optimizer_type == "AdamW":
        return torch.optim.AdamW(parameters, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def flatten_optimization_modules(modules: OptimizationModules) -> nn.ModuleList:
    """Concatenate per-component module lists in key order; a module list is
    returned unchanged.
    """
    if isinstance(modules, Mapping):
        return nn.ModuleList(
            [module for component in modules.values() for module in component]
        )
    return modules


def _step_scheduler(
    scheduler: LRScheduler, valid_loss: float | None, is_iteration: bool
) -> None:
    if scheduler.should_step(is_iteration):
        try:
            if valid_loss is not None:
                scheduler.step(metrics=valid_loss)
            else:
                scheduler.step()
        except TypeError:
            # Some schedulers don't accept metrics argument
            scheduler.step()


def _load_optimizer_state_for_finetuning(
    optimizer: torch.optim.Optimizer, optimizer_state_dict: dict[str, Any]
) -> None:
    """Load per-parameter running state, keeping ``optimizer``'s per-group
    hyperparameters. See ``Optimization.load_optimizer_state_for_finetuning``.
    """
    check_muon_state_dict_compatible(optimizer, optimizer_state_dict)
    fresh_hparams = [
        {k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups
    ]
    try:
        optimizer.load_state_dict(optimizer_state_dict)
    except ValueError as e:
        raise ValueError(
            "Failed to load optimizer state for fine-tuning: parameter "
            "groups in the checkpoint are incompatible with the "
            "freshly-built optimizer (e.g. group count or per-group "
            "parameter count mismatch). This typically indicates the "
            "model architecture or trainable-parameter set changed "
            "between the source checkpoint and the current run. "
            f"Underlying error: {e}"
        ) from e
    for group, hparams in zip(optimizer.param_groups, fresh_hparams):
        for k in list(group.keys()):
            if k != "params":
                del group[k]
        group.update(hparams)


class Optimization(OptimizationABC):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        enable_automatic_mixed_precision: bool,
        use_gradient_accumulation: bool = False,
        get_checkpoint: Callable[
            [int], Checkpoint | NoCheckpoint
        ] = lambda _: NoCheckpoint(),
        max_grad_norm: float | None = None,
    ):
        self.optimizer = optimizer
        if enable_automatic_mixed_precision:
            self.gscaler: torch.amp.GradScaler | None = torch.amp.GradScaler("cuda")
        else:
            self.gscaler = None
        self.scheduler = scheduler
        self._accumulated_loss = torch.tensor(0.0, device=get_device())
        self._use_gradient_accumulation = use_gradient_accumulation
        self._get_checkpoint = get_checkpoint
        self._max_grad_norm = max_grad_norm
        self._last_grad_norm: float | None = None

    def checkpoint(self, module: nn.Module, step: int) -> nn.Module:
        return self._get_checkpoint(step)(module)

    @contextlib.contextmanager
    def autocast(self):
        enabled = self.gscaler is not None
        dtype = torch.bfloat16 if enabled else None
        with torch.amp.autocast("cuda", enabled=enabled, dtype=dtype):
            yield

    @property
    def learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def set_mode(self, modules: nn.ModuleList):
        """
        Sets the mode of the module to train.
        """
        for m in modules:
            m.train()

    def step_scheduler(
        self,
        valid_loss: float | None = None,
        is_iteration: bool = False,
    ):
        """
        Step the scheduler.

        Args:
            valid_loss: The validation loss. Used in schedulers which change the
                learning rate based on whether the validation loss is decreasing.
                If None, this indicates the call is from within a training iteration
                rather than at the end of an epoch.
            is_iteration: Whether the step is called from a training iteration or at
                the end of an epoch. Default is epoch.
        """
        _step_scheduler(self.scheduler, valid_loss, is_iteration)

    def detach_if_using_gradient_accumulation(self, state: TensorMapping) -> TensorDict:
        if self._use_gradient_accumulation:
            return {k: v.detach() for k, v in state.items()}
        return dict(state)

    def accumulate_loss(self, loss: torch.Tensor):
        self._validate_loss(loss)
        self._accumulated_loss += loss
        if self._use_gradient_accumulation:
            self._backward(loss)

    def get_accumulated_loss(self) -> torch.Tensor:
        return self._accumulated_loss

    def _backward(self, loss: torch.Tensor):
        if self.gscaler is not None:
            self.gscaler.scale(loss).backward()
        else:
            loss.backward()

    def _clip_gradients(self):
        self._last_grad_norm = None
        if self._max_grad_norm is not None:
            if self.gscaler is not None:
                self.gscaler.unscale_(self.optimizer)
            params = itertools.chain.from_iterable(
                group["params"] for group in self.optimizer.param_groups
            )
            self._last_grad_norm = torch.nn.utils.clip_grad_norm_(
                params, self._max_grad_norm
            ).item()

    def _step_weights(self):
        if self.gscaler is not None:
            self.gscaler.step(self.optimizer)
        else:
            self.optimizer.step()

    def step_weights(self):
        if not self._use_gradient_accumulation:
            self._backward(self._accumulated_loss)
        self._clip_gradients()
        self._step_weights()
        self.optimizer.zero_grad()
        if self.gscaler is not None:
            self.gscaler.update()
        self._accumulated_loss = torch.tensor(0.0, device=get_device())

    def set_learning_rate(self, lr: float):
        """Set the learning rate of the first parameter group to ``lr``.

        Any other parameter groups (e.g. Muon's AdamW group) are rescaled by
        the same factor, preserving their ratio to the first group. If the
        first group's learning rate is zero, every group is set to ``lr``.
        """
        current = self.optimizer.param_groups[0]["lr"]
        for param_group in self.optimizer.param_groups:
            if current == 0:
                param_group["lr"] = lr
            else:
                param_group["lr"] = param_group["lr"] * (lr / current)

    def load_optimizer_state_for_finetuning(self, state: dict):
        """Load per-parameter optimizer running state and grad scaler state from a
        checkpoint for fine-tuning.

        Restores per-parameter optimizer state (e.g. Adam moment estimates) and,
        if available, the grad scaler state. The freshly-built optimizer's
        per-group hyperparameters (``lr``, ``weight_decay``, ``betas``, ``eps``,
        ...) from the current finetune config are authoritative; any per-group
        hyperparameters from the checkpoint (including optimizer-type-specific
        flags like ``fused``/``amsgrad`` and scheduler-injected keys like
        ``initial_lr``) are discarded. Scheduler state is not restored, so the
        configured schedule starts from scratch.

        Args:
            state: The optimization state dict as saved by ``get_state()``,
                containing at least ``"optimizer_state_dict"``.

        Raises:
            ValueError: If the checkpoint's parameter groups are not
                structurally compatible with the freshly-built optimizer
                (e.g. different group count or per-group parameter count),
                or if one but not both of the checkpoint and the current
                optimizer is Muon.
        """
        _load_optimizer_state_for_finetuning(
            self.optimizer, state["optimizer_state_dict"]
        )
        if self.gscaler is not None and state.get("gscaler_state_dict") is not None:
            self.gscaler.load_state_dict(state["gscaler_state_dict"])

    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "gscaler_state_dict": (
                self.gscaler.state_dict() if self.gscaler is not None else None
            ),
        }
        return state

    def load_state(self, state):
        """
        Loads state from a serializable data structure.

        Raises:
            ValueError: If one but not both of the saved and the current
                optimizer is Muon.
        """
        if "components" in state:
            raise ValueError(
                "Optimization state holds one optimizer per component "
                f"({sorted(state['components'])}); it cannot load into a single "
                "optimizer. Use the per-component optimization config."
            )
        check_muon_state_dict_compatible(self.optimizer, state["optimizer_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if self.gscaler is not None:
            self.gscaler.load_state_dict(state["gscaler_state_dict"])

    def _validate_loss(self, loss: torch.Tensor):
        with torch.no_grad():
            if torch.isnan(loss):
                raise ValueError("Loss is NaN-valued during training.")


@dataclasses.dataclass
class _OptimizerEntry:
    optimizer: torch.optim.Optimizer
    scheduler: LRScheduler
    max_grad_norm: float | None


def _has_grad(optimizer: torch.optim.Optimizer) -> bool:
    return any(
        param.grad is not None
        for group in optimizer.param_groups
        for param in group["params"]
    )


class OptimizationDict(OptimizationABC):
    """One optimizer and LR scheduler per component over one shared backward
    pass and one shared GradScaler.

    ``learning_rate`` is the first component's first-group learning rate.
    Each component's scheduler steps that component's learning rate alone.
    Setting one learning rate for all components (LR tuning) is not supported.
    """

    def __init__(
        self,
        entries: Mapping[str, _OptimizerEntry],
        enable_automatic_mixed_precision: bool,
        use_gradient_accumulation: bool = False,
        get_checkpoint: Callable[
            [int], Checkpoint | NoCheckpoint
        ] = lambda _: NoCheckpoint(),
    ):
        if len(entries) == 0:
            raise ValueError("OptimizationDict requires at least one component.")
        self._entries = dict(entries)
        if enable_automatic_mixed_precision:
            self.gscaler: torch.amp.GradScaler | None = torch.amp.GradScaler("cuda")
        else:
            self.gscaler = None
        self._accumulated_loss = torch.tensor(0.0, device=get_device())
        self._use_gradient_accumulation = use_gradient_accumulation
        self._get_checkpoint = get_checkpoint

    @property
    def _first(self) -> _OptimizerEntry:
        return next(iter(self._entries.values()))

    def checkpoint(self, module: nn.Module, step: int) -> nn.Module:
        return self._get_checkpoint(step)(module)

    @contextlib.contextmanager
    def autocast(self):
        enabled = self.gscaler is not None
        dtype = torch.bfloat16 if enabled else None
        with torch.amp.autocast("cuda", enabled=enabled, dtype=dtype):
            yield

    @property
    def learning_rate(self) -> float:
        return self._first.optimizer.param_groups[0]["lr"]

    @property
    def learning_rates(self) -> dict[str, float]:
        return {
            key: entry.optimizer.param_groups[0]["lr"]
            for key, entry in self._entries.items()
        }

    def set_learning_rate(self, lr: float):
        raise NotImplementedError(
            "Per-component optimization has one learning rate per component; "
            "setting a single learning rate (LR tuning) is not supported."
        )

    def set_mode(self, modules: nn.ModuleList):
        for m in modules:
            m.train()

    def step_scheduler(
        self, valid_loss: float | None = None, is_iteration: bool = False
    ):
        for entry in self._entries.values():
            _step_scheduler(entry.scheduler, valid_loss, is_iteration)

    def detach_if_using_gradient_accumulation(self, state: TensorMapping) -> TensorDict:
        if self._use_gradient_accumulation:
            return {k: v.detach() for k, v in state.items()}
        return dict(state)

    def accumulate_loss(self, loss: torch.Tensor):
        with torch.no_grad():
            if torch.isnan(loss):
                raise ValueError("Loss is NaN-valued during training.")
        self._accumulated_loss += loss
        if self._use_gradient_accumulation:
            self._backward(loss)

    def get_accumulated_loss(self) -> torch.Tensor:
        return self._accumulated_loss

    def _backward(self, loss: torch.Tensor):
        if self.gscaler is not None:
            self.gscaler.scale(loss).backward()
        else:
            loss.backward()

    def _clip_gradients(self, entry: _OptimizerEntry):
        if entry.max_grad_norm is None:
            return
        if self.gscaler is not None:
            self.gscaler.unscale_(entry.optimizer)
        params = itertools.chain.from_iterable(
            group["params"] for group in entry.optimizer.param_groups
        )
        torch.nn.utils.clip_grad_norm_(params, entry.max_grad_norm)

    def step_weights(self):
        if not self._use_gradient_accumulation:
            self._backward(self._accumulated_loss)
        stepped = False
        for entry in self._entries.values():
            # GradScaler.step raises "No inf checks were recorded for this
            # optimizer" when no param has a grad: a frozen component, or a
            # single-component batch whose loss does not reach it.
            if not _has_grad(entry.optimizer):
                continue
            self._clip_gradients(entry)
            if self.gscaler is not None:
                self.gscaler.step(entry.optimizer)
            else:
                entry.optimizer.step()
            stepped = True
        for entry in self._entries.values():
            entry.optimizer.zero_grad()
        # GradScaler.update raises "No inf checks were recorded prior to
        # update" when no optimizer stepped.
        if self.gscaler is not None and stepped:
            self.gscaler.update()
        self._accumulated_loss = torch.tensor(0.0, device=get_device())

    def load_optimizer_state_for_finetuning(self, key: str, state: Mapping[str, Any]):
        """Load one component's per-parameter running state from a saved
        optimization state, keeping this config's hyperparameters and schedule.

        ``state`` is either a single-optimizer state (e.g. an uncoupled
        ``ckpt.tar``) or a per-component state holding ``key``. The shared
        GradScaler starts fresh.
        """
        if "components" in state:
            if key not in state["components"]:
                raise ValueError(
                    f"Saved per-component optimization state has no {key!r} "
                    f"entry; it has {sorted(state['components'])}."
                )
            state = state["components"][key]
        _load_optimizer_state_for_finetuning(
            self._entries[key].optimizer, state["optimizer_state_dict"]
        )

    def get_state(self):
        return {
            "components": {
                key: {
                    "optimizer_state_dict": entry.optimizer.state_dict(),
                    "scheduler_state_dict": entry.scheduler.state_dict(),
                }
                for key, entry in self._entries.items()
            },
            "gscaler_state_dict": (
                self.gscaler.state_dict() if self.gscaler is not None else None
            ),
        }

    def load_state(self, state):
        """
        Raises:
            ValueError: If ``state`` is a single-optimizer state, or its
                components differ from this optimization's, or one but not
                both of a saved and a current optimizer is Muon.
        """
        if "components" not in state:
            raise ValueError(
                "Optimization state holds a single optimizer over all "
                "components; it cannot load into per-component optimizers. "
                "Start the optimizers fresh, or set each component's "
                "resume_optimizer_ckpt_path."
            )
        saved = set(state["components"])
        if saved != set(self._entries):
            raise ValueError(
                f"Saved optimization components {sorted(saved)} differ from "
                f"configured components {sorted(self._entries)}."
            )
        for key, entry in self._entries.items():
            component_state = state["components"][key]
            check_muon_state_dict_compatible(
                entry.optimizer, component_state["optimizer_state_dict"]
            )
            entry.optimizer.load_state_dict(component_state["optimizer_state_dict"])
            entry.scheduler.load_state_dict(component_state["scheduler_state_dict"])
        if self.gscaler is not None:
            self.gscaler.load_state_dict(state["gscaler_state_dict"])


@dataclasses.dataclass
class OptimizationConfig:
    """
    Configuration for optimization.

    Parameters:
        optimizer_type: The type of optimizer to use. ``"Muon"`` updates every
            parameter with ``ndim >= 2`` (conv kernels are flattened to
            ``(out_channels, -1)``) with Muon's orthogonalized momentum, and
            all other parameters (biases, norm affines) plus those selected
            by ``adamw_names`` with AdamW, in a second parameter group.
            Grouping is by ``ndim`` alone, so a norm affine over several
            dimensions (e.g. ``LayerNorm([C, H, W])``) goes to Muon unless
            listed in ``adamw_names``.
        lr: The learning rate. For Muon, the Muon group's learning rate; the
            AdamW group's is ``kwargs["adamw_lr"]``. The logged learning rate
            and LR tuning act on the first (Muon) group, and LR tuning
            rescales the AdamW group by the same factor; LR schedulers
            scale each group from its own initial learning rate.
        kwargs: Additional keyword arguments to pass to the optimizer. For
            Muon the allowed keys are ``momentum`` (default 0.95),
            ``nesterov`` (default True), ``weight_decay`` (decoupled, Muon
            group, default 0.0), ``ns_steps`` (Newton-Schulz iterations,
            default 5), ``adjust_lr_fn`` (Muon step-size rule, ``None``/
            ``"original"`` or ``"match_rms_adamw"``, default None; see
            :class:`fme.core.muon.Muon`), ``adamw_lr`` (required),
            ``adamw_betas`` (default ``[0.9, 0.999]``), ``adamw_eps``
            (default 1e-8) and ``adamw_weight_decay`` (decoupled, default
            0.0).
        adamw_names: Muon only. Substrings of parameter names (as given by
            ``named_parameters()``, prefixed by the module index, e.g.
            ``"0.module.film.W_scale.weight"``) whose matrix parameters are
            routed to the AdamW group instead of Muon, e.g. zero-initialized
            projections that Muon's unit-scale first step would move off
            zero. Every entry must match at least one parameter.
        enable_automatic_mixed_precision: Whether to use automatic mixed
            precision.
        scheduler: The type of scheduler to use. If none is given, no scheduler
            will be used.
        use_gradient_accumulation: Whether to use gradient accumulation. This must be
            supported by the stepper being optimized, which may accumulate gradients
            from separate losses to reduce memory consumption. The stepper may choose
            to accumulate gradients differently when this is enabled, such as by
            detaching the computational graph between steps. See the documentation of
            your stepper (e.g. Stepper) for more details.
        max_grad_norm: Maximum norm for gradient clipping. If None, no gradient
            clipping is applied. When set, gradients are clipped to this global
            norm before each optimizer step. Compatible with automatic mixed
            precision. When use_gradient_accumulation is enabled, clipping is
            applied to the full N-step accumulated gradient (i.e. the gradient
            the optimizer sees), not per accumulation sub-step. With Muon the
            clip is applied unchanged to the global norm over all
            parameters, but Muon normalizes each matrix's update, so the
            clip changes the size of only the AdamW group's update.
        resume_optimizer_ckpt_path: Optional path to a training checkpoint
            (``ckpt.tar``) whose per-parameter optimizer running state (e.g.
            Adam moment estimates) and grad scaler state should be loaded into
            the freshly-built ``Optimization`` for fine-tuning. The current
            config's per-group hyperparameters (``lr``, ``weight_decay``,
            ``betas``, ...) and scheduler are kept; only the running state is
            transferred. Intended for non-resuming jobs; preemption resume in
            the Trainer overrides this state via ``Optimization.load_state``.
            Loading a non-Muon optimizer state into Muon, or vice versa, is
            an error.
    """

    optimizer_type: Literal["Adam", "AdamW", "FusedAdam", "Muon"] = "Adam"
    lr: float = 0.001
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    enable_automatic_mixed_precision: bool = False
    scheduler: SchedulerConfig | SequentialSchedulerConfig = dataclasses.field(
        default_factory=lambda: SchedulerConfig()
    )
    use_gradient_accumulation: bool = False
    max_grad_norm: float | None = None
    checkpoint: CheckpointConfig = dataclasses.field(
        default_factory=lambda: CheckpointConfig()
    )
    resume_optimizer_ckpt_path: str | None = None
    adamw_names: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.optimizer_type == "FusedAdam":
            warnings.warn(
                "FusedAdam is deprecated. Use AdamW with fused=True in kwargs instead.",
                DeprecationWarning,
            )
        if self.optimizer_type == "Muon":
            unknown = set(self.kwargs) - _MUON_KWARGS
            if unknown:
                raise ValueError(
                    f"Unknown kwargs for Muon: {sorted(unknown)}. "
                    f"Allowed: {sorted(_MUON_KWARGS)}."
                )
            adjust_lr_fn = self.kwargs.get("adjust_lr_fn")
            if adjust_lr_fn not in MUON_ADJUST_LR_FNS:
                raise ValueError(
                    "Muon kwargs['adjust_lr_fn'] must be one of "
                    f"{list(MUON_ADJUST_LR_FNS)}, got {adjust_lr_fn!r}."
                )
            if "adamw_lr" not in self.kwargs:
                raise ValueError(
                    "Muon requires kwargs['adamw_lr'], the learning rate of "
                    "its AdamW parameter group."
                )
        elif len(self.adamw_names) > 0:
            raise ValueError(
                "adamw_names is only used with optimizer_type='Muon', got "
                f"optimizer_type={self.optimizer_type!r}."
            )

    @property
    def has_lr_schedule(self) -> bool:
        """Whether a learning rate scheduler is configured."""
        if isinstance(self.scheduler, SequentialSchedulerConfig):
            return True
        return self.scheduler.type is not None

    def build_optimizer_and_scheduler(
        self, modules: OptimizationModules, max_epochs: int
    ) -> tuple[torch.optim.Optimizer, LRScheduler]:
        named_parameters = [
            (f"{i}.{name}", param)
            for i, module in enumerate(flatten_optimization_modules(modules))
            for name, param in module.named_parameters()
        ]
        optimizer = _build_optimizer(
            self.optimizer_type,
            named_parameters,
            self.lr,
            self.kwargs,
            adamw_names=self.adamw_names,
        )
        return optimizer, self.scheduler.build(optimizer, max_epochs)

    def build(self, modules: OptimizationModules, max_epochs: int) -> Optimization:
        """A per-component mapping is flattened in key order into one optimizer."""
        optimizer, scheduler = self.build_optimizer_and_scheduler(modules, max_epochs)
        optimization = Optimization(
            optimizer=optimizer,
            scheduler=scheduler,
            enable_automatic_mixed_precision=self.enable_automatic_mixed_precision,
            use_gradient_accumulation=self.use_gradient_accumulation,
            get_checkpoint=self.checkpoint.build,
            max_grad_norm=self.max_grad_norm,
        )
        if self.resume_optimizer_ckpt_path is not None:
            optimization.load_optimizer_state_for_finetuning(
                _read_optimization_state(self.resume_optimizer_ckpt_path)
            )
        return optimization

    def get_state(self) -> Mapping[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "OptimizationConfig":
        return cls(**state)


OptimizationConfigDict: TypeAlias = dict[str, OptimizationConfig]
"""One OptimizationConfig per independently optimized component."""

_SHARED_OPTIMIZATION_FIELDS = (
    "enable_automatic_mixed_precision",
    "use_gradient_accumulation",
    "checkpoint",
)


def validate_optimization_config_dict(
    configs: OptimizationConfigDict, keys: Collection[str]
) -> None:
    """
    Raises:
        ValueError: If the keys of ``configs`` differ from ``keys``, or a
            field that acts on the shared backward pass differs across entries.
    """
    missing = set(keys) - set(configs)
    unknown = set(configs) - set(keys)
    if missing or unknown:
        raise ValueError(
            f"Per-component optimization needs exactly the keys {sorted(keys)}; "
            f"missing {sorted(missing)}, unknown {sorted(unknown)}."
        )
    for name in _SHARED_OPTIMIZATION_FIELDS:
        values = {key: getattr(config, name) for key, config in configs.items()}
        first, *rest = values.values()
        if any(value != first for value in rest):
            raise ValueError(
                f"optimization.<component>.{name} must agree across components, "
                f"got {values}."
            )


def build_optimization_dict(
    configs: OptimizationConfigDict,
    modules: Mapping[str, nn.ModuleList],
    max_epochs: int,
) -> OptimizationDict:
    """Build one optimizer and scheduler per key of ``modules``, in its order.

    Parameter names restart at ``0.<name>`` within each component, as for an
    uncoupled stepper, so ``adamw_names`` and ``resume_optimizer_ckpt_path``
    carry over from uncoupled configs.
    """
    validate_optimization_config_dict(configs, keys=modules.keys())
    entries: dict[str, _OptimizerEntry] = {}
    for key, component_modules in modules.items():
        optimizer, scheduler = configs[key].build_optimizer_and_scheduler(
            component_modules, max_epochs
        )
        entries[key] = _OptimizerEntry(
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=configs[key].max_grad_norm,
        )
    shared = next(iter(configs.values()))
    optimization = OptimizationDict(
        entries,
        enable_automatic_mixed_precision=shared.enable_automatic_mixed_precision,
        use_gradient_accumulation=shared.use_gradient_accumulation,
        get_checkpoint=shared.checkpoint.build,
    )
    for key in modules:
        path = configs[key].resume_optimizer_ckpt_path
        if path is not None:
            optimization.load_optimizer_state_for_finetuning(
                key, _read_optimization_state(path)
            )
    return optimization


NestedTensor: TypeAlias = (
    "torch.Tensor | dict[str, NestedTensor] | list[NestedTensor] | tuple[NestedTensor]"
)


def _tensors_to_device(obj: NestedTensor, device: torch.device):
    """Recursively move all tensors in a nested dict/list to *device*."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: _tensors_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return type(obj)(_tensors_to_device(v, device) for v in obj)
    return obj


def _read_optimization_state(checkpoint_path: str) -> dict[str, Any]:
    """Read a ckpt.tar's optimization state onto the training device.

    The checkpoint is loaded on CPU so that only the optimization state
    (not model weights, EMA, etc.) is transferred to the training device.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "optimization" not in checkpoint:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain optimization "
            "state. Only checkpoints saved with include_optimization=True "
            "(i.e. ckpt.tar) support fine-tune optimization loading."
        )
    optim_state = checkpoint["optimization"]
    del checkpoint
    return _tensors_to_device(optim_state, get_device())


class NullOptimization(OptimizationABC):
    def __init__(self):
        self._accumulated_loss = torch.tensor(0.0, device=get_device())

    @contextlib.contextmanager
    def autocast(self):
        yield

    @property
    def learning_rate(self) -> float:
        return float("nan")

    def set_learning_rate(self, lr: float):
        pass

    def checkpoint(self, module: nn.Module, step: int) -> nn.Module:
        return module

    def step_scheduler(
        self, valid_loss: float | None = None, is_iteration: bool = False
    ):
        return

    def detach_if_using_gradient_accumulation(self, state: TensorMapping) -> TensorDict:
        return dict(state)

    def accumulate_loss(self, loss: torch.Tensor):
        self._accumulated_loss += loss

    def get_accumulated_loss(self) -> torch.Tensor:
        return self._accumulated_loss

    def step_weights(self):
        self._accumulated_loss = torch.tensor(0.0, device=get_device())
        return

    def get_state(self):
        return {}

    def load_state(self, state):
        return

    def set_mode(self, modules: nn.ModuleList):
        """
        Sets the mode of the module to eval.
        """
        for m in modules:
            m.eval()
