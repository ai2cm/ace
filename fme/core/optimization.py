import contextlib
import dataclasses
import itertools
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import torch
from torch import nn

from fme.core.device import get_device
from fme.core.generics.optimization import OptimizationABC
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
        if self.scheduler.should_step(is_iteration):
            try:
                if valid_loss is not None:
                    self.scheduler.step(metrics=valid_loss)
                else:
                    self.scheduler.step()
            except TypeError:
                # Some schedulers don't accept metrics argument
                self.scheduler.step()

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
                or if exactly one of the checkpoint and the current
                optimizer is Muon.
        """
        check_muon_state_dict_compatible(self.optimizer, state["optimizer_state_dict"])
        fresh_hparams = [
            {k: v for k, v in g.items() if k != "params"}
            for g in self.optimizer.param_groups
        ]
        try:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
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
        for group, hparams in zip(self.optimizer.param_groups, fresh_hparams):
            for k in list(group.keys()):
                if k != "params":
                    del group[k]
            group.update(hparams)
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
            ValueError: If exactly one of the saved and the current optimizer
                is Muon.
        """
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
            default 5), ``adamw_lr`` (required), ``adamw_betas`` (default
            ``[0.9, 0.999]``), ``adamw_eps`` (default 1e-8) and
            ``adamw_weight_decay`` (decoupled, default 0.0).
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

    def build(self, modules: torch.nn.ModuleList, max_epochs: int) -> Optimization:
        named_parameters = [
            (f"{i}.{name}", param)
            for i, module in enumerate(modules)
            for name, param in module.named_parameters()
        ]
        optimizer = _build_optimizer(
            self.optimizer_type,
            named_parameters,
            self.lr,
            self.kwargs,
            adamw_names=self.adamw_names,
        )
        scheduler = self.scheduler.build(optimizer, max_epochs)
        optimization = Optimization(
            optimizer=optimizer,
            scheduler=scheduler,
            enable_automatic_mixed_precision=self.enable_automatic_mixed_precision,
            use_gradient_accumulation=self.use_gradient_accumulation,
            get_checkpoint=self.checkpoint.build,
            max_grad_norm=self.max_grad_norm,
        )
        if self.resume_optimizer_ckpt_path is not None:
            _load_finetune_optimization_state(
                optimization, self.resume_optimizer_ckpt_path
            )
        return optimization

    def get_state(self) -> Mapping[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "OptimizationConfig":
        return cls(**state)


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


def _load_finetune_optimization_state(optimization: Optimization, checkpoint_path: str):
    """Load optimizer (and optionally grad scaler) state for fine-tuning.

    Only loads the optimizer state dict and grad scaler state from the
    checkpoint. Scheduler state and training counters are not restored, so
    the current config's schedule starts from scratch. All freshly-built
    optimizer per-group hyperparameters (lr, weight_decay, betas, eps, ...)
    are preserved from the current job's TrainConfig.

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
    optim_state = _tensors_to_device(optim_state, get_device())
    optimization.load_optimizer_state_for_finetuning(optim_state)


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
