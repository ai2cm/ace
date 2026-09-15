import contextlib
import dataclasses
import itertools
import logging
import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, TypeAlias

import numpy as np
import torch
from torch import nn

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.generics.optimization import OptimizationABC
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


def _build_optimizer(
    optimizer_type: Literal["Adam", "FusedAdam", "AdamW"],
    parameters: Iterable[torch.nn.Parameter],
    lr: float,
    kwargs: Mapping[str, Any],
) -> torch.optim.Optimizer:
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
        max_consecutive_non_finite_losses: int = 0,
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
        self._max_consecutive_non_finite_losses = max_consecutive_non_finite_losses
        # set on this rank by accumulate_loss, combined across ranks in
        # step_weights, and reset there for the next batch
        self._saw_non_finite_loss = False
        self._consecutive_non_finite_losses = 0
        self._skipped_non_finite_batches = 0

    @property
    def skipped_non_finite_batches(self) -> int:
        """Total number of batches whose optimizer step was skipped because
        the loss was non-finite on at least one rank.
        """
        return self._skipped_non_finite_batches

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

    def _any_rank_saw_non_finite_loss(self) -> bool:
        """Combine the per-rank non-finite loss flag across all ranks.

        Every rank must take the same branch in step_weights, otherwise the
        ranks would disagree about which collectives to run.
        """
        if self._max_consecutive_non_finite_losses == 0:
            # accumulate_loss already raised in this mode, so no rank can carry
            # the flag; skip the collective entirely
            return False
        dist = Distributed.get_instance()
        flag = torch.tensor(
            float(self._saw_non_finite_loss), device=self._accumulated_loss.device
        )
        return bool(dist.reduce_max(flag).item() > 0.0)

    def step_weights(self):
        if not self._use_gradient_accumulation:
            # backward is run even for a non-finite loss so that all ranks
            # participate in DDP's gradient all-reduce
            self._backward(self._accumulated_loss)
        skip = self._any_rank_saw_non_finite_loss()
        if skip:
            self._skipped_non_finite_batches += 1
            self._consecutive_non_finite_losses += 1
            self._last_grad_norm = None
            if self.gscaler is not None:
                # record an inf check for this iteration (normally done by
                # gscaler.step) so that gscaler.update() below has state to
                # consume and lowers the scale
                self.gscaler.unscale_(self.optimizer)
        else:
            self._consecutive_non_finite_losses = 0
            self._clip_gradients()
            self._step_weights()
        self.optimizer.zero_grad()
        if self.gscaler is not None:
            self.gscaler.update()
        self._saw_non_finite_loss = False
        self._accumulated_loss = torch.tensor(0.0, device=get_device())
        if skip:
            logging.warning(
                "Skipping optimizer step: loss was non-finite (NaN or inf) on at "
                f"least one rank. This batch is the "
                f"{self._consecutive_non_finite_losses} in a row with a non-finite "
                f"loss, and the {self._skipped_non_finite_batches} skipped in total."
            )
            if (
                self._consecutive_non_finite_losses
                > self._max_consecutive_non_finite_losses
            ):
                raise ValueError(
                    "Loss is non-finite (NaN or inf) during training for "
                    f"{self._consecutive_non_finite_losses} consecutive batches, "
                    "exceeding max_consecutive_non_finite_losses="
                    f"{self._max_consecutive_non_finite_losses} "
                    f"({self._skipped_non_finite_batches} batches skipped in total)."
                )

    def set_learning_rate(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

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
                (e.g. different group count or per-group parameter count).
        """
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
        """
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if self.gscaler is not None:
            self.gscaler.load_state_dict(state["gscaler_state_dict"])

    def _validate_loss(self, loss: torch.Tensor):
        """Check the loss for NaN or inf values.

        When no non-finite losses are tolerated this raises immediately, as it
        has historically done for NaN. Otherwise it only records a flag, and
        the optimizer step is skipped in step_weights once the flag has been
        combined across ranks.
        """
        with torch.no_grad():
            if not torch.isfinite(loss):
                if self._max_consecutive_non_finite_losses == 0:
                    raise ValueError("Loss is non-finite (NaN or inf) during training.")
                self._saw_non_finite_loss = True


@dataclasses.dataclass
class OptimizationConfig:
    """
    Configuration for optimization.

    Parameters:
        optimizer_type: The type of optimizer to use.
        lr: The learning rate.
        kwargs: Additional keyword arguments to pass to the optimizer.
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
            the optimizer sees), not per accumulation sub-step.
        resume_optimizer_ckpt_path: Optional path to a training checkpoint
            (``ckpt.tar``) whose per-parameter optimizer running state (e.g.
            Adam moment estimates) and grad scaler state should be loaded into
            the freshly-built ``Optimization`` for fine-tuning. The current
            config's per-group hyperparameters (``lr``, ``weight_decay``,
            ``betas``, ...) and scheduler are kept; only the running state is
            transferred. Intended for non-resuming jobs; preemption resume in
            the Trainer overrides this state via ``Optimization.load_state``.
        float32_matmul_precision: If set, passed to
            ``torch.set_float32_matmul_precision`` when the optimization is
            built. ``"high"`` enables TensorFloat-32 for float32 matrix
            multiplies (``nn.Linear``, ``torch.matmul``) on Ampere and newer
            GPUs, giving tensor-core throughput at the cost of rounding the
            matmul inputs to a 10-bit mantissa; accumulation stays in
            float32. Convolutions already use TensorFloat-32 by default in
            PyTorch, so this mostly matters for transformer-style models.
            ``None`` (default) leaves the process-wide PyTorch setting
            untouched. Has no effect when automatic mixed precision is enabled.
        max_consecutive_non_finite_losses: How many consecutive batches with a
            non-finite (NaN or inf) loss are tolerated before training aborts.
            ``0`` (the default) keeps the historical behavior of raising on the
            first one. When greater than zero, a batch whose loss is non-finite
            on any rank contributes no optimizer step (its gradients are
            discarded), a warning is logged, and the total number of skipped
            batches is exposed as ``Optimization.skipped_non_finite_batches``;
            the consecutive count resets on the next finite batch, and a
            ``ValueError`` naming the count is raised once it exceeds this
            limit. The per-iteration learning rate scheduler is still stepped on
            a skipped batch so the schedule stays aligned with the batch count.
            Note that inf losses previously slipped through the NaN-only check;
            they are now treated like NaN, which is a deliberate tightening.
    """

    optimizer_type: Literal["Adam", "AdamW", "FusedAdam"] = "Adam"
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
    float32_matmul_precision: Literal["highest", "high", "medium"] | None = None
    max_consecutive_non_finite_losses: int = 0

    def __post_init__(self):
        if self.max_consecutive_non_finite_losses < 0:
            raise ValueError(
                "max_consecutive_non_finite_losses must be >= 0, got "
                f"{self.max_consecutive_non_finite_losses}."
            )
        if self.optimizer_type == "FusedAdam":
            warnings.warn(
                "FusedAdam is deprecated. Use AdamW with fused=True in kwargs instead.",
                DeprecationWarning,
            )

    @property
    def has_lr_schedule(self) -> bool:
        """Whether a learning rate scheduler is configured."""
        if isinstance(self.scheduler, SequentialSchedulerConfig):
            return True
        return self.scheduler.type is not None

    def build(self, modules: torch.nn.ModuleList, max_epochs: int) -> Optimization:
        if self.float32_matmul_precision is not None:
            logging.info(
                "Setting torch float32 matmul precision to "
                f"'{self.float32_matmul_precision}'"
            )
            torch.set_float32_matmul_precision(self.float32_matmul_precision)
        parameters = itertools.chain(*[module.parameters() for module in modules])
        optimizer = _build_optimizer(
            self.optimizer_type, parameters, self.lr, self.kwargs
        )
        scheduler = self.scheduler.build(optimizer, max_epochs)
        optimization = Optimization(
            optimizer=optimizer,
            scheduler=scheduler,
            enable_automatic_mixed_precision=self.enable_automatic_mixed_precision,
            use_gradient_accumulation=self.use_gradient_accumulation,
            get_checkpoint=self.checkpoint.build,
            max_grad_norm=self.max_grad_norm,
            max_consecutive_non_finite_losses=self.max_consecutive_non_finite_losses,
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
