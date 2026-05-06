"""
Exponential Moving Average (EMA) module.

Copied from https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/ema.py
and modified.

MIT License

Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import contextlib
import dataclasses
import logging
from collections.abc import Iterable, Iterator
from typing import Protocol

import torch
from torch import nn

from fme.core.device import get_device


class HasNamedParameters(Protocol):
    def named_parameters(
        self, recurse: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]: ...

    def parameters(self) -> Iterator[nn.Parameter]: ...


@dataclasses.dataclass
class EMAConfig:
    """
    Configuration for exponential moving average of model weights.

    Parameters:
        decay: The decay rate of the moving average.
        faster_decay_at_start: Whether to use the number of updates to determine
            the decay rate. If True, the decay rate will be min(decay, (1 +
            num_updates) / (10 + num_updates)). If False, the decay rate
            will be decay.
        resume_ema_ckpt_path: Optional path to a training checkpoint
            (e.g., ``ckpt.tar``) whose EMA running state (averaged weights and
            update counter) should be loaded into the freshly-built ``EMATracker``
            for fine-tuning. The current config's ``decay`` and
            ``faster_decay_at_start`` are kept; only the running state is
            transferred. Intended for non-resuming jobs; preemption resume in
            the Trainer overrides this state via ``EMATracker.from_state``.
    """

    decay: float = 0.9999
    faster_decay_at_start: bool = True
    resume_ema_ckpt_path: str | None = None

    def build(self, model: HasNamedParameters):
        ema = EMATracker(
            model,
            decay=self.decay,
            faster_decay_at_start=self.faster_decay_at_start,
        )
        if self.resume_ema_ckpt_path is not None:
            _load_finetune_ema_state(ema, self.resume_ema_ckpt_path)
        return ema


class EMATracker:
    """
    Exponential Moving Average (EMA) tracker.

    This tracks the moving average of the parameters of a model, and has methods
    that can be used to temporarily replace the parameters of the model with its EMA.
    """

    def __init__(
        self, model: HasNamedParameters, decay: float, faster_decay_at_start=True
    ):
        """
        Create a new EMA tracker.

        Args:
            model: The model whose parameters should be tracked.
            decay: The decay rate of the moving average.
            faster_decay_at_start: Whether to use the number of updates to determine
                the decay rate. If True, the decay rate will be min(decay, (1 +
                num_updates) / (10 + num_updates)). If False, the decay rate
                will be decay.
        """
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self._module_name_to_ema_name = {}
        self.decay = torch.tensor(decay, dtype=torch.float32).to(get_device())
        self._faster_decay_at_start = faster_decay_at_start
        self.num_updates = torch.tensor(0, dtype=torch.int).to(get_device())

        self._ema_params: dict[str, torch.Tensor] = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                ema_name = name.replace(".", "")
                self._module_name_to_ema_name.update({name: ema_name})
                self._ema_params[ema_name] = p.clone().detach().data

        self._stored_params: list[nn.Parameter] = []
        self._in_context = False

    def __call__(self, model: HasNamedParameters):
        """
        Update the moving average of the parameters.

        Does not mutate the input, only updates the moving average.

        Args:
            model: The model whose parameters should be updated. Should be a model
                specified identically to the one passed when this object was
                instantiated.
        """
        decay = self.decay

        self.num_updates += 1
        if self._faster_decay_at_start:
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        with torch.no_grad():
            module_parameters = dict(model.named_parameters())

            for key in module_parameters:
                if module_parameters[key].requires_grad:
                    ema_name = self._module_name_to_ema_name[key]
                    self._ema_params[ema_name] = self._ema_params[ema_name].type_as(
                        module_parameters[key]
                    )
                    self._ema_params[ema_name].sub_(
                        (1.0 - decay)
                        * (self._ema_params[ema_name] - module_parameters[key])
                    )
                elif key in self._module_name_to_ema_name:
                    raise ValueError(
                        f"Expected model parameter {key} to require gradient, "
                        "but it does not"
                    )

    @contextlib.contextmanager
    def applied_params(self, model: HasNamedParameters) -> Iterator[None]:
        self.store(parameters=model.parameters())
        if self._in_context:
            raise RuntimeError("Cannot nest EMA contexts")
        self._in_context = True
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(parameters=model.parameters())
            self._in_context = False

    def copy_to(self, model: HasNamedParameters) -> None:
        """
        Copy the averaged parameters to the model, overwriting its values.
        """
        m_param = dict(model.named_parameters())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(
                    self._ema_params[self._module_name_to_ema_name[key]].data
                )
            else:
                assert key not in self._module_name_to_ema_name

    def store(self, parameters: Iterable[nn.Parameter]):
        """
        Save the current parameters for restoring later.

        Args:
            parameters: The parameters to be stored for later restoration by `restore`
        """
        self._stored_params = [param.clone() for param in parameters]

    def restore(self, parameters: Iterable[nn.Parameter]):
        """
        Restore the parameters stored with the `store` method.

        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: The parameters to be updated with the values stored by `store`
        """
        for c_param, param in zip(self._stored_params, parameters):
            param.data.copy_(c_param.data)

    def get_state(self):
        """
        Get the state of the EMA tracker.

        Returns:
            The state of the EMA tracker.
        """
        return {
            "decay": self.decay.clone(),
            "num_updates": self.num_updates.clone(),
            "faster_decay_at_start": self._faster_decay_at_start,
            "module_name_to_ema_name": dict(self._module_name_to_ema_name),
            "ema_params": {
                name: param.clone().detach() for name, param in self._ema_params.items()
            },
        }

    def load_ema_state_for_finetuning(self, state: dict):
        """Load EMA running state from a checkpoint for fine-tuning.

        Restores the averaged parameter weights and update counter from
        a previously saved EMA state. The current tracker's ``decay`` and
        ``faster_decay_at_start`` (set at construction from the current
        config) are preserved; only the running state is transferred.

        Args:
            state: The EMA state dict as saved by ``get_state()``,
                containing at least ``"ema_params"``, ``"num_updates"``,
                and ``"module_name_to_ema_name"``.

        Raises:
            ValueError: If the state does not contain ``"ema_params"``
                (e.g. from a checkpoint saved without
                ``include_optimization=True``).
        """
        if "ema_params" not in state:
            raise ValueError(
                "EMA state does not contain ema_params. Only ckpt.tar "
                "checkpoints (saved with include_optimization=True) "
                "contain the full EMA state needed for fine-tuning."
            )
        device = get_device()
        self.num_updates = state["num_updates"].to(device, copy=True)
        self._module_name_to_ema_name = state["module_name_to_ema_name"]
        self._ema_params = {
            name: param.to(device, copy=True)
            for name, param in state["ema_params"].items()
        }

    @classmethod
    def from_state(cls, state, model) -> "EMATracker":
        """
        Create an EMA tracker from a state.

        Args:
            state: The state of the EMA tracker.
            model: The model whose parameters should be tracked, used to
                initialize the EMA weights.

        Returns:
            The EMA tracker.
        """
        device = get_device()
        ema = cls(model, float(state["decay"]), state["faster_decay_at_start"])
        ema.num_updates = state["num_updates"].to(device, copy=True)
        ema._module_name_to_ema_name = state["module_name_to_ema_name"]
        if "ema_params" in state:
            ema._ema_params = {
                name: param.to(device, copy=True)
                for name, param in state["ema_params"].items()
            }
        else:
            logging.warning("EMA params not found in state and will not be restored.")
        return ema


def _load_finetune_ema_state(ema: EMATracker, checkpoint_path: str):
    """Load EMA running state from a training checkpoint for fine-tuning.

    Only loads the EMA averaged weights and update counter from the
    checkpoint. The current tracker's decay and faster_decay_at_start
    are preserved from the current config.

    The checkpoint is loaded on CPU so that only the EMA state (not model
    weights, optimizer, etc.) is transferred to the training device.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "ema" not in checkpoint:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain EMA state. "
            "Only training checkpoints (ckpt.tar) contain EMA state."
        )
    ema_state = checkpoint["ema"]
    del checkpoint
    ema.load_ema_state_for_finetuning(ema_state)
