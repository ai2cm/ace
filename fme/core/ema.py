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

import dataclasses
from collections.abc import Iterable, Iterator
from typing import Protocol

import torch
from torch import nn


class HasNamedParameters(Protocol):
    def named_parameters(
        self, recurse: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]: ...


@dataclasses.dataclass
class EMAConfig:
    """
    Configuration for exponential moving average of model weights.

    Parameters:
        decay: decay rate for the moving average
    """

    decay: float = 0.9999

    def build(self, model: HasNamedParameters):
        return EMATracker(model, decay=self.decay, faster_decay_at_start=True)


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
        self.decay = torch.tensor(decay, dtype=torch.float32)
        self._faster_decay_at_start = faster_decay_at_start
        self.num_updates = torch.tensor(0, dtype=torch.int)

        self._ema_params = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                ema_name = name.replace(".", "")
                self._module_name_to_ema_name.update({name: ema_name})
                self._ema_params[ema_name] = p.clone().detach().data

        self._stored_params: list[nn.Parameter] = []

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

    def copy_to(self, model: HasNamedParameters):
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
        Get the state of the EMA tracker, excluding weights.

        Returns:
            The state of the EMA tracker.
        """
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "faster_decay_at_start": self._faster_decay_at_start,
            "module_name_to_ema_name": self._module_name_to_ema_name,
        }

    @classmethod
    def from_state(cls, state, model) -> "EMATracker":
        """
        Create an EMA tracker from a state.

        Args:
            state: The state of the EMA tracker.
            model: The model whose parameters should be tracked, used to
                initialize the EMA weights. Should come from an EMA checkpoint.

        Returns:
            The EMA tracker.
        """
        ema = cls(model, float(state["decay"]), state["faster_decay_at_start"])
        ema.num_updates = state["num_updates"]
        ema._module_name_to_ema_name = state["module_name_to_ema_name"]
        return ema
