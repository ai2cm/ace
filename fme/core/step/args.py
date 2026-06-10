from collections.abc import Callable

import torch

from fme.core.labels import BatchLabels
from fme.core.stepper_state import StepperState
from fme.core.typing_ import TensorMapping


class StepArgs:
    """
    Arguments to the step function.

    Parameters:
        input: Mapping from variable name to tensor of shape
            [n_batch, n_lat, n_lon]. This data is used as input for pytorch
            module(s) and is assumed to contain all input variables
            and be denormalized.
        next_step_input_data: Mapping from variable name to tensor of shape
            [n_batch, n_lat, n_lon]. This must contain the necessary input
            data at the output timestep, such as might be needed to prescribe
            sea surface temperature or use a corrector.
        labels: Labels for each batch member.
        data_mask: Per-variable, per-sample masks indicating variable
            presence. Keys are variable names, values are [n_batch] bool
            tensors where True means present and False means masked.
        stepper_state: Per-sample state carried across step calls (e.g.
            corrector references seeded from the IC). ``None`` if no state
            has been seeded yet. The step returns an updated stepper_state
            alongside its output dict.
    """

    def __init__(
        self,
        input: TensorMapping,
        next_step_input_data: TensorMapping,
        labels: BatchLabels | None = None,
        data_mask: TensorMapping | None = None,
        stepper_state: StepperState | None = None,
        forward_time: torch.Tensor | None = None,
    ):
        self.input = input
        self.next_step_input_data = next_step_input_data
        self.labels = labels
        self.data_mask = data_mask
        self.stepper_state = stepper_state
        self.forward_time = forward_time

    def apply_input_process_func(
        self, func: Callable[[TensorMapping], TensorMapping]
    ) -> "StepArgs":
        input = func(self.input)
        next_step_input_data = func(self.next_step_input_data)
        return StepArgs(
            input=input,
            next_step_input_data=next_step_input_data,
            labels=self.labels,
            data_mask=self.data_mask,
            stepper_state=self.stepper_state,
            forward_time=self.forward_time,
        )
