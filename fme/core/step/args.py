from collections.abc import Callable

from fme.core.labels import BatchLabels
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
    """

    def __init__(
        self,
        input: TensorMapping,
        next_step_input_data: TensorMapping,
        labels: BatchLabels | None = None,
    ):
        self.input = input
        self.next_step_input_data = next_step_input_data
        self.labels = labels

    def apply_input_process_func(
        self, func: Callable[[TensorMapping], TensorMapping]
    ) -> "StepArgs":
        input = func(self.input)
        next_step_input_data = func(self.next_step_input_data)
        return StepArgs(
            input=input, next_step_input_data=next_step_input_data, labels=self.labels
        )
