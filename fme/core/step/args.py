from collections.abc import Callable

from fme.core.labels import BatchLabels
from fme.core.typing_ import TensorMapping


class StepArgs:
    """
    Arguments to the step function.

    Parameters:
        input: Mapping from variable name to tensor. Prognostic variables have
            shape [n_batch, n_ic_timesteps, n_lat, n_lon]; forcing variables
            have shape [n_batch, n_lat, n_lon].
        next_step_input_data: Mapping from variable name to tensor of shape
            [n_batch, n_lat, n_lon]. This must contain the necessary input
            data at the output timestep, such as might be needed to prescribe
            sea surface temperature or use a corrector.
        labels: Labels for each batch member.
        data_mask: Per-variable, per-sample masks indicating variable
            presence. Keys are variable names, values are [n_batch] bool
            tensors where True means present and False means masked.
        channel_mask: Like data_mask but also includes training-time masking.
            Values may be [n_batch] or [n_batch, n_ic_timesteps] bool tensors.
            Used only to build channel-mask indicator inputs; does not affect
            loss weighting. Falls back to data_mask when None.
        next_step_prognostic_obs: Ground-truth observations at the next
            timestep for selected prognostic variables. Shape
            [n_batch, n_lat, n_lon] per variable. Present only during
            training; None at inference.
    """

    def __init__(
        self,
        input: TensorMapping,
        next_step_input_data: TensorMapping,
        labels: BatchLabels | None = None,
        data_mask: TensorMapping | None = None,
        channel_mask: TensorMapping | None = None,
        next_step_prognostic_obs: TensorMapping | None = None,
    ):
        self.input = input
        self.next_step_input_data = next_step_input_data
        self.labels = labels
        self.data_mask = data_mask
        self.channel_mask = channel_mask
        self.next_step_prognostic_obs = next_step_prognostic_obs

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
            channel_mask=self.channel_mask,
            next_step_prognostic_obs=self.next_step_prognostic_obs,
        )
