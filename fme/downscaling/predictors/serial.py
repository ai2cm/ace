import torch

from fme.core.typing_ import TensorDict
from fme.downscaling.data import BatchData, PairedBatchData
from fme.downscaling.models import DiffusionModel, ModelOutputs


class SerialPredictor:
    """Chains two DiffusionModels so the first model's outputs feed the second.

    The first model generates fine-resolution fields from coarse input.
    Fields that match the second model's ``high_res_conditioning`` are
    forwarded as conditioning channels.  Any remaining first-model outputs
    are included in the final prediction alongside the second model's
    outputs.

    Both models must share the same ``coarse_shape`` and
    ``downscale_factor``.  The second model's ``high_res_conditioning``
    must be a subset of the first model's ``out_names``, and the two
    models must not share any ``out_names``.
    """

    def __init__(
        self,
        first_model: DiffusionModel,
        second_model: DiffusionModel,
    ):
        high_res_cond = set(second_model.config.high_res_conditioning or [])
        if not high_res_cond:
            raise ValueError("second_model must have high_res_conditioning configured")
        first_out = set(first_model.config.out_names)
        missing = high_res_cond - first_out
        if missing:
            raise ValueError(
                f"second_model high_res_conditioning {missing} not found in "
                f"first_model out_names {first_model.config.out_names}"
            )
        second_out = set(second_model.config.out_names)
        overlap = first_out & second_out
        if overlap:
            raise ValueError(
                f"first_model and second_model share out_names {overlap}; "
                f"outputs must not overlap"
            )
        if first_model.coarse_shape != second_model.coarse_shape:
            raise ValueError(
                f"Models must share coarse_shape: "
                f"{first_model.coarse_shape} != {second_model.coarse_shape}"
            )
        if first_model.downscale_factor != second_model.downscale_factor:
            raise ValueError(
                f"Models must share downscale_factor: "
                f"{first_model.downscale_factor} != "
                f"{second_model.downscale_factor}"
            )
        self.first_model = first_model
        self.second_model = second_model
        self._high_res_cond_names = high_res_cond
        # First-model outputs that are not consumed as conditioning
        self._first_model_output_names = first_out - high_res_cond

    @property
    def modules(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList(
            [*self.first_model.modules, *self.second_model.modules]
        )

    @property
    def coarse_shape(self):
        return self.second_model.coarse_shape

    @property
    def downscale_factor(self):
        return self.second_model.downscale_factor

    @property
    def fine_shape(self):
        return self.second_model.fine_shape

    @property
    def static_inputs(self):
        return self.second_model.static_inputs

    def get_fine_coords_for_batch(self, batch: BatchData):
        return self.second_model.get_fine_coords_for_batch(batch)

    def _run_first_model(
        self,
        batch: BatchData,
    ) -> TensorDict:
        """Run first model and squeeze the sample dimension."""
        generated = self.first_model.generate_on_batch_no_target(batch, n_samples=1)
        # generated values have shape [batch, 1, lat, lon]; squeeze sample dim
        return {k: v.squeeze(1) for k, v in generated.items()}

    @torch.no_grad()
    def generate_on_batch_no_target(
        self,
        batch: BatchData,
        n_samples: int = 1,
    ) -> TensorDict:
        first_output = self._run_first_model(batch)
        fine_data = {
            k: v for k, v in first_output.items() if k in self._high_res_cond_names
        }
        second_output = self.second_model.generate_on_batch_no_target(
            batch, n_samples=n_samples, fine_data=fine_data
        )
        # Include first-model passthrough outputs with a sample dimension
        for name in self._first_model_output_names:
            second_output[name] = (
                first_output[name].unsqueeze(1).expand(-1, n_samples, -1, -1)
            )
        return second_output

    @torch.no_grad()
    def generate_on_batch(
        self,
        batch: PairedBatchData,
        n_samples: int = 1,
    ) -> ModelOutputs:
        first_output = self._run_first_model(batch.coarse)
        fine_data = {
            k: v for k, v in first_output.items() if k in self._high_res_cond_names
        }
        # Merge conditioning into fine data for the second model
        merged_fine = {**batch.fine.data, **fine_data}
        merged_batch = PairedBatchData(
            fine=BatchData(
                data=merged_fine,
                time=batch.fine.time,
                latlon_coordinates=batch.fine.latlon_coordinates,
            ),
            coarse=batch.coarse,
        )
        result = self.second_model.generate_on_batch(merged_batch, n_samples)
        # Add first-model passthrough outputs to prediction and target
        for name in self._first_model_output_names:
            result.prediction[name] = (
                first_output[name].unsqueeze(1).expand(-1, n_samples, -1, -1)
            )
            if name in batch.fine.data:
                result.target[name] = batch.fine.data[name].unsqueeze(1)
        return result
