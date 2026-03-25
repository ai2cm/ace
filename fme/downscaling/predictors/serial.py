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
        self._first_model_output_names = first_out

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
        n_samples: int = 1,
    ) -> TensorDict:
        """Run first model returning shape [batch, n_samples, lat, lon]."""
        return self.first_model.generate_on_batch_no_target(batch, n_samples=n_samples)

    def _fold_first_output_conditioning(
        self, first_output: TensorDict, B: int, n_samples: int
    ) -> TensorDict:
        """Reshape conditioning from [B, S, H, W] to [B*S, H, W]."""
        return {
            k: v.reshape(B * n_samples, *v.shape[2:])
            for k, v in first_output.items()
            if k in self._high_res_cond_names
        }

    @staticmethod
    def _unfold_samples(flat: TensorDict, B: int, n_samples: int) -> TensorDict:
        """Reshape from [B*S, 1, H, W] to [B, S, H, W]."""
        result = {}
        for k, v in flat.items():
            squeezed = v.squeeze(1)  # [B*S, H, W]
            result[k] = squeezed.reshape(B, n_samples, *squeezed.shape[1:])
        return result

    @torch.no_grad()
    def generate_on_batch_no_target(
        self,
        batch: BatchData,
        n_samples: int = 1,
    ) -> TensorDict:
        first_output = self._run_first_model(batch, n_samples=n_samples)
        B = next(iter(first_output.values())).shape[0]

        fine_data = self._fold_first_output_conditioning(first_output, B, n_samples)
        expanded_batch = batch.expand_and_fold(n_samples, sample_dim=1)
        second_output_flat = self.second_model.generate_on_batch_no_target(
            expanded_batch, n_samples=1, fine_data=fine_data
        )
        second_output = self._unfold_samples(second_output_flat, B, n_samples)
        for name in self._first_model_output_names:
            second_output[name] = first_output[name]
        return second_output

    @torch.no_grad()
    def generate_on_batch(
        self,
        batch: PairedBatchData,
        n_samples: int = 1,
    ) -> ModelOutputs:
        first_output = self._run_first_model(batch.coarse, n_samples=n_samples)
        B = next(iter(first_output.values())).shape[0]

        folded_cond = self._fold_first_output_conditioning(first_output, B, n_samples)
        expanded = batch.expand_and_fold(n_samples, sample_dim=1)
        merged_fine_data = {**expanded.fine.data, **folded_cond}
        merged_batch = PairedBatchData(
            fine=BatchData(
                data=merged_fine_data,
                time=expanded.fine.time,
                latlon_coordinates=expanded.fine.latlon_coordinates,
            ),
            coarse=expanded.coarse,
        )
        result = self.second_model.generate_on_batch(merged_batch, n_samples=1)

        result.prediction = self._unfold_samples(result.prediction, B, n_samples)
        result.target = {k: v[::n_samples] for k, v in result.target.items()}

        for name in self._first_model_output_names:
            result.prediction[name] = first_output[name]
            if name in batch.fine.data:
                result.target[name] = batch.fine.data[name].unsqueeze(1)
        return result
