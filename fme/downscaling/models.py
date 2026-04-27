import dataclasses
import warnings
from collections.abc import Mapping
from functools import cached_property
from typing import Any

import dacite
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.optimization import NullOptimization, Optimization
from fme.core.packer import Packer
from fme.core.typing_ import TensorDict, TensorMapping
from fme.downscaling.data import (
    BatchData,
    ClosedInterval,
    PairedBatchData,
    StaticInputs,
    adjust_fine_coord_range,
    load_coords_from_path,
)
from fme.downscaling.metrics_and_maths import filter_tensor_mapping, interpolate
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.noise import (
    LogNormalNoiseDistribution,
    LogUniformNoiseDistribution,
    condition_with_noise_for_training,
)
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.samplers import stochastic_sampler as edm_sampler
from fme.downscaling.typing_ import FineResCoarseResPair


@dataclasses.dataclass
class ModelOutputs:
    prediction: TensorDict
    target: TensorDict
    loss: torch.Tensor
    latent_steps: list[torch.Tensor] = dataclasses.field(default_factory=list)
    channel_losses: TensorDict = dataclasses.field(default_factory=dict)
    sigma: torch.Tensor | None = None
    per_sample_channel_loss: TensorDict = dataclasses.field(default_factory=dict)


def _rename_normalizer(
    normalizer: StandardNormalizer, rename: dict[str, str] | None
) -> StandardNormalizer:
    if not rename:
        return normalizer
    new_means = {
        rename.get(name, name): value for name, value in normalizer.means.items()
    }
    new_stds = {
        rename.get(name, name): value for name, value in normalizer.stds.items()
    }
    return StandardNormalizer(means=new_means, stds=new_stds)


def _build_variable_loss_weight_tensor(
    weights: dict[str, float], out_names: list[str]
) -> torch.Tensor:
    for name in weights:
        if name not in out_names:
            raise ValueError(
                f"Name {name} in loss_weights.output_channels is not in out_names"
            )
    values = [weights.get(name, 1.0) for name in out_names]
    return torch.tensor(values, dtype=torch.float32, device=get_device()).reshape(
        1, len(out_names), 1, 1
    )


@dataclasses.dataclass
class LossWeightsConfig:
    """
    Configuration for loss weighting during training.

    Parameters:
        output_channels: Per-variable multiplicative weights applied to the loss.
            Keys are variable names from out_names; variables not listed default to 1.0.
        noise_weight_exponent: Exponent applied to the EDM noise-level loss weight
            ``(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2``. The default
            of 1.0 gives the standard EDM weighting (~1/sigma^2 for small sigma).
            Use values less than 1.0 to reduce relative weighting of low-noise steps.
            We find that 0.75 improves performance for winds and sea level pressure,
            which are dominated by low-noise samples in the default EDM weighting.
    """

    output_channels: dict[str, float] = dataclasses.field(default_factory=dict)
    noise_weight_exponent: float = 1.0


@dataclasses.dataclass
class PairedNormalizationConfig:
    fine: NormalizationConfig
    coarse: NormalizationConfig

    def build(
        self,
        in_names: list[str],
        out_names: list[str],
        rename: dict[str, str] | None = None,
    ) -> FineResCoarseResPair[StandardNormalizer]:
        coarse = self.coarse.build(list(set(in_names).union(out_names)))
        fine = self.fine.build(out_names)

        return FineResCoarseResPair[StandardNormalizer](
            coarse=_rename_normalizer(coarse, rename),
            fine=_rename_normalizer(fine, rename),
        )

    def load(self):
        """
        Load the normalization configuration from the netCDF files.

        Updates the configuration so it no longer requires external files.
        """
        self.fine.load()
        self.coarse.load()


@dataclasses.dataclass(frozen=True)
class DiffusionModelMetadata:
    in_names: list[str]
    out_names: list[str]
    coarse_shape: tuple[int, int]
    downscale_factor: int
    sigma_data: float
    predict_residual: bool
    use_fine_topography: bool
    full_fine_coords: LatLonCoordinates
    num_static_inputs: int
    # Avoid runtime failures for frozen class with unhashable field
    __hash__ = None  # type: ignore[assignment]


@dataclasses.dataclass
class DiffusionModelConfig:
    """
    This class implements or wraps the algorithms described in `EDM`_.

    .. _EDM: https://arxiv.org/abs/2206.00364

    Parameters:
        module: The module registry selector for the diffusion model.
        loss:  The loss configuration for the diffusion model.
        in_names: The input variable names for the diffusion model.
        out_names: The output variable names for the diffusion model.
        normalization: The normalization configurations for the diffusion model.
        sigma_min: Min noise level for generation.
        sigma_max: Max noise level for generation.
        churn: The amount of stochasticity during generation.
        num_diffusion_generation_steps: Number of diffusion generation steps
        use_fine_topography: Whether to use fine topography in the model.
        use_amp_bf16: Whether to use automatic mixed precision (bfloat16) in the
            UNetDiffusionModule.
        loss_weights: Weighting configuration for the training loss, including
            per-variable channel weights and the noise-level weight exponent.
        training_noise_distribution: Noise distribution to use during training.
        p_mean: The mean of noise distribution used during training.
            Deprecated. Use training_noise_distribution field instead.
            This is kept for backwards compatibility.
        p_std: The std of the noise distribution used during training.
            Deprecated. Use training_noise_distribution field instead.
            This is kept for backwards compatibility.
    """

    module: DiffusionModuleRegistrySelector
    loss: LossConfig
    in_names: list[str]
    out_names: list[str]
    normalization: PairedNormalizationConfig
    sigma_min: float
    sigma_max: float
    churn: float
    num_diffusion_generation_steps: int
    predict_residual: bool
    use_fine_topography: bool = False
    use_amp_bf16: bool = False
    loss_weights: LossWeightsConfig = dataclasses.field(
        default_factory=LossWeightsConfig
    )
    training_noise_distribution: (
        LogNormalNoiseDistribution | LogUniformNoiseDistribution | None
    ) = None
    p_mean: float | None = None
    p_std: float | None = None

    def __post_init__(self):
        self._interpolate_input = self.module.expects_interpolated_input
        if self.use_fine_topography and not self._interpolate_input:
            raise ValueError(
                "Fine topography can only be used when predicting on interpolated"
                " coarse input"
            )
        if self.p_mean is not None and self.p_std is not None:
            if self.training_noise_distribution is None:
                warnings.warn(
                    "p_mean and p_std are deprecated. "
                    f"Use training_noise_distribution field instead."
                )
            else:
                raise ValueError(
                    "Training noise should be specified in training_noise_distribution "
                    "field only. Both training_noise_distribution and p_mean, p_std "
                    "were specified. The latter two fields are deprecated."
                )
        if self.training_noise_distribution is None and (
            self.p_mean is None or self.p_std is None
        ):
            raise ValueError(
                "Noise distribution must be specified in training_noise_distribution "
                "field or in p_mean and p_std fields."
            )

    @property
    def noise_distribution(
        self,
    ) -> LogNormalNoiseDistribution | LogUniformNoiseDistribution:
        """
        Returns NoiseDistribution object to use for sampling noise in training.
        """
        if self.training_noise_distribution is not None:
            return self.training_noise_distribution
        elif self.p_mean is not None and self.p_std is not None:
            return LogNormalNoiseDistribution(p_mean=self.p_mean, p_std=self.p_std)
        else:
            raise ValueError(
                "Noise distribution must be specified in training_noise_distribution "
                "or in p_mean and p_std fields."
            )

    def build(
        self,
        coarse_shape: tuple[int, int],
        downscale_factor: int,
        full_fine_coords: LatLonCoordinates,
        rename: dict[str, str] | None = None,
        static_inputs: StaticInputs | None = None,
    ) -> "DiffusionModel":
        invert_rename = {v: k for k, v in (rename or {}).items()}
        orig_in_names = [invert_rename.get(name, name) for name in self.in_names]
        orig_out_names = [invert_rename.get(name, name) for name in self.out_names]
        normalizer = self.normalization.build(orig_in_names, orig_out_names, rename)
        loss = self.loss.build(reduction="none", gridded_operations=None)
        # We always use standard score normalization, so sigma_data is
        # always 1.0. See below for standard score normalization:
        # https://en.wikipedia.org/wiki/Standard_score
        sigma_data = 1.0

        num_static_in_channels = len(static_inputs.fields) if static_inputs else 0
        n_in_channels = len(self.in_names) + num_static_in_channels
        if self.use_fine_topography and (
            not static_inputs or len(static_inputs.fields) == 0
        ):
            raise ValueError(
                "use_fine_topography is enabled but no static input fields were found. "
                "At least one static input field must be provided when using fine "
                "topography."
            )

        if static_inputs and static_inputs.coords != full_fine_coords:
            raise ValueError(
                "static_inputs coordinates do not match full_fine_coords. "
                "Static inputs must be defined on the same grid as the model output."
            )

        module = self.module.build(
            n_in_channels=n_in_channels,
            n_out_channels=len(self.out_names),
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
            sigma_data=sigma_data,
            use_amp_bf16=self.use_amp_bf16,
        )

        return DiffusionModel(
            config=self,
            module=module,
            normalizer=normalizer,
            loss=loss,
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
            sigma_data=sigma_data,
            full_fine_coords=full_fine_coords,
            static_inputs=static_inputs,
        )

    def get_state(self) -> Mapping[str, Any]:
        # Update normalization configuration so it no longer requires external files.
        self.normalization.load()
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DiffusionModelConfig":
        return dacite.from_dict(data_class=cls, data=state)

    @property
    def data_requirements(self) -> DataRequirements:
        # Requires output names in coarse for aggregators checking relative measures
        return DataRequirements(
            fine_names=self.out_names,
            coarse_names=list(set(self.in_names).union(self.out_names)),
            n_timesteps=1,
            use_fine_topography=self.use_fine_topography,
        )


def _repeat_batch_by_samples(tensor: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Repeat the batch dimension of a tensor n_samples times.  Used to parallelize
    sample generation in the diffusion model, but done so such that the tensor still
    only leads with a single batch dimension.  Added as a separate function for
    round-trip testing.
    """
    return tensor.repeat_interleave(dim=0, repeats=n_samples)


def _separate_interleaved_samples(tensor: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Reshape an interleaved tensor to have a leading [batch, sample, ...] dimension.
    """
    if tensor.shape[0] % n_samples != 0:
        raise ValueError(
            "The interleaved batch+sample dimension of the tensor must be divisible "
            "by n_samples."
        )
    n_batch = tensor.shape[0] // n_samples
    return tensor.reshape(n_batch, n_samples, *tensor.shape[1:])


class DiffusionModel:
    def __init__(
        self,
        config: DiffusionModelConfig,
        module: torch.nn.Module,
        normalizer: FineResCoarseResPair[StandardNormalizer],
        loss: torch.nn.Module,
        coarse_shape: tuple[int, int],
        downscale_factor: int,
        sigma_data: float,
        full_fine_coords: LatLonCoordinates,
        static_inputs: StaticInputs | None = None,
    ) -> None:
        """
        Args:
            config: The configuration object for the diffusion model.
            module: The neural network module used for downscaling. Note: this
                should *not* be DistributedDataParallel since it is wrapped by
                it in this init method.
            normalizer: The normalizer object used for data normalization.
            loss: The loss function used for training the model.
            coarse_shape: The height (lat) and width (lon) of the
                coarse-resolution input data used to train the model
                (same as patch extent, if training on patches).
            downscale_factor: The factor by which the data is downscaled from
                coarse to fine.
            sigma_data: The standard deviation of the data, used for diffusion
                model preconditioning.
            full_fine_coords: The full fine-resolution domain coordinates.
                Serves as the canonical source of truth for the model output grid.
            static_inputs: Static inputs to the model. May be None when
                no static data is needed. If present, coordinates
                must match full_fine_coords.
        """
        self.coarse_shape = coarse_shape
        self.downscale_factor = downscale_factor
        self.sigma_data = sigma_data
        dist = Distributed.get_instance()
        self.module = dist.wrap_module(module.to(get_device()))
        self.normalizer = normalizer
        self.loss = loss
        self.in_packer = Packer(config.in_names)
        self.out_packer = Packer(config.out_names)
        self.config = config
        self._channel_axis = -3
        self.full_fine_coords = full_fine_coords.to(get_device())
        self.static_inputs = static_inputs.to_device() if static_inputs else None
        self._loss_weight_tensor = _build_variable_loss_weight_tensor(
            config.loss_weights.output_channels, config.out_names
        )

    @property
    def modules(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([self.module])

    def _get_fine_interval_from_batch(
        self, batch: BatchData
    ) -> tuple[ClosedInterval, ClosedInterval]:
        coarse_lat = batch.latlon_coordinates.lat[0]
        coarse_lon = batch.latlon_coordinates.lon[0]
        fine_lat_interval = adjust_fine_coord_range(
            batch.lat_interval,
            full_coarse_coord=coarse_lat,
            full_fine_coord=self.full_fine_coords.lat,
            downscale_factor=self.downscale_factor,
        )
        fine_lon_interval = adjust_fine_coord_range(
            batch.lon_interval,
            full_coarse_coord=coarse_lon,
            full_fine_coord=self.full_fine_coords.lon,
            downscale_factor=self.downscale_factor,
        )
        return fine_lat_interval, fine_lon_interval

    def get_fine_coords_for_batch(self, batch: BatchData) -> LatLonCoordinates:
        """Return fine-resolution coordinates matching the spatial extent of batch."""
        lat_interval, lon_interval = self._get_fine_interval_from_batch(batch)
        return LatLonCoordinates(
            lat=lat_interval.subset_of(self.full_fine_coords.lat),
            lon=lon_interval.subset_of(self.full_fine_coords.lon),
        )

    def _subset_static_if_available(self, batch: BatchData) -> StaticInputs | None:
        if self.static_inputs is None:
            return None
        fine_lat_interval, fine_lon_interval = self._get_fine_interval_from_batch(batch)
        return self.static_inputs.subset(
            lat_interval=fine_lat_interval,
            lon_interval=fine_lon_interval,
        )

    @property
    def fine_shape(self) -> tuple[int, int]:
        return self._get_fine_shape(self.coarse_shape)

    def _get_fine_shape(self, coarse_shape: tuple[int, int]) -> tuple[int, int]:
        """
        Calculate the fine shape based on the coarse shape of data used to train
        the model and the downscaling factor.
        """
        return (
            coarse_shape[0] * self.downscale_factor,
            coarse_shape[1] * self.downscale_factor,
        )

    def _get_input_from_coarse(
        self, coarse: TensorMapping, static_inputs: StaticInputs | None
    ) -> torch.Tensor:
        inputs = filter_tensor_mapping(coarse, self.in_packer.names)
        normalized = self.in_packer.pack(
            self.normalizer.coarse.normalize(inputs), axis=self._channel_axis
        )
        interpolated = interpolate(normalized, self.downscale_factor)

        if self.config.use_fine_topography and static_inputs is not None:
            expected_shape = interpolated.shape[-2:]
            if static_inputs.shape != expected_shape:
                raise ValueError(
                    f"Subsetted static input shape {static_inputs.shape} does not "
                    f"match expected fine spatial shape {expected_shape}."
                )
            n_batches = normalized.shape[0]
            # Join normalized static inputs to input (see dataset for details)
            fields: list[torch.Tensor] = [interpolated]
            for field in static_inputs.fields:
                static_field = field.data.unsqueeze(0).repeat(n_batches, 1, 1)
                static_field = static_field.unsqueeze(self._channel_axis)
                fields.append(static_field)
            interpolated = torch.concat(fields, dim=self._channel_axis)

        if self.config._interpolate_input:
            return interpolated
        return normalized

    def train_on_batch(
        self,
        batch: PairedBatchData,
        optimizer: Optimization | NullOptimization,
    ) -> ModelOutputs:
        """Performs a denoising training step on a batch of data."""
        _static_inputs = self._subset_static_if_available(batch.coarse)
        coarse, fine = batch.coarse.data, batch.fine.data
        inputs_norm = self._get_input_from_coarse(coarse, _static_inputs)
        targets_norm = self.out_packer.pack(
            self.normalizer.fine.normalize(dict(fine)), axis=self._channel_axis
        )

        if self.config.predict_residual:
            base_prediction = interpolate(
                self.out_packer.pack(
                    self.normalizer.coarse.normalize(
                        {k: coarse[k] for k in self.out_packer.names}
                    ),
                    axis=self._channel_axis,
                ),
                self.downscale_factor,
            )
            targets_norm = targets_norm - base_prediction

        conditioned_target = condition_with_noise_for_training(
            targets_norm,
            self.config.noise_distribution,
            self.sigma_data,
            loss_weight_exponent=self.config.loss_weights.noise_weight_exponent,
        )

        denoised_norm = self.module(
            conditioned_target.latents, inputs_norm, conditioned_target.sigma
        )
        weighted_loss = (  # has dims (batch, channels, lat, lon)
            conditioned_target.weight
            * self._loss_weight_tensor
            * self.loss(denoised_norm, targets_norm)
        )
        loss = torch.mean(weighted_loss)
        optimizer.accumulate_loss(loss)
        optimizer.step_weights()

        with torch.no_grad():
            channel_losses = {
                name: torch.mean(weighted_loss[:, i, :, :])
                for i, name in enumerate(self.out_packer.names)
            }
            per_sample_channel_loss = {
                name: torch.mean(weighted_loss[:, i, :, :], dim=(-2, -1))
                for i, name in enumerate(self.out_packer.names)
            }
            sigma = conditioned_target.sigma[:, 0, 0, 0]

        if self.config.predict_residual:
            denoised_norm = denoised_norm + base_prediction

        target = filter_tensor_mapping(batch.fine.data, set(self.out_packer.names))
        denoised = self.normalizer.fine.denormalize(
            self.out_packer.unpack(denoised_norm, axis=self._channel_axis)
        )
        return ModelOutputs(
            prediction=denoised,
            target=target,
            loss=loss,
            channel_losses=channel_losses,
            sigma=sigma,
            per_sample_channel_loss=per_sample_channel_loss,
            latent_steps=[],
        )

    def prepare_generation_inputs(
        self,
        coarse_data: TensorMapping,
        static_inputs: StaticInputs | None,
        n_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize coarse input and build random latents for generation.

        Returns:
            inputs: Normalized (and optionally interpolated) coarse input,
                repeated ``n_samples`` times along the batch dimension.
            latents: Random noise tensor shaped for the fine output grid.
        """
        inputs = self._get_input_from_coarse(coarse_data, static_inputs)
        inputs = _repeat_batch_by_samples(inputs, n_samples)
        coarse_input_shape = next(iter(coarse_data.values())).shape[-2:]
        outputs_shape = (
            inputs.shape[0],
            len(self.out_packer.names),
            *self._get_fine_shape(coarse_input_shape),
        )
        latents = torch.randn(outputs_shape).to(device=get_device())
        return inputs, latents

    def postprocess_generated(
        self,
        generated_norm: torch.Tensor,
        coarse_data: TensorMapping,
        n_samples: int,
    ) -> tuple[TensorDict, torch.Tensor]:
        """Add residual, separate samples, and denormalize sampler output."""
        if self.config.predict_residual:
            base_prediction = interpolate(
                self.out_packer.pack(
                    self.normalizer.coarse.normalize(
                        {k: coarse_data[k] for k in self.out_packer.names}
                    ),
                    axis=self._channel_axis,
                ),
                self.downscale_factor,
            )
            generated_norm = generated_norm + _repeat_batch_by_samples(
                base_prediction, n_samples
            )
        generated_norm_reshaped = _separate_interleaved_samples(
            generated_norm, n_samples
        )
        generated = self.normalizer.fine.denormalize(
            self.out_packer.unpack(generated_norm_reshaped, axis=self._channel_axis)
        )
        return generated, generated_norm

    @torch.no_grad()
    def generate(
        self,
        coarse_data: TensorMapping,
        static_inputs: StaticInputs | None,
        n_samples: int = 1,
    ) -> tuple[TensorDict, torch.Tensor, list[torch.Tensor]]:
        inputs, latents = self.prepare_generation_inputs(
            coarse_data, static_inputs, n_samples
        )
        generated_norm, latent_steps = edm_sampler(
            self.module,
            latents,
            inputs,
            S_churn=self.config.churn,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            num_steps=self.config.num_diffusion_generation_steps,
        )
        generated, generated_norm = self.postprocess_generated(
            generated_norm, coarse_data, n_samples
        )
        return generated, generated_norm, latent_steps

    @torch.no_grad()
    def generate_on_batch_no_target(
        self,
        batch: BatchData,
        n_samples: int = 1,
    ) -> TensorDict:
        _static_inputs = self._subset_static_if_available(batch)
        generated, _, _ = self.generate(batch.data, _static_inputs, n_samples)
        return generated

    @torch.no_grad()
    def generate_on_batch(
        self,
        batch: PairedBatchData,
        n_samples: int = 1,
    ) -> ModelOutputs:
        _static_inputs = self._subset_static_if_available(batch.coarse)
        coarse, fine = batch.coarse.data, batch.fine.data
        generated, generated_norm, latent_steps = self.generate(
            coarse, _static_inputs, n_samples
        )

        targets_norm = self.out_packer.pack(
            self.normalizer.fine.normalize(dict(fine)), axis=self._channel_axis
        )
        targets_norm = _repeat_batch_by_samples(targets_norm, n_samples)

        targets = filter_tensor_mapping(batch.fine.data, set(self.out_packer.names))
        targets = {k: v.unsqueeze(1) for k, v in targets.items()}

        loss = self.loss(generated_norm, targets_norm)
        return ModelOutputs(
            prediction=generated, target=targets, loss=loss, latent_steps=latent_steps
        )

    def get_state(self) -> Mapping[str, Any]:
        if self.static_inputs is not None:
            static_inputs_state = self.static_inputs.get_state()
        else:
            static_inputs_state = None

        return {
            "config": self.config.get_state(),
            "module": self.module.state_dict(),
            "coarse_shape": self.coarse_shape,
            "downscale_factor": self.downscale_factor,
            "full_fine_coords": self.full_fine_coords.get_state(),
            "static_inputs": static_inputs_state,
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
    ) -> "DiffusionModel":
        """
        Reconstruct model from state (used during training checkpoint resumption).
        Requires full_fine_coords in state. For old checkpoints without it, use
        CheckpointModelConfig with fine_coordinates_path for backwards compatibility.
        """
        static_inputs_state = state.get("static_inputs")
        static_inputs = (
            StaticInputs.from_state(static_inputs_state)
            if static_inputs_state
            else None
        )
        full_fine_coords_state = state.get("full_fine_coords")
        if full_fine_coords_state is not None:
            full_fine_coords = LatLonCoordinates(
                lat=full_fine_coords_state["lat"].to(get_device(), copy=True),
                lon=full_fine_coords_state["lon"].to(get_device(), copy=True),
            )
        else:
            raise ValueError(
                "No full_fine_coords found in loaded state for DiffusionModel. "
                "Must use CheckpointModelConfig with fine_coordinates_path provided "
                "for backwards compatibility loading of old checkpoints without "
                "full_fine_coords in state."
            )
        config = DiffusionModelConfig.from_state(state["config"])
        model = config.build(
            state["coarse_shape"],
            state["downscale_factor"],
            full_fine_coords=full_fine_coords,
            static_inputs=static_inputs,
        )
        model.module.load_state_dict(state["module"], strict=True)
        return model

    @cached_property
    def metadata(self):
        return DiffusionModelMetadata(
            in_names=self.config.in_names,
            out_names=self.config.out_names,
            coarse_shape=self.coarse_shape,
            downscale_factor=self.downscale_factor,
            sigma_data=self.sigma_data,
            predict_residual=self.config.predict_residual,
            use_fine_topography=self.config.use_fine_topography,
            full_fine_coords=self.full_fine_coords,
            num_static_inputs=len(self.static_inputs.fields)
            if self.static_inputs
            else 0,
        )


@dataclasses.dataclass
class _CheckpointModelConfigSelector:
    wrapper: DiffusionModelConfig

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> DiffusionModelConfig:
        return dacite.from_dict(
            data={"wrapper": state}, data_class=cls, config=dacite.Config(strict=True)
        ).wrapper


@dataclasses.dataclass
class CheckpointModelConfig:
    """
    This class specifies a diffusion model loaded from a checkpoint file.

    Parameters:
        checkpoint_path: The path to the checkpoint file.
        rename: Optional mapping of {old: new} model input/output names to rename.
        static_inputs: Optional mapping of {field_name: path} for static inputs to
            the model. Useful when no fine res data is available during evaluation
            but the model requires static input data. Raises an error if the
            checkpoint already has static inputs from training.
        fine_topography_path: Deprecated. Use static_inputs instead.
        fine_coordinates_path: Optional path to a netCDF/zarr file containing lat/lon
            coordinates for the full fine domain. Used for old checkpoints that have
            no static_inputs and no stored fine_coords.
        model_updates: Optional mapping of {key: new_value} model config updates to
            apply when loading the model. This is useful for running evaluation with
            updated parameters than at training time. Use with caution; not all
            parameters can or should be updated at evaluation time.
    """

    checkpoint_path: str
    rename: dict[str, str] | None = None
    static_inputs: dict[str, str] | None = None
    fine_topography_path: str | None = None
    fine_coordinates_path: str | None = None
    model_updates: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # For config validation testing, we don't want to load immediately
        # so we defer until build or properties are accessed.
        self._checkpoint_is_loaded = False
        self._rename = self.rename or {}
        if "module" in (self.model_updates or {}):
            raise ValueError("'module' cannot be updated in model_updates.")
        if self.fine_topography_path is not None:
            raise ValueError(
                "fine_topography_path is deprecated and will be removed in "
                "a future release. Use static_inputs instead, "
                "e.g., static_inputs: {HGTsfc: <path>}.",
            )

    @property
    def _checkpoint(self) -> Mapping[str, Any]:
        if not self._checkpoint_is_loaded:
            checkpoint_data = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
            checkpoint_data["model"]["config"]["in_names"] = [
                self._rename.get(name, name)
                for name in checkpoint_data["model"]["config"]["in_names"]
            ]
            checkpoint_data["model"]["config"]["out_names"] = [
                self._rename.get(name, name)
                for name in checkpoint_data["model"]["config"]["out_names"]
            ]
            self._checkpoint_data = checkpoint_data
            self._checkpoint_is_loaded = True
            if self.model_updates is not None:
                for k, v in self.model_updates.items():
                    checkpoint_data["model"]["config"][k] = v
        return self._checkpoint_data

    @staticmethod
    def _get_coords_backwards_compatible(
        coords_from_state: dict | None,
        fine_coordinates_path: str | None,
    ) -> LatLonCoordinates:
        if coords_from_state and fine_coordinates_path:
            raise ValueError(
                "Checkpoint contains fine coordinates but fine_coordinates_path is also"
                " provided. Backwards compatibility loading only supports a single "
                "source of fine coordinates info."
            )
        if coords_from_state is not None:
            return LatLonCoordinates(
                lat=coords_from_state["lat"].to(get_device(), copy=True),
                lon=coords_from_state["lon"].to(get_device(), copy=True),
            )
        elif fine_coordinates_path is not None:
            return load_coords_from_path(fine_coordinates_path)
        else:
            raise ValueError(
                "No fine coordinates found in checkpoint state and no "
                "fine_coordinates_path provided. One of these must be provided to "
                "load the model using CheckpointModelConfig."
            )

    def build(
        self,
    ) -> DiffusionModel:
        checkpoint_model: dict = self._checkpoint["model"]
        full_fine_coords = self._get_coords_backwards_compatible(
            checkpoint_model.get("full_fine_coords"),
            self.fine_coordinates_path,
        )
        static_inputs = StaticInputs.from_state_backwards_compatible(
            state=checkpoint_model.get("static_inputs") or {},
            static_inputs_config=self.static_inputs or {},
        )
        model = _CheckpointModelConfigSelector.from_state(
            self._checkpoint["model"]["config"]
        ).build(
            coarse_shape=self._checkpoint["model"]["coarse_shape"],
            downscale_factor=self._checkpoint["model"]["downscale_factor"],
            full_fine_coords=full_fine_coords,
            rename=self._rename,
            static_inputs=static_inputs,
        )
        model.module.load_state_dict(self._checkpoint["model"]["module"])
        model.module.eval()
        return model

    @property
    def data_requirements(self) -> DataRequirements:
        in_names = self.in_names
        out_names = self.out_names
        return DataRequirements(
            fine_names=out_names,
            coarse_names=list(set(in_names).union(out_names)),
            n_timesteps=1,
            use_fine_topography=self._checkpoint["model"]["config"][
                "use_fine_topography"
            ],
        )

    @property
    def in_names(self):
        return self._checkpoint["model"]["config"]["in_names"]

    @property
    def out_names(self):
        return self._checkpoint["model"]["config"]["out_names"]
