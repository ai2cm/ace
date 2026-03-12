import dataclasses
import warnings
from collections.abc import Mapping
from typing import Any

import dacite
import torch
import xarray as xr

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
    load_static_inputs,
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
        rename: dict[str, str] | None = None,
        static_inputs: StaticInputs | None = None,
        fine_coords: LatLonCoordinates | None = None,
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

        n_in_channels = len(self.in_names)
        if static_inputs is not None:
            n_in_channels += len(static_inputs.fields)
        elif self.use_fine_topography:
            # Old checkpoints may not have static inputs serialized, but if
            # use_fine_topography is True, we still need to account for the topography
            # channel, which was the only static input at the time
            n_in_channels += 1

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
            static_inputs=static_inputs,
            fine_coords=fine_coords,
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
        static_inputs: StaticInputs | None = None,
        fine_coords: LatLonCoordinates | None = None,
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
            static_inputs: Static inputs to the model, loaded from the trainer
                config or checkpoint. Must be set when use_fine_topography is True.
            fine_coords: Full-domain fine-resolution coordinates. Used as the
                single coordinate authority for output spatial metadata.
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
        self.static_inputs = (
            static_inputs.to_device() if static_inputs is not None else None
        )
        self.fine_coords = fine_coords
        if fine_coords is not None and static_inputs is not None:
            expected = (len(fine_coords.lat), len(fine_coords.lon))
            if static_inputs.shape != expected:
                raise ValueError(
                    f"static_inputs shape {static_inputs.shape} does not match "
                    f"fine_coords grid {expected}"
                )

    @property
    def modules(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([self.module])

    def _subset_static_inputs(
        self,
        lat_interval: ClosedInterval,
        lon_interval: ClosedInterval,
    ) -> StaticInputs | None:
        """Subset self.static_inputs to the given fine lat/lon interval.

        Returns None if use_fine_topography is False.
        Raises ValueError if use_fine_topography is True but self.static_inputs is None.
        """
        if not self.config.use_fine_topography:
            return None
        if self.static_inputs is None:
            raise ValueError(
                "Static inputs must be provided for each batch when use of fine "
                "static inputs is enabled."
            )
        if self.fine_coords is None:
            raise ValueError(
                "fine_coords must be set on the model to subset static inputs."
            )
        lat_slice = lat_interval.slice_of(self.fine_coords.lat)
        lon_slice = lon_interval.slice_of(self.fine_coords.lon)
        return self.static_inputs.subset(lat_slice, lon_slice)

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

        if self.config.use_fine_topography:
            if static_inputs is None:
                raise ValueError(
                    "Static inputs must be provided for each batch when use of fine "
                    "static inputs is enabled."
                )
            else:
                expected_shape = interpolated.shape[-2:]
                if static_inputs.shape != expected_shape:
                    raise ValueError(
                        f"Subsetted static input shape {static_inputs.shape} does not "
                        f"match expected fine spatial shape {expected_shape}."
                    )
                n_batches = normalized.shape[0]
                # Join normalized static inputs to input (see dataset for details)
                for field in static_inputs.fields:
                    topo = field.data.unsqueeze(0).repeat(n_batches, 1, 1)
                    topo = topo.unsqueeze(self._channel_axis)
                    interpolated = torch.concat(
                        [interpolated, topo], axis=self._channel_axis
                    )

        if self.config._interpolate_input:
            return interpolated
        return normalized

    def train_on_batch(
        self,
        batch: PairedBatchData,
        optimizer: Optimization | NullOptimization,
    ) -> ModelOutputs:
        """Performs a denoising training step on a batch of data."""
        _static_inputs = self._subset_static_inputs(
            batch.fine.lat_interval, batch.fine.lon_interval
        )
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
            targets_norm, self.config.noise_distribution, self.sigma_data
        )

        denoised_norm = self.module(
            conditioned_target.latents, inputs_norm, conditioned_target.sigma
        )
        weighted_loss = conditioned_target.weight * self.loss(
            denoised_norm, targets_norm
        )
        loss = torch.mean(weighted_loss)
        optimizer.accumulate_loss(loss)
        optimizer.step_weights()

        with torch.no_grad():
            channel_losses = {
                name: torch.mean(weighted_loss[:, i, :, :])
                for i, name in enumerate(self.out_packer.names)
            }

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
            latent_steps=[],
        )

    @torch.no_grad()
    def generate(
        self,
        coarse_data: TensorMapping,
        static_inputs: StaticInputs | None,
        n_samples: int = 1,
    ) -> tuple[TensorDict, torch.Tensor, list[torch.Tensor]]:
        # Internal method; external callers should use generate_on_batch /
        # generate_on_batch_no_target.
        inputs_ = self._get_input_from_coarse(coarse_data, static_inputs)
        # expand samples and fold to
        # [batch * n_samples, output_channels, height, width]
        inputs_ = _repeat_batch_by_samples(inputs_, n_samples)
        coarse_input_shape = next(iter(coarse_data.values())).shape[-2:]

        outputs_shape = (
            inputs_.shape[0],
            len(self.out_packer.names),
            *self._get_fine_shape(coarse_input_shape),
        )
        latents = torch.randn(outputs_shape).to(device=get_device())

        generated_norm, latent_steps = edm_sampler(
            self.module,
            latents,
            inputs_,
            S_churn=self.config.churn,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            num_steps=self.config.num_diffusion_generation_steps,
        )

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
        return generated, generated_norm, latent_steps

    @torch.no_grad()
    def generate_on_batch_no_target(
        self,
        batch: BatchData,
        n_samples: int = 1,
    ) -> TensorDict:
        if self.config.use_fine_topography:
            if self.static_inputs is None:
                raise ValueError(
                    "Static inputs must be provided for each batch when use of fine "
                    "static inputs is enabled."
                )
            if self.fine_coords is None:
                raise ValueError(
                    "fine_coords must be set on the model when use_fine_topography "
                    "is enabled."
                )
            coarse_lat = batch.latlon_coordinates.lat[0]
            coarse_lon = batch.latlon_coordinates.lon[0]
            fine_lat_interval = adjust_fine_coord_range(
                batch.lat_interval,
                full_coarse_coord=coarse_lat,
                full_fine_coord=self.fine_coords.lat,
                downscale_factor=self.downscale_factor,
            )
            fine_lon_interval = adjust_fine_coord_range(
                batch.lon_interval,
                full_coarse_coord=coarse_lon,
                full_fine_coord=self.fine_coords.lon,
                downscale_factor=self.downscale_factor,
            )
            lat_slice = fine_lat_interval.slice_of(self.fine_coords.lat)
            lon_slice = fine_lon_interval.slice_of(self.fine_coords.lon)
            _static_inputs = self.static_inputs.subset(lat_slice, lon_slice)
        else:
            _static_inputs = None
        generated, _, _ = self.generate(batch.data, _static_inputs, n_samples)
        return generated

    @torch.no_grad()
    def generate_on_batch(
        self,
        batch: PairedBatchData,
        n_samples: int = 1,
    ) -> ModelOutputs:
        _static_inputs = self._subset_static_inputs(
            batch.fine.lat_interval, batch.fine.lon_interval
        )
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
            "static_inputs": static_inputs_state,
            "fine_coords": (
                self.fine_coords.get_state() if self.fine_coords is not None else None
            ),
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
    ) -> "DiffusionModel":
        config = DiffusionModelConfig.from_state(state["config"])
        # backwards compatibility for models before static inputs serialization
        if state.get("static_inputs") is not None:
            static_inputs = StaticInputs.from_state(state["static_inputs"]).to_device()
        else:
            static_inputs = None

        # Load fine_coords: new checkpoints store it directly; old checkpoints
        # that had static_inputs with coords can auto-migrate from raw state.
        if state.get("fine_coords") is not None:
            fine_coords = LatLonCoordinates(
                lat=state["fine_coords"]["lat"],
                lon=state["fine_coords"]["lon"],
            )
        elif (
            state.get("static_inputs") is not None
            and len(state["static_inputs"].get("fields", [])) > 0
            and "coords" in state["static_inputs"]["fields"][0]
        ):
            # Backward compat: old checkpoints stored coords inside static_inputs fields
            coords_state = state["static_inputs"]["fields"][0]["coords"]
            fine_coords = LatLonCoordinates(
                lat=coords_state["lat"],
                lon=coords_state["lon"],
            )
        else:
            fine_coords = None

        model = config.build(
            state["coarse_shape"],
            state["downscale_factor"],
            static_inputs=static_inputs,
            fine_coords=fine_coords,
        )
        model.module.load_state_dict(state["module"], strict=True)
        return model


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
            checkpoint_data = torch.load(self.checkpoint_path, weights_only=False)
            checkpoint_data["model"]["config"]["in_names"] = [
                self._rename.get(name, name)
                for name in checkpoint_data["model"]["config"]["in_names"]
            ]
            checkpoint_data["model"]["config"]["out_names"] = [
                self._rename.get(name, name)
                for name in checkpoint_data["model"]["config"]["out_names"]
            ]
            # backwards compatibility for models before static inputs serialization
            checkpoint_data["model"].setdefault("static_inputs", None)
            # backwards compatibility for models before fine_coords serialization
            checkpoint_data["model"].setdefault("fine_coords", None)

            self._checkpoint_data = checkpoint_data
            self._checkpoint_is_loaded = True
            if self.model_updates is not None:
                for k, v in self.model_updates.items():
                    checkpoint_data["model"]["config"][k] = v
        return self._checkpoint_data

    def _load_fine_coords_from_path(self, path: str) -> LatLonCoordinates:
        if path.endswith(".zarr"):
            ds = xr.open_zarr(path)
        else:
            ds = xr.open_dataset(path)
        lat_name = next((n for n in ["lat", "latitude"] if n in ds.coords), None)
        lon_name = next((n for n in ["lon", "longitude"] if n in ds.coords), None)
        if lat_name is None or lon_name is None:
            raise ValueError(
                f"Could not find lat/lon coordinates in {path}. "
                "Expected 'lat'/'latitude' and 'lon'/'longitude'."
            )
        return LatLonCoordinates(
            lat=torch.tensor(ds[lat_name].values, dtype=torch.float32),
            lon=torch.tensor(ds[lon_name].values, dtype=torch.float32),
        )

    def build(
        self,
    ) -> DiffusionModel:
        static_inputs: StaticInputs | None
        if self._checkpoint["model"]["static_inputs"] is not None:
            if self.static_inputs is not None:
                raise ValueError(
                    "The model checkpoint already has static inputs from training. "
                    "static_inputs should not be provided in checkpoint model config."
                    "static inputs from training."
                )
            static_inputs = StaticInputs.from_state(
                self._checkpoint["model"]["static_inputs"]
            )
        elif self.static_inputs is not None:
            static_inputs = load_static_inputs(self.static_inputs)
        else:
            static_inputs = None
        model = _CheckpointModelConfigSelector.from_state(
            self._checkpoint["model"]["config"]
        ).build(
            coarse_shape=self._checkpoint["model"]["coarse_shape"],
            downscale_factor=self._checkpoint["model"]["downscale_factor"],
            rename=self._rename,
            static_inputs=static_inputs,
        )
        model.module.load_state_dict(self._checkpoint["model"]["module"])

        # Restore fine_coords: new checkpoints have it stored directly; old
        # checkpoints may have coords embedded in static_inputs fields.
        model_state = self._checkpoint["model"]
        if model_state.get("fine_coords") is not None:
            fine_coords_state = model_state["fine_coords"]
            model.fine_coords = LatLonCoordinates(
                lat=fine_coords_state["lat"],
                lon=fine_coords_state["lon"],
            )
        elif (
            model_state.get("static_inputs") is not None
            and len(model_state["static_inputs"].get("fields", [])) > 0
            and "coords" in model_state["static_inputs"]["fields"][0]
        ):
            coords_state = model_state["static_inputs"]["fields"][0]["coords"]
            model.fine_coords = LatLonCoordinates(
                lat=coords_state["lat"],
                lon=coords_state["lon"],
            )
        elif self.fine_coordinates_path is not None:
            model.fine_coords = self._load_fine_coords_from_path(
                self.fine_coordinates_path
            )

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
