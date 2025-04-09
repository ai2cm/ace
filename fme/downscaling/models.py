import dataclasses
import logging
from typing import Any, List, Mapping, Optional, Tuple, Union

import dacite
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.optimization import NullOptimization, Optimization
from fme.core.packer import Packer
from fme.core.typing_ import TensorDict, TensorMapping
from fme.downscaling.datasets import PairedBatchData
from fme.downscaling.metrics_and_maths import filter_tensor_mapping, interpolate
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.modules.registry import ModuleRegistrySelector
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.samplers import edm_sampler
from fme.downscaling.typing_ import FineResCoarseResPair


@dataclasses.dataclass
class ModelOutputs:
    prediction: TensorDict
    target: TensorDict
    loss: torch.Tensor
    latent_steps: List[torch.Tensor] = dataclasses.field(default_factory=list)


def _tensor_mapping_to_device(
    tensor_mapping: TensorMapping, device: torch.device
) -> TensorMapping:
    return {k: v.to(device) for k, v in tensor_mapping.items()}


@dataclasses.dataclass
class PairedNormalizationConfig:
    fine: NormalizationConfig
    coarse: NormalizationConfig

    def build(
        self, in_names: List[str], out_names: List[str]
    ) -> FineResCoarseResPair[StandardNormalizer]:
        return FineResCoarseResPair[StandardNormalizer](
            coarse=self.coarse.build(list(set(in_names).union(out_names))),
            fine=self.fine.build(out_names),
        )

    def load(self):
        """
        Load the normalization configuration from the netCDF files.

        Updates the configuration so it no longer requires external files.
        """
        self.fine.load()
        self.coarse.load()


@dataclasses.dataclass
class DownscalingModelConfig:
    module: ModuleRegistrySelector
    loss: LossConfig
    in_names: List[str]
    out_names: List[str]
    normalization: PairedNormalizationConfig
    use_fine_topography: bool = False

    def __post_init__(self):
        self._interpolate_input = self.module.expects_interpolated_input
        if self.use_fine_topography and not self._interpolate_input:
            raise ValueError(
                "Fine topography can only be used when predicting on interpolated"
                " coarse input"
            )

    def build(
        self,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
    ) -> "Model":
        normalizer = self.normalization.build(self.in_names, self.out_names)
        loss = self.loss.build(reduction="mean")
        n_in_channels = len(self.in_names)

        if self.use_fine_topography:
            n_in_channels += 1

        module = self.module.build(
            n_in_channels=n_in_channels,
            n_out_channels=len(self.out_names),
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
        )
        return Model(
            module,
            normalizer,
            loss,
            coarse_shape,
            downscale_factor,
            self,
        )

    def get_state(self) -> Mapping[str, Any]:
        # Update normalization configuration so it no longer requires external files.
        self.normalization.load()
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DownscalingModelConfig":
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


class Model:
    def __init__(
        self,
        module: torch.nn.Module,
        normalizer: FineResCoarseResPair[StandardNormalizer],
        loss: torch.nn.Module,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        config: DownscalingModelConfig,
    ) -> None:
        self.coarse_shape = coarse_shape
        self.downscale_factor = downscale_factor
        dist = Distributed.get_instance()
        self.module = dist.wrap_module(module.to(get_device()))
        self.normalizer = normalizer
        self.loss = loss
        self.in_packer = Packer(config.in_names)
        self.out_packer = Packer(config.out_names)
        self.config = config
        self.null_optimization = NullOptimization()
        self._channel_axis = -3

    @property
    def modules(self) -> torch.nn.ModuleList:
        """
        Returns:
            A list of modules being trained.
        """
        return torch.nn.ModuleList([self.module])

    def train_on_batch(
        self,
        batch: PairedBatchData,
        optimization: Union[Optimization, NullOptimization],
    ) -> ModelOutputs:
        return self._run_on_batch(batch, optimization)

    def generate_on_batch(
        self,
        batch: PairedBatchData,
        n_samples: int = 1,
    ) -> ModelOutputs:
        if n_samples != 1:
            raise ValueError("n_samples must be 1 for deterministic models")
        result = self._run_on_batch(batch, self.null_optimization)
        for k, v in result.prediction.items():
            result.prediction[k] = v.unsqueeze(1)  # insert sample dimension
        for k, v in result.target.items():
            result.target[k] = v.unsqueeze(1)
        return result

    def _run_on_batch(
        self,
        batch: PairedBatchData,
        optimizer: Union[Optimization, NullOptimization],
    ) -> ModelOutputs:
        coarse, fine = batch.coarse.data, batch.fine.data

        coarse_inputs = filter_tensor_mapping(coarse, self.in_packer.names)
        coarse_norm = self.in_packer.pack(
            self.normalizer.coarse.normalize(coarse_inputs), axis=self._channel_axis
        )
        interpolated = interpolate(coarse_norm, self.downscale_factor)

        if self.config.use_fine_topography:
            if batch.fine.topography is None:
                raise ValueError(
                    "Topography must be provided for each batch when use of fine "
                    "topography is enabled."
                )

            # Join the normalized topography to the input (see dataset for details)
            topo = batch.fine.topography.unsqueeze(self._channel_axis)
            coarse_norm = torch.concat([interpolated, topo], axis=self._channel_axis)
        elif self.config._interpolate_input:
            coarse_norm = interpolated

        targets_norm = self.out_packer.pack(
            self.normalizer.fine.normalize(dict(fine)), axis=self._channel_axis
        )
        predicted_norm = self.module(coarse_norm)
        loss = self.loss(predicted_norm, targets_norm)
        optimizer.accumulate_loss(loss)
        optimizer.step_weights()
        target = filter_tensor_mapping(fine, set(self.out_packer.names))
        prediction = self.normalizer.fine.denormalize(
            self.out_packer.unpack(predicted_norm, axis=self._channel_axis)
        )
        return ModelOutputs(
            prediction=prediction, target=target, loss=loss, latent_steps=[]
        )

    def get_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.get_state(),
            "module": self.module.state_dict(),
            "coarse_shape": self.coarse_shape,
            "downscale_factor": self.downscale_factor,
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
    ) -> "Model":
        config = DownscalingModelConfig.from_state(state["config"])
        model = config.build(
            state["coarse_shape"],
            state["downscale_factor"],
        )
        model.module.load_state_dict(state["module"], strict=True)
        return model


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
        p_mean: The mean of noise distribution used during training.
        p_std: The std of the noise distribution used during training.
        sigma_min: Min noise level for generation.
        sigma_max: Max noise level for generation.
        churn: The amount of stochasticity during generation.
        num_diffusion_generation_steps: Number of diffusion generation steps
        use_fine_topography: Whether to use fine topography in the model.
    """

    module: DiffusionModuleRegistrySelector
    loss: LossConfig
    in_names: List[str]
    out_names: List[str]
    normalization: PairedNormalizationConfig
    p_mean: float
    p_std: float
    sigma_min: float
    sigma_max: float
    churn: float
    num_diffusion_generation_steps: int
    predict_residual: bool
    use_fine_topography: bool = False

    def __post_init__(self):
        self._interpolate_input = self.module.expects_interpolated_input
        if self.use_fine_topography and not self._interpolate_input:
            raise ValueError(
                "Fine topography can only be used when predicting on interpolated"
                " coarse input"
            )

    def build(
        self,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
    ) -> "DiffusionModel":
        normalizer = self.normalization.build(self.in_names, self.out_names)
        loss = self.loss.build("none")
        # We always use standard score normalization, so sigma_data is
        # always 1.0. See below for standard score normalization:
        # https://en.wikipedia.org/wiki/Standard_score
        sigma_data = 1.0

        n_in_channels = len(self.in_names)
        # fine topography is already normalized and at fine scale, so needs
        # some special handling for now
        if self.use_fine_topography:
            n_in_channels += 1

        module = self.module.build(
            n_in_channels=n_in_channels,
            n_out_channels=len(self.out_names),
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
            sigma_data=sigma_data,
        )
        return DiffusionModel(
            config=self,
            module=module,
            normalizer=normalizer,
            loss=loss,
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
            sigma_data=sigma_data,
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


def _separate_interleaved_samples(
    tensor: torch.Tensor, n_batch: int, n_samples: int
) -> torch.Tensor:
    """
    Reshape an interleaved tensor to have a leading [batch, sample, ...] dimension.
    """
    return tensor.reshape(n_batch, n_samples, *tensor.shape[1:])


@dataclasses.dataclass
class ConditionedTarget:
    """
    A class to hold the conditioned targets and the loss weighting.

    Attributes:
        latents: The normalized targets with noise added.
        sigma: The noise level.
        weight: The loss weighting.
    """

    latents: torch.Tensor
    sigma: torch.Tensor
    weight: torch.Tensor


def condition_with_noise_for_training(
    targets_norm: torch.Tensor,
    p_std: float,
    p_mean: float,
    sigma_data: float,
) -> ConditionedTarget:
    """
    Condition the targets with noise for training.

    Args:
        targets_norm: The normalized targets.
        p_std: The standard deviation of the noise distribution used during training.
        p_mean: The mean of the noise distribution used during training.
        sigma_data: The standard deviation of the data,
            used to determine loss weighting.

    Returns:
        The conditioned targets and the loss weighting.
    """
    rnd_normal = torch.randn(
        [targets_norm.shape[0], 1, 1, 1], device=targets_norm.device
    )
    # This is taken from EDM's original implementation in EDMLoss:
    # https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py#L72-L80  # noqa: E501
    sigma = (rnd_normal * p_std + p_mean).exp()
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    noise = torch.randn_like(targets_norm) * sigma
    latents = targets_norm + noise
    return ConditionedTarget(latents=latents, sigma=sigma, weight=weight)


class DiffusionModel:
    def __init__(
        self,
        config: DiffusionModelConfig,
        module: torch.nn.Module,
        normalizer: FineResCoarseResPair[StandardNormalizer],
        loss: torch.nn.Module,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        sigma_data: float,
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
                coarse-resolution input data.
            downscale_factor: The factor by which the data is downscaled from
                coarse to fine.
            sigma_data: The standard deviation of the data, used for diffusion
                model preconditioning.
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

    @property
    def modules(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([self.module])

    def _get_input_from_coarse(
        self, coarse: TensorMapping, topography: Optional[torch.Tensor]
    ) -> torch.Tensor:
        inputs = filter_tensor_mapping(coarse, self.in_packer.names)
        normalized = self.in_packer.pack(
            self.normalizer.coarse.normalize(inputs), axis=self._channel_axis
        )
        interpolated = interpolate(normalized, self.downscale_factor)

        if self.config.use_fine_topography:
            if topography is None:
                raise ValueError(
                    "Topography must be provided for each batch when use of fine "
                    "topography is enabled."
                )
            # Join the normalized topography to the input (see dataset for details)
            topo = topography.unsqueeze(self._channel_axis)
            interpolated = torch.concat([interpolated, topo], axis=self._channel_axis)

        if self.config._interpolate_input:
            return interpolated
        return normalized

    def train_on_batch(
        self,
        batch: PairedBatchData,
        optimizer: Union[Optimization, NullOptimization],
    ) -> ModelOutputs:
        """Performs a denoising training step on a batch of data."""
        coarse, fine = batch.coarse.data, batch.fine.data
        inputs_norm = self._get_input_from_coarse(coarse, batch.fine.topography)
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
            targets_norm, self.config.p_std, self.config.p_mean, self.sigma_data
        )

        denoised_norm = self.module(
            conditioned_target.latents, inputs_norm, conditioned_target.sigma
        )
        loss = torch.mean(
            conditioned_target.weight * self.loss(targets_norm, denoised_norm)
        )
        optimizer.accumulate_loss(loss)
        optimizer.step_weights()

        if self.config.predict_residual:
            denoised_norm = denoised_norm + base_prediction

        target = filter_tensor_mapping(batch.fine.data, set(self.out_packer.names))
        denoised = self.normalizer.fine.denormalize(
            self.out_packer.unpack(denoised_norm, axis=self._channel_axis)
        )
        return ModelOutputs(
            prediction=denoised, target=target, loss=loss, latent_steps=[]
        )

    @torch.no_grad()
    def generate_on_batch(
        self,
        batch: PairedBatchData,
        n_samples: int = 1,
    ) -> ModelOutputs:
        coarse, fine = batch.coarse.data, batch.fine.data
        inputs_ = self._get_input_from_coarse(coarse, batch.fine.topography)

        targets_norm = self.out_packer.pack(
            self.normalizer.fine.normalize(dict(fine)), axis=self._channel_axis
        )

        n_batch = targets_norm.shape[0]
        # expand samples and fold to [batch * n_samples, output_channels, height, width]
        inputs_ = _repeat_batch_by_samples(inputs_, n_samples)
        targets_norm = _repeat_batch_by_samples(targets_norm, n_samples)
        latents = torch.randn_like(targets_norm)

        logging.info("Running EDM sampler...")
        samples_norm, latent_steps = edm_sampler(
            self.module,
            latents,
            inputs_,
            S_churn=self.config.churn,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            num_steps=self.config.num_diffusion_generation_steps,
        )
        logging.info("Done running EDM sampler.")

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
            samples_norm += _repeat_batch_by_samples(base_prediction, n_samples)

        loss = self.loss(targets_norm, samples_norm)

        samples_norm_reshaped = _separate_interleaved_samples(
            samples_norm, n_batch, n_samples
        )
        samples = self.normalizer.fine.denormalize(
            self.out_packer.unpack(samples_norm_reshaped, axis=self._channel_axis)
        )
        targets = filter_tensor_mapping(batch.fine.data, set(self.out_packer.names))
        targets = {k: v.unsqueeze(1) for k, v in targets.items()}

        return ModelOutputs(
            prediction=samples, target=targets, loss=loss, latent_steps=latent_steps
        )

    def get_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.get_state(),
            "module": self.module.state_dict(),
            "coarse_shape": self.coarse_shape,
            "downscale_factor": self.downscale_factor,
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
    ) -> "DiffusionModel":
        config = DiffusionModelConfig.from_state(state["config"])
        model = config.build(
            state["coarse_shape"],
            state["downscale_factor"],
        )
        model.module.load_state_dict(state["module"], strict=True)
        return model
