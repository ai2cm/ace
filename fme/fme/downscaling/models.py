import dataclasses
from typing import Any, List, Mapping, Tuple, Union

import dacite
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.optimization import NullOptimization, Optimization
from fme.core.packer import Packer
from fme.core.typing_ import TensorMapping
from fme.downscaling.metrics_and_maths import filter_tensor_mapping
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.modules.registry import ModuleRegistrySelector
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.samplers import edm_sampler
from fme.downscaling.typing_ import FineResCoarseResPair


@dataclasses.dataclass
class ModelOutputs:
    prediction: TensorMapping
    target: TensorMapping
    loss: torch.Tensor


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
            coarse=self.coarse.build(in_names),
            fine=self.fine.build(out_names),
        )


@dataclasses.dataclass
class DownscalingModelConfig:
    module: ModuleRegistrySelector
    loss: LossConfig
    in_names: List[str]
    out_names: List[str]
    normalization: PairedNormalizationConfig
    use_fine_topography: bool

    def build(
        self,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        area_weights: FineResCoarseResPair[torch.Tensor],
        fine_topography: torch.Tensor,
    ) -> "Model":
        normalizer = self.normalization.build(self.in_names, self.out_names)
        loss = self.loss.build(area_weights.fine, reduction="mean")
        module = self.module.build(
            n_in_channels=len(self.in_names),
            n_out_channels=len(self.out_names),
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
            fine_topography=fine_topography if self.use_fine_topography else None,
        )
        return Model(
            module,
            normalizer,
            loss,
            self.in_names,
            self.out_names,
            coarse_shape,
            downscale_factor,
            self,
        )

    def get_state(self) -> Mapping[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DownscalingModelConfig":
        return dacite.from_dict(data_class=cls, data=state)

    @property
    def data_requirements(self) -> DataRequirements:
        return DataRequirements(
            names=list(set(self.in_names).union(self.out_names)),
            n_timesteps=1,
            use_fine_topography=self.use_fine_topography,
        )


class Model:
    def __init__(
        self,
        module: torch.nn.Module,
        normalizer: FineResCoarseResPair[StandardNormalizer],
        loss: torch.nn.Module,
        in_names: List[str],
        out_names: List[str],
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
        self.in_packer = Packer(in_names)
        self.out_packer = Packer(out_names)
        self.config = config
        self.null_optimization = NullOptimization()

    @property
    def modules(self) -> torch.nn.ModuleList:
        """
        Returns:
            A list of modules being trained.
        """
        return torch.nn.ModuleList([self.module])

    def train_on_batch(
        self, batch: FineResCoarseResPair[TensorMapping], optimization: Optimization
    ) -> ModelOutputs:
        return self._run_on_batch(batch, optimization)

    def generate_on_batch(
        self, batch: FineResCoarseResPair[TensorMapping]
    ) -> ModelOutputs:
        return self._run_on_batch(batch, self.null_optimization)

    def _run_on_batch(
        self,
        batch: FineResCoarseResPair[TensorMapping],
        optimizer: Union[Optimization, NullOptimization],
    ) -> ModelOutputs:
        channel_axis = -3
        coarse, fine = _tensor_mapping_to_device(
            batch.coarse, get_device()
        ), _tensor_mapping_to_device(batch.fine, get_device())
        inputs_norm = self.in_packer.pack(
            self.normalizer.coarse.normalize(dict(coarse)), axis=channel_axis
        )
        targets_norm = self.out_packer.pack(
            self.normalizer.fine.normalize(dict(fine)), axis=channel_axis
        )
        predicted_norm = self.module(inputs_norm)
        loss = self.loss(predicted_norm, targets_norm)
        optimizer.step_weights(loss)
        target = filter_tensor_mapping(batch.fine, set(self.out_packer.names))
        prediction = self.normalizer.fine.denormalize(
            self.out_packer.unpack(predicted_norm, axis=channel_axis)
        )
        return ModelOutputs(prediction=prediction, target=target, loss=loss)

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
        area_weights: FineResCoarseResPair[torch.Tensor],
        fine_topography: torch.Tensor,
    ) -> "Model":
        config = DownscalingModelConfig.from_state(state["config"])
        model = config.build(
            state["coarse_shape"],
            state["downscale_factor"],
            area_weights,
            fine_topography,
        )
        model.module.load_state_dict(state["module"], strict=True)
        return model


@dataclasses.dataclass
class DiffusionModelConfig:
    """
    This class implements or wraps the algorithms described in `EDM`_.

    .. _EDM: https://arxiv.org/abs/2206.00364

    Attributes:
        module: The module registry selector for the diffusion model.
        loss:  The loss configuration for the diffusion model.
        in_names: The input variable names for the diffusion model.
        out_names: The output variable names for the diffusion model.
        normalization: The normalization configurations for the diffusion model.
        use_fine_topography: Indicates whether to use the fine topography.
        p_mean: The mean of noise distribution used during training.
        p_std: The std of the noise distribution used during training.
        sigma_min: Min noise level for generation.
        sigma_max: Max noise level for generation.
        churn: The amount of stochasticity during generation.
        num_diffusion_generation_steps: Number of diffusion generation steps.
    """

    module: DiffusionModuleRegistrySelector
    loss: LossConfig
    in_names: List[str]
    out_names: List[str]
    normalization: PairedNormalizationConfig
    use_fine_topography: bool
    p_mean: float
    p_std: float
    sigma_min: float
    sigma_max: float
    churn: float
    num_diffusion_generation_steps: int

    def build(
        self,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        area_weights: FineResCoarseResPair[torch.Tensor],
        fine_topography: torch.Tensor,
    ) -> "DiffusionModel":
        normalizer = self.normalization.build(self.in_names, self.out_names)
        loss = self.loss.build(area_weights.fine, "none")
        # We always use standard score normalization, so sigma_data is
        # always 1.0. See below for standard score normalization:
        # https://en.wikipedia.org/wiki/Standard_score
        sigma_data = 1.0

        module = self.module.build(
            n_in_channels=len(self.in_names),
            n_out_channels=len(self.out_names),
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
            fine_topography=fine_topography if self.use_fine_topography else None,
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
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DiffusionModelConfig":
        return dacite.from_dict(data_class=cls, data=state)

    @property
    def data_requirements(self) -> DataRequirements:
        return DataRequirements(
            names=list(set(self.in_names).union(self.out_names)),
            n_timesteps=1,
            use_fine_topography=self.use_fine_topography,
        )


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

    @property
    def modules(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([self.module])

    def train_on_batch(
        self,
        batch: FineResCoarseResPair[TensorMapping],
        optimizer: Optimization,
    ) -> ModelOutputs:
        """Performs a denoising training step on a batch of data."""

        channel_axis = -3
        coarse, fine = _tensor_mapping_to_device(
            batch.coarse, get_device()
        ), _tensor_mapping_to_device(batch.fine, get_device())
        coarse_norm = self.in_packer.pack(
            self.normalizer.coarse.normalize(dict(coarse)), axis=channel_axis
        )
        targets_norm = self.out_packer.pack(
            self.normalizer.fine.normalize(dict(fine)), axis=channel_axis
        )

        rnd_normal = torch.randn(
            [targets_norm.shape[0], 1, 1, 1], device=targets_norm.device
        )
        # This is taken from EDM's original implementation in EDMLoss:
        # https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py#L72-L80  # noqa: E501
        sigma = (rnd_normal * self.config.p_std + self.config.p_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(targets_norm) * sigma
        latents = targets_norm + noise

        denoised_norm = self.module(latents, coarse_norm, sigma)
        batch_size = len(weight)
        loss = torch.sum(weight * self.loss(targets_norm, denoised_norm)) / batch_size

        optimizer.step_weights(loss)
        target = filter_tensor_mapping(batch.fine, set(self.out_packer.names))
        denoised = self.normalizer.fine.denormalize(
            self.out_packer.unpack(denoised_norm, axis=channel_axis)
        )
        return ModelOutputs(prediction=denoised, target=target, loss=loss)

    @torch.no_grad()
    def generate_on_batch(
        self, batch: FineResCoarseResPair[TensorMapping]
    ) -> ModelOutputs:
        channel_axis = -3
        coarse, fine = _tensor_mapping_to_device(
            batch.coarse, get_device()
        ), _tensor_mapping_to_device(batch.fine, get_device())
        coarse_norm = self.in_packer.pack(
            self.normalizer.coarse.normalize(dict(coarse)), axis=channel_axis
        )
        targets_norm = self.out_packer.pack(
            self.normalizer.fine.normalize(dict(fine)), axis=channel_axis
        )
        latents = torch.rand_like(targets_norm)
        samples_norm = edm_sampler(
            self.module,
            latents,
            coarse_norm,
            S_churn=self.config.churn,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            num_steps=self.config.num_diffusion_generation_steps,
        )
        samples = self.normalizer.fine.denormalize(
            self.out_packer.unpack(samples_norm, axis=channel_axis)
        )
        loss = self.loss(targets_norm, samples_norm)
        return ModelOutputs(
            prediction=samples,
            target=batch.fine,
            loss=loss,
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
        area_weights: FineResCoarseResPair[torch.Tensor],
        fine_topography: torch.Tensor,
    ) -> "DiffusionModel":
        config = DiffusionModelConfig.from_state(state["config"])
        model = config.build(
            state["coarse_shape"],
            state["downscale_factor"],
            area_weights,
            fine_topography,
        )
        model.module.load_state_dict(state["module"], strict=True)
        return model
