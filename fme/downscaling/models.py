import dataclasses
from collections.abc import Mapping
from typing import Any

import dacite
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.optimization import NullOptimization, Optimization
from fme.core.packer import Packer
from fme.core.rand import randn, randn_like
from fme.core.typing_ import TensorDict, TensorMapping
from fme.downscaling.data import (
    BatchData,
    PairedBatchData,
    StaticInputs,
    Topography,
    get_normalized_topography,
)
from fme.downscaling.metrics_and_maths import filter_tensor_mapping, interpolate
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.samplers import stochastic_sampler as edm_sampler
from fme.downscaling.typing_ import FineResCoarseResPair


@dataclasses.dataclass
class ModelOutputs:
    prediction: TensorDict
    target: TensorDict
    loss: torch.Tensor
    latent_steps: list[torch.Tensor] = dataclasses.field(default_factory=list)


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
    in_names: list[str]
    out_names: list[str]
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
        coarse_shape: tuple[int, int],
        downscale_factor: int,
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
    rnd_normal = randn([targets_norm.shape[0], 1, 1, 1], device=targets_norm.device)
    # This is taken from EDM's original implementation in EDMLoss:
    # https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py#L72-L80  # noqa: E501
    sigma = (rnd_normal * p_std + p_mean).exp()
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    noise = randn_like(targets_norm) * sigma
    latents = targets_norm + noise
    return ConditionedTarget(latents=latents, sigma=sigma, weight=weight)


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
            static_inputs: Optional static inputs to the model that may be loaded
                from saved checkpoint. If required by the model but not passed at
                init, they are expected to be provided in the loaded dataset.
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
        self.static_inputs = static_inputs

    @property
    def modules(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([self.module])

    @property
    def fine_shape(self) -> tuple[int, int]:
        return self._get_fine_shape(self.coarse_shape)

    def _get_fine_shape(self, coarse_shape: tuple[int, int]) -> tuple[int, int]:
        """
        Calculate the fine shape based on the coarse shape and downscale factor.
        """
        return (
            coarse_shape[0] * self.downscale_factor,
            coarse_shape[1] * self.downscale_factor,
        )

    def _get_input_from_coarse(
        self, coarse: TensorMapping, topography: Topography | None
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
            else:
                n_batches = normalized.shape[0]
                # Join the normalized topography to the input (see dataset for details)
                topo = topography.data.unsqueeze(0).repeat(n_batches, 1, 1)
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
        topography: Topography | None,
        optimizer: Optimization | NullOptimization,
    ) -> ModelOutputs:
        """Performs a denoising training step on a batch of data."""
        coarse, fine = batch.coarse.data, batch.fine.data
        inputs_norm = self._get_input_from_coarse(coarse, topography)
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
            conditioned_target.weight * self.loss(denoised_norm, targets_norm)
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
    def generate(
        self,
        coarse_data: TensorMapping,
        topography: torch.Tensor | None,
        n_samples: int = 1,
    ) -> tuple[TensorDict, torch.Tensor, list[torch.Tensor]]:
        inputs_ = self._get_input_from_coarse(coarse_data, topography)
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
        topography: Topography | None,
        n_samples: int = 1,
    ) -> TensorDict:
        generated, _, _ = self.generate(batch.data, topography, n_samples)
        return generated

    @torch.no_grad()
    def generate_on_batch(
        self,
        batch: PairedBatchData,
        topography: Topography | None,
        n_samples: int = 1,
    ) -> ModelOutputs:
        coarse, fine = batch.coarse.data, batch.fine.data
        generated, generated_norm, latent_steps = self.generate(
            coarse, topography, n_samples
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
            static_inputs_state = self.static_inputs.to_state()
        else:
            static_inputs_state = None

        return {
            "config": self.config.get_state(),
            "module": self.module.state_dict(),
            "coarse_shape": self.coarse_shape,
            "downscale_factor": self.downscale_factor,
            "static_inputs": static_inputs_state,
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
        fine_topography_path: Optional path to the fine topography file, if needed.
            This is useful when no fine res data is used during evaluation but the
            model still needs fine res static input data.
        model_updates: Optional mapping of {key: new_value} model config updates to
            apply when loading the model. This is useful for running evaluation with
            updated parameters than at training time. Use with caution; not all
            parameters can or should be updated at evaluation time.
    """

    checkpoint_path: str
    rename: dict[str, str] | None = None
    fine_topography_path: str | None = None
    model_updates: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # For config validation testing, we don't want to load immediately
        # so we defer until build or properties are accessed.
        self._checkpoint_is_loaded = False
        self._rename = self.rename or {}
        if "module" in (self.model_updates or {}):
            raise ValueError("'module' cannot be updated in model_updates.")

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

            self._checkpoint_data = checkpoint_data
            self._checkpoint_is_loaded = True
            if self.model_updates is not None:
                for k, v in self.model_updates.items():
                    checkpoint_data["model"]["config"][k] = v
        return self._checkpoint_data

    def build(
        self,
    ) -> DiffusionModel:
        if self._checkpoint["model"]["static_inputs"] is not None:
            static_inputs = StaticInputs.from_state(
                self._checkpoint["model"]["static_inputs"]
            )
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

    def get_topography(self) -> Topography | None:
        if self.data_requirements.use_fine_topography:
            if self.fine_topography_path is None:
                raise ValueError(
                    "Topography path must be provided for model configured "
                    "to use fine topography."
                )
            return get_normalized_topography(self.fine_topography_path).to_device()
        else:
            return None
