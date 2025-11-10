import dataclasses
from collections.abc import Mapping
from typing import Any, Literal

import dacite
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.optimization import NullOptimization, Optimization
from fme.core.packer import Packer
from fme.core.typing_ import TensorDict
from fme.downscaling.data import BatchData, PairedBatchData, Topography
from fme.downscaling.metrics_and_maths import filter_tensor_mapping, interpolate
from fme.downscaling.models import ModelOutputs, PairedNormalizationConfig
from fme.downscaling.modules.registry import ModuleRegistrySelector
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.typing_ import FineResCoarseResPair


@dataclasses.dataclass
class DeterministicModelConfig:
    module: ModuleRegistrySelector
    loss: LossConfig
    in_names: list[str]
    out_names: list[str]
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
        coarse_shape: tuple[int, int],
        downscale_factor: int,
        rename: dict[str, str] | None = None,
    ) -> "DeterministicModel":
        invert_rename = {v: k for k, v in (rename or {}).items()}
        orig_in_names = [invert_rename.get(name, name) for name in self.in_names]
        orig_out_names = [invert_rename.get(name, name) for name in self.out_names]
        normalizer = self.normalization.build(orig_in_names, orig_out_names, rename)
        loss = self.loss.build(reduction="mean", gridded_operations=None)
        n_in_channels = len(self.in_names)

        if self.use_fine_topography:
            n_in_channels += 1

        module = self.module.build(
            n_in_channels=n_in_channels,
            n_out_channels=len(self.out_names),
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
        )
        return DeterministicModel(
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
    def from_state(cls, state: Mapping[str, Any]) -> "DeterministicModelConfig":
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


class DeterministicModel:
    def __init__(
        self,
        module: torch.nn.Module,
        normalizer: FineResCoarseResPair[StandardNormalizer],
        loss: torch.nn.Module,
        coarse_shape: tuple[int, int],
        downscale_factor: int,
        config: DeterministicModelConfig,
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
        topography: Topography | None,
        optimization: Optimization | NullOptimization,
    ) -> ModelOutputs:
        return self._run_on_batch(batch, topography, optimization)

    def generate_on_batch(
        self,
        batch: PairedBatchData,
        topography: Topography | None,
        n_samples: int = 1,
    ) -> ModelOutputs:
        if n_samples != 1:
            raise ValueError("n_samples must be 1 for deterministic models")
        result = self._run_on_batch(batch, topography, self.null_optimization)
        for k, v in result.prediction.items():
            result.prediction[k] = v.unsqueeze(1)  # insert sample dimension
        for k, v in result.target.items():
            result.target[k] = v.unsqueeze(1)
        return result

    def generate_on_batch_no_target(
        self,
        batch: BatchData,
        topography: Topography | None,
        n_samples: int = 1,
    ) -> TensorDict:
        raise NotImplementedError(
            "This method is not implemented for the base Model class. "
        )

    def _run_on_batch(
        self,
        batch: PairedBatchData,
        topography: Topography | None,
        optimizer: Optimization | NullOptimization,
    ) -> ModelOutputs:
        coarse, fine = batch.coarse.data, batch.fine.data

        coarse_inputs = filter_tensor_mapping(coarse, self.in_packer.names)
        coarse_norm = self.in_packer.pack(
            self.normalizer.coarse.normalize(coarse_inputs), axis=self._channel_axis
        )
        interpolated = interpolate(coarse_norm, self.downscale_factor)

        if self.config.use_fine_topography:
            if topography is None:
                raise ValueError(
                    "Topography must be provided for each batch when use of fine "
                    "topography is enabled."
                )
            else:
                # Join the normalized topography to the input (see dataset for details)
                topo = topography.data.unsqueeze(self._channel_axis)
                coarse_norm = torch.concat(
                    [interpolated, topo], axis=self._channel_axis
                )
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
    ) -> "DeterministicModel":
        config = DeterministicModelConfig.from_state(state["config"])
        model = config.build(
            state["coarse_shape"],
            state["downscale_factor"],
        )
        model.module.load_state_dict(state["module"], strict=True)
        return model


@dataclasses.dataclass
class InterpolateModelConfig:
    mode: Literal["bicubic", "nearest"]
    downscale_factor: int
    in_names: list[str]
    out_names: list[str]

    def build(
        self,
    ) -> DeterministicModel:
        module = ModuleRegistrySelector(type="interpolate", config={"mode": self.mode})
        var_names = list(set(self.in_names).union(set(self.out_names)))
        normalization_config = PairedNormalizationConfig(
            NormalizationConfig(
                means={var_name: 0.0 for var_name in var_names},
                stds={var_name: 1.0 for var_name in var_names},
            ),
            NormalizationConfig(
                means={var_name: 0.0 for var_name in var_names},
                stds={var_name: 1.0 for var_name in var_names},
            ),
        )
        return DeterministicModelConfig(
            module,
            LossConfig("NaN"),
            self.in_names,
            self.out_names,
            normalization_config,
        ).build(
            (-1, -1),
            self.downscale_factor,
        )

    @property
    def data_requirements(self) -> DataRequirements:
        return DataRequirements(
            fine_names=self.out_names,
            coarse_names=list(set(self.in_names).union(self.out_names)),
            n_timesteps=1,
        )
