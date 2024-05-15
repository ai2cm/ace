import dataclasses
from typing import Any, List, Mapping, Tuple, Union

import dacite
import torch

from fme.core.device import get_device
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.optimization import NullOptimization, Optimization
from fme.core.packer import Packer
from fme.core.typing_ import TensorMapping
from fme.downscaling.metrics_and_maths import filter_tensor_mapping
from fme.downscaling.modules.registry import ModuleRegistrySelector
from fme.downscaling.requirements import DataRequirements
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
        loss = self.loss.build(area_weights.fine)

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
        self.module = module.to(get_device())
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
