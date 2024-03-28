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
from fme.downscaling.typing_ import HighResLowResPair


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
    highres: NormalizationConfig
    lowres: NormalizationConfig

    def build(
        self, in_names: List[str], out_names: List[str]
    ) -> HighResLowResPair[StandardNormalizer]:
        return HighResLowResPair[StandardNormalizer](
            lowres=self.lowres.build(in_names),
            highres=self.highres.build(out_names),
        )


@dataclasses.dataclass
class DownscalingModelConfig:
    module: ModuleRegistrySelector
    loss: LossConfig
    in_names: List[str]
    out_names: List[str]
    normalization: PairedNormalizationConfig

    def build(
        self,
        lowres_shape: Tuple[int, int],
        downscale_factor: int,
        area_weights: HighResLowResPair[torch.Tensor],
    ) -> "Model":
        normalizer = self.normalization.build(self.in_names, self.out_names)
        loss = self.loss.build(area_weights.highres)
        module = self.module.build(
            n_in_channels=len(self.in_names),
            n_out_channels=len(self.out_names),
            lowres_shape=lowres_shape,
            downscale_factor=downscale_factor,
        )
        return Model(
            module,
            normalizer,
            loss,
            self.in_names,
            self.out_names,
            lowres_shape,
            downscale_factor,
            self,
        )

    def get_state(self) -> Mapping[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DownscalingModelConfig":
        return dacite.from_dict(data_class=cls, data=state)


class Model:
    def __init__(
        self,
        module: torch.nn.Module,
        normalizer: HighResLowResPair[StandardNormalizer],
        loss: torch.nn.Module,
        in_names: List[str],
        out_names: List[str],
        lowres_shape: Tuple[int, int],
        downscale_factor: int,
        config: DownscalingModelConfig,
    ) -> None:
        self.lowres_shape = lowres_shape
        self.downscale_factor = downscale_factor
        self.module = module.to(get_device())
        self.normalizer = normalizer
        self.loss = loss
        self.in_packer = Packer(in_names)
        self.out_packer = Packer(out_names)
        self.config = config

    @property
    def modules(self) -> torch.nn.ModuleList:
        """
        Returns:
            A list of modules being trained.
        """
        return torch.nn.ModuleList([self.module])

    def count_parameters(self) -> int:
        """Counts the number of differentiable parameters in the model."""
        num_parameters = 0
        for parameter in self.module.parameters():
            if parameter.requires_grad:
                num_parameters += parameter.numel()
        return num_parameters

    def run_on_batch(
        self,
        batch: HighResLowResPair[TensorMapping],
        optimizer: Union[Optimization, NullOptimization],
    ) -> ModelOutputs:
        channel_axis = -3
        lowres, highres = _tensor_mapping_to_device(
            batch.lowres, get_device()
        ), _tensor_mapping_to_device(batch.highres, get_device())
        inputs_norm = self.in_packer.pack(
            self.normalizer.lowres.normalize(dict(lowres)), axis=channel_axis
        )
        targets_norm = self.out_packer.pack(
            self.normalizer.highres.normalize(dict(highres)), axis=channel_axis
        )
        predicted_norm = self.module(inputs_norm)
        loss = self.loss(predicted_norm, targets_norm)
        optimizer.step_weights(loss)
        target = filter_tensor_mapping(batch.highres, set(self.out_packer.names))
        prediction = self.normalizer.highres.denormalize(
            self.out_packer.unpack(predicted_norm, axis=channel_axis)
        )
        return ModelOutputs(prediction=prediction, target=target, loss=loss)

    def get_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.get_state(),
            "module": self.module.state_dict(),
            "lowres_shape": self.lowres_shape,
            "downscale_factor": self.downscale_factor,
        }

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any], area_weights: HighResLowResPair[torch.Tensor]
    ) -> "Model":
        config = DownscalingModelConfig.from_state(state["config"])
        model = config.build(
            state["lowres_shape"], state["downscale_factor"], area_weights
        )
        model.module.load_state_dict(state["module"], strict=True)
        return model
