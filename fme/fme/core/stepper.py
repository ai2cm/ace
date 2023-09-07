import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import dacite
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from fme.core.aggregator import InferenceAggregator, NullAggregator, OneStepAggregator
from fme.core.data_loading.requirements import DataRequirements
from fme.core.device import get_device, using_gpu
from fme.core.distributed import Distributed
from fme.core.loss import LossConfig
from fme.core.normalizer import (
    FromStateNormalizer,
    NormalizationConfig,
    StandardNormalizer,
)
from fme.core.packer import Packer
from fme.core.prescriber import NullPrescriber, Prescriber, PrescriberConfig
from fme.fcn_training.registry import ModuleSelector

from .optimization import DisabledOptimizationConfig, NullOptimization, Optimization


@dataclasses.dataclass
class SingleModuleStepperConfig:
    builder: ModuleSelector
    in_names: List[str]
    out_names: List[str]
    normalization: Union[NormalizationConfig, FromStateNormalizer]
    optimization: Optional[DisabledOptimizationConfig] = None
    prescriber: Optional[PrescriberConfig] = None
    loss: LossConfig = dataclasses.field(default_factory=lambda: LossConfig())

    def get_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        return DataRequirements(
            names=self.all_names,
            in_names=self.in_names,
            out_names=self.out_names,
            n_timesteps=n_forward_steps + 1,
        )

    def get_state(self):
        return dataclasses.asdict(self)

    def get_stepper(
        self,
        shapes: Dict[str, Tuple[int, ...]],
        area: Optional[torch.Tensor],
    ):
        return SingleModuleStepper(
            config=self,
            data_shapes=shapes,
            area=area,
        )

    @classmethod
    def from_state(cls, state) -> "SingleModuleStepperConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @property
    def all_names(self):
        if self.prescriber is not None:
            mask_name = [self.prescriber.mask_name]
        else:
            mask_name = []
        all_names = list(set(self.in_names).union(self.out_names).union(mask_name))
        return all_names

    @property
    def normalize_names(self):
        return list(set(self.in_names).union(self.out_names))


@dataclasses.dataclass
class ExistingStepperConfig:
    checkpoint_path: str

    def _load_checkpoint(self) -> Mapping[str, Any]:
        return torch.load(self.checkpoint_path, map_location=get_device())

    def get_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        return SingleModuleStepperConfig.from_state(
            self._load_checkpoint()["stepper"]["config"]
        ).get_data_requirements(n_forward_steps)

    def get_stepper(self, shapes, area):
        del shapes  # unused
        return SingleModuleStepper.from_state(self._load_checkpoint()["stepper"], area)


class DummyWrapper(nn.Module):
    """
    Wrapper class for a single pytorch module, which does nothing.

    Exists so we have an identical module structure to the case where we use
    a DistributedDataParallel wrapper.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@dataclasses.dataclass
class SteppedData:
    loss: float
    gen_data: Dict[str, torch.Tensor]
    target_data: Dict[str, torch.Tensor]
    gen_data_norm: Dict[str, torch.Tensor]
    target_data_norm: Dict[str, torch.Tensor]

    def remove_initial_condition(self) -> "SteppedData":
        return SteppedData(
            loss=self.loss,
            gen_data={k: v[:, 1:] for k, v in self.gen_data.items()},
            target_data={k: v[:, 1:] for k, v in self.target_data.items()},
            gen_data_norm={k: v[:, 1:] for k, v in self.gen_data_norm.items()},
            target_data_norm={k: v[:, 1:] for k, v in self.target_data_norm.items()},
        )

    def copy(self) -> "SteppedData":
        """Creates new dictionaries for the data but with the same tensors."""
        return SteppedData(
            loss=self.loss,
            gen_data={k: v for k, v in self.gen_data.items()},
            target_data={k: v for k, v in self.target_data.items()},
            gen_data_norm={k: v for k, v in self.gen_data_norm.items()},
            target_data_norm={k: v for k, v in self.target_data_norm.items()},
        )


class SingleModuleStepper:
    """
    Stepper class for a single pytorch module.
    """

    def __init__(
        self,
        config: SingleModuleStepperConfig,
        data_shapes: Dict[str, Tuple[int, ...]],
        area: torch.Tensor,
    ):
        """
        Args:
            config: The configuration.
            data_shapes: The shapes of the data.
            area: (n_lat, n_lon) array containing relative gridcell area,
                in any units including unitless.
        """
        dist = Distributed.get_instance()
        n_in_channels = len(config.in_names)
        n_out_channels = len(config.out_names)
        self.in_packer = Packer(config.in_names)
        self.out_packer = Packer(config.out_names)
        self.normalizer = config.normalization.build(config.normalize_names)
        if config.prescriber is not None:
            self.prescriber = config.prescriber.build(config.in_names, config.out_names)
        else:
            self.prescriber = NullPrescriber()
        example_name = list(data_shapes.keys())[0]
        self.module = config.builder.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=cast(Tuple[int, int], tuple(data_shapes[example_name][-2:])),
        ).to(get_device())
        self.data_shapes = data_shapes
        self._config = config
        self._no_optimization = NullOptimization()

        if dist.is_distributed():
            if using_gpu():
                device_ids = [dist.local_rank]
                output_device = [dist.local_rank]
            else:
                device_ids = None
                output_device = None
            self.module = DistributedDataParallel(
                self.module,
                device_ids=device_ids,
                output_device=output_device,
                find_unused_parameters=True,
            )
        else:
            self.module = DummyWrapper(self.module)
        self._is_distributed = dist.is_distributed()

        self.loss_obj = config.loss.build(area)

    def get_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        return self._config.get_data_requirements(n_forward_steps)

    @property
    def modules(self) -> List[nn.Module]:
        """
        Returns:
            A list of modules being trained.
        """
        return [self.module]

    def run_on_batch(
        self,
        data: Dict[str, torch.Tensor],
        optimization: Union[Optimization, NullOptimization],
        n_forward_steps: int = 1,
        aggregator: Optional[OneStepAggregator] = None,
    ) -> SteppedData:
        """
        Step the model forward on a batch of data.

        Args:
            data: The batch data of shape [n_sample, n_timesteps, n_channels, n_x, n_y].
            optimization: The optimization class to use for updating the module.
                Use `NullOptimization` to disable training.
            n_forward_steps: The number of timesteps to run the model for.
            aggregator: The data aggregator.

        Returns:
            The loss, the generated data, the normalized generated data,
                and the normalized batch data.
        """
        if aggregator is None:
            non_none_aggregator: Union[
                OneStepAggregator, InferenceAggregator, NullAggregator
            ] = NullAggregator()
        else:
            non_none_aggregator = aggregator

        device = get_device()
        device_data = {
            name: value.to(device, dtype=torch.float) for name, value in data.items()
        }
        return run_on_batch(
            data=device_data,
            module=self.module,
            normalizer=self.normalizer,
            in_packer=self.in_packer,
            out_packer=self.out_packer,
            optimization=optimization,
            loss_obj=self.loss_obj,
            n_forward_steps=n_forward_steps,
            prescriber=self.prescriber,
            aggregator=non_none_aggregator,
        )

    def get_state(self):
        """
        Returns:
            The state of the stepper.
        """
        return {
            "module": self.module.state_dict(),
            "normalizer": self.normalizer.get_state(),
            "in_packer": self.in_packer.get_state(),
            "out_packer": self.out_packer.get_state(),
            "data_shapes": self.data_shapes,
            "config": self._config.get_state(),
            "prescriber": self.prescriber.get_state(),
        }

    def load_state(self, state):
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
        """
        if "module" in state:
            self.module.load_state_dict(state["module"])
        self.in_packer = Packer.from_state(state["in_packer"])
        self.out_packer = Packer.from_state(state["out_packer"])
        self.prescriber.load_state(state["prescriber"])

    @classmethod
    def from_state(cls, state, area: torch.Tensor) -> "SingleModuleStepper":
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
            area: (n_lat, n_lon) array containing relative gridcell area, in any
                units including unitless.

        Returns:
            The stepper.
        """
        config = {**state["config"]}  # make a copy to avoid mutating input
        config["normalization"] = FromStateNormalizer(state["normalizer"])
        stepper = cls(
            config=SingleModuleStepperConfig.from_state(config),
            data_shapes=state["data_shapes"],
            area=area,
        )
        stepper.load_state(state)
        return stepper


def run_on_batch(
    data: Dict[str, torch.Tensor],
    module: nn.Module,
    normalizer: StandardNormalizer,
    in_packer: Packer,
    out_packer: Packer,
    optimization: Union[Optimization, NullOptimization],
    loss_obj: nn.Module,
    prescriber: Union[Prescriber, NullPrescriber],
    aggregator: Union[OneStepAggregator, InferenceAggregator, NullAggregator],
    n_forward_steps: int = 1,
) -> SteppedData:
    """
    Run the model on a batch of data.

    The module is assumed to require packed (concatenated into a tensor with
    a channel dimension) and normalized data, as provided by the given packer
    and normalizer.

    Args:
        data: The denormalized batch data. The second dimension of each tensor
            should be the time dimension.
        module: The module to run.
        normalizer: The normalizer.
        in_packer: The packer for the input data.
        out_packer: The packer for the output data.
        optimization: The optimization object. If it is NullOptimization,
            then the model is not trained.
        loss_obj: The loss object.
        prescriber: Overwrite an output with target value in specified region.
        aggregator: The data aggregator.
        n_forward_steps: The number of timesteps to run the model for.
        i_time_start: The index of the first timestep of the data time window,
            passed to the aggregator.

    Returns:
        The loss, the generated data, the normalized generated data,
            and the normalized batch data. The generated data contains
            the initial input data as its first timestep.
    """
    # must be negative-indexed so it works with or without a time dim
    channel_dim = -3
    time_dim = 1
    example_shape = data[list(data.keys())[0]].shape
    assert len(example_shape) == 4
    assert example_shape[1] == n_forward_steps + 1
    full_data_norm = normalizer.normalize(data)
    time_input = 0
    time_target = 1

    def get_input_data_norm(names, time_index):
        return {
            name: full_data_norm[name].select(dim=time_dim, index=time_index)
            for name in names
        }

    full_target_tensor_norm = out_packer.pack(full_data_norm, axis=channel_dim)
    loss = torch.tensor(0.0, device=get_device())
    input_data_norm = get_input_data_norm(in_packer.names, time_input)
    gen_data_norm = []
    optimization.set_mode(module)
    for _ in range(n_forward_steps):
        input_tensor_norm = in_packer.pack(input_data_norm, axis=channel_dim)
        target_tensor_norm = full_target_tensor_norm.select(
            dim=time_dim, index=time_target
        )

        with optimization.autocast():
            gen_tensor_norm = module(input_tensor_norm).to(
                get_device(), dtype=torch.float
            )
            loss += loss_obj(gen_tensor_norm, target_tensor_norm)
        gen_norm = out_packer.unpack(gen_tensor_norm, axis=channel_dim)
        target_norm = out_packer.unpack(target_tensor_norm, axis=channel_dim)
        data_time = {
            k: v.select(dim=time_dim, index=time_target) for k, v in data.items()
        }
        gen_norm = prescriber(data_time, gen_norm, target_norm)
        time_input += 1
        time_target += 1
        gen_data_norm.append(gen_norm)
        # update input data with generated outputs, and forcings for missing outputs
        forcing_names = list(set(in_packer.names).difference(gen_norm.keys()))
        forcing_data_norm = get_input_data_norm(forcing_names, time_input)
        input_data_norm = {**forcing_data_norm, **gen_norm}

    optimization.step_weights(loss)
    # prepend the initial (pre-first-timestep) output data to the generated data
    initial = get_input_data_norm(out_packer.names, 0)
    gen_data_norm = [initial] + gen_data_norm
    gen_data_norm_timeseries = {}
    for name in out_packer.names:
        gen_data_norm_timeseries[name] = torch.stack(
            [x[name] for x in gen_data_norm], dim=time_dim
        )
    gen_data = normalizer.denormalize(gen_data_norm_timeseries)
    aggregator.record_batch(
        loss,
        target_data=data,
        gen_data=gen_data,
        target_data_norm=full_data_norm,
        gen_data_norm=gen_data_norm_timeseries,
    )
    return SteppedData(
        loss=loss,
        gen_data=gen_data,
        target_data=data,
        gen_data_norm=gen_data_norm_timeseries,
        target_data_norm=full_data_norm,
    )
