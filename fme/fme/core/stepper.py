from typing import List, Dict, Tuple, Optional, Union
from fme.fcn_training.utils.darcy_loss import LpLoss

from fme.fcn_training.utils.data_requirements import DataRequirements
from torch.nn.parallel import DistributedDataParallel
from fme.core.packer import Packer
from fme.core.device import get_device
from fme.core.normalizer import StandardNormalizer
import torch.distributed as dist
import dataclasses
from fme.fcn_training.registry import ModuleSelector
from torch import nn
import torch
import os
from .optimization import Optimization, OptimizationConfig, NullOptimization


@dataclasses.dataclass
class SingleModuleStepperConfig:
    builder: ModuleSelector
    in_names: List[str]
    out_names: List[str]
    optimization: OptimizationConfig

    def get_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        return DataRequirements(
            names=list(self.in_names) + list(self.out_names),
            in_names=self.in_names,
            out_names=self.out_names,
            n_timesteps=n_forward_steps + 1,
        )

    def get_stepper(
        self,
        shapes: Dict[str, Tuple[int, ...]],
        normalizer: StandardNormalizer,
    ):
        return SingleModuleStepper(
            builder=self.builder,
            data_shapes=shapes,
            normalizer=normalizer,
            in_names=self.in_names,
            out_names=self.out_names,
            optimization_builder=self.optimization,
        )


class SingleModuleStepper:
    """
    Stepper class for a single pytorch module.
    """

    def __init__(
        self,
        builder: ModuleSelector,
        data_shapes: Dict[str, Tuple[int, ...]],
        normalizer: StandardNormalizer,
        in_names: List[str],
        out_names: List[str],
        optimization_builder: Optional[OptimizationConfig] = None,
    ):
        """
        Args:
            builder: The module builder.
            data_shapes: The shapes of the data.
            normalizer: The normalizer.
            in_names: The names of the input fields.
            out_names: The names of the output fields.
            loss_obj: The loss object to use.
            optimization_builder: The optimization builder.
        """
        n_in_channels = len(in_names)
        n_out_channels = len(out_names)
        self.in_packer = Packer(in_names)
        self.out_packer = Packer(out_names)
        self.normalizer = normalizer
        example_name = list(data_shapes.keys())[0]
        self.module = builder.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape_x=data_shapes[example_name][-2],
            img_shape_y=data_shapes[example_name][-1],
        ).to(get_device())
        self._data_shapes = data_shapes
        self._builder = builder
        self._optimization_builder = optimization_builder
        if optimization_builder is not None:
            self.optimization: Union[
                Optimization, NullOptimization
            ] = optimization_builder.build(parameters=self.module.parameters())
        else:
            self.optimization = NullOptimization()

        self._no_optimization = NullOptimization()

        if dist.is_initialized():
            self.module = DistributedDataParallel(
                self.module,
                device_ids=[int(os.environ["LOCAL_RANK"])],
                output_device=[int(os.environ["LOCAL_RANK"])],
                find_unused_parameters=True,
            )

        self.loss_obj = LpLoss()

    @property
    def modules(self) -> List[nn.Module]:
        """
        Returns:
            A list of modules being trained.
        """
        return [self.module]

    def step_scheduler(self, valid_loss: float):
        """
        Step the scheduler.

        Args:
            valid_loss: The validation loss. Used in schedulers which change the
                learning rate based on whether the validation loss is decreasing.
        """
        self.optimization.step_scheduler(valid_loss)

    def run_on_batch(
        self,
        data: Dict[str, torch.Tensor],
        train: bool,
        n_forward_steps: int = 1,
    ) -> Tuple[
        float,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """
        Step the model forward on a batch of data.

        Args:
            data: The batch data of shape [n_sample, n_timesteps, n_channels, n_x, n_y].
            train: Whether to train the model.
            n_forward_steps: The number of timesteps to run the model for.

        Returns:
            The loss, the generated data, the normalized generated data,
                and the normalized batch data.
        """
        if train:
            if self.optimization is None:
                raise ValueError(
                    "Cannot train without an optimizer, "
                    "should be passed at initialization."
                )
            optimization: Union[Optimization, NullOptimization] = self.optimization
        else:
            optimization = self._no_optimization

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
        )

    def get_state(self):
        """
        Returns:
            The state of the stepper.
        """
        return {
            "module": self.module.state_dict(),
            "optimization": self.optimization.get_state(),
            "normalizer": self.normalizer.get_state(),
            "in_packer": self.in_packer.get_state(),
            "out_packer": self.out_packer.get_state(),
            "builder": self._builder.get_state(),
            "optimization_builder": self._optimization_builder.get_state(),
            "data_shapes": self._data_shapes,
        }

    def load_state(self, state, load_optimizer: bool = True):
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
            load_optimizer: Whether to load the optimizer state.
        """
        if "module" in state:
            module_state = {}
            for key in state["module"]:
                if key.startswith("module.") and not dist.is_initialized():
                    # model was stored using ddp which prepends 'module.' if training
                    # with multiple GPUs
                    name = key[7:]
                else:
                    name = key
                module_state[name] = state["module"][key]
            self.module.load_state_dict(module_state)
        if load_optimizer and "optimization" in state:
            self.optimization.load_state(state["optimization"])
        self.normalizer = StandardNormalizer.from_state(state["normalizer"])
        self.in_packer = Packer.from_state(state["in_packer"])
        self.out_packer = Packer.from_state(state["out_packer"])
        self._builder = ModuleSelector.from_state(state["builder"])
        self._optimization_builder = OptimizationConfig.from_state(
            state["optimization_builder"]
        )
        self._data_shapes = state["data_shapes"]

    @classmethod
    def from_state(cls, state, load_optimizer: bool = True) -> "SingleModuleStepper":
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
            load_optimizer: Whether to load the optimizer state.

        Returns:
            The stepper.
        """
        stepper = cls(
            in_names=state["in_packer"]["names"],
            out_names=state["out_packer"]["names"],
            normalizer=StandardNormalizer.from_state(state["normalizer"]),
            builder=ModuleSelector.from_state(state["builder"]),
            data_shapes=state["data_shapes"],
            optimization_builder=None,  # will be handled by load_state
        )
        stepper.load_state(state, load_optimizer=load_optimizer)
        return stepper


def run_on_batch(
    data: Dict[str, torch.Tensor],
    module: nn.Module,
    normalizer: StandardNormalizer,
    in_packer: Packer,
    out_packer: Packer,
    optimization: Union[Optimization, NullOptimization],
    loss_obj: nn.Module,
    n_forward_steps: int = 1,
) -> Tuple[
    float,
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]:
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
        n_forward_steps: The number of timesteps to run the model for.

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
    gen_tensor_norms = []
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
        time_input += 1
        time_target += 1
        gen_tensor_norms.append(gen_tensor_norm)
        # update input data with generated outputs, and forcings for missing outputs
        gen_norm = out_packer.unpack(gen_tensor_norm, axis=channel_dim)
        forcing_names = list(set(in_packer.names).difference(gen_norm.keys()))
        forcing_data_norm = get_input_data_norm(forcing_names, time_input)
        input_data_norm = {**forcing_data_norm, **gen_norm}

    optimization.step_weights(loss)
    # prepend the initial (pre-first-timestep) output data to the generated data
    initial_tensor = full_target_tensor_norm.select(dim=time_dim, index=0)
    gen_tensor_norms = [initial_tensor] + gen_tensor_norms
    gen_tensor_norm = torch.stack(gen_tensor_norms, dim=time_dim)
    gen_data_norm = out_packer.unpack(
        gen_tensor_norm, axis=channel_dim  # - 1 because no time dim
    )
    gen_data = normalizer.denormalize(gen_data_norm)
    # TODO: use an aggregator as input instead of returning these sample outputs
    return loss, gen_data, gen_data_norm, full_data_norm
