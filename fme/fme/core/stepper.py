import dataclasses
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
)

import dacite
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from fme.core import metrics
from fme.core.aggregator import InferenceAggregator, NullAggregator, OneStepAggregator
from fme.core.climate_data import ClimateData
from fme.core.data_loading.requirements import DataRequirements
from fme.core.data_loading.typing import SigmaCoordinates
from fme.core.device import get_device, using_gpu
from fme.core.distributed import Distributed
from fme.core.loss import ConservationLoss, ConservationLossConfig, LossConfig
from fme.core.normalizer import (
    FromStateNormalizer,
    NormalizationConfig,
    StandardNormalizer,
)
from fme.core.packer import DataShapesNotUniform, Packer
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
    conserve_dry_air: bool = False
    conservation_loss: ConservationLossConfig = dataclasses.field(
        default_factory=lambda: ConservationLossConfig()
    )

    def get_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        return DataRequirements(
            names=self.all_names,
            n_timesteps=n_forward_steps + 1,
        )

    def get_state(self):
        return dataclasses.asdict(self)

    def get_stepper(
        self,
        shapes: Dict[str, Tuple[int, ...]],
        area: Optional[torch.Tensor],
        sigma_coordinates: SigmaCoordinates,
    ):
        return SingleModuleStepper(
            config=self,
            data_shapes=shapes,
            area=area,
            sigma_coordinates=sigma_coordinates,
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

    def get_stepper(self, shapes, area, sigma_coordinates):
        del shapes  # unused
        return SingleModuleStepper.from_state(
            self._load_checkpoint()["stepper"],
            area=area,
            sigma_coordinates=sigma_coordinates,
        )


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
    metrics: Dict[str, torch.Tensor]
    gen_data: Dict[str, torch.Tensor]
    target_data: Dict[str, torch.Tensor]
    gen_data_norm: Dict[str, torch.Tensor]
    target_data_norm: Dict[str, torch.Tensor]

    def remove_initial_condition(self) -> "SteppedData":
        return SteppedData(
            metrics=self.metrics,
            gen_data={k: v[:, 1:] for k, v in self.gen_data.items()},
            target_data={k: v[:, 1:] for k, v in self.target_data.items()},
            gen_data_norm={k: v[:, 1:] for k, v in self.gen_data_norm.items()},
            target_data_norm={k: v[:, 1:] for k, v in self.target_data_norm.items()},
        )

    def copy(self) -> "SteppedData":
        """Creates new dictionaries for the data but with the same tensors."""
        return SteppedData(
            metrics=self.metrics,
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
        sigma_coordinates: SigmaCoordinates,
    ):
        """
        Args:
            config: The configuration.
            data_shapes: Shapes of the module inputs.
            area: (n_lat, n_lon) array containing relative gridcell area,
                in any units including unitless.
            sigma_coordinates: The sigma coordinates.
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
            )
        else:
            self.module = DummyWrapper(self.module)
        self._is_distributed = dist.is_distributed()

        self.area = area
        self.sigma_coordinates = sigma_coordinates.to(get_device())
        self.loss_obj = config.loss.build(self.area)
        self._conservation_loss = config.conservation_loss.build(
            area_weights=self.area,
            sigma_coordinates=self.sigma_coordinates,
        )

    def get_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        return self._config.get_data_requirements(n_forward_steps)

    @property
    def modules(self) -> nn.ModuleList:
        """
        Returns:
            A list of modules being trained.
        """
        return nn.ModuleList([self.module])

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
            conserve_dry_air=self._config.conserve_dry_air,
            sigma_coordinates=self.sigma_coordinates,
            area=self.area,
            conservation_loss=self._conservation_loss,
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
            "area": self.area,
            "sigma_coordinates": self.sigma_coordinates.as_dict(),
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
    def from_state(
        cls, state, area: torch.Tensor, sigma_coordinates: SigmaCoordinates
    ) -> "SingleModuleStepper":
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
            area: (n_lat, n_lon) array containing relative gridcell area, in any
                units including unitless.
            sigma_coordinates: The sigma coordinates.

        Returns:
            The stepper.
        """
        config = {**state["config"]}  # make a copy to avoid mutating input
        config["normalization"] = FromStateNormalizer(state["normalizer"])
        area = state.get("area", area)
        if "sigma_coordinates" in state:
            sigma_coordinates = dacite.from_dict(
                data_class=SigmaCoordinates,
                data=state["sigma_coordinates"],
                config=dacite.Config(strict=True),
            )
        stepper = cls(
            config=SingleModuleStepperConfig.from_state(config),
            data_shapes=state["data_shapes"],
            area=area,
            sigma_coordinates=sigma_coordinates,
        )
        stepper.load_state(state)
        return stepper


class NameAndTimeQueryFunction(Protocol):
    def __call__(
        self,
        names: Iterable[str],
        time_index: int,
        norm_mode: Literal["norm", "denorm"],
    ) -> Dict[str, torch.Tensor]:
        ...


def get_name_and_time_query_fn(
    data: Dict[str, torch.Tensor], data_norm: Dict[str, torch.Tensor], time_dim: int
) -> NameAndTimeQueryFunction:
    """Construct a function for querying `data` by name and time and whether it
    is normalized or not. (Note: that the `names` argument can contain None values
    to handle NullPrescriber)."""

    norm_mode_to_data = {"norm": data_norm, "denorm": data}

    def name_and_time_query_fn(names, time_index, norm_mode):
        _data = norm_mode_to_data[norm_mode]
        query_results = {}
        for name in names:
            try:
                query_results[name] = _data[name].select(dim=time_dim, index=time_index)
            except IndexError as err:
                raise ValueError(
                    f'tensor "{name}" does not have values at t={time_index}'
                ) from err
        return query_results

    return name_and_time_query_fn


def _pack_data_if_available(
    packer: Packer,
    data: Dict[str, torch.Tensor],
    axis: int,
) -> Optional[torch.Tensor]:
    try:
        return packer.pack(data, axis=axis)
    except DataShapesNotUniform:
        return None


def _force_conserve_dry_air(
    input_data: Mapping[str, torch.Tensor],
    gen_data: Mapping[str, torch.Tensor],
    area: torch.Tensor,
    sigma_coordinates: SigmaCoordinates,
) -> Dict[str, torch.Tensor]:
    """
    Update the generated data to conserve dry air.

    This is done by adding a constant correction to the dry air pressure of
    each column, and may result in changes in per-mass values such as
    total water or energy.

    We first compute the target dry air pressure by computing the globally
    averaged difference in dry air pressure between the input_data and gen_data,
    and then add this offset to the fully-resolved gen_data dry air pressure.
    We can then solve for the surface pressure corresponding to this new dry air
    pressure.

    We start from the expression for dry air pressure:

        dry_air = ps - sum_k((ak_diff + bk_diff * ps) * wat_k)

    To update the dry air, we compute and update the surface pressure:

        ps = (
            dry_air + sum_k(ak_diff * wat_k)
        ) / (
            1 - sum_k(bk_diff * wat_k)
        )
    """
    input = ClimateData(input_data)
    if input.surface_pressure is None:
        raise ValueError("surface_pressure is required to force dry air conservation")
    gen = ClimateData(gen_data)
    gen_dry_air = gen.surface_pressure_due_to_dry_air(sigma_coordinates)
    global_gen_dry_air = metrics.weighted_mean(gen_dry_air, weights=area, dim=(-2, -1))
    global_target_gen_dry_air = metrics.weighted_mean(
        input.surface_pressure_due_to_dry_air(sigma_coordinates),
        weights=area,
        dim=(-2, -1),
    )
    error = global_gen_dry_air - global_target_gen_dry_air
    new_gen_dry_air = gen_dry_air - error[..., None, None]
    wat = gen.specific_total_water
    if wat is None:
        raise ValueError("specific_total_water is required for conservation")
    ak_diff = sigma_coordinates.ak.diff()
    bk_diff = sigma_coordinates.bk.diff()
    new_pressure = (new_gen_dry_air + (ak_diff * wat).sum(-1)) / (
        1 - (bk_diff * wat).sum(-1)
    )
    gen.surface_pressure = new_pressure.to(dtype=input.surface_pressure.dtype)
    return gen.data


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
    sigma_coordinates: SigmaCoordinates,
    area: torch.Tensor,
    conservation_loss: ConservationLoss,
    n_forward_steps: int = 1,
    conserve_dry_air: bool = False,
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
        sigma_coordinates: The sigma coordinates.
        area: (n_lat, n_lon) array containing relative gridcell area, in any
            units including unitless.
        conservation_loss: Computes conservation-related losses, if any.
        n_forward_steps: The number of timesteps to run the model for.
        conserve_dry_air: if True, force global dry air mass conservation

    Returns:
        The loss, the generated data, the normalized generated data,
            and the normalized batch data. The generated data contains
            the initial input data as its first timestep.
    """
    channel_dim = -3
    time_dim = 1
    full_data_norm = normalizer.normalize(data)
    get_input_data = get_name_and_time_query_fn(data, full_data_norm, time_dim)

    full_target_tensor_norm = _pack_data_if_available(
        out_packer,
        full_data_norm,
        channel_dim,
    )

    loss = torch.tensor(0.0, device=get_device())
    metrics = {}
    input_data_norm = get_input_data(in_packer.names, time_index=0, norm_mode="norm")
    gen_data_norm_list = []
    optimization.set_mode(module)
    for step in range(n_forward_steps):
        input_tensor_norm = in_packer.pack(input_data_norm, axis=channel_dim)

        if full_target_tensor_norm is None:
            target_tensor_norm: Optional[torch.Tensor] = None
        else:
            target_tensor_norm = full_target_tensor_norm.select(
                dim=time_dim, index=step + 1
            )

        with optimization.autocast():
            gen_tensor_norm = module(input_tensor_norm).to(
                get_device(), dtype=torch.float
            )
            if conserve_dry_air:
                gen_norm = out_packer.unpack(gen_tensor_norm, axis=channel_dim)
                gen_data = normalizer.denormalize(gen_norm)
                input_data = get_input_data(
                    in_packer.names, time_index=step, norm_mode="denorm"
                )
                gen_data = _force_conserve_dry_air(
                    input_data, gen_data, area=area, sigma_coordinates=sigma_coordinates
                )
                gen_norm = normalizer.normalize(gen_data)
                gen_tensor_norm = out_packer.pack(gen_norm, axis=channel_dim).to(
                    get_device(), dtype=torch.float
                )
            if target_tensor_norm is None:
                step_loss = torch.tensor(torch.nan)
            else:
                step_loss = loss_obj(gen_tensor_norm, target_tensor_norm)
            loss += step_loss
            metrics[f"loss_step_{step}"] = step_loss.detach()
        gen_norm = out_packer.unpack(gen_tensor_norm, axis=channel_dim)

        gen_norm = prescriber(
            get_input_data(
                prescriber.mask_names, time_index=step + 1, norm_mode="denorm"
            ),
            gen_norm,
            get_input_data(
                prescriber.prescribed_names, time_index=step + 1, norm_mode="norm"
            ),
        )

        gen_data_norm_list.append(gen_norm)
        # update input data with generated outputs, and forcings for missing outputs
        forcing_names = list(set(in_packer.names).difference(gen_norm.keys()))
        forcing_data_norm = get_input_data(
            forcing_names, time_index=step + 1, norm_mode="norm"
        )
        input_data_norm = {**forcing_data_norm, **gen_norm}

    # prepend the initial (pre-first-timestep) output data to the generated data
    initial = get_input_data(out_packer.names, time_index=0, norm_mode="norm")
    gen_data_norm_list = [initial] + gen_data_norm_list
    gen_data_norm_timeseries = {}
    for name in out_packer.names:
        gen_data_norm_timeseries[name] = torch.stack(
            [x[name] for x in gen_data_norm_list], dim=time_dim
        )
    gen_data = normalizer.denormalize(gen_data_norm_timeseries)

    conservation_metrics, conservation_loss = conservation_loss(gen_data)
    metrics.update(conservation_metrics)
    loss += conservation_loss

    metrics["loss"] = loss.detach()
    optimization.step_weights(loss)

    aggregator.record_batch(
        float(loss),
        target_data=data,
        gen_data=gen_data,
        target_data_norm=full_data_norm,
        gen_data_norm=gen_data_norm_timeseries,
    )

    return SteppedData(
        metrics=metrics,
        gen_data=gen_data,
        target_data=data,
        gen_data_norm=gen_data_norm_timeseries,
        target_data_norm=full_data_norm,
    )
