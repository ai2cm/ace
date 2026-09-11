import dataclasses
from collections.abc import Callable
from typing import Any

import dacite
import torch
from torch import nn

from fme.core.dataset_info import DatasetInfo
from fme.core.normalizer import StandardNormalizer
from fme.core.ocean import OceanConfig
from fme.core.step.args import StepArgs
from fme.core.step.output import StepOutput
from fme.core.step.step import StepABC, StepConfigABC, StepSelector
from fme.core.typing_ import TensorDict, TensorMapping


@StepSelector.register("ensemble")
@dataclasses.dataclass
class EnsembleStepConfig(StepConfigABC):
    """
    Configuration for a step whose output is a weighted sum of member steps.

    Every member is called on the same inputs, and each output variable is the
    weighted sum of the members' predictions for it. The weights are used as
    given (not normalized), so a weighted mean needs weights summing to one.
    Members must agree on their input, output, next-step input, and next-step
    forcing names, number of initial-condition timesteps, and ocean
    configuration; anything else that a step exposes once (normalizer, surface
    temperature name, SST prescription, carried stepper state) is taken from
    the first member. Every member is stepped on the first member's carried
    state, and the other members' state updates are discarded, so stochastic
    members share the first member's random state rather than sampling
    independently. Corrector diagnostics from the members are dropped.

    Parameters:
        members: The member step configurations.
        weights: One weight per member.
    """

    members: list[StepSelector]
    weights: list[float]

    def __post_init__(self):
        if len(self.members) == 0:
            raise ValueError("ensemble step requires at least one member")
        if len(self.weights) != len(self.members):
            raise ValueError(
                f"ensemble step has {len(self.members)} members but "
                f"{len(self.weights)} weights"
            )
        first = self.members[0]
        for i, member in enumerate(self.members[1:], start=1):
            _assert_members_agree(first, member, i)

    def get_step(
        self,
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None],
    ) -> "EnsembleStep":
        return EnsembleStep(
            members=[
                member.get_step(dataset_info, init_weights) for member in self.members
            ],
            config=self,
        )

    @property
    def n_ic_timesteps(self) -> int:
        return self.members[0].n_ic_timesteps

    @property
    def input_names(self) -> frozenset[str]:
        return self.members[0].input_names

    @property
    def output_names(self) -> frozenset[str]:
        return self.members[0].output_names

    @property
    def next_step_input_names(self) -> frozenset[str]:
        return self.members[0].next_step_input_names

    @property
    def loss_names(self) -> list[str]:
        return self.members[0].loss_names

    def get_next_step_forcing_names(self) -> list[str]:
        return self.members[0].get_next_step_forcing_names()

    def get_loss_normalizer(
        self,
        extra_names: list[str] | None = None,
        extra_residual_scaled_names: list[str] | None = None,
    ) -> StandardNormalizer:
        return self.members[0].get_loss_normalizer(
            extra_names=extra_names,
            extra_residual_scaled_names=extra_residual_scaled_names,
        )

    def replace_ocean(self, ocean: OceanConfig | None):
        for member in self.members:
            member.replace_ocean(ocean)

    def get_ocean(self) -> OceanConfig | None:
        return self.members[0].get_ocean()

    def replace_prescribed_prognostic_names(self, names: list[str]) -> None:
        for member in self.members:
            member.replace_prescribed_prognostic_names(names)

    def get_prescribed_prognostic_names(self) -> list[str]:
        return self.members[0].get_prescribed_prognostic_names()

    @property
    def allow_missing_variables(self) -> bool:
        # A variable may only be missing if every member tolerates it missing.
        return all(member.allow_missing_variables for member in self.members)

    def load(self):
        for member in self.members:
            member.load()

    @classmethod
    def from_state(cls, state) -> "EnsembleStepConfig":
        return dacite.from_dict(cls, state, config=dacite.Config(strict=True))


def _assert_members_agree(first: StepSelector, other: StepSelector, index: int):
    checks: list[tuple[str, Any, Any]] = [
        ("input_names", first.input_names, other.input_names),
        ("output_names", first.output_names, other.output_names),
        (
            "next_step_input_names",
            first.next_step_input_names,
            other.next_step_input_names,
        ),
        (
            "next_step_forcing_names",
            frozenset(first.get_next_step_forcing_names()),
            frozenset(other.get_next_step_forcing_names()),
        ),
        ("n_ic_timesteps", first.n_ic_timesteps, other.n_ic_timesteps),
        ("ocean", first.get_ocean(), other.get_ocean()),
    ]
    for name, expected, actual in checks:
        if expected != actual:
            raise ValueError(
                f"ensemble member {index} disagrees with member 0 on {name}: "
                f"{actual!r} != {expected!r}"
            )


class EnsembleStep(StepABC):
    """
    Step whose output is a weighted sum of its member steps' outputs.
    """

    def __init__(self, members: list[StepABC], config: EnsembleStepConfig):
        """
        Args:
            members: The member steps, in the same order as the config's
                members and weights.
            config: The ensemble step configuration.
        """
        super().__init__()
        self._members = members
        self._config = config

    @property
    def config(self) -> EnsembleStepConfig:
        return self._config

    @property
    def modules(self) -> nn.ModuleList:
        return nn.ModuleList(
            [module for member in self._members for module in member.modules]
        )

    @property
    def normalizer(self) -> StandardNormalizer:
        return self._members[0].normalizer

    @property
    def surface_temperature_name(self) -> str | None:
        return self._members[0].surface_temperature_name

    @property
    def ocean_fraction_name(self) -> str | None:
        return self._members[0].ocean_fraction_name

    def prescribe_sst(
        self,
        mask_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> TensorDict:
        return self._members[0].prescribe_sst(mask_data, gen_data, target_data)

    def get_regularizer_loss(self) -> torch.Tensor:
        total = self._members[0].get_regularizer_loss()
        for member in self._members[1:]:
            total = total + member.get_regularizer_loss()
        return total

    def train(self, mode: bool = True) -> StepABC:
        super().train(mode)
        for member in self._members:
            member.train(mode)
        return self

    def set_epoch(self, epoch: int) -> None:
        for member in self._members:
            member.set_epoch(epoch)

    def step(
        self,
        args: StepArgs,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> StepOutput:
        outputs = [member.step(args=args, wrapper=wrapper) for member in self._members]
        for i, member_output in enumerate(outputs[1:], start=1):
            if set(member_output.output) != set(outputs[0].output):
                raise ValueError(
                    f"ensemble member {i} returned variables "
                    f"{sorted(member_output.output)}, member 0 returned "
                    f"{sorted(outputs[0].output)}"
                )
        weights = self._config.weights
        output: TensorDict = {}
        for name in outputs[0].output:  # keep member 0's variable order
            total = weights[0] * outputs[0].output[name]
            for weight, member_output in zip(weights[1:], outputs[1:]):
                total = total + weight * member_output.output[name]
            output[name] = total
        # Carried state comes from the first member; the members' corrector
        # diagnostics are dropped since no single delta describes the sum.
        return StepOutput(output=output, stepper_state=outputs[0].stepper_state)

    def get_state(self) -> dict[str, Any]:
        return {"members": [member.get_state() for member in self._members]}

    def load_state(self, state: dict[str, Any]):
        member_states = state["members"]
        if len(member_states) != len(self._members):
            raise ValueError(
                f"state has {len(member_states)} members, step has "
                f"{len(self._members)}"
            )
        for member, member_state in zip(self._members, member_states):
            member.load_state(member_state)
