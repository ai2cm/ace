import dataclasses
import re
from collections.abc import Callable

from torch import nn

from fme.core.typing_ import TensorDict, TensorMapping

LEVEL_PATTERN = re.compile(r"_(\d+)$")
TEMPLATE = "{name}{suffix}"


StepMethod = Callable[
    [TensorMapping, TensorMapping, Callable[[nn.Module], nn.Module]], TensorDict
]


def get_multi_call_name(name: str, suffix: str) -> str:
    """Get multi-call name, appropriately handling 3D variables.

    For 2D fields this trivially appends the suffix to the name; for 3D fields
    we insert the suffix between the variable name and vertical level label,
    following how these variables are pre-processed.

    Args:
       name: Name of field.
       suffix: Suffix to append indicating multi-call multiplier.

    Returns:
       Name of multi-call field associated with the given suffix.

    Examples:
       >>> fme.core.multi_call.get_multi_call_name("foo", "_with_quartered_co2")
       'foo_with_quartered_co2'
       >>> fme.core.multi_call.get_multi_call_name("bar_0", "_with_quartered_co2")
       'bar_with_quartered_co2_0'
    """
    match = LEVEL_PATTERN.search(name)
    if match is not None:
        name = name[: match.start()]
        suffix = suffix + match.group(0)
    return TEMPLATE.format(name=name, suffix=suffix)


@dataclasses.dataclass
class MultiCallConfig:
    """Configuration for doing 'multi-call' predictions where an input variable (e.g.
    CO2) is varied by multiplying by floats and then certain output variables (e.g.
    radiative heating or fluxes) are predicted.

    Parameters:
        forcing_name: name of the variable to perturb in the forcing data, e.g. "co2".
        forcing_multipliers: mapping from a label suffix to a multiplier that is applied
            to the 'forcing_name' variable. For example, could be
            {"_quadrupled_co2": 4, "_halved_co2": 0.5}. The suffixes will be appended to
            the output_names below.
        output_names: names of the variables to predict given perturbed forcing. For
            example, ["ULWRFtoa", "USWRFsfc"].
    """

    forcing_name: str
    forcing_multipliers: dict[str, float]
    output_names: list[str]

    def __post_init__(self):
        self._names = []
        for name in self.output_names:
            self._names.extend(self.get_multi_called_names(name))

    def get_multi_called_names(self, name: str) -> list[str]:
        names = []
        for suffix in self.forcing_multipliers:
            names.append(get_multi_call_name(name, suffix))
        return names

    def validate(self, in_names: list[str], out_names: list[str]):
        if self.forcing_name not in in_names:
            raise ValueError(
                f"forcing name {self.forcing_name} not in input names. It is required "
                "as a forcing given provided radiation multi call configuration."
            )
        if self.forcing_name in out_names:
            raise ValueError(
                f"forcing name {self.forcing_name} is in the output names, "
                "but it must be a forcing variable, not an output."
            )
        for name in self.output_names:
            if name not in out_names:
                raise ValueError(
                    f"{name} not in output names. It is required "
                    "as an output given provided radiation multi call configuration."
                )
        for multi_called_name in self.names:
            if multi_called_name in in_names:
                raise ValueError(
                    f"The multi-call output {multi_called_name} is already in in_names."
                    " This will lead to a conflict--please rename the input or "
                    "use a different multi-call suffix label."
                )
            if multi_called_name in out_names:
                raise ValueError(
                    f"The multi-call output {multi_called_name} is already in "
                    "out_names. This will lead to a conflict--please rename the output "
                    "or use a different multi-call suffix label."
                )

    @property
    def names(self) -> list[str]:
        """
        Return the names of all multi-called output variables,
        often radiative fluxes.

        E.g. ['ULWRFtoa_quadrupled_co2'].
        """
        return self._names

    def build(self, step_method: StepMethod) -> "MultiCall":
        return MultiCall(self, step_method)


class MultiCall:
    """Class for doing 'multi-call' predictions where a forcing variable is varied
    and certain outputs are saved for each value.

    Given a 'step' method that takes the input data and next-step forcing data, this
    class will call the step method multiple times, changing the value of the desired
    forcing variable each time. Specified outputs are saved for each value with their
    names modified to indicate the forcing value cahnge.
    """

    def __init__(
        self,
        config: MultiCallConfig,
        step_method: StepMethod,
    ):
        self.forcing_name = config.forcing_name
        self.forcing_multipliers = config.forcing_multipliers
        self.output_names = config.output_names
        self._names = config.names
        self._step = step_method

    @property
    def names(self) -> list[str]:
        return self._names

    def step(
        self,
        input: TensorMapping,
        next_step_forcing_data: TensorMapping,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> TensorDict:
        predictions = {}
        unscaled_forcing = input[self.forcing_name]
        for suffix, multiplier in self.forcing_multipliers.items():
            scaled_input = dict(input)
            scaled_input[self.forcing_name] = multiplier * unscaled_forcing
            output = self._step(scaled_input, next_step_forcing_data, wrapper)

            for name in self.output_names:
                new_name = get_multi_call_name(name, suffix)
                predictions[new_name] = output[name]

        return predictions
