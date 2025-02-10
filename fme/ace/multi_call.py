import dataclasses
from typing import Callable, Dict, List

from fme.core.typing_ import TensorDict, TensorMapping

TEMPLATE = "{name}{suffix}"


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
    forcing_multipliers: Dict[str, float]
    output_names: List[str]

    def __post_init__(self):
        self._names = []
        for name in self.output_names:
            for suffix in self.forcing_multipliers:
                self._names.append(TEMPLATE.format(name=name, suffix=suffix))

    def validate(self, in_names: List[str], out_names: List[str]):
        if self.forcing_name not in in_names:
            raise ValueError(
                f"{self.forcing_name} not in input names. It is required "
                "as a forcing given provided radiation multi call configuration."
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
    def names(self) -> List[str]:
        """Return the names of all multi-called variables, often radiative fluxes.

        E.g. ['ULWRFtoa_quadrupled_co2'].
        """
        return self._names

    def build(
        self,
        step_method: Callable[[TensorMapping, TensorMapping, bool], TensorDict],
    ):
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
        step_method: Callable[[TensorMapping, TensorMapping, bool], TensorDict],
    ):
        self.forcing_name = config.forcing_name
        self.forcing_multipliers = config.forcing_multipliers
        self.output_names = config.output_names
        self._names = config.names
        self._step = step_method

    @property
    def names(self) -> List[str]:
        return self._names

    def step(
        self,
        input: TensorMapping,
        next_step_forcing_data: TensorMapping,
        use_activation_checkpointing: bool = False,
    ) -> TensorDict:
        predictions = {}
        unscaled_forcing = input[self.forcing_name]
        for suffix, multiplier in self.forcing_multipliers.items():
            scaled_input = input | {self.forcing_name: multiplier * unscaled_forcing}
            output = self._step(
                scaled_input, next_step_forcing_data, use_activation_checkpointing
            )

            for name in self.output_names:
                new_name = TEMPLATE.format(name=name, suffix=suffix)
                predictions[new_name] = output[name]

        return predictions
