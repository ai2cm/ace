import re
from typing import List, Mapping, Union

import torch

from fme.core.typing_ import TensorMapping


def natural_sort(alist: List[str]) -> List[str]:
    """Sort to alphabetical order but with numbers sorted
    numerically, e.g. a11 comes after a2. See [1] and [2].

    [1] https://stackoverflow.com/questions/11150239/natural-sorting
    [2] https://en.wikipedia.org/wiki/Natural_sort_order
    """

    def convert(text: str) -> Union[str, int]:
        if text.isdigit():
            return int(text)
        else:
            return text.lower()

    def alphanum_key(item: str) -> List[Union[str, int]]:
        return [convert(c) for c in re.split("([0-9]+)", item)]

    return sorted(alist, key=alphanum_key)


def unstack(tensor: torch.Tensor, names: List[str], dim: int = -1) -> TensorMapping:
    """Unstack a 3D variable to a dictionary of 2D variables.

    Args:
        tensor: 3D tensor to unstack, such as output by a Stacker.
        names: List of names in natural order to assign to the unstacked variables.
            Stacker.get_all_level_names can help in retrieving these names when the
            input tensor was stacked via a Stacker.
        dim: Dimension along which to unstack.

    """
    if len(names) != tensor.size(dim):
        raise ValueError(
            f"Received {len(names)} names, but 3D tensor has "
            f"{tensor.size(-1)} levels."
        )
    if len(names) == 1:
        return {names[0]: tensor.select(dim=dim, index=0)}
    # split the output tensor along the vertical dimension
    tensors = torch.split(tensor, 1, dim=dim)
    return {name: tensor.squeeze(dim=dim) for name, tensor in zip(names, tensors)}


class Stacker:
    """Handles extraction and stacking of data tensors for 3D variables."""

    LEVEL_PATTERN = re.compile(r"_(\d+)$")

    def __init__(
        self,
        prefix_map: Mapping[str, List[str]],
    ):
        """
        Args:
            prefix_map: Mapping which defines the correspondence between an arbitrary
                set of "standard" names (e.g., "surface_pressure" or "air_temperature")
                and lists of possible names or prefix variants (e.g., ["PRESsfc", "PS"]
                or ["air_temperature_", "T_"]) found in the data.
        """
        self._prefix_map = prefix_map

    @property
    def prefix_map(self) -> Mapping[str, List[str]]:
        """Mapping which defines the correspondence between an arbitrary set of
        "standard" names (e.g., "surface_pressure" or "air_temperature") and
        lists of possible names or prefix variants (e.g., ["PRESsfc", "PS"] or
        ["air_temperature_", "T_"]) found in the data.
        """
        return self._prefix_map

    @property
    def standard_names(self) -> List[str]:
        return list(self._prefix_map.keys())

    def get_all_level_names(self, standard_name: str, data: TensorMapping) -> List[str]:
        """Get the names of all variables in the data that match one of the
        prefixes associated with the given standard name. If the standard name
        corresponds to a 3D variable, returns all vertical level names in their
        natural order.
        """
        if standard_name not in self.standard_names:
            raise ValueError(f"{standard_name} is not a standard name.")
        for prefix_or_name in self._prefix_map[standard_name]:
            if prefix_or_name in data:
                return [prefix_or_name]
            try:
                return self._natural_sort_names(prefix_or_name, data)
            except KeyError:
                pass
        raise KeyError(
            f"No prefix associated with {standard_name} was found in data keys."
        )

    def __call__(self, standard_name: str, data: TensorMapping) -> torch.Tensor:
        """Extract the variable corresponding to standard name and return as a
        3D tensor.

        """
        return self._stack_levels_try(standard_name, data)

    def _stack_levels(self, prefix_or_name: str, data: TensorMapping) -> torch.Tensor:
        names = self._natural_sort_names(prefix_or_name, data)
        # stack along the final dimension
        return torch.stack([data[name] for name in names], dim=-1)

    def _stack_levels_try(
        self, standard_name: str, data: TensorMapping
    ) -> torch.Tensor:
        prefixes_or_names = self._prefix_map[standard_name]
        for prefix_or_name in prefixes_or_names:
            if prefix_or_name in data:
                # 2D variable, return as 1-level 3D tensor
                return data[prefix_or_name].unsqueeze(-1)
            try:
                return self._stack_levels(prefix_or_name, data)
            except KeyError:
                pass
        raise KeyError(
            f"Found no matches for any of {prefixes_or_names} "
            f"among the data names {list(data.keys())}."
        )

    def _natural_sort_names(self, prefix: str, data: TensorMapping) -> List[str]:
        names = [field_name for field_name in data if field_name.startswith(prefix)]

        levels = []
        for name in names:
            match = self.LEVEL_PATTERN.search(name)
            if match is None:
                raise ValueError(
                    f"Invalid field name {name}, is a prefix variable "
                    "but does not end in _{number}."
                )
            levels.append(int(match.group(1)))

        for i, level in enumerate(sorted(levels)):
            if i != level:
                raise ValueError(f"Missing level {i} in {prefix} levels {levels}.")

        if len(names) == 0:
            raise KeyError(prefix)

        return natural_sort(names)
