import re
from collections.abc import Iterable, Mapping

import torch

from fme.core.typing_ import TensorMapping


def unstack(tensor: torch.Tensor, names: list[str], dim: int = -1) -> TensorMapping:
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
            f"Received {len(names)} names, but 3D tensor has {tensor.size(-1)} levels."
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
        prefix_map: Mapping[str, list[str]] | None = None,
    ):
        """
        Args:
            prefix_map: Mapping which defines the correspondence between an arbitrary
                set of "standard" names (e.g., "surface_pressure" or "air_temperature")
                and lists of possible names or prefix variants (e.g., ["PRESsfc", "PS"]
                or ["air_temperature_", "T_"]) found in the data.
        """
        self._prefix_map: Mapping[str, list[str]] | None = prefix_map

    def infer_prefix_map(self, names: Iterable[str]):
        """
        Infer the prefix map from data names.

        Args:
             names: Variable names, such as the keys from a tensor mapping.
        """
        if self.has_prefix_map:
            raise RuntimeError("Prefix map has already been built.")
        prefix_map = {}
        for name in names:
            match = self.LEVEL_PATTERN.search(name)
            if match is None:
                # 2D variable
                prefix_map[name] = [name]
            else:
                # 3D variable, +1 to include "_" in the prefix
                prefix_name = name[: match.start() + 1]
                prefix_map[prefix_name] = [prefix_name]
        self._prefix_map = prefix_map

    @property
    def has_prefix_map(self) -> bool:
        return self._prefix_map is not None

    @property
    def prefix_map(self) -> Mapping[str, list[str]]:
        """Mapping which defines the correspondence between an arbitrary set of
        "standard" names (e.g., "surface_pressure" or "air_temperature") and
        lists of possible names or prefix variants (e.g., ["PRESsfc", "PS"] or
        ["air_temperature_", "T_"]) found in the data.
        """
        if self._prefix_map is None:
            raise RuntimeError(
                "Stacker's prefix map hasn't yet been built. Build it at runtime by "
                "first calling stacker.infer_prefix_map(data) with the data."
            )
        return self._prefix_map

    @property
    def standard_names(self) -> list[str]:
        return list(self.prefix_map.keys())

    def get_all_level_names(self, standard_name: str, data: TensorMapping) -> list[str]:
        """Get the names of all variables in the data that match one of the
        prefixes associated with the given standard name. If the standard name
        corresponds to a 3D variable, returns all vertical level names in their
        natural order.
        """
        if standard_name not in self.standard_names:
            raise ValueError(f"{standard_name} is not a standard name.")
        for prefix_or_name in self.prefix_map[standard_name]:
            if prefix_or_name in data:
                return [prefix_or_name]
            try:
                return self._natural_sort_names(prefix_or_name, data)
            except KeyError:
                pass
        raise KeyError(
            f"No prefix associated with '{standard_name}' was found in data keys."
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
        prefixes_or_names = self.prefix_map[standard_name]
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

    def _natural_sort_names(self, prefix: str, data: TensorMapping) -> list[str]:
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

        return sorted(names, key=lambda name: levels[names.index(name)])
