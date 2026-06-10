import dataclasses
from typing import Literal

import torch

from fme.core.gridded_ops import GriddedOperations
from fme.core.normalizer import StandardNormalizer
from fme.core.typing_ import TensorDict, TensorMapping

USE_NORMALIZATION_MEAN: Literal["mean"] = "mean"


@dataclasses.dataclass
class GlobalMeanRelaxationVariableConfig:
    """
    Per-variable target and timescale for global-mean Newtonian relaxation.

    Parameters:
        target: Target global mean. Either a float in the variable's
            physical units, or the literal string ``"mean"`` to use the
            variable's normalization mean as the target.
        timescale_steps: Relaxation timescale in steps. Each step shifts the
            field by ``-(area_weighted_mean(x) - target) / timescale_steps``,
            i.e. an exponential approach to ``target`` with e-folding time of
            ``timescale_steps`` steps. Must be positive.
    """

    target: float | Literal["mean"]
    timescale_steps: float

    def __post_init__(self):
        if self.timescale_steps <= 0:
            raise ValueError(
                f"timescale_steps must be positive, got {self.timescale_steps}."
            )
        if isinstance(self.target, str) and self.target != USE_NORMALIZATION_MEAN:
            raise ValueError(
                f"target must be a number or '{USE_NORMALIZATION_MEAN}', got "
                f"{self.target!r}."
            )


@dataclasses.dataclass
class GlobalMeanRelaxationConfig:
    """
    Newtonian relaxation of the global mean of selected output variables
    toward configured target values.

    The relaxation is applied during ``.step`` in eval mode only, after
    the network call (and any global-mean-removal inverse transform) and
    before the corrector. For each named variable, the step subtracts a
    uniform offset of ``(area_weighted_mean(x) - target) / timescale_steps``
    from the field.

    Names in ``variables`` must appear in ``out_names`` of the enclosing
    step config. Variables whose ``target`` is ``"mean"`` must have a
    normalization mean defined in the network normalizer.

    Parameters:
        variables: Mapping from output variable name to its per-variable
            relaxation configuration.
    """

    variables: dict[str, GlobalMeanRelaxationVariableConfig]

    def __post_init__(self):
        if len(self.variables) == 0:
            raise ValueError(
                "global_mean_relaxation.variables must contain at least one entry; "
                "use null/None to disable relaxation."
            )

    def validate_names(self, out_names: list[str]) -> None:
        out_set = set(out_names)
        missing = [name for name in self.variables if name not in out_set]
        if missing:
            raise ValueError(
                "global_mean_relaxation variable names must be in out_names; "
                f"missing: {missing}. out_names: {out_names}."
            )

    def build(
        self,
        gridded_operations: GriddedOperations,
        normalizer: StandardNormalizer,
    ) -> "GlobalMeanRelaxation":
        targets: dict[str, float] = {}
        timescales: dict[str, float] = {}
        for name, var_config in self.variables.items():
            if var_config.target == USE_NORMALIZATION_MEAN:
                if name not in normalizer.means:
                    raise ValueError(
                        f"global_mean_relaxation target='{USE_NORMALIZATION_MEAN}' "
                        f"requires '{name}' to have a normalization mean; "
                        "none found in the network normalizer."
                    )
                targets[name] = float(normalizer.means[name].item())
            else:
                targets[name] = float(var_config.target)
            timescales[name] = var_config.timescale_steps
        return GlobalMeanRelaxation(
            targets=targets,
            timescales=timescales,
            gridded_operations=gridded_operations,
        )


class GlobalMeanRelaxation:
    """Apply Newtonian relaxation of the global mean toward target values.

    For each configured variable, computes the area-weighted global mean
    and subtracts a uniform offset that nudges the field's mean toward
    the configured target on the configured timescale (in steps).
    Variables not in the configuration are returned unchanged. Variables
    in the configuration but absent from the input are skipped silently
    so that the relaxation can be a no-op for steps that don't produce
    that field on a given call.
    """

    def __init__(
        self,
        targets: dict[str, float],
        timescales: dict[str, float],
        gridded_operations: GriddedOperations,
    ):
        if set(targets) != set(timescales):
            raise ValueError(
                "targets and timescales must have the same keys: "
                f"targets={list(targets)}, timescales={list(timescales)}."
            )
        self._targets = targets
        self._timescales = timescales
        self._gridded_operations = gridded_operations

    def __call__(self, data: TensorMapping) -> TensorDict:
        result: TensorDict = dict(data)
        for name, target_value in self._targets.items():
            if name not in result:
                continue
            field = result[name]
            mean = self._gridded_operations.area_weighted_mean(
                field, keepdim=True, name=name
            )
            target = torch.as_tensor(
                target_value, dtype=field.dtype, device=field.device
            )
            offset = (mean - target) / self._timescales[name]
            result[name] = field - offset
        return result
