"""
Snapshot residual loss: optimize a loss on the absolute one-step difference
between predicted states versus the corresponding target difference.

For each configured step ``k``, the loss compares
``|gen[k] - gen[k-1]|`` to ``|target[k] - target[k-1]|``, where ``step=0``
denotes the initial condition (IC). The reference endpoint ``k-1`` is
always detached so gradients flow only through the later prediction.
Using absolute residuals penalizes the magnitude of temporal change
regardless of sign.
"""

import dataclasses
import pathlib
from collections.abc import Mapping

import fsspec
import torch
import xarray as xr

from fme.core.device import get_device
from fme.core.gridded_ops import GriddedOperations
from fme.core.loss import LossOutput, StepLossConfig, WeightedMappingLoss
from fme.core.normalizer import StandardNormalizer
from fme.core.typing_ import TensorDict, TensorMapping


def step_label(step: int) -> str:
    """Stable suffix used in metric keys, e.g. ``"step_1_minus_0"``."""
    return f"step_{step}_minus_{step - 1}"


def load_variance_maps(
    path: str | pathlib.Path,
    names: list[str],
) -> TensorDict:
    """Load per-variable spatial variance maps from a netCDF file.

    Each variable in the file should be a 2D ``(lat, lon)`` DataArray
    representing the climatological variance of temporal residuals at each
    grid cell.

    Args:
        path: Path to the netCDF file.
        names: Variable names to load.

    Returns:
        Dictionary mapping variable names to tensors of shape ``(1, 1, lat, lon)``
        (broadcastable to ``(batch, channel, lat, lon)``).
    """
    with fsspec.open(path, "rb") as f:
        ds = xr.load_dataset(f, mask_and_scale=False)
    result: TensorDict = {}
    for name in names:
        if name not in ds:
            raise ValueError(
                f"Variable {name!r} not found in variance file {path}. "
                f"Available: {sorted(ds.data_vars)}"
            )
        arr = ds[name].values
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D (lat, lon) array for {name!r}, got shape {arr.shape}."
            )
        tensor = torch.as_tensor(arr, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        result[name] = tensor.to(get_device())
    ds.close()
    return result


@dataclasses.dataclass
class SnapshotResidualLossConfig:
    """Configuration for the snapshot residual loss term.

    Each entry ``k`` in :attr:`steps` means: penalize the absolute difference
    ``|gen[k] - gen[k-1]|`` relative to ``|target[k] - target[k-1]|``.
    The reference endpoint ``k-1`` is always detached so the residual
    term constrains only the later prediction.

    Parameters:
        steps: List of step indices (each >= 1). For step ``k``, the
            loss is computed on the residual ``gen[k] - gen[k-1]``.
        loss: Inner loss configuration.
        weight: Multiplier applied to the aggregated residual loss term
            before adding it to the total loss.
        variance_path: Path to a netCDF file containing the climatological
            variance of true temporal residuals at each grid cell, with one
            2D ``(lat, lon)`` variable per output name. When provided, residuals
            are divided by ``sqrt(variance)`` at each grid cell before the
            inner loss, implementing local variance normalization.
    """

    steps: list[int]
    loss: StepLossConfig = dataclasses.field(default_factory=lambda: StepLossConfig())
    weight: float = 1.0
    variance_path: str | pathlib.Path | None = None

    def __post_init__(self):
        if len(self.steps) == 0:
            raise ValueError(
                "SnapshotResidualLossConfig.steps must contain at least one entry."
            )
        for step in self.steps:
            if step < 1:
                raise ValueError(
                    f"Each step must be >= 1, got {step}. The reference is "
                    f"implicitly step - 1."
                )

    @property
    def max_step(self) -> int:
        """Largest step over all configured entries."""
        return max(self.steps)

    def build(
        self,
        gridded_ops: GriddedOperations | None,
        out_names: list[str],
        normalizer: StandardNormalizer,
        channel_dim: int = -3,
    ) -> "SnapshotResidualLoss":
        """Construct a :class:`SnapshotResidualLoss`.

        Args:
            gridded_ops: Gridded operations, required for area-weighted or
                spectral loss types.
            out_names: Variable names the residual loss applies to. Typically
                the intersection of the stepper's ``loss_names`` and
                ``prognostic_names``.
            normalizer: Loss normalizer to use. Means cancel under
                subtraction, so the per-variable stds are what determines the
                scale of normalized residuals.
            channel_dim: Channel dimension of the input tensors.
        """
        inner_loss = self.loss.loss_config.build(
            reduction="none",
            gridded_operations=gridded_ops,
        )
        weighted_mapping_loss = WeightedMappingLoss(
            loss=inner_loss,
            weights=self.loss.weights,
            out_names=out_names,
            channel_dim=channel_dim,
            normalizer=normalizer,
        )
        variance_maps: TensorDict | None = None
        if self.variance_path is not None:
            variance_maps = load_variance_maps(self.variance_path, out_names)
        return SnapshotResidualLoss(
            steps=list(self.steps),
            loss=weighted_mapping_loss,
            out_names=list(out_names),
            weight=self.weight,
            variance_maps=variance_maps,
        )


class SnapshotResidualLoss:
    """Computes a residual loss across configured rollout steps.

    For each active step ``k``, the loss is
    ``inner_loss(|gen[k] - gen[k-1].detach()|, |target[k] - target[k-1].detach()|)``.

    The active set of steps is intersected with the per-batch sampled rollout
    length via :meth:`update_max_n_forward_steps`. Steps exceeding the sampled
    length are silently skipped for that batch.
    """

    def __init__(
        self,
        steps: list[int],
        loss: WeightedMappingLoss,
        out_names: list[str],
        weight: float,
        variance_maps: TensorDict | None = None,
    ):
        self._all_steps: list[int] = sorted(steps)
        self._active_steps: list[int] = list(self._all_steps)
        self._loss = loss
        self._out_names = list(out_names)
        self._weight = weight
        self._variance_maps = variance_maps

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def out_names(self) -> list[str]:
        return list(self._out_names)

    @property
    def all_steps(self) -> list[int]:
        return list(self._all_steps)

    @property
    def active_steps(self) -> list[int]:
        return list(self._active_steps)

    @property
    def effective_loss_scaling(self) -> dict[str, float]:
        return self._loss.effective_loss_scaling

    def max_step(self) -> int:
        return max(self._all_steps)

    def update_max_n_forward_steps(self, n_forward_steps: int) -> None:
        """Filter to steps reachable in this batch.

        A step ``k`` is reachable if ``k <= n_forward_steps``.
        """
        self._active_steps = [s for s in self._all_steps if s <= n_forward_steps]

    def needed_steps(self) -> set[int]:
        """Set of step indices required as inputs for any active step."""
        result: set[int] = set()
        for s in self._active_steps:
            result.add(s)
            result.add(s - 1)
        return result

    def steps_completing_at(self, step: int) -> list[int]:
        """Active steps equal to ``step``.

        Since each step ``k`` needs both ``k`` and ``k-1``, the step
        completes (becomes computable) when the loop reaches ``k``.
        """
        return [s for s in self._active_steps if s == step]

    def compute_residuals(
        self,
        step: int,
        predictions: Mapping[int, TensorMapping],
        targets: Mapping[int, TensorMapping],
    ) -> tuple[TensorMapping, TensorMapping]:
        """Return the absolute residuals for a single step.

        Computes ``|predictions[step] - predictions[step-1].detach()|`` and
        the analogous target residual. The reference endpoint ``step - 1`` is
        always detached so gradients flow only through the later prediction.

        Args:
            step: The step index (>= 1).
            predictions: Must contain keys ``step`` and ``step - 1``.
            targets: Must contain keys ``step`` and ``step - 1``.

        Returns:
            ``(gen_residual, target_residual)`` where each is a
            ``TensorMapping`` keyed by variable name.
        """
        ref = step - 1
        self._validate_step_inputs(step, predictions, targets)
        gen_residual: TensorMapping = {
            name: (predictions[step][name] - predictions[ref][name].detach()).abs()
            for name in self._out_names
        }
        target_residual: TensorMapping = {
            name: (targets[step][name] - targets[ref][name].detach()).abs()
            for name in self._out_names
        }
        if self._variance_maps is not None:
            gen_residual = {
                name: gen_residual[name] / torch.sqrt(self._variance_maps[name])
                for name in self._out_names
            }
            target_residual = {
                name: target_residual[name] / torch.sqrt(self._variance_maps[name])
                for name in self._out_names
            }
        return gen_residual, target_residual

    def compute_step_loss(
        self,
        step: int,
        predictions: Mapping[int, TensorMapping],
        targets: Mapping[int, TensorMapping],
    ) -> torch.Tensor:
        """Compute one step's residual loss as a scalar tensor.

        Builds the absolute residuals via :meth:`compute_residuals` and
        passes them to the inner loss.

        Args:
            step: The step index (>= 1).
            predictions: Must contain keys ``step`` and ``step - 1``.
            targets: Must contain keys ``step`` and ``step - 1``.

        Returns:
            Unweighted residual loss scalar; callers apply :attr:`weight`.
        """
        gen_residual, target_residual = self.compute_residuals(
            step, predictions, targets
        )
        loss_output: LossOutput = self._loss(gen_residual, target_residual)
        return loss_output.total()

    def __call__(
        self,
        predictions: Mapping[int, TensorMapping],
        targets: Mapping[int, TensorMapping],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the aggregated residual loss and per-step scalars.

        Returns:
            ``(total, per_step)`` where ``total`` is the sum over active
            steps, and ``per_step`` maps a label (e.g.
            ``"residual_loss_step_1_minus_0"``) to that step's scalar
            contribution. The returned ``total`` does not include the
            configured ``weight`` multiplier; callers apply it.
        """
        per_step: dict[str, torch.Tensor] = {}
        running_total: torch.Tensor | None = None
        for s in self._active_steps:
            scalar = self.compute_step_loss(s, predictions, targets)
            per_step[f"residual_loss_{step_label(s)}"] = scalar
            running_total = scalar if running_total is None else running_total + scalar
        if running_total is None:
            running_total = torch.zeros((), device=self._zero_device())
        return running_total, per_step

    def _validate_step_inputs(
        self,
        step: int,
        predictions: Mapping[int, TensorMapping],
        targets: Mapping[int, TensorMapping],
    ) -> None:
        ref = step - 1
        for s in (step, ref):
            if s not in predictions:
                raise KeyError(
                    f"Missing prediction at step {s} required by residual "
                    f"step {step}; available: {sorted(predictions)}"
                )
            if s not in targets:
                raise KeyError(
                    f"Missing target at step {s} required by residual "
                    f"step {step}; available: {sorted(targets)}"
                )
            for name in self._out_names:
                if name not in predictions[s]:
                    raise KeyError(
                        f"Missing variable {name!r} in prediction at step "
                        f"{s} required by residual step {step}."
                    )
                if name not in targets[s]:
                    raise KeyError(
                        f"Missing variable {name!r} in target at step "
                        f"{s} required by residual step {step}."
                    )

    def _zero_device(self) -> torch.device:
        for std in self._loss.normalizer.stds.values():
            return std.device
        return torch.device("cpu")
