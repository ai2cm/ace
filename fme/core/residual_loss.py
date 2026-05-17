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
import math
import pathlib
from collections.abc import Mapping

import fsspec
import torch
import xarray as xr

from fme.core.device import get_device
from fme.core.gridded_ops import GriddedOperations
from fme.core.loss import LossOutput, StepLossConfig, WeightedMappingLoss
from fme.core.normalizer import StandardNormalizer, load_dict_from_netcdf
from fme.core.typing_ import TensorDict, TensorMapping


def step_label(step: int) -> str:
    """Stable suffix used in metric keys, e.g. ``"step_1_minus_0"``."""
    return f"step_{step}_minus_{step - 1}"


def load_std_maps(
    path: str | pathlib.Path,
    names: list[str],
) -> TensorDict:
    """Load per-variable spatial standard-deviation maps from a netCDF file.

    Each variable in the file should be a 2D ``(lat, lon)`` DataArray
    representing the climatological standard deviation of temporal residuals
    at each grid cell.

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
                f"Variable {name!r} not found in std maps file {path}. "
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

    The residual loss penalizes the absolute difference
    ``|gen[k] - gen[k-1]|`` relative to ``|target[k] - target[k-1]|``.
    The reference endpoint ``k-1`` is always detached so the residual
    term constrains only the later prediction.

    Which steps receive the residual loss is determined at runtime by the
    training stepper based on ``n_forward_steps`` and
    ``optimize_last_step_only``.

    Parameters:
        residual_stds_path: Path to a netCDF file containing scalar
            per-variable standard deviations of temporal residuals
            (the "global" std ``sigma_g``).
        loss: Inner loss configuration.
        weight: Multiplier applied to the aggregated residual loss term
            before adding it to the total loss.
        std_maps_path: Path to a netCDF file containing the climatological
            standard deviation of temporal residuals at each grid cell,
            with one 2D ``(lat, lon)`` variable per output name. When
            provided, normalized residuals are divided by
            ``sigma_l + eps * sigma_g`` at each grid cell before the
            inner loss, where ``sigma_l`` is the local std and
            ``sigma_g`` is the scalar std from ``residual_stds_path``.
        full_state_stds_path: Path to a netCDF file containing scalar
            per-variable standard deviations of the full state. When
            provided, the per-variable loss is scaled by
            ``sigma_residual / sigma_full`` (applied as ``sqrt`` on both
            residual inputs to exploit MSE's quadratic nature). This
            brings the residual loss to the same scale as the MSE loss.
    """

    residual_stds_path: str | pathlib.Path
    loss: StepLossConfig = dataclasses.field(default_factory=lambda: StepLossConfig())
    weight: float = 1.0
    std_maps_path: str | pathlib.Path | None = None
    full_state_stds_path: str | pathlib.Path | None = None

    def build(
        self,
        gridded_ops: GriddedOperations | None,
        out_names: list[str],
        channel_dim: int = -3,
    ) -> "SnapshotResidualLoss":
        """Construct a :class:`SnapshotResidualLoss`.

        Args:
            gridded_ops: Gridded operations, required for area-weighted or
                spectral loss types.
            out_names: Variable names the residual loss applies to. Typically
                the intersection of the stepper's ``loss_names`` and
                ``prognostic_names``.
            channel_dim: Channel dimension of the input tensors.
        """
        residual_stds = load_dict_from_netcdf(
            self.residual_stds_path,
            names=out_names,
            defaults={},
        )
        identity_normalizer = StandardNormalizer(
            means={name: torch.zeros(()) for name in out_names},
            stds={name: torch.ones(()) for name in out_names},
        )
        inner_loss = self.loss.loss_config.build(
            gridded_operations=gridded_ops,
        )
        weighted_mapping_loss = WeightedMappingLoss(
            loss=inner_loss,
            weights=self.loss.weights,
            out_names=out_names,
            channel_dim=channel_dim,
            normalizer=identity_normalizer,
        )
        residual_std_maps: TensorDict | None = None
        if self.std_maps_path is not None:
            residual_std_maps = load_std_maps(self.std_maps_path, out_names)
        residual_scale: dict[str, float] | None = None
        if self.full_state_stds_path is not None:
            full_state_stds = load_dict_from_netcdf(
                self.full_state_stds_path,
                names=out_names,
                defaults={},
            )
            residual_scale = {
                name: math.sqrt(residual_stds[name] / full_state_stds[name])
                for name in out_names
            }
        return SnapshotResidualLoss(
            loss=weighted_mapping_loss,
            residual_stds=residual_stds,
            out_names=list(out_names),
            weight=self.weight,
            residual_std_maps=residual_std_maps,
            residual_scale=residual_scale,
        )


class SnapshotResidualLoss:
    """Computes a residual loss across configured rollout steps.

    For each active step ``k``, the loss is
    ``inner_loss(|gen[k] - gen[k-1].detach()|, |target[k] - target[k-1].detach()|)``.

    When ``residual_std_maps`` are provided, the normalization formula is
    ``|r| / (sigma_l + eps * sigma_g)`` where ``r`` is the raw snapshot
    residual, ``sigma_l`` is the local (spatial) std, and ``sigma_g`` is
    the corresponding scalar std. Without std maps, residuals are simply
    divided by ``sigma_g``.

    When ``residual_scale`` is provided, normalized residuals are further
    multiplied by ``sqrt(sigma_residual / sigma_full)`` per variable so
    that the MSE loss is scaled by ``sigma_residual / sigma_full``,
    bringing it to the same magnitude as the full-state MSE loss.

    Active steps are set each batch via :meth:`set_active_steps`.
    """

    def __init__(
        self,
        loss: WeightedMappingLoss,
        residual_stds: dict[str, float],
        out_names: list[str],
        weight: float,
        residual_std_maps: TensorDict | None = None,
        eps: float = 1e-2,
        residual_scale: dict[str, float] | None = None,
    ):
        self._active_steps: list[int] = []
        self._loss = loss
        self._residual_stds = residual_stds
        self._out_names = list(out_names)
        self._weight = weight
        self._residual_std_maps = residual_std_maps
        self._eps = eps
        self._residual_scale = residual_scale

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def out_names(self) -> list[str]:
        return list(self._out_names)

    @property
    def active_steps(self) -> list[int]:
        return list(self._active_steps)

    @property
    def effective_loss_scaling(self) -> dict[str, float]:
        weights = dict(
            zip(self._loss.packer.names, self._loss._weight_tensor.flatten())
        )
        return {k: self._residual_stds[k] / weights[k] for k in self._out_names}

    def set_active_steps(self, steps: list[int]) -> None:
        """Set the active steps for this batch.

        Args:
            steps: 1-indexed step indices where residual loss will be
                computed. Each step ``k`` computes the residual
                ``gen[k] - gen[k-1]``.
        """
        for s in steps:
            if s < 1:
                raise ValueError(f"Each residual step must be >= 1, got {s}.")
        self._active_steps = sorted(steps)

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
        """Return the normalized absolute residuals for a single step.

        Operation order:
        1. Raw signed residual: ``pred[k] - pred[k-1].detach()``
        2. Take absolute value
        3. Divide by ``sigma_l + eps * sigma_g`` if std maps are provided,
           otherwise divide by ``sigma_g``

        The reference endpoint ``step - 1`` is always detached so gradients
        flow only through the later prediction.

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
        if self._residual_std_maps is not None:
            gen_residual = {
                name: gen_residual[name]
                / (
                    self._residual_std_maps[name]
                    + self._eps * self._residual_stds[name]
                )
                for name in self._out_names
            }
            target_residual = {
                name: target_residual[name]
                / (
                    self._residual_std_maps[name]
                    + self._eps * self._residual_stds[name]
                )
                for name in self._out_names
            }
        else:
            gen_residual = {
                name: gen_residual[name] / self._residual_stds[name]
                for name in self._out_names
            }
            target_residual = {
                name: target_residual[name] / self._residual_stds[name]
                for name in self._out_names
            }
        if self._residual_scale is not None:
            gen_residual = {
                name: gen_residual[name] * self._residual_scale[name]
                for name in self._out_names
            }
            target_residual = {
                name: target_residual[name] * self._residual_scale[name]
                for name in self._out_names
            }
        return gen_residual, target_residual

    def compute_step_loss(
        self,
        step: int,
        predictions: Mapping[int, TensorMapping],
        targets: Mapping[int, TensorMapping],
    ) -> torch.Tensor:
        """Compute one step's residual loss as a weighted scalar tensor.

        Builds the absolute residuals via :meth:`compute_residuals` and
        passes them to the inner loss, then applies :attr:`weight`.

        Args:
            step: The step index (>= 1).
            predictions: Must contain keys ``step`` and ``step - 1``.
            targets: Must contain keys ``step`` and ``step - 1``.

        Returns:
            Weighted residual loss scalar.
        """
        gen_residual, target_residual = self.compute_residuals(
            step, predictions, targets
        )
        loss_output: LossOutput = self._loss(gen_residual, target_residual)
        return self._weight * loss_output.total()

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
