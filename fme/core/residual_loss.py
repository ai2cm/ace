"""
Snapshot residual loss: optimize a loss on the difference between predicted
states at two rollout timesteps versus the corresponding target difference.

For a configured pair ``(step_a, step_b)``, the loss compares
``gen[step_a] - gen[step_b]`` to ``target[step_a] - target[step_b]``, where
``step=0`` denotes the initial condition (IC). This is the optimizer-side
counterpart to the residual snapshot panel rendered by
:class:`fme.ace.aggregator.one_step.snapshot.SnapshotAggregator`, generalized
to arbitrary pairs of timesteps and constrained at runtime by the per-batch
sampled rollout length.
"""

import dataclasses
from collections.abc import Mapping

import torch

from fme.core.gridded_ops import GriddedOperations
from fme.core.loss import LossOutput, StepLossConfig, WeightedMappingLoss
from fme.core.normalizer import StandardNormalizer
from fme.core.typing_ import TensorMapping


@dataclasses.dataclass(frozen=True)
class ResidualPair:
    """A pair of rollout timesteps for the snapshot residual loss.

    The convention is ``step=0`` denotes the initial condition (IC) and
    ``step=k`` for ``k >= 1`` denotes the ``k``-th forward prediction. This
    matches :class:`fme.ace.aggregator.one_step.snapshot.SnapshotAggregator`
    where the residual panel for step ``k`` is ``gen[k] - input[0]``, i.e. pair
    ``(k, 0)``.

    Parameters:
        step_a: First endpoint, ``>= 0``.
        step_b: Second endpoint, ``>= 0``. Must differ from ``step_a``.
    """

    step_a: int
    step_b: int

    def __post_init__(self):
        if self.step_a < 0 or self.step_b < 0:
            raise ValueError(
                f"ResidualPair endpoints must be non-negative, got "
                f"step_a={self.step_a}, step_b={self.step_b}."
            )
        if self.step_a == self.step_b:
            raise ValueError(
                f"ResidualPair endpoints must differ, got step_a={self.step_a} "
                f"== step_b={self.step_b}."
            )

    @property
    def max_step(self) -> int:
        return max(self.step_a, self.step_b)

    @property
    def label(self) -> str:
        """Stable suffix used in metric keys, e.g. ``"step_3_minus_1"``."""
        return f"step_{self.step_a}_minus_{self.step_b}"


@dataclasses.dataclass
class SnapshotResidualLossConfig:
    """Configuration for the snapshot residual loss term.

    The residual loss is computed by reusing the inner loss machinery of
    :class:`StepLossConfig` (``type``, ``kwargs``, ``weights``,
    ``global_mean_*``), applied to the residual dictionaries
    ``{name: gen[step_a] - gen[step_b]}`` and the analogous target.

    Parameters:
        pairs: List of timestep pairs to evaluate. Pairs with
            ``max(step_a, step_b)`` exceeding the maximum sampled rollout
            length (``TimeLengthProbabilities.max_n_forward_steps``) are
            rejected at config build time.
        loss: Inner loss configuration. The
            :attr:`StepLossConfig.sqrt_loss_step_decay_constant` field is not
            used here (no notion of single "current step").
        weight: Multiplier applied to the aggregated residual loss term
            before adding it to the total loss.
    """

    pairs: list[ResidualPair]
    loss: StepLossConfig = dataclasses.field(default_factory=lambda: StepLossConfig())
    weight: float = 1.0

    def __post_init__(self):
        if len(self.pairs) == 0:
            raise ValueError(
                "SnapshotResidualLossConfig.pairs must contain at least one "
                "ResidualPair."
            )

    @property
    def max_pair_step(self) -> int:
        """Largest endpoint over all configured pairs."""
        return max(pair.max_step for pair in self.pairs)

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
        return SnapshotResidualLoss(
            pairs=list(self.pairs),
            loss=weighted_mapping_loss,
            out_names=list(out_names),
            weight=self.weight,
        )


class SnapshotResidualLoss:
    """Computes a residual loss across configured rollout-timestep pairs.

    The active set of pairs is intersected with the per-batch sampled rollout
    length via :meth:`update_max_n_forward_steps`. Pairs whose maximum
    endpoint exceeds the sampled length are silently skipped for that batch.
    """

    def __init__(
        self,
        pairs: list[ResidualPair],
        loss: WeightedMappingLoss,
        out_names: list[str],
        weight: float,
    ):
        self._all_pairs: list[ResidualPair] = list(pairs)
        self._active_pairs: list[ResidualPair] = list(pairs)
        self._loss = loss
        self._out_names = list(out_names)
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def out_names(self) -> list[str]:
        return list(self._out_names)

    @property
    def all_pairs(self) -> list[ResidualPair]:
        return list(self._all_pairs)

    @property
    def active_pairs(self) -> list[ResidualPair]:
        return list(self._active_pairs)

    @property
    def effective_loss_scaling(self) -> dict[str, float]:
        return self._loss.effective_loss_scaling

    def max_pair_step(self) -> int:
        return max(pair.max_step for pair in self._all_pairs)

    def update_max_n_forward_steps(self, n_forward_steps: int) -> None:
        """Filter ``all_pairs`` to those reachable in this batch.

        A pair is reachable if ``max(step_a, step_b) <= n_forward_steps``.
        ``step=0`` (the IC) is always reachable.
        """
        self._active_pairs = [
            pair for pair in self._all_pairs if pair.max_step <= n_forward_steps
        ]

    def needed_steps(self) -> set[int]:
        """Set of step indices required as inputs for any active pair."""
        steps: set[int] = set()
        for pair in self._active_pairs:
            steps.add(pair.step_a)
            steps.add(pair.step_b)
        return steps

    def pairs_completing_at(self, step: int) -> list[ResidualPair]:
        """Active pairs whose ``max_step`` equals ``step``.

        Used by callers that fold each pair's residual contribution into
        the loss at the step where the pair becomes computable.
        """
        return [pair for pair in self._active_pairs if pair.max_step == step]

    def compute_pair_loss(
        self,
        pair: ResidualPair,
        predictions: Mapping[int, TensorMapping],
        targets: Mapping[int, TensorMapping],
    ) -> torch.Tensor:
        """Compute a single pair's residual loss as a scalar tensor.

        The earlier endpoint (the one with the smaller step index) is
        ``.detach()``-ed on both the prediction and target sides before
        forming the residual difference. This means the residual term
        only propagates gradients through the later endpoint -- the
        earlier endpoint is treated as a fixed reference. This semantic
        is shared regardless of whether the caller uses gradient
        accumulation; see :class:`SnapshotResidualLossConfig` for
        background.

        Args:
            pair: The pair to evaluate. Must be an active pair.
            predictions: Mapping from step index to a dict of predicted
                tensors. Must contain ``pair.step_a`` and ``pair.step_b``.
                Tensors should be denormalized; the inner
                :class:`WeightedMappingLoss` applies normalization.
            targets: Mapping from step index to a dict of target tensors.
                Same key requirements as ``predictions``.

        Returns:
            The (unweighted) residual loss for this pair as a scalar
            tensor; callers apply the configured :attr:`weight`.
        """
        self._validate_pair_inputs(pair, predictions, targets)
        gen_residual = self._build_residual_dict(predictions, pair)
        target_residual = self._build_residual_dict(targets, pair)
        loss_output: LossOutput = self._loss(gen_residual, target_residual)
        return loss_output.total()

    def __call__(
        self,
        predictions: Mapping[int, TensorMapping],
        targets: Mapping[int, TensorMapping],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the aggregated residual loss and per-pair scalars.

        Args:
            predictions: Mapping from step index to a dict of predicted
                tensors. Must contain every step referenced by an active
                pair. Tensors should be denormalized; the inner
                :class:`WeightedMappingLoss` applies normalization.
            targets: Mapping from step index to a dict of target tensors.
                Same key requirements as ``predictions``.

        Returns:
            ``(total, per_pair)`` where ``total`` is the sum over active
            pairs of the inner loss applied to ``(gen_residual,
            target_residual)``, and ``per_pair`` maps a stable label
            (e.g. ``"residual_loss_step_3_minus_1"``) to that pair's scalar
            contribution to ``total`` (before applying the configured
            ``weight``).

            If ``active_pairs`` is empty, ``total`` is a zero scalar tensor
            and ``per_pair`` is empty. The returned ``total`` does not
            include the configured ``weight`` multiplier; callers apply it.

            The earlier endpoint of each pair is detached -- see
            :meth:`compute_pair_loss` for the rationale.
        """
        per_pair: dict[str, torch.Tensor] = {}
        running_total: torch.Tensor | None = None
        for pair in self._active_pairs:
            scalar = self.compute_pair_loss(pair, predictions, targets)
            per_pair[f"residual_loss_{pair.label}"] = scalar
            running_total = scalar if running_total is None else running_total + scalar
        if running_total is None:
            running_total = torch.zeros((), device=self._zero_device())
        return running_total, per_pair

    def _build_residual_dict(
        self,
        step_to_data: Mapping[int, TensorMapping],
        pair: ResidualPair,
    ) -> dict[str, torch.Tensor]:
        if pair.step_a >= pair.step_b:
            later, earlier = pair.step_a, pair.step_b
        else:
            later, earlier = pair.step_b, pair.step_a
        late = step_to_data[later]
        early = step_to_data[earlier]
        # The earlier endpoint is detached so this residual term only
        # propagates gradient through the later prediction. With gradient
        # accumulation, the earlier endpoint's autograd graph has typically
        # already been freed by its own per-step backward when this is
        # invoked from the stepper; without gradient accumulation, the
        # detach makes the residual a one-sided constraint by design.
        if pair.step_a >= pair.step_b:
            # Preserve the original sign convention: gen[step_a] - gen[step_b].
            return {name: late[name] - early[name].detach() for name in self._out_names}
        else:
            # step_a is the earlier (detached) endpoint.
            return {name: early[name].detach() - late[name] for name in self._out_names}

    def _validate_pair_inputs(
        self,
        pair: ResidualPair,
        predictions: Mapping[int, TensorMapping],
        targets: Mapping[int, TensorMapping],
    ) -> None:
        for step in (pair.step_a, pair.step_b):
            if step not in predictions:
                raise KeyError(
                    f"Missing prediction at step {step} required by pair "
                    f"{pair}; available: {sorted(predictions)}"
                )
            if step not in targets:
                raise KeyError(
                    f"Missing target at step {step} required by pair "
                    f"{pair}; available: {sorted(targets)}"
                )
            for name in self._out_names:
                if name not in predictions[step]:
                    raise KeyError(
                        f"Missing variable {name!r} in prediction at step "
                        f"{step} required by pair {pair}."
                    )
                if name not in targets[step]:
                    raise KeyError(
                        f"Missing variable {name!r} in target at step "
                        f"{step} required by pair {pair}."
                    )

    def _zero_device(self) -> torch.device:
        # use any normalizer std tensor to determine the device for the
        # zero-result fallback
        for std in self._loss.normalizer.stds.values():
            return std.device
        return torch.device("cpu")
