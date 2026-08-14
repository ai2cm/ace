import dataclasses
from collections.abc import Collection

from fme.core.gridded_ops import GriddedOperations
from fme.core.loss import CorrectorLoss, LossConfig, WeightedMappingLoss
from fme.core.name_and_prefix_matcher import NameAndPrefixSelection
from fme.core.normalizer import StandardNormalizer


@dataclasses.dataclass
class PreCorrectorOptimizationConfig:
    """Selects corrector-modified variables whose main-loss prediction is the
    pre-corrector network output ``prediction - delta``.

    Selection is pure opt-in; this feature may be enabled together with
    corrector regularization.

    Parameters:
        names_and_prefixes: Required ``NameAndPrefixMatcher`` entries selecting
            the corrector-modified variables to optimize pre-corrector.
    """

    names_and_prefixes: list[str] | None = None

    def __post_init__(self):
        if not self.names_and_prefixes:
            raise ValueError(
                "precorrector_optimization requires names_and_prefixes: "
                "configuring the feature while selecting nothing is a "
                "contradiction, not a no-op."
            )


@dataclasses.dataclass
class CorrectorRegularizationConfig:
    """A penalty pushing selected correction deltas toward zero in
    loss-normalized space.

    The penalty is not subject to the main loss's per-step sqrt decay
    (``sqrt_loss_step_decay_constant``); only ``weight`` scales it. This
    feature may be enabled together with pre-corrector optimization.

    Parameters:
        loss: The loss applied to the normalized deltas against zeros. The
            ``EnsembleLoss`` and ``NaN`` types and ``global_mean_type`` are not
            supported.
        weight: The positive weight applied to the penalty in the total loss.
        names_and_prefixes: Required ``NameAndPrefixMatcher`` entries selecting
            the corrector-modified variables whose deltas are penalized.
    """

    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    weight: float = 1.0
    names_and_prefixes: list[str] | None = None

    def __post_init__(self):
        if not self.names_and_prefixes:
            raise ValueError(
                "regularization requires names_and_prefixes: configuring "
                "the feature while selecting nothing is a contradiction, "
                "not a no-op."
            )
        if self.loss.type in ("EnsembleLoss", "NaN"):
            raise ValueError(
                f"loss type {self.loss.type!r} is not supported for "
                "corrector regularization."
            )
        if self.loss.global_mean_type is not None:
            raise ValueError(
                "global_mean_type is not supported for corrector regularization."
            )
        if self.weight <= 0:
            raise ValueError(
                f"regularization weight must be positive, got {self.weight}"
            )


@dataclasses.dataclass
class CorrectorLossConfig:
    """Training-only consumption of correction deltas by the step loss.

    Parameters:
        precorrector_optimization: Optimize selected corrector-modified
            variables against their pre-corrector network outputs.
        regularization: Penalize selected correction deltas toward zero.
    """

    precorrector_optimization: PreCorrectorOptimizationConfig | None = None
    regularization: CorrectorRegularizationConfig | None = None

    def __post_init__(self):
        if self.precorrector_optimization is None and self.regularization is None:
            raise ValueError(
                "corrector_loss requires at least one of "
                "precorrector_optimization or regularization: configuring "
                "it while selecting no feature is a contradiction, not a "
                "no-op."
            )

    def build(
        self,
        corrector_modified_names: frozenset[str],
        prescribed_prognostic_names: Collection[str],
        normalizer: StandardNormalizer,
        gridded_operations: GriddedOperations | None,
        channel_dim: int = -3,
    ) -> CorrectorLoss:
        """Validate the configured selections and build the corrector loss.

        All name validation happens here, when the run starts: entries are
        checked against the modified names minus the prescribed prognostics.

        Args:
            corrector_modified_names: The delta keys the step's corrector
                produces when active.
            prescribed_prognostic_names: Names whose deltas are dropped by
                ``step_with_adjustments`` after the prescribed overwrite.
            normalizer: The loss normalizer, used to normalize the deltas.
            gridded_operations: Gridded operations for losses that need the
                horizontal dimensions.
            channel_dim: The channel dimension of the loss inputs.
        """
        if len(corrector_modified_names) == 0:
            raise ValueError(
                "corrector_loss is configured but the corrector modifies no "
                "variables, so there are no correction deltas to consume."
            )
        prescribed = frozenset(prescribed_prognostic_names)
        loss_visible_names = corrector_modified_names - prescribed
        dropped_names = corrector_modified_names & prescribed

        def _validated_matches(
            selection: NameAndPrefixSelection, feature: str
        ) -> list[str]:
            unmatched = selection.unmatched_entries(loss_visible_names)
            if unmatched:
                reasons = []
                for entry in unmatched:
                    entry_selection = NameAndPrefixSelection((entry,))
                    dropped = entry_selection.matched(dropped_names)
                    if dropped:
                        reasons.append(
                            f"{entry!r} selects only prescribed prognostic "
                            f"variables {dropped}, whose correction deltas "
                            "step_with_adjustments drops after the prescribed "
                            "overwrite, so they never reach the loss"
                        )
                    else:
                        reasons.append(
                            f"{entry!r} selects no variable the corrector modifies"
                        )
                raise ValueError(
                    f"{feature} has entries that select nothing usable: "
                    + "; ".join(reasons)
                    + f". The corrector modifies {sorted(corrector_modified_names)}."
                )
            return selection.matched(loss_visible_names)

        precorrector_names = None
        if self.precorrector_optimization is not None:
            assert self.precorrector_optimization.names_and_prefixes is not None
            precorrector_names = _validated_matches(
                NameAndPrefixSelection(
                    tuple(self.precorrector_optimization.names_and_prefixes)
                ),
                "precorrector_optimization",
            )
        regularizer = None
        penalty_weight = 1.0
        if self.regularization is not None:
            assert self.regularization.names_and_prefixes is not None
            regularizer_names = _validated_matches(
                NameAndPrefixSelection(tuple(self.regularization.names_and_prefixes)),
                "regularization",
            )
            regularizer = WeightedMappingLoss(
                loss=self.regularization.loss.build(gridded_operations),
                weights={},
                out_names=regularizer_names,
                normalizer=normalizer,
                channel_dim=channel_dim,
            )
            penalty_weight = self.regularization.weight
        return CorrectorLoss(
            precorrector_names=precorrector_names,
            regularizer=regularizer,
            penalty_weight=penalty_weight,
        )
