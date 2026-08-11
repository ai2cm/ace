import dataclasses
from collections.abc import Collection

from fme.core.corrector.registry import CorrectorABC
from fme.core.gridded_ops import GriddedOperations
from fme.core.loss import LossConfig, StepLossCorrectorArgs, WeightedMappingLoss
from fme.core.name_and_prefix_matcher import NameAndPrefixSelection
from fme.core.normalizer import StandardNormalizer


@dataclasses.dataclass
class PreCorrectorOptimizationConfig:
    """Selects corrector-modified variables whose main-loss prediction is
    the pre-corrector value ``prediction - delta``.

    Selection is pure opt-in; this feature may be enabled together with
    corrector regularization.

    Parameters:
        names_and_prefixes: Names and prefixes (following the
            ``NameAndPrefixMatcher`` convention) of corrector-modified
            variables to optimize pre-corrector. Required: configuring the
            feature while selecting nothing is a contradiction, not a no-op.
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
            ``EnsembleLoss`` and ``NaN`` types and any ``global_mean_type``
            are not supported.
        weight: The weight applied to the penalty when it is added to the
            total loss. Must be positive.
        names_and_prefixes: Names and prefixes (following the
            ``NameAndPrefixMatcher`` convention) of corrector-modified
            variables whose deltas are penalized. Required: configuring the
            feature while selecting nothing is a contradiction, not a no-op.
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
            variables against their pre-corrector predictions.
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
        corrector: CorrectorABC | None,
        prescribed_prognostic_names: Collection[str],
        normalizer: StandardNormalizer,
        gridded_operations: GriddedOperations | None,
        channel_dim: int = -3,
    ) -> StepLossCorrectorArgs:
        """Validate the configured selections and build the loss arguments.

        All name validation happens here, when the run starts; no runtime
        name check remains. Entries are validated against the loss-visible
        names: the corrector's modified names minus the prescribed
        prognostic names.
        """
        if corrector is None:
            raise ValueError(
                "corrector_loss is configured but the step has no corrector, "
                "so there are no correction deltas to consume."
            )
        modified_names = corrector.modified_names
        if modified_names is None:
            raise RuntimeError(
                "corrector_loss requires the corrector's modified names, but "
                "modified-name discovery (discover_modified_names) never "
                "ran; this is a programming error in the step type, not a "
                "config error."
            )
        if len(modified_names) == 0:
            raise ValueError(
                "corrector_loss is configured but the corrector modifies no "
                "variables, so there are no correction deltas to consume."
            )
        loss_visible_names = modified_names - frozenset(prescribed_prognostic_names)
        keep_gradient_names = corrector.keep_gradient_names

        def _validated_matches(
            selection: NameAndPrefixSelection, feature: str
        ) -> list[str]:
            unmatched = selection.unmatched_entries(loss_visible_names)
            if unmatched:
                raise ValueError(
                    f"{feature} entries {unmatched} match no loss-visible "
                    "corrector-modified variable (corrector-modified names "
                    "minus prescribed prognostic names: "
                    f"{sorted(loss_visible_names)})."
                )
            keep_gradient_matched = selection.matched(keep_gradient_names)
            if keep_gradient_matched:
                raise ValueError(
                    f"{feature} selects {keep_gradient_matched}, which are "
                    "corrected via straight-through (keep-gradient) clamps; "
                    "their correction deltas are detached, so corrector_loss "
                    "features cannot act on them."
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
        regularization_weight = 1.0
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
            regularization_weight = self.regularization.weight
        return StepLossCorrectorArgs(
            precorrector_names=precorrector_names,
            regularizer=regularizer,
            regularization_weight=regularization_weight,
        )
