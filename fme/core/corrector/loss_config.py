import dataclasses
from collections.abc import Collection
from typing import Literal

from fme.core.loss import (
    CorrectorLoss,
    CorrectorRegularizer,
    WeightedMappingLoss,
    _L1Loss,
    _MSELoss,
    require_matched_entries,
)
from fme.core.name_and_prefix_matcher import NameAndPrefixSelection
from fme.core.normalizer import StandardNormalizer

_LOSS_NAMES_SUBJECT = "variables the loss covers"


@dataclasses.dataclass
class PreCorrectorOptimizationConfig:
    """Selects corrector-modified variables whose main-loss prediction is the
    pre-corrector network output ``prediction - delta``.

    Selection is pure opt-in; this feature may be enabled together with
    corrector regularization.

    Parameters:
        names_and_prefixes: ``NameAndPrefixMatcher`` entries selecting the
            corrector-modified variables to optimize pre-corrector.
    """

    names_and_prefixes: list[str]

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
    (``sqrt_loss_step_decay_constant``); only ``weight`` scales it. Nor does it
    take the main loss's per-variable ``weights``: every selected channel enters
    the mean with weight 1.0. This feature may be enabled together with
    pre-corrector optimization.

    Parameters:
        names_and_prefixes: ``NameAndPrefixMatcher`` entries selecting the
            corrector-modified variables whose deltas are penalized.
        norm: Which norm of the normalized deltas is penalized. ``"L1"`` is the
            mean absolute delta. ``"L2"`` is the mean squared delta, the
            squared L2 norm: the square root is not taken, which keeps the
            gradient defined at a zero delta.
        weight: The positive weight applied to the penalty in the total loss.
    """

    names_and_prefixes: list[str]
    norm: Literal["L1", "L2"]
    weight: float = 1.0

    def __post_init__(self):
        if not self.names_and_prefixes:
            raise ValueError(
                "regularization requires names_and_prefixes: configuring "
                "the feature while selecting nothing is a contradiction, "
                "not a no-op."
            )
        if self.weight <= 0:
            raise ValueError(
                f"regularization weight must be positive, got {self.weight}"
            )

    def build(
        self,
        normalizer: StandardNormalizer,
        channel_dim: int,
    ) -> CorrectorRegularizer:
        """Build the penalty, deferring its channels to the first delta.

        Args:
            normalizer: The loss normalizer, used to normalize the deltas.
            channel_dim: The channel dimension of the loss inputs.
        """

        def build_loss(names: list[str]) -> WeightedMappingLoss:
            return WeightedMappingLoss(
                loss=_L1Loss() if self.norm == "L1" else _MSELoss(),
                weights={},
                out_names=names,
                normalizer=normalizer,
                channel_dim=channel_dim,
            )

        return CorrectorRegularizer(
            selection=NameAndPrefixSelection(tuple(self.names_and_prefixes)),
            build_loss=build_loss,
            weight=self.weight,
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
        loss_names: Collection[str],
        normalizer: StandardNormalizer,
        channel_dim: int,
    ) -> CorrectorLoss:
        """Check the configured entries against the names the loss covers, then
        build.

        ``loss_names`` is exactly the set the loss can pack and the loss
        normalizer covers, so an entry matching nothing in it can never be
        penalized under any corrector. What it cannot catch -- that the
        corrector modifies a name, and that the step does not then drop that
        name from the delta -- is why the entries are checked again, against
        the corrector's actual delta keys, the first time the corrector loss
        runs on a non-empty delta.

        Args:
            loss_names: The names the step loss covers.
            normalizer: The loss normalizer, used to normalize the deltas.
            channel_dim: The channel dimension of the loss inputs.
        """
        precorrector_selection = None
        if self.precorrector_optimization is not None:
            precorrector_selection = NameAndPrefixSelection(
                tuple(self.precorrector_optimization.names_and_prefixes)
            )
            require_matched_entries(
                precorrector_selection,
                loss_names,
                "precorrector_optimization",
                _LOSS_NAMES_SUBJECT,
            )
        regularizer = None
        if self.regularization is not None:
            require_matched_entries(
                NameAndPrefixSelection(tuple(self.regularization.names_and_prefixes)),
                loss_names,
                "regularization",
                _LOSS_NAMES_SUBJECT,
            )
            regularizer = self.regularization.build(normalizer, channel_dim)
        return CorrectorLoss(
            precorrector_selection=precorrector_selection,
            regularizer=regularizer,
        )
