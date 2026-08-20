import dataclasses
import logging
from typing import Literal

from fme.core.loss import (
    CorrectorLoss,
    CorrectorRegularizer,
    WeightedMappingLoss,
    _L1Loss,
    _MSELoss,
)
from fme.core.normalizer import StandardNormalizer


@dataclasses.dataclass
class CorrectorRegularizationConfig:
    """A penalty pushing every correction delta toward zero in
    loss-normalized space.

    The penalty is not subject to the main loss's per-step sqrt decay
    (``sqrt_loss_step_decay_constant``); only ``weight`` scales it. Nor does it
    take the main loss's per-variable ``weights``: every channel enters
    the mean with weight 1.0. This feature may be enabled together with
    pre-corrector optimization.

    Parameters:
        norm: Which norm of the normalized deltas is penalized. ``"L1"`` is the
            mean absolute delta. ``"L2"`` is the mean squared delta, the
            squared L2 norm: the square root is not taken, which keeps the
            gradient defined at a zero delta.
        weight: The positive weight applied to the penalty in the total loss.
    """

    norm: Literal["L1", "L2"]
    weight: float = 1.0

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError(
                f"regularization weight must be positive, got {self.weight}"
            )

    def build(
        self,
        normalizer: StandardNormalizer,
        channel_dim: int,
    ) -> CorrectorRegularizer:
        """Build the penalty. Its channels are whatever delta keys each step's
        corrector produces.

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
            build_loss=build_loss,
            weight=self.weight,
        )


@dataclasses.dataclass
class CorrectorLossConfig:
    """Training-only consumption of correction deltas by the step loss.

    Both features act on every delta the corrector produces at each step;
    neither selects names.

    Parameters:
        precorrector_optimization: Optimize the corrector-modified variables
            against their pre-corrector network outputs.
        regularization: Penalize every correction delta toward zero.
    """

    precorrector_optimization: bool = False
    regularization: CorrectorRegularizationConfig | None = None

    def __post_init__(self):
        if not self.precorrector_optimization and self.regularization is None:
            raise ValueError(
                "corrector_loss requires at least one of "
                "precorrector_optimization or regularization: configuring "
                "it while selecting no feature is a contradiction, not a "
                "no-op."
            )

    def build(
        self,
        normalizer: StandardNormalizer,
        channel_dim: int,
    ) -> CorrectorLoss:
        """Build the corrector loss, logging each enabled feature.

        Args:
            normalizer: The loss normalizer, used to normalize the deltas.
            channel_dim: The channel dimension of the loss inputs.
        """
        if self.precorrector_optimization:
            logging.info("corrector_loss: optimizing pre-corrector outputs")
        regularizer = None
        if self.regularization is not None:
            logging.info(
                "corrector_loss: penalizing correction deltas with norm "
                f"{self.regularization.norm} and weight {self.regularization.weight}"
            )
            regularizer = self.regularization.build(normalizer, channel_dim)
        return CorrectorLoss(
            precorrector_optimization=self.precorrector_optimization,
            regularizer=regularizer,
        )
