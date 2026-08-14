import dataclasses
from collections.abc import Collection

import torch

from fme.core.gridded_ops import GriddedOperations
from fme.core.loss import (
    ChannelLossInfo,
    LossConfig,
    LossOutput,
    StepLoss,
    WeightedMappingLoss,
)
from fme.core.name_and_prefix_matcher import NameAndPrefixSelection
from fme.core.normalizer import StandardNormalizer
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class PreCorrectorOptimizationConfig:
    """Selects corrector-modified variables whose main-loss prediction is the
    pre-corrector network output ``prediction - delta``.

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
    ) -> "CorrectorLoss":
        """Validate the configured selections and build the corrector loss.

        All name validation happens here, when the run starts; no runtime name
        check remains. Entries are validated against the loss-visible names:
        the corrector's modified names minus the prescribed prognostic names.

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
                # Name the reason for each unmatched entry rather than
                # restating the set arithmetic: an entry that only reaches a
                # prescribed prognostic is a different mistake from an entry
                # that reaches nothing the corrector touches.
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
        return CorrectorLoss(
            precorrector_names=precorrector_names,
            regularizer=regularizer,
            regularization_weight=regularization_weight,
        )


class CorrectorLoss(torch.nn.Module):
    """The corrector-delta half of the training loss.

    Owns both features that consume the correction deltas carried on a
    ``StepOutput``: replacing selected predictions with their pre-corrector
    network outputs, and penalizing selected deltas toward zero.
    """

    def __init__(
        self,
        precorrector_names: list[str] | None,
        regularizer: WeightedMappingLoss | None,
        regularization_weight: float,
    ):
        """
        Args:
            precorrector_names: Names whose main-loss prediction is the
                pre-corrector network output, or None when the feature is off.
            regularizer: The penalty over the selected deltas, or None when
                the feature is off.
            regularization_weight: The weight applied to the penalty.
        """
        super().__init__()
        self._precorrector_names = precorrector_names
        self._regularizer = regularizer
        self._regularization_weight = regularization_weight

    @property
    def regularization_weight(self) -> float:
        return self._regularization_weight

    def pre_corrector_outputs(
        self, predict_dict: TensorMapping, deltas: TensorMapping
    ) -> TensorDict:
        """Replace the selected predictions with their pre-corrector outputs.

        Returns ``predict_dict[k] - deltas[k]`` for the selected names and
        ``predict_dict[k]`` for every other name. A selected name missing from
        a non-empty delta dict raises: an active corrector must produce deltas
        for every selected name.
        """
        net_output = dict(predict_dict)
        if self._precorrector_names is None or len(deltas) == 0:
            return net_output
        for name in self._precorrector_names:
            _require_delta(deltas, name, "precorrector_optimization")
            net_output[name] = predict_dict[name] - deltas[name]
        return net_output

    def regularization(self, deltas: TensorMapping) -> LossOutput | None:
        """The per-channel penalty over the selected deltas.

        Returns None when the feature is off or the delta dict is empty. The
        deltas are compared against zeros in loss-normalized space, so with an
        affine normalizer the means cancel and this penalizes ``delta / std``.
        Masked (NaN-filled) delta points are dropped by copying the delta's NaN
        pattern onto the zeros target, which triggers the NaN-target zeroing
        already in ``WeightedMappingLoss``.
        """
        if self._regularizer is None or len(deltas) == 0:
            return None
        selected: TensorDict = {}
        targets: TensorDict = {}
        for name in self._regularizer.packer.names:
            _require_delta(deltas, name, "regularization")
            delta = deltas[name]
            selected[name] = delta
            targets[name] = torch.where(
                delta.isnan(),
                torch.full_like(delta, torch.nan),
                torch.zeros_like(delta),
            )
        return self._regularizer(selected, targets)


def _require_delta(deltas: TensorMapping, name: str, feature: str) -> None:
    if name not in deltas:
        raise ValueError(
            f"{feature} selects {name!r}, but the corrector produced no delta "
            f"for it; it produced deltas for {sorted(deltas)}. An active "
            "corrector must produce deltas for every selected name."
        )


@dataclasses.dataclass
class StepOutputLossOutput:
    """The loss of one step, main term plus the corrector penalty.

    Parameters:
        main: The main ``StepLoss`` output.
        corrector_regularization: The penalty's own per-channel ``LossOutput``,
            or None when there is no penalty.
        corrector_regularization_weight: The weight applied to the penalty in
            ``total()``. Per-channel penalties are reported unweighted.
    """

    main: LossOutput
    corrector_regularization: LossOutput | None = None
    corrector_regularization_weight: float = 1.0

    def total(self) -> torch.Tensor:
        """``main.total() + weight * corrector_regularization.total()``."""
        total = self.main.total()
        if self.corrector_regularization is not None:
            total = (
                total
                + self.corrector_regularization_weight
                * self.corrector_regularization.total()
            )
        return total

    def get_channel_losses(self) -> dict[str, ChannelLossInfo]:
        """Per-channel main-loss values; the penalty is reported separately."""
        return self.main.get_channel_losses()

    def get_corrector_channel_losses(self) -> dict[str, ChannelLossInfo]:
        """Unweighted per-channel penalties, empty when there is no penalty."""
        if self.corrector_regularization is None:
            return {}
        return self.corrector_regularization.get_channel_losses()


class StepOutputLoss(torch.nn.Module):
    """``StepLoss`` plus the corrector-delta terms of a ``StepOutput``.

    Lives here rather than in ``fme/core/loss.py`` so that module keeps
    importing nothing from ``fme/core/corrector/``.
    """

    def __init__(self, step_loss: StepLoss, corrector_loss: CorrectorLoss | None):
        super().__init__()
        self.step_loss = step_loss
        self.corrector_loss = corrector_loss

    def forward(
        self,
        predict_dict: TensorMapping,
        target_dict: TensorMapping,
        step: int,
        data_mask: TensorMapping | None = None,
        deltas: TensorMapping | None = None,
    ) -> StepOutputLossOutput:
        """
        Args:
            predict_dict: The predicted (corrected) data.
            target_dict: The target data.
            step: The step number, indexed from 0 for the first step.
            data_mask: Optional per-variable boolean masks forwarded to the
                main loss.
            deltas: The corrector's per-variable correction deltas, empty or
                None when the corrector was inactive.
        """
        if self.corrector_loss is None or deltas is None or len(deltas) == 0:
            # Inert path: exactly today's StepLoss result. An epoch-disabled
            # corrector lands here.
            return StepOutputLossOutput(
                main=self.step_loss(predict_dict, target_dict, step, data_mask)
            )
        # Pre-corrector outputs first, so StepLoss never sees a delta.
        net_output = self.corrector_loss.pre_corrector_outputs(predict_dict, deltas)
        main = self.step_loss(net_output, target_dict, step, data_mask)
        # The penalty comes from the original deltas, not the pre-corrector
        # outputs the main loss saw.
        return StepOutputLossOutput(
            main=main,
            corrector_regularization=self.corrector_loss.regularization(deltas),
            corrector_regularization_weight=self.corrector_loss.regularization_weight,
        )
