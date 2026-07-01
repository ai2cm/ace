import dataclasses
import datetime

import torch

from fme.core.corrector.registry import (
    Correction,
    CorrectionSequence,
    CorrectorConfigABC,
)
from fme.core.corrector.state import CorrectorState
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import GriddedOperations
from fme.core.registry.corrector import CorrectorSelector
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class IceBudgetCorrectionConfig:
    """
    Reconstruct prognostic sea ice concentration, ice mass,
    and snow mass from predicted area and mass budget terms.

    Corrected variables in the config must be ordered as:
    {'variable': ['source_term', 'sink_term', 'transport_term']}.
    For example: {'siconc': ['LSRCc', 'LSNKc', 'XPRTc']}.
    """

    corrected_variables: dict[str, list[str]] | None = None

    def constrain_budgets(
        self,
        old_mass: torch.Tensor,
        source: torch.Tensor,
        sink: torch.Tensor,
        transport: torch.Tensor,
        timestep: float,
        area_mode: bool = False,
        ice_mask: torch.Tensor | None = None,
    ):
        """
        Adjust budget terms so that new_mass = old_mass + source +
        sink + transport. If the new mass is less than zero, then add
        a positive correction across non-zero terms to make it zero.
        Similarly, if area_mode=True and the new mass is greater than 1,
        then remove 'area' across non-zero terms to make it 1. This code
        always respects that source is >= 0 and sink <= 0. Any violation
        is fixed by moving residual to the transport term.

        Args:
        old_mass (torch.Tensor): mass or concentration at time t
        source (torch.Tensor): source term
        sink (torch.Tensor): sink term
        transport (torch.Tensor): transport/export term
        timestep (float): the timestep of the data (default 6 hours)
        area_mode (bool): when True, cap maximum concentration to 1
        ice_mask (torch.Tensor | None): a mask indicating the presence of
                                        ice to ensure that when ice_mass
                                        == 0, siconc and snow also == 0

        Returns:
        Source, sink, transport terms adjusted to conserve mass.
        """
        dtype = old_mass.dtype
        s = source.clone() * timestep
        k = sink.clone() * timestep
        t = transport.clone() * timestep

        def _rebalance(s, k, t, mask, mass, sign=1):
            nz_s = s.abs() > 0
            nz_k = k.abs() > 0
            nz_t = t.abs() > 0
            n_active = nz_s.to(dtype) + nz_k.to(dtype) + nz_t.to(dtype)
            share = torch.where(
                mask & (n_active > 0),
                mass / n_active.clamp(min=1),
                torch.zeros_like(mass, dtype=dtype),
            )

            resid_s = torch.where(
                mask & nz_s, share, torch.zeros_like(share, dtype=dtype)
            )
            resid_k = torch.where(
                mask & nz_k, share, torch.zeros_like(share, dtype=dtype)
            )
            resid_t = torch.where(
                mask & nz_t, share, torch.zeros_like(share, dtype=dtype)
            )
            all_zero = mask & (n_active == 0)
            resid_t = torch.where(all_zero, mass, resid_t)

            tmp = k + sign * resid_k
            k_overshoot = torch.where(tmp > 0, tmp, torch.zeros_like(k, dtype=dtype))
            resid_k = resid_k - k_overshoot
            resid_t = resid_t + k_overshoot

            tmp = s + sign * resid_s
            s_overshoot = torch.where(tmp < 0, tmp, torch.zeros_like(s, dtype=dtype))
            resid_s = resid_s - sign * s_overshoot
            resid_t = resid_t + sign * s_overshoot

            s = s + sign * resid_s
            k = k + sign * resid_k
            t = t + sign * resid_t

            return s, k, t

        new_mass = old_mass + (s + k + t)
        neg_mask = new_mass < 0
        if torch.any(neg_mask):
            deficit = torch.where(
                neg_mask, -new_mass, torch.zeros_like(new_mass, dtype=dtype)
            )
            s, k, t = _rebalance(s, k, t, neg_mask, deficit, sign=1)

        if area_mode:
            new_mass = old_mass + (s + k + t)
            high_mask = new_mass > 1
            if torch.any(high_mask):
                excess = torch.where(
                    high_mask, new_mass - 1, torch.zeros_like(new_mass, dtype=dtype)
                )
                s, k, t = _rebalance(s, k, t, high_mask, excess, sign=-1)

        if ice_mask is not None:
            new_mass = old_mass + (s + k + t)
            high_mask = (ice_mask == 0) & (new_mass > 0)
            if torch.any(high_mask):
                excess = torch.where(
                    high_mask, new_mass, torch.zeros_like(new_mass, dtype=dtype)
                )
                s, k, t = _rebalance(s, k, t, high_mask, excess, sign=-1)

        return s / timestep, k / timestep, t / timestep

    def _processing_order(self) -> list[str]:
        """Prognostic variables this correction reconstructs, in write order."""
        if self.corrected_variables is None:
            return []
        sic_vars = {"siconc", "sea_ice_fraction", "ocean_sea_ice_fraction"}
        processing_order: list[str] = []
        if "simass" in self.corrected_variables:
            processing_order.append("simass")
        for var in sic_vars:
            if var in self.corrected_variables:
                processing_order.append(var)
        if "sisnmass" in self.corrected_variables:
            processing_order.append("sisnmass")
        return processing_order

    def __call__(
        self, gen_data: TensorMapping, input_data: TensorMapping, timestep: float
    ) -> TensorDict:
        """
        Returns:
            A ``TensorDict`` containing only the fields modified by this
            correction (each reconstructed prognostic variable and its three
            budget terms); empty when ``corrected_variables`` is None.
        """
        if self.corrected_variables is None:
            return {}

        x_in = {key: value.double() for key, value in input_data.items()}
        # A full-precision working copy is needed for the read-after-write
        # dependencies below (the ice mask reads a prognostic written in an
        # earlier iteration); only the fields actually written are returned.
        work = {key: value.double() for key, value in gen_data.items()}
        modified: TensorDict = {}

        sic_vars = {"siconc", "sea_ice_fraction", "ocean_sea_ice_fraction"}
        mask_var = None
        if "simass" in self.corrected_variables:
            mask_var = "simass"
        else:
            sic_in_corrected = sic_vars.intersection(self.corrected_variables.keys())
            if sic_in_corrected:
                mask_var = next(iter(sic_in_corrected))

        processing_order = self._processing_order()

        for key in processing_order:
            area_mode = key in sic_vars
            ice_mask = None
            if key != processing_order[0] and mask_var is not None:
                ice_mask = work[mask_var]

            terms = self.corrected_variables[key]
            budgets = self.constrain_budgets(
                x_in[key],
                work[terms[0]],
                work[terms[1]],
                work[terms[2]],
                area_mode=area_mode,
                timestep=timestep,
                ice_mask=ice_mask,
            )
            work[terms[0]] = budgets[0]
            work[terms[1]] = budgets[1]
            work[terms[2]] = budgets[2]
            work[key] = x_in[key] + timestep * (budgets[0] + budgets[1] + budgets[2])
            # record each field as it is written -- the writes determine the
            # modified set.
            for name in (terms[0], terms[1], terms[2], key):
                modified[name] = work[name]

        return {key: value.float() for key, value in modified.items()}


@dataclasses.dataclass
class IceBudgetCorrection:
    """Correction that reconstructs ice/snow state from budget terms.

    Wraps ``IceBudgetCorrectionConfig`` so the corrector applies the operation
    without reading config fields. ``forcing_data`` and ``corrector_state`` are
    unused and passed through.
    """

    config: IceBudgetCorrectionConfig
    timestep_seconds: float

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        """
        Returns:
            A tuple whose ``TensorDict`` contains only the fields modified by
            this correction (each reconstructed prognostic variable and its
            budget terms); empty when no variables are corrected.
            ``IceBudgetCorrectionConfig.__call__`` already returns only those
            fields.
        """
        corrected = self.config(gen_data, input_data, self.timestep_seconds)
        return corrected, corrector_state


@CorrectorSelector.register("ice_corrector")
@dataclasses.dataclass
class IceCorrectorConfig(CorrectorConfigABC):
    budget_correction: IceBudgetCorrectionConfig | None = None

    def _get_corrector(
        self,
        dataset_info: DatasetInfo,
    ) -> "IceCorrector":
        return self._build(
            dataset_info.gridded_operations,
            dataset_info.timestep,
        )

    def _build(
        self,
        gridded_operations: GriddedOperations,
        timestep: datetime.timedelta,
    ) -> "IceCorrector":
        corrections: list[Correction] = []
        if self.budget_correction is not None:
            corrections.append(
                IceBudgetCorrection(self.budget_correction, timestep.total_seconds())
            )
        return IceCorrector(corrections)


class IceCorrector(CorrectionSequence):
    """
    Implement choice of sea ice corrector.
    """
