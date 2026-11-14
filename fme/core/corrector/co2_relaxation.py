import dataclasses
import datetime
import logging
import math
import pathlib

import torch

from fme.core.cloud import open_dataset_via_inter_filesystem_copy
from fme.core.corrector.state import CorrectorState
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorDict, TensorMapping

SECONDS_PER_DAY = 86400.0
CO2_NAME_ATTR = "co2_name"
CO2_REF_ATTR = "co2_ref"


def intercept_name(name: str) -> str:
    """Name of the intercept variable for ``name`` in a reference file."""
    return f"{name}_intercept"


def slope_name(name: str) -> str:
    """Name of the slope variable for ``name`` in a reference file."""
    return f"{name}_slope"


@dataclasses.dataclass
class _Reference:
    intercept: dict[str, torch.Tensor]
    slope: dict[str, torch.Tensor]
    co2_ref: float


def _read_reference(
    path: str | pathlib.Path, names: list[str], co2_name: str
) -> _Reference:
    ds = open_dataset_via_inter_filesystem_copy(path)
    if CO2_REF_ATTR not in ds.attrs:
        raise ValueError(
            f"CO2 regression reference file {path} is missing the "
            f"'{CO2_REF_ATTR}' attribute."
        )
    file_co2_name = ds.attrs.get(CO2_NAME_ATTR)
    if file_co2_name is not None and file_co2_name != co2_name:
        logging.warning(
            f"CO2 regression reference file {path} was fit against CO2 variable "
            f"'{file_co2_name}', but the corrector reads CO2 from '{co2_name}'."
        )
    intercept = {}
    slope = {}
    for name in names:
        for var_name in (intercept_name(name), slope_name(name)):
            if var_name not in ds.variables:
                raise ValueError(
                    f"Variable '{var_name}' not found in CO2 regression "
                    f"reference file {path}."
                )
        intercept[name] = torch.as_tensor(
            ds[intercept_name(name)].values, dtype=torch.float32
        )
        slope[name] = torch.as_tensor(ds[slope_name(name)].values, dtype=torch.float32)
        if intercept[name].shape != slope[name].shape:
            raise ValueError(
                f"Intercept and slope for '{name}' in {path} have different "
                f"shapes: {intercept[name].shape} and {slope[name].shape}."
            )
        for label, values in (("intercept", intercept), ("slope", slope)):
            if not torch.all(torch.isfinite(values[name])):
                raise ValueError(
                    f"The {label} for '{name}' in {path} has non-finite values, "
                    "which the relaxation would propagate into the field."
                )
    co2_ref = float(ds.attrs[CO2_REF_ATTR])
    ds.close()
    return _Reference(intercept=intercept, slope=slope, co2_ref=co2_ref)


@dataclasses.dataclass
class CO2RegressionRelaxationConfig:
    r"""Configuration for Newtonian relaxation of generated fields toward a
    per-gridpoint linear function of ln(CO2).

    At each step, each named field ``x`` is relaxed toward the target

    .. math::

        target = intercept + slope \cdot \ln(co2 / co2\_ref)

    using the exact exponential solution of :math:`dx/dt = -(x - target) / \tau`
    over one timestep :math:`\Delta t`:

    .. math::

        x \leftarrow target + (x - target) \exp(-\Delta t / \tau)

    ``intercept`` and ``slope`` are single (lat, lon) maps per field, e.g. an
    annual-mean fit over a reference period produced by
    ``scripts/data_process/compute_co2_regression.py``. CO2 is read pointwise
    from the corrector's forcing data, so it may be spatially uniform or not.

    The relaxation is only applied when the corrector is in evaluation mode
    (validation, inline inference and standalone inference), never to
    train-mode steps.

    The reference is read from ``path``. When the configuration is serialized
    into a checkpoint it is first loaded (see ``load``), so the checkpoint
    carries the reference values and no longer depends on the file.

    Parameters:
        timescale_days: Mapping from the name of each relaxed field to its
            relaxation timescale in days. Relaxed fields must be generated
            (prognostic) outputs.
        co2_name: Name of the CO2 forcing variable.
        path: Path to a netCDF file with ``{name}_intercept`` and
            ``{name}_slope`` variables for every relaxed field (dimensions
            matching the horizontal shape of the field) and a ``co2_ref``
            attribute. Set to None once the reference is loaded.
        intercept: Loaded intercept maps, by field name. Not specified by
            users; filled by ``load``.
        slope: Loaded slope maps, by field name. Not specified by users;
            filled by ``load``.
        co2_ref: Reference CO2 concentration of the fit. Not specified by users;
            filled by ``load``.
    """

    timescale_days: dict[str, float]
    co2_name: str
    path: str | None = None
    intercept: dict[str, torch.Tensor] | None = dataclasses.field(
        default=None, compare=False
    )
    slope: dict[str, torch.Tensor] | None = dataclasses.field(
        default=None, compare=False
    )
    co2_ref: float | None = None

    def __post_init__(self):
        if len(self.timescale_days) == 0:
            raise ValueError("timescale_days must name at least one field.")
        for name, timescale in self.timescale_days.items():
            if timescale <= 0:
                raise ValueError(
                    f"timescale_days must be positive, got {timescale} for '{name}'."
                )
        loaded = (self.intercept, self.slope, self.co2_ref)
        is_loaded = all(value is not None for value in loaded)
        if any(value is not None for value in loaded) and not is_loaded:
            raise ValueError(
                "intercept, slope and co2_ref must be given together; they are "
                "set by loading the reference from path."
            )
        if (self.path is None) == (not is_loaded):
            raise ValueError(
                "Exactly one of path or a loaded reference (intercept, slope "
                "and co2_ref) must be given."
            )
        if self.intercept is not None and self.slope is not None:
            names = set(self.timescale_days)
            for label, values in (("intercept", self.intercept), ("slope", self.slope)):
                if set(values) != names:
                    raise ValueError(
                        f"Loaded {label} names {sorted(values)} do not match "
                        f"timescale_days names {sorted(names)}."
                    )

    def load(self):
        """Read the reference from ``path`` into the configuration, so it no
        longer requires the external file.
        """
        if self.path is not None:
            reference = self._read()
            self.intercept = reference.intercept
            self.slope = reference.slope
            self.co2_ref = reference.co2_ref
            self.path = None

    def _read(self) -> _Reference:
        if self.path is not None:
            return _read_reference(self.path, list(self.timescale_days), self.co2_name)
        assert self.intercept is not None
        assert self.slope is not None
        assert self.co2_ref is not None
        return _Reference(
            intercept=self.intercept, slope=self.slope, co2_ref=self.co2_ref
        )

    def build(self, timestep: datetime.timedelta) -> "CO2RegressionRelaxation":
        reference = self._read()
        return CO2RegressionRelaxation(
            timescale_seconds={
                name: days * SECONDS_PER_DAY
                for name, days in self.timescale_days.items()
            },
            co2_name=self.co2_name,
            co2_ref=reference.co2_ref,
            intercept=reference.intercept,
            slope=reference.slope,
            timestep_seconds=timestep.total_seconds(),
        )


class CO2RegressionRelaxation:
    """Correction relaxing generated fields toward a linear function of
    ln(CO2). See ``CO2RegressionRelaxationConfig``.

    Skipped (returns no modified fields) in training mode, which is the
    initial mode.
    """

    def __init__(
        self,
        timescale_seconds: dict[str, float],
        co2_name: str,
        co2_ref: float,
        intercept: dict[str, torch.Tensor],
        slope: dict[str, torch.Tensor],
        timestep_seconds: float,
    ):
        self._decay = {
            name: math.exp(-timestep_seconds / timescale)
            for name, timescale in timescale_seconds.items()
        }
        self._co2_name = co2_name
        self._co2_ref = co2_ref
        self._global_shape = {name: tuple(intercept[name].shape) for name in intercept}
        # Under spatial parallelism each rank holds a local chunk of the grid.
        dist = Distributed.get_instance()
        device = get_device()
        self._intercept = {
            name: value[dist.get_local_slices(value.shape)].to(device)
            for name, value in intercept.items()
        }
        self._slope = {
            name: value[dist.get_local_slices(value.shape)].to(device)
            for name, value in slope.items()
        }
        self._training = True

    def train(self, mode: bool = True):
        self._training = mode

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        """
        Returns:
            A tuple whose ``TensorDict`` contains only the relaxed fields, which
            is empty in training mode.
        """
        if self._training:
            return {}, corrector_state
        if self._co2_name not in forcing_data:
            raise ValueError(
                f"CO2 forcing variable '{self._co2_name}' for the CO2 regression "
                "relaxation is not available in the forcing data, which has "
                f"{sorted(forcing_data)}. It must be an input-only (forcing) "
                "variable of the model."
            )
        log_co2_ratio = torch.log(forcing_data[self._co2_name] / self._co2_ref)
        relaxed: TensorDict = {}
        for name, decay in self._decay.items():
            if name not in gen_data or name not in input_data:
                raise ValueError(
                    f"CO2 regression relaxation field '{name}' must be a "
                    "generated prognostic variable."
                )
            x = gen_data[name]
            intercept = self._intercept[name]
            if x.shape[-2:] != intercept.shape:
                raise ValueError(
                    f"CO2 regression reference for '{name}' has horizontal shape "
                    f"{self._global_shape[name]} (local {tuple(intercept.shape)}), "
                    f"which does not match the field's local horizontal shape "
                    f"{tuple(x.shape[-2:])}."
                )
            intercept = intercept.to(device=x.device, dtype=x.dtype)
            slope = self._slope[name].to(device=x.device, dtype=x.dtype)
            target = intercept + slope * log_co2_ratio.to(dtype=x.dtype)
            relaxed[name] = target + (x - target) * decay
        return relaxed, corrector_state
