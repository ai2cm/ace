"""Quick NaN/inf checks for debugging forward passes.

Enable with FME_LOG_NAN_DIAGNOSTICS=1.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from fme.core.typing_ import TensorMapping

_log = logging.getLogger(__name__)

_stepper_context: str = ""


def enabled() -> bool:
    return os.environ.get("FME_LOG_NAN_DIAGNOSTICS", "").lower() in ("1", "true", "yes")


def set_stepper_context(context: str) -> None:
    global _stepper_context
    _stepper_context = context


def clear_stepper_context() -> None:
    global _stepper_context
    _stepper_context = ""


def stepper_context() -> str:
    return _stepper_context or "stepper"


def info(message: str, *args: Any) -> None:
    if enabled():
        _log.warning("nan_diag %s", message % args if args else message)


def check(label: str, data: TensorMapping | torch.Tensor) -> None:
    """Log a warning for any non-finite values in a tensor or tensor mapping."""
    if not enabled():
        return
    if isinstance(data, torch.Tensor):
        _check_tensor(label, data)
        return
    for key in sorted(data.keys()):
        value = data[key]
        if isinstance(value, torch.Tensor):
            _check_tensor(f"{label}/{key}", value)


def summarize_tensor(label: str, tensor: torch.Tensor) -> None:
    """Log stats for a tensor even when finite (for targeted step-0 debugging)."""
    if not enabled():
        return
    flat = tensor.detach().float().reshape(-1)
    n = flat.numel()
    n_nan = int(torch.isnan(flat).sum().item())
    n_inf = int(torch.isinf(flat).sum().item())
    finite = flat[torch.isfinite(flat)]
    if finite.numel():
        stats = (
            f"min={finite.min().item():.4g} max={finite.max().item():.4g} "
            f"mean={finite.mean().item():.4g}"
        )
    else:
        stats = "all_non_finite"
    _log.warning(
        "nan_diag %s: nan=%d/%d inf=%d shape=%s %s",
        label,
        n_nan,
        n,
        n_inf,
        tuple(tensor.shape),
        stats,
    )


def summarize_mapping(
    label: str, data: TensorMapping, keys: list[str] | None = None
) -> None:
    if not enabled():
        return
    names = keys if keys is not None else sorted(data.keys())
    for key in names:
        if key not in data:
            info("%s/%s: MISSING", label, key)
            continue
        value = data[key]
        if isinstance(value, torch.Tensor):
            summarize_tensor(f"{label}/{key}", value)


def log_coupled_ocean_step0(
    *,
    ocean_input_only_names: list[str],
    ocean_prognostic_names: list[str],
    ocean_next_step_forcing_names: list[str],
    atmosphere_to_ocean_forcing_names: list[str],
    atmosphere_output_names: list[str],
    ocean_forcing_exogenous_names: list[str],
    shared_forcing_exogenous_names: list[str],
    exogenous_from_ocean_zarr: TensorMapping,
    from_atmos_gen: TensorMapping,
    from_atmos_forcings_shared: TensorMapping,
    before_nan_pad: TensorMapping,
    final_ocean_forcings: TensorMapping,
    ocean_ic: TensorMapping,
    ocean_zarr_window: TensorMapping,
    atmos_gen_keys: list[str],
) -> None:
    """Targeted audit for coupled ocean outer step 0 vs ocean-only training."""
    if not enabled():
        return

    info("=== coupled ocean step 0 audit (compare to ocean-only pretrain) ===")
    info(
        "coupling: atmosphere_to_ocean=%s shared_exogenous=%s "
        "ocean_exogenous_only=%s ocean_next_step=%s",
        sorted(atmosphere_to_ocean_forcing_names),
        sorted(shared_forcing_exogenous_names),
        sorted(
            set(ocean_forcing_exogenous_names).difference(
                shared_forcing_exogenous_names
            )
        ),
        sorted(ocean_next_step_forcing_names),
    )

    wind_stress = {
        "eastward_surface_wind_stress",
        "northward_surface_wind_stress",
    }
    atmos_stress = {
        "eastward_surface_stress",
        "northward_surface_stress",
    }
    missing_coupled = wind_stress - set(atmosphere_to_ocean_forcing_names)
    if missing_coupled:
        info(
            "coupling: wind stress NOT from atmosphere (ocean zarr replay instead): "
            "missing=%s atmosphere_outputs=%s",
            sorted(missing_coupled),
            sorted(atmos_stress & set(atmosphere_output_names)),
        )

    expected_forcings = set(ocean_input_only_names)
    got_forcings = set(final_ocean_forcings.keys())
    missing_forcings = sorted(expected_forcings - got_forcings)
    extra_forcings = sorted(got_forcings - expected_forcings)
    if missing_forcings:
        info("ocean step0: MISSING forcing keys vs input_only: %s", missing_forcings)
    if extra_forcings:
        info("ocean step0: EXTRA forcing keys vs input_only: %s", extra_forcings)

    for key in sorted(ocean_input_only_names):
        if key in exogenous_from_ocean_zarr:
            source = "ocean_zarr"
        elif key in from_atmos_gen:
            source = "atmos_gen_avg"
        elif key in from_atmos_forcings_shared:
            source = "atmos_forcing_avg"
        else:
            source = "UNKNOWN"
        info("ocean step0 forcing source %s -> %s", key, source)

    summarize_mapping("ocean step0/ic", ocean_ic, list(ocean_prognostic_names))

    step0_forcing_keys = sorted(
        set(ocean_input_only_names)
        | set(atmosphere_to_ocean_forcing_names)
        | set(shared_forcing_exogenous_names)
    )
    summarize_mapping(
        "ocean step0/forcings_final", final_ocean_forcings, step0_forcing_keys
    )

    time_dim = 1
    for key in sorted(ocean_next_step_forcing_names):
        if key not in final_ocean_forcings:
            info("ocean step0 next_step %s: absent from final forcings", key)
            continue
        tensor = final_ocean_forcings[key]
        if tensor.shape[time_dim] < 2:
            info(
                "ocean step0 next_step %s: time dim %d < 2 shape=%s",
                key,
                tensor.shape[time_dim],
                tuple(tensor.shape),
            )
            continue
        summarize_tensor(
            f"ocean step0/next_step {key} t=0 (step input)",
            tensor.select(time_dim, 0),
        )
        summarize_tensor(
            f"ocean step0/next_step {key} t=1 (next-step input at fwd step 0)",
            tensor.select(time_dim, 1),
        )

    # Compare shared forcings: ocean-only would use ocean zarr; coupled uses ERA5 avg.
    for key in sorted(shared_forcing_exogenous_names):
        if key not in ocean_zarr_window or key not in final_ocean_forcings:
            continue
        zarr_t = ocean_zarr_window[key]
        coupled_t = final_ocean_forcings[key]
        # ocean zarr window has IC+1 times; compare time index used at ocean fwd step 0
        zarr_slice = zarr_t.select(time_dim, 0)
        if coupled_t.shape[time_dim] >= 2 and key in ocean_next_step_forcing_names:
            coupled_slice = coupled_t.select(time_dim, 1)
            slice_label = "t=1 next_step"
        else:
            coupled_slice = coupled_t.select(time_dim, 0)
            slice_label = "t=0"
        if zarr_slice.shape != coupled_slice.shape:
            info(
                "ocean step0 compare %s: shape mismatch zarr=%s coupled_%s=%s",
                key,
                tuple(zarr_slice.shape),
                slice_label,
                tuple(coupled_slice.shape),
            )
            continue
        diff = (zarr_slice.detach().float() - coupled_slice.detach().float()).abs()
        finite = diff[torch.isfinite(diff)]
        if finite.numel():
            info(
                "ocean step0 compare %s ocean_zarr vs coupled_%s: "
                "max_abs_diff=%.4g mean_abs_diff=%.4g",
                key,
                slice_label,
                finite.max().item(),
                finite.mean().item(),
            )
        check(f"ocean step0/compare {key} zarr", zarr_slice)
        check(f"ocean step0/compare {key} coupled", coupled_slice)

    for key in sorted(from_atmos_gen.keys()):
        if key not in atmos_gen_keys:
            info(
                "ocean step0 atmos_gen %s: NOT in atmos_gen keys %s",
                key,
                atmos_gen_keys,
            )

    summarize_mapping("ocean step0/before_nan_pad", before_nan_pad, step0_forcing_keys)
    check("ocean step0/before_nan_pad", before_nan_pad)
    check("ocean step0/forcings_final", final_ocean_forcings)
    check("ocean step0/ic", ocean_ic)


def _check_tensor(label: str, tensor: torch.Tensor) -> None:
    flat = tensor.detach().reshape(-1)
    n_nan = int(torch.isnan(flat).sum().item())
    n_inf = int(torch.isinf(flat).sum().item())
    if n_nan == 0 and n_inf == 0:
        return
    finite = flat[torch.isfinite(flat)]
    if finite.numel():
        stats = (
            f" finite_min={finite.min().item():.4g}"
            f" finite_max={finite.max().item():.4g}"
            f" finite_mean={finite.mean().item():.4g}"
        )
    else:
        stats = " (all non-finite)"
    _log.warning(
        "nan_diag %s: nan=%d/%d inf=%d shape=%s%s",
        label,
        n_nan,
        flat.numel(),
        n_inf,
        tuple(tensor.shape),
        stats,
    )
