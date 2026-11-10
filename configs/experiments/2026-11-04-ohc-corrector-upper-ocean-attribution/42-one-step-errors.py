"""Issue 42 (i): one-step T and S errors of a `ppad_fmean_nohc` parent.

One step of the network from each target state t_n, OHC and salt correctors off,
reduced on the device to e_T,k,n, e_S,k,n, E_H,n, E_S,n, r_H,n, r_S,n (definitions:
.scratch/2026-11-04-1337-ohc-corrector-upper-ocean-attribution/issues/42
§ Definitions, § Approach). No maps are saved.

    python -I 42-one-step-errors.py --checkpoint /ckpt.tar \
        --out /results/42-one-step-errors.nc --start-time 0241-01-01T00:00:00 \
        [--stop-time ...] [--data-path /climate-default]
"""

import argparse
import dataclasses
import logging
import subprocess
from collections.abc import Mapping
from typing import Any

import dacite
import numpy as np
import torch
import xarray as xr

from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.data_loading.getters import get_gridded_data
from fme.ace.stepper.single_module import Stepper, load_stepper
from fme.core.constants import (
    DENSITY_OF_SEA_WATER_CM4,
    REFERENCE_SALINITY,
    SPECIFIC_HEAT_OF_SEA_WATER_CM4,
    SPHERE_AREA_M2,
)
from fme.core.coordinates import DepthCoordinate
from fme.core.corrector.ocean import (
    OceanHeatContentCorrection,
    OceanSaltContentCorrection,
    SeaSurfaceHeightSaltBudget,
)
from fme.core.corrector.registry import CorrectionSequence, EpochScheduledCorrector
from fme.core.device import get_device
from fme.core.gridded_ops import GriddedOperations
from fme.core.ocean_data import OceanData

RHO_CP = DENSITY_OF_SEA_WATER_CM4 * SPECIFIC_HEAT_OF_SEA_WATER_CM4  # J m-3 K-1
RHO = DENSITY_OF_SEA_WATER_CM4  # kg m-3
F64 = torch.float64

STORES = (
    "2026-07-15-om4-picontrol-1deg-coupled-ocean.zarr",
    "2026-07-15-om4-picontrol-1deg-ocean-5daily.zarr",
)

PER_SAMPLE = {  # name: (units, long_name)
    "E_H": ("J", "one-step heat-content error, sum_k h_T,k e_T,k"),
    "E_S": ("g", "one-step salt-content error, sum_k h_S,k e_S,k"),
    "r_H": (
        "J",
        "heat-budget residual the OHC corrector would remove, "
        "A_norm (G_target - G_gen)",
    ),
    "r_S": ("g", "salt-budget residual, rho (Salt_in + dSalt_SSH - Salt_gen)"),
    "r_S_fixed": (
        "g",
        "salt-budget residual with budget_config None, rho (Salt_in - Salt_gen)",
    ),
    "E_H_direct": (
        "J",
        "A_norm (G_gen - <s H(theta_target)>_a), closure check of E_H",
    ),
    "E_S_direct": ("g", "rho (Salt_gen - Salt_target), closure check of E_S"),
    "G_in": ("J m-2", "<s H(theta_n)>_a, ocean_heat_content mask"),
    "G_gen": ("J m-2", "<s H(theta_hat)>_a"),
    "G_target": ("J m-2", "G_in + <hfds_total_area_gen + s hfgeou>_a dt"),
    "G_tgt_state": ("J m-2", "<s H(theta at t_n + dt)>_a"),
    "Salt_in": ("psu m3", "global_salt_content of S at t_n"),
    "Salt_gen": ("psu m3", "global_salt_content of S_hat"),
    "Salt_target": ("psu m3", "global_salt_content of S at t_n + dt"),
    "dSalt_SSH": (
        "psu m3",
        "SeaSurfaceHeightSaltBudget, SSH_in from the target at t_n",
    ),
}
PER_LEVEL_SAMPLE = {
    "e_T": ("K", "one-step thetao error, area s dz weighted level mean"),
    "e_S": ("g kg-1", "one-step so error, area s dz weighted level mean"),
    "thetao_target_mean": (
        "degC",
        "target thetao at t_n + dt, area s dz weighted level mean",
    ),
    "so_target_mean": (
        "g kg-1",
        "target so at t_n + dt, area s dz weighted level mean",
    ),
}


@dataclasses.dataclass
class Reducer:
    gridded_ops: GriddedOperations
    vertical_coordinate: DepthCoordinate
    area_weights: torch.Tensor  # [lat, lon], sum 1 over the sphere
    timestep_seconds: float

    @property
    def n_levels(self) -> int:
        return self.vertical_coordinate.dz.shape[-1]

    def a_norm(self) -> torch.Tensor:
        ones = torch.ones_like(self.area_weights, dtype=F64)
        return SPHERE_AREA_M2 * self.gridded_ops.area_weighted_sum(
            ones, name="ocean_heat_content"
        )

    def level_weights(self, sea_surface_fraction: torch.Tensor) -> torch.Tensor:
        """W_jk = area_j s_j dz_jk [m3], shape [batch, lat, lon, level]."""
        area_m2 = self.area_weights.to(F64) * SPHERE_AREA_M2
        dz = torch.nan_to_num(self.vertical_coordinate.dz.to(F64))
        s = torch.nan_to_num(sea_surface_fraction.to(F64))
        return (area_m2 * s)[..., None] * dz

    def level_mean(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = torch.where(w > 0, torch.nan_to_num(x.to(F64)), 0.0)
        return (w * x).sum(dim=(-3, -2)) / w.sum(dim=(-3, -2))

    def __call__(
        self, input: Mapping, gen: Mapping, target: Mapping
    ) -> dict[str, torch.Tensor]:
        """input: target at t_n; gen, target: at t_n + dt. [batch, lat, lon]."""
        vc = self.vertical_coordinate
        inp, g, t = (OceanData(d, vc) for d in (input, gen, target))
        ssf = inp.sea_surface_fraction
        w = self.level_weights(ssf)
        den = w.sum(dim=(-3, -2))  # [batch, level], m3
        h_T, h_S = RHO_CP * den, RHO * den
        out: dict[str, torch.Tensor] = {
            "e_T": self.level_mean(
                g.sea_water_potential_temperature - t.sea_water_potential_temperature, w
            ),
            "e_S": self.level_mean(g.sea_water_salinity - t.sea_water_salinity, w),
            "thetao_target_mean": self.level_mean(t.sea_water_potential_temperature, w),
            "so_target_mean": self.level_mean(t.sea_water_salinity, w),
            "h_T": h_T,
            "h_S": h_S,
        }
        out["E_H"] = (h_T * out["e_T"]).sum(-1)
        out["E_S"] = (h_S * out["e_S"]).sum(-1)  # kg x g kg-1 = g

        # heat residual, as _force_conserve_ocean_heat_content (scaled_temperature,
        # no constant). nan_to_num: the raw target and the output-masked prediction
        # hold NaN on land, where the corrector sees input-masked fill values;
        # s = 0 there, so the sums are unchanged.
        s64 = torch.nan_to_num(ssf.to(F64))

        def ohc_mean(data: OceanData) -> torch.Tensor:
            theta = data.sea_water_potential_temperature.to(F64)
            # = OceanData.ocean_heat_content, in float64
            column = torch.nan_to_num(vc.depth_integral(theta * RHO_CP))
            return self.gridded_ops.area_weighted_mean(
                column * s64, name="ocean_heat_content"
            )

        flux = (
            g.net_downward_surface_heat_flux_total_area.to(F64)
            + inp.geothermal_heat_flux.to(F64) * s64
        )
        flux_mean = self.gridded_ops.area_weighted_mean(
            torch.nan_to_num(flux), name="ocean_heat_content"
        )
        a_norm = self.a_norm()
        out["G_in"], out["G_gen"], out["G_tgt_state"] = (
            ohc_mean(inp),
            ohc_mean(g),
            ohc_mean(t),
        )
        out["G_target"] = out["G_in"] + flux_mean * self.timestep_seconds
        out["r_H"] = a_norm * (out["G_target"] - out["G_gen"])
        out["E_H_direct"] = a_norm * (out["G_gen"] - out["G_tgt_state"])

        # salt residual, as _force_conserve_ocean_salt_content
        # (weight_by_sea_surface_fraction, float64)
        def global_total(x: torch.Tensor) -> torch.Tensor:
            return (
                self.gridded_ops.area_weighted_sum(
                    x, keepdim=True, name="ocean_salt_content"
                )
                * SPHERE_AREA_M2
            )

        def salt(data: OceanData) -> torch.Tensor:
            column = torch.nan_to_num(
                vc.depth_integral(data.sea_water_salinity.to(F64))
            )
            return global_total(column * s64).flatten()

        out["Salt_in"], out["Salt_gen"], out["Salt_target"] = (
            salt(inp),
            salt(g),
            salt(t),
        )
        out["dSalt_SSH"] = SeaSurfaceHeightSaltBudget(REFERENCE_SALINITY)(
            inp, g, inp, global_total, self.timestep_seconds, F64
        ).flatten()
        out["r_S"] = RHO * (out["Salt_in"] + out["dSalt_SSH"] - out["Salt_gen"])
        out["r_S_fixed"] = RHO * (out["Salt_in"] - out["Salt_gen"])
        out["E_S_direct"] = RHO * (out["Salt_gen"] - out["Salt_target"])
        return out


def ocean_corrections(stepper: Stepper) -> list:
    step: Any = stepper._step_obj  # SingleModuleStep; StepABC exposes no corrector
    corrector = step._corrector
    if isinstance(corrector, EpochScheduledCorrector):
        corrector = corrector._wrapped
    assert isinstance(corrector, CorrectionSequence), type(corrector)
    return list(corrector._corrections)


def assert_uncorrected(stepper: Stepper) -> list[str]:
    corrections = ocean_corrections(stepper)
    banned = (OceanHeatContentCorrection, OceanSaltContentCorrection)
    assert not any(isinstance(c, banned) for c in corrections), corrections
    return [type(c).__name__ for c in corrections]


def loader_config(
    data_path: str, start: str, stop: str | None, batch_size: int, workers: int
) -> DataLoaderConfig:
    subset = {"start_time": start, "stop_time": stop}
    merge = [
        {
            "data_path": data_path,
            "file_pattern": store,
            "engine": "zarr",
            "subset": subset,
        }
        for store in STORES
    ]
    return dacite.from_dict(
        DataLoaderConfig,
        {
            "dataset": {"merge": merge},
            "batch_size": batch_size,
            "num_data_workers": workers,
        },
        config=dacite.Config(strict=True),
    )


def build_reducer(stepper: Stepper, data_vertical_coordinate) -> tuple[Reducer, str]:
    ckpt_vc = stepper.training_dataset_info.vertical_coordinate
    assert isinstance(ckpt_vc, DepthCoordinate), ckpt_vc
    deptho_source = "checkpoint" if ckpt_vc.deptho is not None else "dataset"
    stepper.backfill_deptho(data_vertical_coordinate)
    info = stepper.training_dataset_info
    vc = info.vertical_coordinate
    assert isinstance(vc, DepthCoordinate) and vc.deptho is not None, vc
    area_weights = info.horizontal_coordinates.area_weights
    assert area_weights is not None
    reducer = Reducer(
        gridded_ops=info.gridded_operations,
        vertical_coordinate=vc.to(get_device()),
        area_weights=area_weights.to(get_device()),
        timestep_seconds=info.timestep.total_seconds(),
    )
    return reducer, deptho_source


def at(batch_data: Mapping, i: int) -> dict:
    return {k: v[:, i] for k, v in batch_data.items()}


@torch.no_grad()
def run(
    stepper: Stepper, data, reducer: Reducer
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    stepper.set_eval()
    n_ic = stepper.n_ic_timesteps
    assert n_ic == 1, n_ic
    chunks: dict[str, list] = {}
    times = []
    for i, batch in enumerate(data.loader):
        paired, _ = stepper.predict_paired(
            batch.get_start(stepper.prognostic_names, n_ic), batch
        )
        assert paired.prediction[next(iter(paired.prediction))].shape[1] == 1
        reduced = reducer(
            at(batch.data, 0), at(paired.prediction, 0), at(paired.reference, 0)
        )
        for k, v in reduced.items():
            chunks.setdefault(k, []).append(v.cpu().numpy())
        times.append(batch.time.isel(time=0).values)
        if i % 50 == 0:
            logging.info(f"batch {i}: t_n = {times[-1][0]}")
    return {k: np.concatenate(v) for k, v in chunks.items()}, np.concatenate(times)


def to_dataset(
    reduced: dict[str, np.ndarray], times: np.ndarray, reducer: Reducer, attrs: dict
) -> xr.Dataset:
    h_T, h_S = reduced.pop("h_T"), reduced.pop("h_S")
    np.testing.assert_allclose(
        h_T, np.broadcast_to(h_T[:1], h_T.shape), rtol=1e-10
    )  # s, dz static
    idepth = reducer.vertical_coordinate.idepth.cpu().numpy().astype(np.float64)
    ds = xr.Dataset(
        coords={"time": ("sample", times), "level": np.arange(reducer.n_levels)}
    )
    for k, (units, long_name) in PER_LEVEL_SAMPLE.items():
        ds[k] = (
            ("sample", "level"),
            reduced[k],
            {"units": units, "long_name": long_name},
        )
    for k, (units, long_name) in PER_SAMPLE.items():
        ds[k] = ("sample", reduced[k], {"units": units, "long_name": long_name})
    ds["h_T"] = (
        "level",
        h_T[0],
        {"units": "J K-1", "long_name": "rho_cp sum_j area_j s_j dz_jk"},
    )
    ds["h_S"] = (
        "level",
        h_S[0],
        {"units": "kg", "long_name": "rho sum_j area_j s_j dz_jk"},
    )
    ds["depth"] = (
        "level",
        0.5 * (idepth[:-1] + idepth[1:]),
        {"units": "m", "long_name": "level mid-depth"},
    )
    ds["A_norm"] = (
        (),
        float(reducer.a_norm()),
        {"units": "m2", "long_name": "SPHERE_AREA_M2 sum_j w_j mask_OHC,j"},
    )
    ds.attrs.update(attrs)
    return ds


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None, stepper: Stepper | None = None) -> xr.Dataset:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/ckpt.tar")
    p.add_argument("--out", required=True)
    p.add_argument("--start-time", required=True)
    p.add_argument("--stop-time", default=None)
    p.add_argument("--data-path", default="/climate-default")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-data-workers", type=int, default=8)
    p.add_argument("--no-ema", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if stepper is None:  # tests pass a random-init stepper
        stepper = load_stepper(args.checkpoint, use_ema_if_available=not args.no_ema)
    corrections = assert_uncorrected(stepper)
    requirements = stepper.config.get_evaluation_window_data_requirements(
        n_forward_steps=1
    )
    config = loader_config(
        args.data_path,
        args.start_time,
        args.stop_time,
        args.batch_size,
        args.num_data_workers,
    )
    data = get_gridded_data(config, train=False, requirements=requirements)
    stepper.training_dataset_info.assert_compatible_with(data.dataset_info)
    reducer, deptho_source = build_reducer(
        stepper, data.dataset_info.vertical_coordinate
    )

    reduced, times = run(stepper, data, reducer)
    attrs = {
        "issue": ".scratch/2026-11-04-1337-ohc-corrector-upper-ocean-attribution"
        "/issues/42",
        "checkpoint": args.checkpoint,
        "ema": str(not args.no_ema),
        "corrections": ", ".join(corrections),
        "ohc_corrector": "off (not in the checkpoint corrector)",
        "salt_corrector": "off (not in the checkpoint corrector)",
        "deptho_source": deptho_source,
        "data": ", ".join(f"{args.data_path}/{s}" for s in STORES),
        "subset": f"[{args.start_time}, {args.stop_time}]",
        "timestep_seconds": reducer.timestep_seconds,
        "area": "area_j = fme spherical_area_weights x SPHERE_AREA_M2",
        "salt_budget": "SeaSurfaceHeightSaltBudget, "
        f"reference_salinity_psu={REFERENCE_SALINITY}",
        "rho_cp": RHO_CP,
        "rho": RHO,
        "ace_exper_commit": git_commit(),
    }
    ds = to_dataset(reduced, times, reducer, attrs)
    ds.to_netcdf(args.out)
    logging.info(f"wrote {args.out}: {ds.sizes}")
    return ds


if __name__ == "__main__":
    main()
