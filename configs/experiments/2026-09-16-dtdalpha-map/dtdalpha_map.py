"""dT and its sensitivity to the moisture rescale alpha through the frozen clip.

One forward step of a checkpointed stepper per validation batch. The corrector
sequence is replayed with every network output detached and ``P_raw`` a leaf,
stopping before the total-energy step, so that

    P_final   = alpha * P_pos,           alpha = <P_final> / <P_pos>
    F_final   = min(F_pos, alpha P_pos)  # ceiling is the rescaled precip
    dT        = T_final_k - T_raw_k      # one scalar per sample, every cell and level
    dT_alpha  = dT(F_final) - dT(F_pos)  # the energy step run with and without the clip

can be read off, and the energy step re-run with

    F(s) = min(F_pos, s P_pos),  s a leaf     ->  dT(s),  d dT / d s  (autograd)
    ddT_dalpha_closed = L_f dt <P_pos [F_pos > alpha P_pos]> / <factor>

Per-cell sensitivity of dT to the raw precipitation, autograd against

    ddT_dPraw = (L_f dt / <factor>) alpha a [P_raw > 0]
                * ( [clip] - <P_pos [clip]> / <P_pos> )

where ``[clip] = [F_pos > alpha P_pos]``. ``<factor>`` is the corrector's own
``_energy_correction_factor`` on the pre-energy state, area-weighted;
``dT_alpha = L_f dt <F_final - F_pos> / <factor>`` and the energy step run with
and without the clip (``dT_alpha_from_T``, float32 temperature difference)
checks it.

Writes ``dtdalpha_map.nc`` and ``facts.json`` to ``output_dir``.
"""

import argparse
import dataclasses
import json
import logging
import os

import dacite
import numpy as np
import torch
import xarray as xr

from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.data_loading.getters import get_gridded_data
from fme.ace.stepper.single_module import Stepper, TrainStepper, TrainStepperConfig
from fme.core.atmosphere_data import AtmosphereData
from fme.core.cli import prepare_config
from fme.core.constants import LATENT_HEAT_OF_FREEZING
from fme.core.coordinates import LatLonCoordinates
from fme.core.corrector.atmosphere import (
    MoistureBudgetCorrection,
    TotalEnergyBudgetCorrection,
    _energy_correction_factor,
)
from fme.core.corrector.output import CorrectorOutput, build_corrector_diagnostics
from fme.core.corrector.registry import CorrectionSequence, CorrectorABC
from fme.core.device import get_device
from fme.core.optimization import NullOptimization
from fme.core.step.single_module import SingleModuleStep

PRECIP = "PRATEsfc"
FROZEN = "total_frozen_precipitation_rate"
T0 = "air_temperature_0"
HDIMS = (-2, -1)


@dataclasses.dataclass
class Config:
    checkpoint_path: str
    loader: DataLoaderConfig
    stepper_training: TrainStepperConfig
    output_dir: str = "/results"
    n_batches: int = 2
    train_mode: bool = True
    sweep_min: float = 0.4
    sweep_max: float = 1.4
    sweep_n: int = 51


def gm(a, x):
    return (a * x).sum(HDIMS, keepdim=True)


class ReplayingCorrector:
    """Replays the sequence's corrections with detached outputs and ``P_raw``
    a leaf, recording the state before the total-energy step.
    """

    def __init__(self, inner: CorrectorABC):
        assert isinstance(inner, CorrectionSequence), type(inner)
        self._inner = inner
        corrections = inner._corrections
        energy = corrections[-1]
        assert isinstance(energy, TotalEnergyBudgetCorrection), corrections
        assert energy.vertical_coordinate is not None
        assert any(isinstance(c, MoistureBudgetCorrection) for c in corrections)
        self.energy = energy
        self.vertical_coordinate = energy.vertical_coordinate
        self.pre_energy: dict = {}
        self.raw: dict = {}
        self.final: dict = {}
        self.args: tuple = ()

    def __call__(self, input, output, next_step_input_data, corrector_state):
        torch.set_grad_enabled(True)
        output = {k: v.detach() for k, v in output.items()}
        output[PRECIP] = output[PRECIP].requires_grad_(True)
        self.raw = dict(output)
        snapshot = dict(output)
        gen = dict(output)
        modified: set[str] = set()
        for correction in self._inner._corrections:
            if correction is self.energy:
                self.pre_energy = dict(gen)
            changed, corrector_state = correction(
                input, gen, next_step_input_data, corrector_state
            )
            gen.update(changed)
            modified |= changed.keys()
        self.final = dict(gen)
        self.args = (input, next_step_input_data, corrector_state)
        return CorrectorOutput(
            corrected=gen,
            diagnostics=build_corrector_diagnostics(snapshot, gen, modified),
            corrector_state=corrector_state,
        )

    def dT_with_frozen(self, F):
        """DT of the energy step with the frozen field replaced by ``F``."""
        assert self.pre_energy and self.args
        input, forcing, state = self.args
        gen = {**self.pre_energy, FROZEN: F}
        out, _ = self.energy(input, gen, forcing, state)
        return (out[T0] - self.pre_energy[T0]).mean(HDIMS)  # uniform: mean == value

    def factor(self):
        """Area-weighted ``_energy_correction_factor`` of the pre-energy state."""
        assert self.pre_energy
        gen = AtmosphereData(self.pre_energy, self.vertical_coordinate)
        fac = _energy_correction_factor(gen, self.vertical_coordinate)
        return self.energy.area_weighted_mean(fac, keepdim=False)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def main(yaml_path: str):
    logging.basicConfig(level=logging.INFO)
    cfg = dacite.from_dict(
        Config, prepare_config(yaml_path), config=dacite.Config(strict=True)
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = get_device()

    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    stepper = Stepper.from_state(checkpoint["stepper"])
    ckpt_epoch = checkpoint.get("epoch")
    del checkpoint
    stepper.set_train() if cfg.train_mode else stepper.set_eval()
    step_obj = stepper._step_obj
    assert isinstance(step_obj, SingleModuleStep), type(step_obj)
    rep = ReplayingCorrector(step_obj._corrector)
    step_obj._corrector = rep  # type: ignore[assignment]
    train_stepper = TrainStepper(stepper=stepper, config=cfg.stepper_training)
    n_ens = cfg.stepper_training.n_ensemble
    dt = rep.energy.timestep_seconds

    requirements = stepper.config.get_evaluation_window_data_requirements(
        n_forward_steps=1
    )
    data = get_gridded_data(cfg.loader, train=False, requirements=requirements)
    hc = data.dataset_info.horizontal_coordinates
    assert isinstance(hc, LatLonCoordinates), type(hc)
    a = hc.area_weights.to(device)
    a = a / a.sum()

    sweep = torch.linspace(cfg.sweep_min, cfg.sweep_max, cfg.sweep_n, device=device)
    fields: dict[str, list] = {
        k: []
        for k in [
            "P_raw",
            "P_final",
            "F_raw",
            "F_final",
            "clip",
            "grad_dT_dPraw",
            "closed_dT_dPraw",
        ]
    }
    scalars: dict[str, list] = {
        k: []
        for k in [
            "alpha",
            "dT",
            "dT_alpha",
            "ddT_dalpha_autograd",
            "ddT_dalpha_closed",
            "factor",
            "a_clip",
            "g",
            "f",
            "dT_alpha_from_T",
        ]
    }
    curves: list = []

    for i, batch in enumerate(data.loader):
        if i >= cfg.n_batches:
            break
        batch = batch.to_device()
        torch.set_grad_enabled(False)
        train_stepper.train_on_batch(batch, NullOptimization())
        assert torch.is_grad_enabled() and rep.pre_energy
        P_raw, P_final = rep.raw[PRECIP], rep.final[PRECIP]
        F_raw, F_final = rep.raw[FROZEN], rep.final[FROZEN]
        F_pos = torch.clamp(
            F_raw, min=0
        )  # ForcePositive's output; pre_energy holds the clipped F
        P_pos = torch.clamp(P_raw, min=0)
        alpha = gm(a, P_final) / gm(a, P_pos)  # (n, 1, 1)
        clip = (F_pos > alpha * P_pos).to(P_raw.dtype)

        # dT and the clip's share of it, from the energy step itself
        dT = rep.dT_with_frozen(F_final)  # (n,)
        g = gm(a, F_final - F_pos).squeeze(HDIMS)  # <= 0
        f = gm(a, F_pos - F_raw).squeeze(HDIMS)  # >= 0
        with torch.no_grad():
            factor = rep.factor()  # (n,)
            dT_alpha = LATENT_HEAT_OF_FREEZING * dt * g / factor
            dT_alpha_from_T = dT - rep.dT_with_frozen(F_pos)  # float32 T difference

        # per-cell sensitivity of dT to P_raw through alpha and the clip ceiling
        grad = torch.autograd.grad(dT.sum(), P_raw, retain_graph=True)[0]
        with torch.no_grad():
            pos = (P_raw > 0).to(P_raw.dtype)
            closed = (
                (LATENT_HEAT_OF_FREEZING * dt / factor.view(-1, 1, 1))
                * alpha
                * a
                * pos
                * (clip - gm(a, P_pos * clip) / gm(a, P_pos))
            )
            ddT_dalpha_closed = (
                LATENT_HEAT_OF_FREEZING
                * dt
                * gm(a, P_pos * clip).squeeze(HDIMS)
                / factor
            )

        # dT(s) with s a leaf: the clip ceiling at a hypothetical rescale s
        P_pos_d, F_pos_d = P_pos.detach(), F_pos.detach()
        s0 = alpha.detach().squeeze(HDIMS).clone().requires_grad_(True)
        dT_s0 = rep.dT_with_frozen(torch.minimum(F_pos_d, s0.view(-1, 1, 1) * P_pos_d))
        ddT_dalpha_auto = torch.autograd.grad(dT_s0.sum(), s0)[0]
        with torch.no_grad():
            curve = torch.stack(
                [
                    rep.dT_with_frozen(torch.minimum(F_pos_d, s * P_pos_d))
                    for s in sweep
                ],
                -1,
            )  # (n, sweep)

        with torch.no_grad():
            for k, v in [
                ("P_raw", P_raw),
                ("P_final", P_final),
                ("F_raw", F_raw),
                ("F_final", F_final),
                ("clip", clip),
                ("grad_dT_dPraw", grad),
                ("closed_dT_dPraw", closed),
            ]:
                fields[k].append(v.detach().cpu())
            for k, v in [
                ("alpha", alpha.squeeze(HDIMS)),
                ("dT", dT),
                ("dT_alpha", dT_alpha),
                ("ddT_dalpha_autograd", ddT_dalpha_auto),
                ("ddT_dalpha_closed", ddT_dalpha_closed),
                ("factor", factor),
                ("a_clip", gm(a, clip).squeeze(HDIMS)),
                ("g", g),
                ("f", f),
                ("dT_alpha_from_T", dT_alpha_from_T),
            ]:
                scalars[k].append(v.detach().cpu())
            curves.append(curve.cpu())
        logging.info(
            "batch %d: alpha %s dT %s dT_alpha %s",
            i,
            alpha.flatten().tolist(),
            dT.tolist(),
            dT_alpha.tolist(),
        )
        torch.cuda.empty_cache()

    dims = ("sample", "ensemble", "lat", "lon")
    ds = xr.Dataset(
        {
            k: (
                dims,
                torch.cat(v, 0)
                .reshape(-1, n_ens, *v[0].shape[-2:])
                .numpy()
                .astype(np.float32),
            )
            for k, v in fields.items()
        }
        | {
            k: (dims[:2], torch.cat(v, 0).reshape(-1, n_ens).numpy())
            for k, v in scalars.items()
        }
        | {
            "dT_of_s": (
                ("sample", "ensemble", "s"),
                torch.cat(curves, 0).reshape(-1, n_ens, cfg.sweep_n).numpy(),
            ),
            "a": (("lat", "lon"), a.cpu().numpy()),
        },
        coords={
            "lat": hc.lat.cpu().numpy(),
            "lon": hc.lon.cpu().numpy(),
            "s": sweep.cpu().numpy(),
        },
    )
    ds.attrs.update(
        checkpoint_path=cfg.checkpoint_path,
        checkpoint_epoch=-1 if ckpt_epoch is None else int(ckpt_epoch),
        train_mode=int(cfg.train_mode),
        dt_seconds=dt,
        latent_heat_of_freezing=LATENT_HEAT_OF_FREEZING,
    )
    ds.to_netcdf(os.path.join(cfg.output_dir, "dtdalpha_map.nc"))

    g_, c_ = ds["grad_dT_dPraw"].values, ds["closed_dT_dPraw"].values
    neg = ds["P_raw"].values < 0
    facts = {
        "checkpoint_path": cfg.checkpoint_path,
        "checkpoint_epoch": ckpt_epoch,
        "n_samples": int(ds.sizes["sample"]),
        "n_ensemble": n_ens,
        "dt_seconds": dt,
        "alpha_mean": float(ds["alpha"].mean()),
        "alpha_min": float(ds["alpha"].min()),
        "alpha_max": float(ds["alpha"].max()),
        "dT_mean": float(ds["dT"].mean()),
        "dT_alpha_mean": float(ds["dT_alpha"].mean()),
        "a_clip_mean": float(ds["a_clip"].mean()),
        "g_mean": float(ds["g"].mean()),
        "f_mean": float(ds["f"].mean()),
        "factor_mean": float(ds["factor"].mean()),
        "factor_rel_spread": float(
            (ds["factor"].max() - ds["factor"].min()) / ds["factor"].mean()
        ),
        "dT_alpha_from_T_max_rel_err": float(
            np.abs(ds["dT_alpha_from_T"] - ds["dT_alpha"]).max()
            / np.abs(ds["dT_alpha"]).max()
        ),
        "ddT_dalpha_autograd_mean": float(ds["ddT_dalpha_autograd"].mean()),
        "ddT_dalpha_closed_max_rel_err": float(
            np.abs(ds["ddT_dalpha_autograd"] - ds["ddT_dalpha_closed"]).max()
            / np.abs(ds["ddT_dalpha_closed"]).max()
        ),
        "dT_dPraw_closed_max_rel_err": float(np.abs(g_ - c_).max() / np.abs(c_).max()),
        "dT_dPraw_max_abs_grad_on_clipped": float(np.abs(g_[neg]).max())
        if neg.any()
        else 0.0,
        "dT_dPraw_ppos_weighted_sum_relative": float(
            (np.clip(ds["P_raw"].values, 0, None) * g_).sum()
            / (np.clip(ds["P_raw"].values, 0, None) * np.abs(g_)).sum()
        ),
    }
    with open(os.path.join(cfg.output_dir, "facts.json"), "w") as fh:
        json.dump(facts, fh, indent=2)
    logging.info("facts: %s", json.dumps(facts, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("yaml_config")
    main(p.parse_args().yaml_config)
