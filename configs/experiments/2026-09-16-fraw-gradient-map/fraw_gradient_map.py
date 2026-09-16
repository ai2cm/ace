"""Gradient of the loss at the pre-corrector frozen-precipitation output.

One forward step of a checkpointed stepper per validation batch, corrector on
the autograd graph, ``F_raw`` and ``P_raw`` leaves. Three regimes of ``F_raw``:

    neg       : F_raw < 0                      ForcePositive zeroes it
    clipped   : F_pos > alpha P_pos            the clip replaces it with alpha P_pos
    trainable : 0 <= F_raw <= alpha P_pos      F_final = F_raw

Post-corrector scoring of ``F_final`` passes no gradient to ``F_raw`` in the
first two regimes; pre-corrector scoring passes it everywhere. Variants
(``_n`` = loss-normalizer units):

    trained : the run's training loss as configured (stepper_training in the
              config, including its corrector_loss.precorrector_optimization)
    post_F  : sum 0.5 (F_final_n - F_target_n)^2       # F scored after the corrector
    pre_F   : sum 0.5 (F_raw_n   - F_target_n)^2       # F scored before the corrector

    g_<variant> = dL_variant / dF_raw                  # physical units of F_raw

Writes ``fraw_gradient_map.nc`` (fields, masks, gradients; dims
sample/ensemble/lat/lon) and ``facts.json`` (regime fractions and per-regime
gradient statistics per variant) to ``output_dir``.
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
from fme.core.cli import prepare_config
from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.optimization import NullOptimization
from fme.core.step.single_module import SingleModuleStep

PRECIP = "PRATEsfc"
FROZEN = "total_frozen_precipitation_rate"
HDIMS = (-2, -1)
REGIMES = ("neg", "clipped", "trainable")


@dataclasses.dataclass
class Config:
    checkpoint_path: str
    run_label: str
    loader: DataLoaderConfig
    stepper_training: TrainStepperConfig
    output_dir: str = "/results"
    n_batches: int = 2
    train_mode: bool = True


class CapturingCorrector:
    """Wraps the step's corrector; detaches every network output, makes
    ``F_raw`` and ``P_raw`` leaves, keeps the corrected fields on the graph.
    """

    def __init__(self, inner):
        self._inner = inner
        self.raw: dict[str, torch.Tensor] = {}
        self.final: dict[str, torch.Tensor] = {}

    def __call__(self, input, output, next_step_input_data, corrector_state):
        torch.set_grad_enabled(True)
        output = {k: v.detach() for k, v in output.items()}
        for name in (PRECIP, FROZEN):
            output[name] = output[name].requires_grad_(True)
        self.raw = {name: output[name] for name in (PRECIP, FROZEN)}
        result = self._inner(input, output, next_step_input_data, corrector_state)
        self.final = {name: result.corrected[name] for name in (PRECIP, FROZEN)}
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


class RecordingOptimization(NullOptimization):
    """NullOptimization that keeps the accumulated losses on the graph."""

    def __init__(self):
        super().__init__()
        self.losses: list[torch.Tensor] = []

    def accumulate_loss(self, loss: torch.Tensor):
        self.losses.append(loss)
        super().accumulate_loss(loss)


def gm(a, x):
    return (a * x).sum(HDIMS, keepdim=True)


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
    cap = CapturingCorrector(step_obj._corrector)
    step_obj._corrector = cap  # type: ignore[assignment]
    train_stepper = TrainStepper(stepper=stepper, config=cfg.stepper_training)
    n_ens = cfg.stepper_training.n_ensemble
    n_ic = stepper.n_ic_timesteps

    requirements = stepper.config.get_evaluation_window_data_requirements(
        n_forward_steps=1
    )
    data = get_gridded_data(cfg.loader, train=False, requirements=requirements)
    hc = data.dataset_info.horizontal_coordinates
    assert isinstance(hc, LatLonCoordinates), type(hc)
    a = hc.area_weights.to(device)
    a = a / a.sum()
    loss_norm = step_obj.get_loss_normalizer()

    def norm_f(x):
        return loss_norm.normalize({FROZEN: x})[FROZEN]

    fields: dict[str, list[torch.Tensor]] = {
        k: [] for k in ["F_raw", "F_final", "F_target", "P_raw", "P_final"]
    }
    grads: dict[str, list[torch.Tensor]] = {}
    alphas: list[torch.Tensor] = []

    for i, batch in enumerate(data.loader):
        if i >= cfg.n_batches:
            break
        batch = batch.to_device()
        opt = RecordingOptimization()
        torch.set_grad_enabled(False)
        train_stepper.train_on_batch(batch, opt)
        assert torch.is_grad_enabled()
        total = sum(opt.losses)
        assert isinstance(total, torch.Tensor) and total.requires_grad
        F_raw, F_final = cap.raw[FROZEN], cap.final[FROZEN]
        P_raw, P_final = cap.raw[PRECIP], cap.final[PRECIP]
        target = batch.data[FROZEN][:, n_ic].repeat_interleave(n_ens, dim=0)
        Ff_n, Fr_n, Ft_n = norm_f(F_final), norm_f(F_raw), norm_f(target)
        losses = {
            "trained": total,
            "post_F": (0.5 * (Ff_n - Ft_n) ** 2).sum(),
            "pre_F": (0.5 * (Fr_n - Ft_n) ** 2).sum(),
        }
        for k, L in losses.items():
            g = torch.autograd.grad(L, F_raw, retain_graph=True)[0]
            grads.setdefault(k, []).append(g.detach().cpu())
        with torch.no_grad():
            P_pos = torch.clamp(P_raw, min=0)
            alphas.append((gm(a, P_final) / gm(a, P_pos)).squeeze(HDIMS).cpu())
            for k, v in [
                ("F_raw", F_raw),
                ("F_final", F_final),
                ("F_target", target),
                ("P_raw", P_raw),
                ("P_final", P_final),
            ]:
                fields[k].append(v.detach().cpu())
        logging.info("batch %d done, loss %s", i, float(total))
        del total, losses, F_raw, F_final, P_raw, P_final, Ff_n, Fr_n, Ft_n
        torch.cuda.empty_cache()

    def stack(lst):
        x = torch.cat(lst, 0)
        return x.reshape(-1, n_ens, *x.shape[-2:]).numpy().astype(np.float32)

    dims = ("sample", "ensemble", "lat", "lon")
    ds = xr.Dataset(
        {k: (dims, stack(v)) for k, v in fields.items()}
        | {f"grad_{k}": (dims, stack(v)) for k, v in grads.items()}
        | {
            "alpha": (dims[:2], torch.cat(alphas).reshape(-1, n_ens).numpy()),
            "a": (("lat", "lon"), a.cpu().numpy()),
        },
        coords={"lat": hc.lat.cpu().numpy(), "lon": hc.lon.cpu().numpy()},
    )
    F_raw_v, P_raw_v = ds["F_raw"].values, ds["P_raw"].values
    alpha_v = ds["alpha"].values[..., None, None]
    neg = F_raw_v < 0
    clipped = (~neg) & (F_raw_v > alpha_v * np.clip(P_raw_v, 0, None))
    trainable = ~(neg | clipped)
    masks = {"neg": neg, "clipped": clipped, "trainable": trainable}
    for k, m in masks.items():
        ds[f"mask_{k}"] = (dims, m.astype(np.int8))
    ds.attrs.update(
        run_label=cfg.run_label,
        checkpoint_path=cfg.checkpoint_path,
        checkpoint_epoch=-1 if ckpt_epoch is None else int(ckpt_epoch),
        train_mode=int(cfg.train_mode),
        loss_type=cfg.stepper_training.loss.type,
        precorrector_names=json.dumps(
            None
            if cfg.stepper_training.corrector_loss is None
            or cfg.stepper_training.corrector_loss.precorrector_optimization is None
            else list(
                cfg.stepper_training.corrector_loss.precorrector_optimization.names_and_prefixes
            )
        ),
    )
    ds.to_netcdf(os.path.join(cfg.output_dir, "fraw_gradient_map.nc"))

    a_v = ds["a"].values
    facts: dict = {
        "run_label": cfg.run_label,
        "checkpoint_path": cfg.checkpoint_path,
        "checkpoint_epoch": ckpt_epoch,
        "precorrector_names": json.loads(ds.attrs["precorrector_names"]),
        "n_samples": int(ds.sizes["sample"]),
        "n_ensemble": n_ens,
        "alpha_mean": float(ds["alpha"].mean()),
    }
    for k, m in masks.items():
        facts[f"area_frac_{k}"] = float((a_v * m).sum() / m.shape[0] / m.shape[1])
    F_pos = np.clip(F_raw_v, 0, None)
    facts["F_pos_gm_over_F_final_gm"] = float(
        (a_v * F_pos).sum() / (a_v * ds["F_final"].values).sum()
    )
    facts["F_final_gm_over_F_target_gm"] = float(
        (a_v * ds["F_final"].values).sum() / (a_v * ds["F_target"].values).sum()
    )
    for k in grads:
        g = ds[f"grad_{k}"].values
        facts[f"{k}/max_abs_grad"] = float(np.abs(g).max())
        for r, m in masks.items():
            facts[f"{k}/mean_abs_grad_{r}"] = (
                float(np.abs(g[m]).mean()) if m.any() else 0.0
            )
            facts[f"{k}/max_abs_grad_{r}"] = (
                float(np.abs(g[m]).max()) if m.any() else 0.0
            )
    with open(os.path.join(cfg.output_dir, "facts.json"), "w") as fh:
        json.dump(facts, fh, indent=2)
    logging.info("facts: %s", json.dumps(facts, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("yaml_config")
    main(p.parse_args().yaml_config)
