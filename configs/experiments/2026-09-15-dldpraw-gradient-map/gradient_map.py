"""Gradient of the training loss at the pre-corrector precipitation output.

One forward step of a checkpointed stepper per validation batch, corrector on
the autograd graph, capturing the raw (pre-corrector) and corrected PRATEsfc
fields. For each loss variant

    g[variant] = dL_variant / dP_raw            # physical units of P_raw

Variants (PRATEsfc only unless noted; ``_n`` = loss-normalizer units; ``a`` =
area weights, sum 1; ``a_rel = a / mean(a)``, sum n_cells):

    trained    : the training loss as configured (all channels, EnsembleLoss)
    mse_unw    : sum 0.5 (P_final_n - P_target_n)^2
    mse_area   : sum a_rel 0.5 (P_final_n - P_target_n)^2
    bias_unw   : sum c P_final_n                  # dL/dP_final_n = c everywhere
    bias_area  : sum a_rel c P_final_n

Closed form checked for the two bias variants:

    dL/dP_raw = alpha [P_raw > 0] ( dL/dP_final - a sum (P_pos / <P_pos>) dL/dP_final )

Writes ``gradient_map.nc`` (fields and gradients, dims sample/ensemble/lat/lon)
and ``facts.json`` (scalar checks) to ``output_dir``.
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
from fme.ace.stepper.single_module import TrainStepper, TrainStepperConfig, load_stepper
from fme.core.cli import prepare_config
from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.optimization import NullOptimization
from fme.core.step.single_module import SingleModuleStep

PRECIP = "PRATEsfc"
HDIMS = (-2, -1)


@dataclasses.dataclass
class Config:
    checkpoint_path: str
    loader: DataLoaderConfig
    stepper_training: TrainStepperConfig
    output_dir: str = "/results"
    n_batches: int = 2
    uniform_bias_c: float = 1.0
    train_mode: bool = True  # module/corrector train() as during optimization


class CapturingCorrector:
    """Wraps the step's corrector; keeps raw and corrected PRATEsfc on the graph.

    The network runs with grad disabled (set by the caller), so no activation
    graph is stored. Here every network output is detached, ``P_raw`` becomes
    a leaf, and grad is switched back on, so the graph the loss builds spans
    only corrector + loss. ``dL/dP_raw`` is the partial with the other network
    outputs held fixed, which is the quantity wanted.
    """

    def __init__(self, inner):
        self._inner = inner
        self.raw: torch.Tensor | None = None
        self.final: torch.Tensor | None = None

    def __call__(self, input, output, next_step_input_data, corrector_state):
        torch.set_grad_enabled(True)
        output = {k: v.detach() for k, v in output.items()}
        output[PRECIP] = output[PRECIP].requires_grad_(True)
        self.raw = output[PRECIP]
        result = self._inner(input, output, next_step_input_data, corrector_state)
        self.final = result.corrected[PRECIP]
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


def gm(a, x):
    return (a * x).sum(HDIMS, keepdim=True)


def closed_form(P_raw, P_final, a, dL_dPfinal):
    P_pos = torch.clamp(P_raw, min=0)
    gm_pos = gm(a, P_pos)
    alpha = gm(a, P_final) / gm_pos
    s = P_pos / gm_pos
    pos = (P_raw > 0).to(P_raw.dtype)
    return alpha * pos * (
        dL_dPfinal - a * (s * dL_dPfinal).sum(HDIMS, keepdim=True)
    ), alpha


def main(yaml_path: str):
    logging.basicConfig(level=logging.INFO)
    cfg = dacite.from_dict(
        Config, prepare_config(yaml_path), config=dacite.Config(strict=True)
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = get_device()

    stepper = load_stepper(cfg.checkpoint_path)
    stepper.set_train() if cfg.train_mode else stepper.set_eval()
    step_obj = stepper._step_obj
    assert isinstance(step_obj, SingleModuleStep), type(step_obj)
    cap = CapturingCorrector(step_obj._corrector)
    step_obj._corrector = cap  # type: ignore[assignment]  # duck-typed wrapper
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
    a_rel = a / a.mean()
    loss_norm = step_obj.get_loss_normalizer()

    def norm_p(x):
        return loss_norm.normalize({PRECIP: x})[PRECIP]

    c = cfg.uniform_bias_c
    fields: dict[str, list[torch.Tensor]] = {
        k: [] for k in ["P_raw", "P_final", "P_target"]
    }
    grads: dict[str, list[torch.Tensor]] = {}
    closed: dict[str, list[torch.Tensor]] = {"bias_unw": [], "bias_area": []}
    alphas: list[torch.Tensor] = []

    for i, batch in enumerate(data.loader):
        if i >= cfg.n_batches:
            break
        batch = batch.to_device()
        opt = NullOptimization()
        torch.set_grad_enabled(False)  # network forward without activation graph
        train_stepper.train_on_batch(batch, opt)  # CapturingCorrector re-enables grad
        assert torch.is_grad_enabled()
        total = opt.get_accumulated_loss()
        assert cap.raw is not None and cap.final is not None
        P_raw, P_final = cap.raw, cap.final  # (n_sample * n_ens, lat, lon)
        target = batch.data[PRECIP][:, n_ic].repeat_interleave(n_ens, dim=0)
        Pf_n, Pt_n = norm_p(P_final), norm_p(target)
        inv_std = torch.autograd.grad(Pf_n.sum(), P_final, retain_graph=True)[0]

        losses = {
            "trained": total,
            "mse_unw": (0.5 * (Pf_n - Pt_n) ** 2).sum(),
            "mse_area": (a_rel * 0.5 * (Pf_n - Pt_n) ** 2).sum(),
            "bias_unw": (c * Pf_n).sum(),
            "bias_area": (a_rel * c * Pf_n).sum(),
        }
        for k, L in losses.items():
            g = torch.autograd.grad(L, P_raw, retain_graph=True)[0]
            grads.setdefault(k, []).append(g.detach().cpu())
        with torch.no_grad():
            cf_u, alpha = closed_form(P_raw, P_final, a, c * inv_std)
            cf_a, _ = closed_form(P_raw, P_final, a, a_rel * c * inv_std)
            closed["bias_unw"].append(cf_u.cpu())
            closed["bias_area"].append(cf_a.cpu())
            alphas.append(alpha.squeeze(HDIMS).cpu())
            fields["P_raw"].append(P_raw.cpu())
            fields["P_final"].append(P_final.cpu())
            fields["P_target"].append(target.cpu())
        logging.info("batch %d done, loss %s", i, float(total))
        del total, losses, P_raw, P_final, Pf_n, Pt_n, inv_std
        torch.cuda.empty_cache()

    def stack(lst):
        x = torch.cat(lst, 0)
        return x.reshape(-1, n_ens, *x.shape[-2:]).numpy().astype(np.float32)

    dims = ("sample", "ensemble", "lat", "lon")
    coords = {"lat": hc.lat.cpu().numpy(), "lon": hc.lon.cpu().numpy()}
    ds = xr.Dataset(
        {k: (dims, stack(v)) for k, v in fields.items()}
        | {f"grad_{k}": (dims, stack(v)) for k, v in grads.items()}
        | {f"closed_{k}": (dims, stack(v)) for k, v in closed.items()}
        | {
            "alpha": (dims[:2], torch.cat(alphas).reshape(-1, n_ens).numpy()),
            "a": (("lat", "lon"), a.cpu().numpy()),
        },
        coords=coords,
    )
    ds.attrs.update(
        checkpoint_path=cfg.checkpoint_path,
        uniform_bias_c=c,
        train_mode=int(cfg.train_mode),
        loss_type=cfg.stepper_training.loss.type,
    )
    ds.to_netcdf(os.path.join(cfg.output_dir, "gradient_map.nc"))

    P_raw = ds["P_raw"].values
    neg = P_raw < 0
    P_pos = np.clip(P_raw, 0, None)
    s = P_pos / (ds["a"].values * P_pos).sum((-2, -1), keepdims=True)
    facts = {
        "n_samples": int(ds.sizes["sample"]),
        "n_ensemble": n_ens,
        "frac_clipped": float(neg.mean()),
        "alpha_min": float(ds["alpha"].min()),
        "alpha_max": float(ds["alpha"].max()),
        "loss_type": cfg.stepper_training.loss.type,
    }
    for k in grads:
        g = ds[f"grad_{k}"].values
        facts[f"{k}/max_abs_grad_on_clipped"] = (
            float(np.abs(g[neg]).max()) if neg.any() else 0.0
        )
        facts[f"{k}/max_abs_grad"] = float(np.abs(g).max())
        facts[f"{k}/ppos_weighted_sum_relative"] = float(
            (s * g).sum() / (s * np.abs(g)).sum()
        )
    for k in closed:
        g, cf = ds[f"grad_{k}"].values, ds[f"closed_{k}"].values
        facts[f"{k}/closed_form_max_rel_err"] = float(
            np.abs(g - cf).max() / np.abs(cf).max()
        )
    with open(os.path.join(cfg.output_dir, "facts.json"), "w") as f:
        json.dump(facts, f, indent=2)
    logging.info("facts: %s", json.dumps(facts, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("yaml_config")
    main(p.parse_args().yaml_config)
