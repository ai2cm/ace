"""Run inference with an intervention on the prognostic state each step."""

import argparse
import logging
import os

import dacite
import numpy as np
import torch

import fme
from fme.ace.data_loading.batch_data import PrognosticState
from fme.ace.data_loading.getters import get_forcing_data
from fme.ace.inference.inference import (
    InferenceConfig,
    get_initial_condition,
    resolve_variable_metadata,
)
from fme.core.cli import prepare_config
from fme.core.distributed.distributed import Distributed
from fme.core.generics.inference import get_record_to_wandb, run_inference
from fme.core.timing import GlobalTimer


def _area_weights_tensor(lat_coords, device):
    """Latitude-based area weights as a tensor broadcastable to (1, 1, lat, lon)."""
    w = torch.cos(torch.deg2rad(torch.tensor(lat_coords, dtype=torch.float64)))
    w = w / w.mean()
    return w.to(device=device, dtype=torch.float32).reshape(1, 1, -1, 1)


def make_clamped_predict(predict_fn, clamp_vars, ic_state):
    """Wrap predict to overwrite clamp_vars with their IC value each step."""
    ic_data = ic_state.as_batch_data().data
    ic_values = {v: ic_data[v][:, -1:].clone() for v in clamp_vars if v in ic_data}
    logging.info(f"Clamping {list(ic_values.keys())} to IC values each step")

    def wrapper(prognostic_state, forcing, compute_derived_variables=True):
        output_data, new_state = predict_fn(
            prognostic_state,
            forcing=forcing,
            compute_derived_variables=compute_derived_variables,
        )
        state_data = new_state.as_batch_data()
        for v, ic_val in ic_values.items():
            if v in state_data.data:
                state_data.data[v] = ic_val.expand_as(state_data.data[v])
        return output_data, PrognosticState(state_data)

    return wrapper


def make_gmean_clamped_predict(predict_fn, var_name, gm_min, gm_max, lat_coords):
    """Wrap predict to nudge the global mean of var_name back into [gm_min, gm_max].

    If the area-weighted global mean after a step falls below gm_min, a uniform
    positive offset is added to the field; if above gm_max, a uniform negative
    offset. This preserves the spatial pattern while bounding the mean.
    """
    logging.info(f"Global-mean clamping {var_name} to [{gm_min:.2e}, {gm_max:.2e}]")
    w = None  # lazy-init on first call to get the right device

    def wrapper(prognostic_state, forcing, compute_derived_variables=True):
        nonlocal w
        output_data, new_state = predict_fn(
            prognostic_state,
            forcing=forcing,
            compute_derived_variables=compute_derived_variables,
        )
        state_data = new_state.as_batch_data()
        if var_name in state_data.data:
            field = state_data.data[var_name]  # (sample, time, lat, lon)
            if w is None:
                w = _area_weights_tensor(lat_coords, field.device)
            gm = (field.double() * w).mean(dim=(-2, -1), keepdim=True).float()
            correction = torch.zeros_like(gm)
            correction = torch.where(gm < gm_min, gm_min - gm, correction)
            correction = torch.where(gm > gm_max, gm_max - gm, correction)
            state_data.data[var_name] = field + correction
        return output_data, PrognosticState(state_data)

    return wrapper


def make_relaxed_gmean_predict(predict_fn, relax_vars, target_means, tau, lat_coords):
    """Wrap predict to apply Newtonian relaxation of global means toward target values.

    After each step, for each variable in relax_vars, nudge the field so that:
        gm(field_corrected) = gm(field) - (1/tau) * (gm(field) - target_mean)
    This is equivalent to exponential relaxation with e-folding timescale tau steps.
    """
    logging.info(
        f"Newtonian relaxation: vars={relax_vars}, tau={tau} days, "
        f"targets={{{', '.join(f'{v}: {target_means[v]:.6e}' for v in relax_vars)}}}"
    )
    w = None
    alpha = 1.0 / tau

    def wrapper(prognostic_state, forcing, compute_derived_variables=True):
        nonlocal w
        output_data, new_state = predict_fn(
            prognostic_state,
            forcing=forcing,
            compute_derived_variables=compute_derived_variables,
        )
        state_data = new_state.as_batch_data()
        for var_name in relax_vars:
            if var_name in state_data.data:
                field = state_data.data[var_name]
                if w is None:
                    w = _area_weights_tensor(lat_coords, field.device)
                gm = (field.double() * w).mean(dim=(-2, -1), keepdim=True).float()
                target = target_means[var_name]
                correction = -alpha * (gm - target)
                state_data.data[var_name] = field + correction
        return output_data, PrognosticState(state_data)

    return wrapper


def build_predict(args, stepper, data):
    """Build the (possibly wrapped) predict function from CLI args."""
    predict_fn = stepper.predict_paired

    if args.mode == "clamp_ic":
        return make_clamped_predict(predict_fn, args.clamp_vars, data.initial_condition)
    elif args.mode == "clamp_gmean":
        ds = data.initial_condition.as_batch_data()
        lat_coords = np.linspace(-90, 90, ds.data[args.clamp_vars[0]].shape[-2])
        return make_gmean_clamped_predict(
            predict_fn,
            args.clamp_vars[0],
            gm_min=args.gm_min,
            gm_max=args.gm_max,
            lat_coords=lat_coords,
        )
    elif args.mode == "relax_gmean":
        ds = data.initial_condition.as_batch_data()
        lat_coords = np.linspace(-90, 90, ds.data[args.clamp_vars[0]].shape[-2])
        target_means = {}
        for var_name in args.clamp_vars:
            target_means[var_name] = stepper.normalizer.means[var_name].item()
        return make_relaxed_gmean_predict(
            predict_fn,
            args.clamp_vars,
            target_means,
            tau=args.relax_tau,
            lat_coords=lat_coords,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config")
    parser.add_argument(
        "--mode", choices=["clamp_ic", "clamp_gmean", "relax_gmean"], required=True
    )
    parser.add_argument("--clamp-vars", nargs="+", required=True)
    parser.add_argument(
        "--gm-min",
        type=float,
        default=None,
        help="Min global-mean bound (for clamp_gmean mode)",
    )
    parser.add_argument(
        "--gm-max",
        type=float,
        default=None,
        help="Max global-mean bound (for clamp_gmean mode)",
    )
    parser.add_argument(
        "--relax-tau",
        type=float,
        default=100.0,
        help="Relaxation timescale in days (for relax_gmean mode)",
    )
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config_data = prepare_config(args.yaml_config, override=args.override)
    config = dacite.from_dict(
        data_class=InferenceConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )

    os.makedirs(config.experiment_dir, exist_ok=True)
    config.configure_logging(log_filename="inference_out.log")

    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True

    stepper_config = config.load_stepper_config()
    data_requirements = stepper_config.get_forcing_window_data_requirements(
        n_forward_steps=config.forward_steps_in_memory
    )
    logging.info("Loading initial condition data")
    initial_condition = get_initial_condition(
        config.initial_condition.get_dataset(),
        stepper_config.prognostic_names,
        labels=config.labels,
        n_ensemble=config.n_ensemble_per_ic,
    )
    stepper = config.load_stepper()
    stepper.set_eval()
    data = get_forcing_data(
        config=config.forcing_loader,
        total_forward_steps=config.n_forward_steps,
        window_requirements=data_requirements,
        initial_condition=initial_condition,
        surface_temperature_name=stepper.surface_temperature_name,
        ocean_fraction_name=stepper.ocean_fraction_name,
        label_override=config.labels,
    )

    variable_metadata = resolve_variable_metadata(
        dataset_metadata=data.variable_metadata,
        stepper_metadata=stepper.training_variable_metadata,
        stepper_all_names=stepper_config.all_names,
    )
    dataset_info = data.dataset_info.update_variable_metadata(variable_metadata)
    aggregator = config.aggregator.build(
        dataset_info=dataset_info,
        n_timesteps=config.n_forward_steps + stepper.n_ic_timesteps,
        output_dir=config.experiment_dir,
    )
    writer = config.get_data_writer(
        initial_condition_times=data.initial_time.to_numpy(),
        timestep=data.timestep,
        coords=data.coords,
        variable_metadata=variable_metadata,
    )

    predict = build_predict(args, stepper, data)

    logging.info("Starting inference with intervention")
    with torch.no_grad(), GlobalTimer():
        logger = get_record_to_wandb(label="inference")
        run_inference(
            predict=predict,
            data=data,
            writer=writer,
            aggregator=aggregator,
            record_logs=logger.log,
        )
        writer.finalize()
        aggregator.flush_diagnostics()


if __name__ == "__main__":
    with Distributed.context():
        main()
