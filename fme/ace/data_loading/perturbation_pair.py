"""Build control/perturbed paired inference data for the perturbation-response eval.

A perturbation-response rollout runs an unperturbed *baseline* and one or more
*perturbed* climates together as separate batch members of a single rollout, so
that differencing their time-means yields a clean warming response. This module
turns an ordinary (unperturbed) inference initial condition and forcing loader
into a paired version: the baseline members are the originals; the perturbed
members are copies with an SST perturbation applied to both the initial
condition and every forcing window.

Only a single perturbation (two groups: baseline + perturbed) is supported for
now; the one-hot group encoding returned alongside the data leaves room to add
more perturbation groups later.
"""

import dataclasses

import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.data_loading.perturbation import SSTPerturbation
from fme.core.generics.data import SimpleInferenceData, SizedMap
from fme.core.labels import BatchLabels


def _concat_along_batch(a: BatchData, b: BatchData) -> BatchData:
    """Concatenate two BatchData along the sample (batch) dimension."""
    if a.n_ensemble != b.n_ensemble:
        raise ValueError("cannot concatenate BatchData with different n_ensemble.")
    data = {name: torch.cat([a.data[name], b.data[name]], dim=0) for name in a.data}
    time = xr.concat([a.time, b.time], dim="sample")
    labels = None
    if a.labels is not None and b.labels is not None:
        labels = BatchLabels(
            torch.cat([a.labels.tensor, b.labels.tensor], dim=0), a.labels.names
        )
    data_mask = None
    if a.data_mask is not None and b.data_mask is not None:
        data_mask = {
            name: torch.cat([a.data_mask[name], b.data_mask[name]], dim=0)
            for name in a.data_mask
        }
    return BatchData(
        data=data,
        time=time,
        horizontal_dims=a.horizontal_dims,
        labels=labels,
        epoch=a.epoch,
        n_ensemble=a.n_ensemble,
        data_mask=data_mask,
    )


def _perturbed_copy(
    batch: BatchData,
    perturbation: SSTPerturbation,
    surface_temperature_name: str,
    lats: torch.Tensor,
    lons: torch.Tensor,
    ocean_fraction: torch.Tensor,
) -> BatchData:
    """Return a copy of ``batch`` with the SST perturbation applied to its
    surface-temperature field. ``ocean_fraction`` selects the cells perturbed
    (pass an all-ones tensor to perturb the whole field).
    """
    new_data = dict(batch.data)
    surface_temperature = batch.data[surface_temperature_name].clone()
    for config in perturbation.perturbations:
        config.apply_perturbation(surface_temperature, lats, lons, ocean_fraction)
    new_data[surface_temperature_name] = surface_temperature
    return dataclasses.replace(batch, data=new_data)


def build_perturbation_pair_data(
    initial_condition: PrognosticState,
    loader,
    perturbation: SSTPerturbation,
    surface_temperature_name: str,
    ocean_fraction_name: str,
    lats: torch.Tensor,
    lons: torch.Tensor,
    perturb_initial_condition_full_field: bool = True,
) -> tuple[SimpleInferenceData, torch.Tensor]:
    """Build paired baseline/perturbed inference data from unperturbed inputs.

    Parameters:
        initial_condition: Unperturbed (prognostic) initial condition.
        loader: Unperturbed forcing loader yielding ``BatchData`` windows.
        perturbation: The SST perturbation applied to the perturbed members.
        surface_temperature_name: Name of the surface-temperature variable.
        ocean_fraction_name: Name of the ocean-fraction forcing variable.
        lats: Latitude meshgrid matching the (local) spatial shape of the data.
        lons: Longitude meshgrid matching the (local) spatial shape of the data.
        perturb_initial_condition_full_field: If True, the initial-condition
            perturbation is applied over the whole field; if False, only over
            the ocean (matching the forcing perturbation). The forcing
            perturbation is always ocean-masked.

    Returns:
        A tuple of the paired inference data and a ``[2 * n_local, 2]`` one-hot
        group encoding (group 0 = baseline, group 1 = perturbed) whose member
        order matches the batch dimension of the paired data.
    """
    base_ic = initial_condition.as_batch_data()
    n_local = next(iter(base_ic.data.values())).shape[0]

    if surface_temperature_name in base_ic.data:
        if perturb_initial_condition_full_field:
            ocean_fraction_ic = torch.ones_like(base_ic.data[surface_temperature_name])
        else:
            n_ic_timesteps = base_ic.data[surface_temperature_name].shape[1]
            first_window = next(iter(loader))
            ocean_fraction_ic = first_window.data[ocean_fraction_name][
                :, :n_ic_timesteps
            ]
        perturbed_ic = _perturbed_copy(
            base_ic,
            perturbation,
            surface_temperature_name,
            lats,
            lons,
            ocean_fraction_ic,
        )
    else:
        # Surface temperature is a forcing variable; perturbing the first
        # forcing window already perturbs the t0 input, so the prognostic
        # initial condition needs no separate perturbation.
        perturbed_ic = base_ic
    paired_ic = PrognosticState(_concat_along_batch(base_ic, perturbed_ic))

    def pair_window(window: BatchData) -> BatchData:
        ocean_fraction = window.data[ocean_fraction_name]
        perturbed = _perturbed_copy(
            window,
            perturbation,
            surface_temperature_name,
            lats,
            lons,
            ocean_fraction,
        )
        return _concat_along_batch(window, perturbed)

    paired_loader = SizedMap(pair_window, loader)

    group_onehot = torch.zeros(2 * n_local, 2)
    group_onehot[:n_local, 0] = 1.0  # baseline block
    group_onehot[n_local:, 1] = 1.0  # perturbed block

    return SimpleInferenceData(paired_ic, paired_loader), group_onehot
