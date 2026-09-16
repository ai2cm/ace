"""Measure how coupled validation memory scales with n_coupled_steps.

Builds a CoupledTrainStepper shaped like the cm4 `fto` recipe -- atmosphere
frozen (loss_weight 0.0, n_steps 0), 20 atmosphere steps per ocean step,
n_ensemble 2 -- and runs one validation batch (train_on_batch with derived
variables, then the validation OneStepAggregator) under torch.no_grad(),
reporting peak live tensor bytes at each requested cap.

The frozen realm's data window is n_inner_steps times the ocean's, so peak
memory is dominated by how many copies of that window one batch is worth. The
reported slope is bytes per atmosphere timestep, which is what sets the
affordable n_coupled_steps for a given card. It is a count of live tensor
bytes, not device memory, so it excludes the caching allocator's overhead;
compare slopes before and after a change rather than to nvidia-smi.

Runs on CPU. Scale the slope to a production grid by the ratio of grid areas:
tensor bytes are linear in n_lat * n_lon and in n_samples.

Example:
    FME_FORCE_CPU=1 python scripts/coupled/coupled_memory_scaling.py \
        --n-lat 24 --n-lon 48 --caps 4 12 16 20
"""

import argparse
import dataclasses
import datetime

import numpy as np
import torch
import xarray as xr

import fme
from fme.ace.data_loading.batch_data import BatchData
from fme.ace.stepper import StepperConfig
from fme.core.coordinates import (
    DepthCoordinate,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
)
from fme.core.dataset_info import DatasetInfo
from fme.core.loss import StepLossConfig
from fme.core.ocean import OceanConfig
from fme.core.optimization import NullOptimization
from fme.core.registry.corrector import CorrectorSelector
from fme.core.registry.module import ModuleSelector
from fme.core.spatial_mask_provider import SpatialMaskProvider
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector
from fme.core.testing import PeakTensorMemory, trivial_network_and_loss_normalization
from fme.coupled.aggregator import OneStepAggregator
from fme.coupled.data_loading.batch_data import CoupledBatchData
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.stepper import (
    ComponentConfig,
    ComponentTrainingConfig,
    CoupledStepperConfig,
    CoupledTrainStepperConfig,
)

N_INNER_STEPS = 20  # 6h atmosphere step inside a 5D ocean step
N_LEVELS = 8  # atmosphere layers, as in the cm4 8-layer recipe
N_OCEAN_LEVELS = 4  # trimmed from 19; the ocean window is n_coupled+1, so tiny

_LEVELED = [
    "air_temperature",
    "specific_total_water",
    "eastward_wind",
    "northward_wind",
]

# Names as in configs/experiments/cm4_1pct_46to125_piC_156to235/coupled_tail.
ATMOS_OUT_NAMES = (
    ["PRESsfc", "surface_temperature"]
    + [f"{name}_{i}" for name in _LEVELED for i in range(N_LEVELS)]
    + [
        "TMP2m",
        "Q2m",
        "UGRD10m",
        "VGRD10m",
        "LHTFLsfc",
        "SHTFLsfc",
        "PRATEsfc",
        "ULWRFsfc",
        "ULWRFtoa",
        "DLWRFsfc",
        "DSWRFsfc",
        "USWRFsfc",
        "USWRFtoa",
        "tendency_of_total_water_path_due_to_advection",
        "TMP850",
        "h500",
        "total_frozen_precipitation_rate",
        "PRMSL",
        "eastward_surface_wind_stress",
        "northward_surface_wind_stress",
    ]
)
ATMOS_EXOGENOUS_NAMES = [
    "carbon_dioxide",
    "land_fraction",
    "lake_fraction",
    "ocean_fraction",
    "sea_ice_fraction",
    "DSWRFtoa",
    "HGTsfc",
]
ATMOS_IN_NAMES = (
    ATMOS_EXOGENOUS_NAMES
    + ["PRESsfc", "surface_temperature"]
    + [f"{name}_{i}" for name in _LEVELED for i in range(N_LEVELS)]
)

ATMOS_TO_OCEAN_NAMES = [
    "DLWRFsfc",
    "DSWRFsfc",
    "ULWRFsfc",
    "USWRFsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "PRATEsfc",
    "eastward_surface_wind_stress",
    "northward_surface_wind_stress",
    "total_frozen_precipitation_rate",
]
OCEAN_PROGNOSTIC_NAMES = (
    ["sst", "zos"]
    + [f"so_{i}" for i in range(N_OCEAN_LEVELS)]
    + [f"thetao_{i}" for i in range(N_OCEAN_LEVELS)]
)
OCEAN_IN_NAMES = (
    ["land_fraction", "sea_surface_fraction", "deptho", "hfgeou"]
    + ATMOS_TO_OCEAN_NAMES
    + OCEAN_PROGNOSTIC_NAMES
)
OCEAN_OUT_NAMES = OCEAN_PROGNOSTIC_NAMES + [
    "hfds_total_area",
    "tauuo",
    "tauvo",
    "wfo",
]


class _ChannelProjection(torch.nn.Module):
    """Stand-in for a trained network: maps n_in channels to n_out, cheaply.

    Peak memory here is set by the data plumbing, not by the architecture:
    validation runs under torch.no_grad(), so a real network's activations are
    per-step transients rather than anything that scales with window depth.
    """

    def __init__(self, n_out: int):
        super().__init__()
        self.n_out = n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_in = x.shape[1]
        if n_in >= self.n_out:
            return x[:, : self.n_out] + 1.0
        repeats = -(-self.n_out // n_in)
        return x.repeat(1, repeats, 1, 1)[:, : self.n_out] + 1.0


def _horizontal_coordinates(n_lat: int, n_lon: int) -> LatLonCoordinates:
    return LatLonCoordinates(
        lat=torch.linspace(-89.5, 89.5, n_lat), lon=torch.linspace(0, 360, n_lon)
    )


def _dataset_info(n_lat: int, n_lon: int) -> CoupledDatasetInfo:
    horizontal = _horizontal_coordinates(n_lat, n_lon)
    n_interfaces = N_LEVELS + 1
    return CoupledDatasetInfo(
        ocean=DatasetInfo(
            horizontal_coordinates=horizontal,
            vertical_coordinate=DepthCoordinate(
                torch.arange(N_OCEAN_LEVELS + 1, dtype=torch.float32),
                torch.ones(n_lat, n_lon, N_OCEAN_LEVELS),
            ).to(fme.get_device()),
            spatial_mask_provider=SpatialMaskProvider(),
            timestep=datetime.timedelta(days=5),
        ),
        atmosphere=DatasetInfo(
            horizontal_coordinates=horizontal,
            vertical_coordinate=HybridSigmaPressureCoordinate(
                torch.arange(n_interfaces, dtype=torch.float32),
                torch.arange(n_interfaces, dtype=torch.float32),
            ).to(fme.get_device()),
            spatial_mask_provider=SpatialMaskProvider(),
            timestep=datetime.timedelta(hours=6),
        ),
    )


def build_train_stepper(n_lat: int, n_lon: int, n_coupled_steps: int, n_ensemble: int):
    next_step_forcing_names = list(set(ATMOS_OUT_NAMES) & set(OCEAN_IN_NAMES))
    stepper_config = CoupledStepperConfig(
        atmosphere=ComponentConfig(
            timedelta="6h",
            stepper=StepperConfig(
                step=StepSelector(
                    type="single_module",
                    config=dataclasses.asdict(
                        SingleModuleStepConfig(
                            builder=ModuleSelector(
                                type="prebuilt",
                                config={
                                    "module": _ChannelProjection(len(ATMOS_OUT_NAMES))
                                },
                            ),
                            in_names=ATMOS_IN_NAMES,
                            out_names=ATMOS_OUT_NAMES,
                            normalization=trivial_network_and_loss_normalization(
                                set(ATMOS_IN_NAMES + ATMOS_OUT_NAMES)
                            ),
                            ocean=OceanConfig(
                                surface_temperature_name="surface_temperature",
                                ocean_fraction_name="ocean_fraction",
                            ),
                        )
                    ),
                )
            ),
        ),
        ocean=ComponentConfig(
            timedelta="5D",
            stepper=StepperConfig(
                step=StepSelector(
                    type="single_module",
                    config=dataclasses.asdict(
                        SingleModuleStepConfig(
                            builder=ModuleSelector(
                                type="prebuilt",
                                config={
                                    "module": _ChannelProjection(len(OCEAN_OUT_NAMES))
                                },
                            ),
                            in_names=OCEAN_IN_NAMES,
                            out_names=OCEAN_OUT_NAMES,
                            next_step_forcing_names=next_step_forcing_names,
                            normalization=trivial_network_and_loss_normalization(
                                set(OCEAN_IN_NAMES + OCEAN_OUT_NAMES)
                            ),
                            corrector=CorrectorSelector("ocean_corrector", {}),
                        )
                    ),
                )
            ),
        ),
        sst_name="sst",
    )
    assert stepper_config.n_inner_steps == N_INNER_STEPS
    train_config = CoupledTrainStepperConfig(
        n_coupled_steps=n_coupled_steps,
        n_ensemble=n_ensemble,
        ocean=ComponentTrainingConfig(
            loss=StepLossConfig(type="MSE"),
            n_steps=n_coupled_steps,
            loss_weight=1.0,
        ),
        atmosphere=ComponentTrainingConfig(
            loss=StepLossConfig(type="MSE"), n_steps=0, loss_weight=0.0
        ),
    )
    return train_config.get_train_stepper(stepper_config, _dataset_info(n_lat, n_lon))


def build_batch(
    n_lat: int, n_lon: int, n_samples: int, n_coupled_steps: int
) -> CoupledBatchData:
    device = fme.get_device()
    # mirror the loader: each realm's store carries the names the other realm
    # does not produce.
    atmos_names = sorted(set(ATMOS_IN_NAMES + ATMOS_OUT_NAMES) - set(OCEAN_OUT_NAMES))
    ocean_names = sorted(set(OCEAN_IN_NAMES + OCEAN_OUT_NAMES) - set(ATMOS_OUT_NAMES))

    def batch_data(names: list[str], n_time: int) -> BatchData:
        return BatchData.new_on_device(
            data={
                name: torch.rand(n_samples, n_time, n_lat, n_lon, device=device)
                for name in names
            },
            time=xr.DataArray(np.zeros((n_samples, n_time)), dims=["sample", "time"]),
            horizontal_dims=["lat", "lon"],
        )

    return CoupledBatchData(
        ocean_data=batch_data(ocean_names, n_coupled_steps + 1),
        atmosphere_data=batch_data(atmos_names, n_coupled_steps * N_INNER_STEPS + 1),
    )


def measure(
    n_lat: int, n_lon: int, n_samples: int, n_ensemble: int, caps: list[int]
) -> None:
    peaks: list[tuple[int, int]] = []
    for cap in caps:
        train_stepper = build_train_stepper(n_lat, n_lon, cap, n_ensemble)
        batch = build_batch(n_lat, n_lon, n_samples, cap)
        train_stepper.set_eval()
        train_stepper.seed_eval(seed=0)
        aggregator = OneStepAggregator(
            dataset_info=_dataset_info(n_lat, n_lon), save_diagnostics=False
        )
        with PeakTensorMemory() as memory:
            memory.track(dict(batch.ocean_data.data), dict(batch.atmosphere_data.data))
            with torch.no_grad():
                stepped = train_stepper.train_on_batch(
                    batch,
                    optimization=NullOptimization(),
                    compute_derived_variables=True,
                    evaluate_all_steps=False,
                )
                aggregator.record_batch(stepped)
            peak = memory.peak
            del stepped
        n_atmos_steps = cap * N_INNER_STEPS
        peaks.append((n_atmos_steps, peak))
        print(
            f"n_coupled_steps {cap:3d}  atmosphere timesteps {n_atmos_steps:4d}  "
            f"peak {peak / 1024**3:7.3f} GiB",
            flush=True,
        )
        del train_stepper, batch, aggregator
    if len(peaks) >= 2:
        (first_steps, first_peak), (last_steps, last_peak) = peaks[0], peaks[-1]
        slope = (last_peak - first_peak) / (last_steps - first_steps)
        print(
            f"\n{slope / 1024**2:.3f} MiB per atmosphere timestep at "
            f"{n_lat}x{n_lon}, {n_samples} samples, n_ensemble={n_ensemble}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-lat", type=int, default=24)
    parser.add_argument("--n-lon", type=int, default=48)
    parser.add_argument("--n-samples", type=int, default=2, help="samples per device")
    parser.add_argument("--n-ensemble", type=int, default=2)
    parser.add_argument("--caps", type=int, nargs="+", default=[4, 12, 16, 20])
    args = parser.parse_args()
    measure(args.n_lat, args.n_lon, args.n_samples, args.n_ensemble, args.caps)


if __name__ == "__main__":
    main()
