import datetime
import pathlib

import cftime
import numpy as np
import torch
import xarray as xr

from fme.ace.aggregator.inference.test_evaluator import get_zero_time
from fme.ace.data_loading.batch_data import PairedData
from fme.ace.testing import DimSizes, MonthlyReferenceData
from fme.core.coordinates import (
    DepthCoordinate,
    DimSize,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
)
from fme.core.device import get_device
from fme.coupled.aggregator import InferenceEvaluatorAggregator
from fme.coupled.data_loading.batch_data import CoupledPairedData
from fme.coupled.data_loading.data_typing import CoupledVerticalCoordinate

TIMESTEP = datetime.timedelta(days=5)


def test_inference_logs_labels_exist(tmpdir):
    n_sample = 2
    # include initial condition
    n_time = 73 * 2 + 1
    nx = 2
    ny = 2
    nz = 3

    vertical_coordinate = CoupledVerticalCoordinate(
        ocean=DepthCoordinate(torch.arange(1)),
        atmosphere=HybridSigmaPressureCoordinate(
            ak=torch.arange(nz + 1), bk=torch.arange(nz + 1)
        ),
    )
    horizontal_coordinates = LatLonCoordinates(
        lon=torch.arange(nx),
        lat=torch.arange(ny),
        loaded_lon_name="lon",
        loaded_lat_name="lat",
    )
    initial_time = get_zero_time(shape=[n_sample, 0], dims=["sample", "time"])

    reference_time_means = xr.Dataset(
        {
            "ocean_var": xr.DataArray(
                np.random.randn(ny, nx),
                dims=["lat", "lon"],
            ),
            "atmos_var": xr.DataArray(
                np.random.randn(ny, nx),
                dims=["lat", "lon"],
            ),
        }
    )

    monthly_reference_data = MonthlyReferenceData(
        path=pathlib.Path(tmpdir),
        names=["ocean_var", "atmos_var"],
        dim_sizes=DimSizes(
            n_time=48,
            horizontal=[DimSize("lat", nx), DimSize("lon", ny)],
            nz_interface=nz + 1,
        ),
        n_ensemble=3,
    )
    monthly_ds = xr.open_dataset(monthly_reference_data.data_filename)

    agg = InferenceEvaluatorAggregator(
        vertical_coordinate=vertical_coordinate,
        horizontal_coordinates=horizontal_coordinates,
        ocean_timestep=TIMESTEP,
        atmosphere_timestep=TIMESTEP,
        n_timesteps_ocean=n_time,
        n_timesteps_atmosphere=n_time,
        initial_time=initial_time,
        ocean_normalize=lambda x: dict(x),
        atmosphere_normalize=lambda x: dict(x),
        log_video=True,
        log_zonal_mean_images=True,
        time_mean_reference_data=reference_time_means,
        monthly_reference_data=monthly_ds,
    )

    time = xr.DataArray(
        [
            [
                (cftime.DatetimeProlepticGregorian(2000, 1, 1) + i * TIMESTEP)
                for i in range(n_time)
            ]
            for _ in range(n_sample)
        ],
        dims=["sample", "time"],
    )

    coupled_data = CoupledPairedData(
        ocean_data=PairedData(
            prediction={
                "ocean_var": torch.randn(n_sample, n_time, nx, ny, device=get_device())
            },
            target={
                "ocean_var": torch.randn(n_sample, n_time, nx, ny, device=get_device())
            },
            time=time,
        ),
        atmosphere_data=PairedData(
            prediction={
                "atmos_var": torch.randn(n_sample, n_time, nx, ny, device=get_device())
            },
            target={
                "atmos_var": torch.randn(n_sample, n_time, nx, ny, device=get_device())
            },
            time=time,
        ),
    )

    logs = agg.record_batch(data=coupled_data)
    assert len(logs) == 0, "TODO: test for correct keys here"

    summary_logs = agg.get_summary_logs()
    expected_keys = [
        # ocean-specific keys
        "annual/ocean_var",
        "annual/r2/ocean_var_gen",
        "annual/r2/ocean_var_target",
        "mean_step_20/weighted_rmse/ocean_var",
        "mean_step_20/weighted_bias/ocean_var",
        "mean_step_20/weighted_grad_mag_percent_diff/ocean_var",
        "spherical_power_spectrum/ocean_var",
        "time_mean/rmse/ocean_var",
        "time_mean/bias/ocean_var",
        "time_mean/bias_map/ocean_var",
        "time_mean/gen_map/ocean_var",
        "time_mean_norm/rmse/ocean_var",
        "time_mean_norm/gen_map/ocean_var",
        "time_mean_norm/rmse/ocean_channel_mean",
        "time_mean/ref_bias/ocean_var",
        "time_mean/ref_rmse/ocean_var",
        "time_mean/ref_bias_map/ocean_var",
        "zonal_mean/error/ocean_var",
        "zonal_mean/gen/ocean_var",
        "video/ocean_var",
        # atmosphere-specific keys
        "annual/atmos_var",
        "annual/r2/atmos_var_gen",
        "annual/r2/atmos_var_target",
        "mean_step_20/weighted_rmse/atmos_var",
        "mean_step_20/weighted_bias/atmos_var",
        "mean_step_20/weighted_grad_mag_percent_diff/atmos_var",
        "spherical_power_spectrum/atmos_var",
        "time_mean/rmse/atmos_var",
        "time_mean/bias/atmos_var",
        "time_mean/bias_map/atmos_var",
        "time_mean/gen_map/atmos_var",
        "time_mean_norm/rmse/atmos_var",
        "time_mean_norm/gen_map/atmos_var",
        "time_mean_norm/rmse/atmosphere_channel_mean",
        "time_mean/ref_bias/atmos_var",
        "time_mean/ref_rmse/atmos_var",
        "time_mean/ref_bias_map/atmos_var",
        "zonal_mean/error/atmos_var",
        "zonal_mean/gen/atmos_var",
        "video/atmos_var",
        # combined key
        "time_mean_norm/rmse/channel_mean",
    ]

    # Check that all expected keys exist in the logs and no extra keys are present
    for key in expected_keys:
        assert key in summary_logs, key
    assert len(summary_logs) == len(
        expected_keys
    ), f"unexpected keys: {set(summary_logs).difference(expected_keys)}"
