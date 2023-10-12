from typing import NamedTuple

import numpy as np
import pytest
import torch
import xarray as xr
from netCDF4 import Dataset

from fme.fcn_training.inference.data_writer import DataWriter


class TestDataWriter:
    class VariableMetadata(NamedTuple):
        units: str
        long_name: str

    @pytest.fixture
    def sample_metadata(self):
        return {
            "temp": self.VariableMetadata(units="K", long_name="Temperature"),
            "humidity": self.VariableMetadata(units="%", long_name="Relative Humidity"),
        }

    @pytest.fixture
    def sample_target_data(self):
        data = {
            # sample, time, lat, lon
            "temp": torch.rand((2, 3, 4, 5)),
            "humidity": torch.rand((2, 3, 4, 5)),  # input-only variable
            "pressure": torch.rand(
                (2, 3, 4, 5)
            ),  # Extra variable for which there's no metadata
            "precipitation": torch.rand((2, 3, 4, 5)),  # optionally saved
        }
        return data

    @pytest.fixture
    def sample_prediction_data(self):
        data = {
            # sample, time, lat, lon
            "temp": torch.rand((2, 3, 4, 5)),
            "pressure": torch.rand(
                (2, 3, 4, 5)
            ),  # Extra variable for which there's no metadata
            "precipitation": torch.rand((2, 3, 4, 5)),  # optionally saved
        }
        return data

    @pytest.mark.parametrize(
        ["save_names"],
        [
            pytest.param(None, id="None"),
            pytest.param(["temp", "humidity", "pressure"], id="subset"),
        ],
    )
    def test_append_batch(
        self,
        sample_metadata,
        sample_target_data,
        sample_prediction_data,
        tmp_path,
        save_names,
    ):
        n_samples = 4
        n_timesteps = 6
        writer = DataWriter(
            str(tmp_path),
            n_samples=n_samples,
            n_timesteps=n_timesteps,
            metadata=sample_metadata,
            coords={"lat": np.arange(4), "lon": np.arange(5)},
            enable_prediction_netcdfs=True,
            enable_video_netcdfs=False,
            save_names=save_names,
        )
        writer.append_batch(
            sample_target_data, sample_prediction_data, start_timestep=0, start_sample=0
        )
        writer.append_batch(
            sample_target_data, sample_prediction_data, start_timestep=0, start_sample=2
        )
        writer.append_batch(
            sample_target_data, sample_prediction_data, start_timestep=3, start_sample=0
        )
        writer.append_batch(
            sample_target_data, sample_prediction_data, start_timestep=3, start_sample=2
        )
        writer.flush()

        # Open the file again and check the data
        dataset = Dataset(tmp_path / "autoregressive_predictions.nc", "r")
        expected_variables = (
            set(save_names)
            if save_names is not None
            else set(sample_target_data.keys())
        )
        assert set(dataset.variables.keys()) == expected_variables.union(
            {"source", "lat", "lon"}
        )
        assert (
            dataset.variables["source"][:] == np.array(["target", "prediction"])
        ).all()

        for var_name in expected_variables:
            var_data = dataset.variables[var_name][:]
            assert var_data.shape == (
                2,
                n_samples,
                n_timesteps,
                4,
                5,
            )  # source, sample, time, lat, lon
            assert not np.isnan(var_data[0, :]).any(), "unexpected NaNs in target data"
            if var_name in sample_prediction_data:
                assert not np.isnan(
                    var_data[1, :]
                ).any(), "unexpected NaNs in prediction data"

            if var_name in sample_metadata:
                assert (
                    dataset.variables[var_name].units == sample_metadata[var_name].units
                )
                assert (
                    dataset.variables[var_name].long_name
                    == sample_metadata[var_name].long_name
                )
            else:
                assert not hasattr(dataset.variables[var_name], "units")
                assert not hasattr(dataset.variables[var_name], "long_name")

        dataset.close()

        histograms = xr.open_dataset(tmp_path / "histograms.nc")
        actual_var_names = sorted([str(k) for k in histograms.keys()])
        assert len(actual_var_names) == 8
        assert "humidity" in actual_var_names
        assert histograms.data_vars["humidity"].attrs["units"] == "count"
        assert "humidity_bin_edges" in actual_var_names
        assert histograms.data_vars["humidity_bin_edges"].attrs["units"] == "%"
        counts_per_timestep = histograms["humidity"].sum(dim=["bin", "source"])
        same_count_each_timestep = np.all(
            counts_per_timestep.values == counts_per_timestep.values[0]
        )
        assert same_count_each_timestep

    def test_append_batch_past_end_of_samples(
        self, sample_metadata, sample_target_data, sample_prediction_data, tmp_path
    ):
        n_samples = 4
        writer = DataWriter(
            str(tmp_path),
            n_samples=n_samples,
            n_timesteps=3,
            metadata=sample_metadata,
            coords={"lat": np.arange(4), "lon": np.arange(5)},
            enable_prediction_netcdfs=True,
            enable_video_netcdfs=False,
            save_names=None,
        )
        writer.append_batch(
            sample_target_data, sample_prediction_data, start_timestep=0, start_sample=0
        )
        writer.append_batch(
            sample_target_data, sample_prediction_data, start_timestep=0, start_sample=2
        )
        with pytest.raises(ValueError):
            writer.append_batch(
                sample_target_data,
                sample_prediction_data,
                start_timestep=0,
                start_sample=4,
            )
