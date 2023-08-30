from typing import NamedTuple

import numpy as np
import pytest
import torch
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
        }
        return data

    def test_append_batch(
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
        )
        writer.append_batch(
            sample_target_data, sample_prediction_data, start_timestep=0, start_sample=0
        )
        writer.append_batch(
            sample_target_data, sample_prediction_data, start_timestep=0, start_sample=2
        )
        writer.append_batch(
            sample_target_data, sample_prediction_data, start_timestep=2, start_sample=0
        )
        writer.append_batch(
            sample_target_data, sample_prediction_data, start_timestep=2, start_sample=2
        )

        # Open the file again and check the data
        dataset = Dataset(tmp_path / "autoregressive_predictions.nc", "r")
        assert set(dataset.variables.keys()) == set(sample_target_data.keys()).union(
            {"source", "lat", "lon"}
        )
        assert (
            dataset.variables["source"][:] == np.array(["target", "prediction"])
        ).all()

        for var_name in sample_target_data.keys():
            var_data = dataset.variables[var_name][:]
            assert var_data.shape == (2, 4, 5, 4, 5)  # source, sample, time, lat, lon
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
