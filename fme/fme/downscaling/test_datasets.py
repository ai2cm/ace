import pytest
import torch
import xarray as xr

from fme.core.data_loading.requirements import DataRequirements
from fme.core.testing.fv3gfs_data import DimSizes, FV3GFSData

from .datasets import BatchData, DataLoaderConfig, PairedDataset


def random_named_tensor(var_names, shape):
    return {var_name: torch.rand(shape) for var_name in var_names}


def data_array(time, dims=["sample"]):
    return xr.DataArray([time], dims=dims)


@pytest.mark.parametrize("n_examples", [1, 2])
def test_downscaling_example_from_sample_tuples(n_examples):
    shape_a = (1, 1, 8, 16)  # C x T x H x W
    shape_b = (1, 1, 4, 8)
    n_examples = 1
    sample_tuples = [
        (
            random_named_tensor(["x"], shape_a),
            random_named_tensor(["x"], shape_b),
            data_array(0),
        )
        for _ in range(n_examples)
    ]
    result = BatchData.from_sample_tuples(sample_tuples)
    assert result.highres["x"].shape == torch.Size((n_examples,) + shape_a)
    assert result.lowres["x"].shape == torch.Size((n_examples,) + shape_b)
    assert len(result.times) == n_examples


@pytest.mark.parametrize("n_examples,index", [(1, 0), (2, 1)])
def test_downscaling_dataset(n_examples, index):
    """Tests getitem and len."""
    data = [
        (
            random_named_tensor(["x"], (1, 1, 8, 16)),
            data_array(0),
        )
        for _ in range(n_examples)
    ]

    dataset = PairedDataset(data, data)
    data1, data2, dataset_times = dataset[index]
    assert torch.equal(data1["x"], data2["x"])
    assert all(dataset_times == data[index][1])
    assert len(dataset) == n_examples


@pytest.mark.parametrize(
    (
        "highres_n_examples, highres_var_names, highres_time,"
        "lowres_n_examples, lowres_var_names, lowres_time"
    ),
    [
        pytest.param(1, ["x"], 0, 2, ["x"], 0, id="num_samples"),
        pytest.param(1, ["x"], 0, 1, ["x", "y"], 0, id="var_names"),
        pytest.param(1, ["x"], 0, 1, ["x"], 1, id="var_names"),
    ],
)
def test_downscaling_dataset_validate(
    highres_n_examples,
    highres_var_names,
    highres_time,
    lowres_n_examples,
    lowres_var_names,
    lowres_time,
):
    all_data = []
    for n_examples, var_names, times in [
        (highres_n_examples, highres_var_names, highres_time),
        (lowres_n_examples, lowres_var_names, lowres_time),
    ]:
        all_data.append(
            [
                (
                    random_named_tensor(var_names, (1, 1, 8, 16)),
                    data_array(times),
                )
                for _ in range(n_examples)
            ]
        )

    highres_data, lowres_data = all_data
    with pytest.raises(AssertionError):
        PairedDataset(highres_data, lowres_data)


@pytest.mark.parametrize(
    "path_extension, data_type",
    [
        pytest.param("", "ensemble_xarray", id="ensemble"),
        pytest.param("data", "xarray", id="xarray"),
    ],
)
def test_dataloader_build(tmp_path, path_extension, data_type):
    """Integration test that creates test data on disk."""

    highres_path, lowres_path = tmp_path / "highres", tmp_path / "lowres"
    highres_path.mkdir()
    lowres_path.mkdir()

    all_names = ["x", "y"]
    num_timesteps = 10
    highres_shape, lowres_shape = (64, 32), (32, 16)
    num_vertical_levels = 1
    highres_data, lowres_data = [
        FV3GFSData(
            path=path,
            names=all_names,
            dim_sizes=DimSizes(num_timesteps, h, w, num_vertical_levels),
            time_varying_values=[float(i) for i in range(num_timesteps)],
        )
        for (path, (h, w)) in zip(
            [highres_path, lowres_path], [highres_shape, lowres_shape]
        )
    ]
    batch_size = 2
    config = DataLoaderConfig(
        str(highres_data.path / path_extension),
        str(lowres_data.path / path_extension),
        data_type,
        batch_size,
        1,
    )

    loader = config.build(True, DataRequirements(all_names, 1), None)
    assert len(loader.loader) == num_timesteps // batch_size
    assert len(loader.loader.dataset) == num_timesteps  # type: ignore
    assert loader.downscale_factor == 2
    batch = next(iter(loader.loader))
    for var_name in all_names:
        assert batch.highres[var_name].shape == (batch_size, 1, *highres_shape)
        assert batch.lowres[var_name].shape == (batch_size, 1, *lowres_shape)
