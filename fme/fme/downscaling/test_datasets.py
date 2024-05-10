import pytest
import torch
import xarray as xr

from fme.core.testing.fv3gfs_data import DimSizes, FV3GFSData
from fme.downscaling.requirements import DataRequirements

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
    assert result.fine["x"].shape == torch.Size((n_examples,) + shape_a)
    assert result.coarse["x"].shape == torch.Size((n_examples,) + shape_b)
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
        "fine_n_examples, fine_var_names, fine_time,"
        "coarse_n_examples, coarse_var_names, coarse_time"
    ),
    [
        pytest.param(1, ["x"], 0, 2, ["x"], 0, id="num_samples"),
        pytest.param(1, ["x"], 0, 1, ["x", "y"], 0, id="var_names"),
        pytest.param(1, ["x"], 0, 1, ["x"], 1, id="var_names"),
    ],
)
def test_downscaling_dataset_validate(
    fine_n_examples,
    fine_var_names,
    fine_time,
    coarse_n_examples,
    coarse_var_names,
    coarse_time,
):
    all_data = []
    for n_examples, var_names, times in [
        (fine_n_examples, fine_var_names, fine_time),
        (coarse_n_examples, coarse_var_names, coarse_time),
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

    fine_data, coarse_data = all_data
    with pytest.raises(AssertionError):
        PairedDataset(fine_data, coarse_data)


@pytest.mark.parametrize(
    "path_extension, data_type",
    [
        pytest.param("", "ensemble_xarray", id="ensemble"),
        pytest.param("data", "xarray", id="xarray"),
    ],
)
def test_dataloader_build(tmp_path, path_extension, data_type):
    """Integration test that creates test data on disk."""

    fine_path, coarse_path = tmp_path / "fine", tmp_path / "coarse"
    fine_path.mkdir()
    coarse_path.mkdir()

    all_names = ["x", "y", "HGTsfc"]
    num_timesteps = 10
    fine_shape, coarse_shape = (64, 32), (32, 16)
    num_vertical_levels = 1
    fine_data, coarse_data = [
        FV3GFSData(
            path=path,
            names=all_names,
            dim_sizes=DimSizes(num_timesteps, h, w, num_vertical_levels),
            time_varying_values=[float(i) for i in range(num_timesteps)],
        )
        for (path, (h, w)) in zip([fine_path, coarse_path], [fine_shape, coarse_shape])
    ]
    batch_size = 2
    config = DataLoaderConfig(
        str(fine_data.path / path_extension),
        str(coarse_data.path / path_extension),
        data_type,
        batch_size,
        1,
    )

    loader = config.build(
        True, DataRequirements(all_names, 1, use_fine_topography=False), None
    )
    assert len(loader.loader) == num_timesteps // batch_size
    assert len(loader.loader.dataset) == num_timesteps  # type: ignore
    assert loader.downscale_factor == 2
    batch = next(iter(loader.loader))
    for var_name in all_names:
        assert batch.fine[var_name].shape == (batch_size, *fine_shape)
        assert batch.coarse[var_name].shape == (batch_size, *coarse_shape)
        assert batch.times.shape == (batch_size,)
