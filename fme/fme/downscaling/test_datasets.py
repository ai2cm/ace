from typing import Dict, Sequence, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
import xarray as xr

import fme.core.data_loading.params
from fme.core.data_loading.requirements import DataRequirements
from fme.core.typing_ import TensorMapping

from .datasets import BatchData, DataLoaderParams, PairedDataset


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


def mock_xarray_dataset_instance(data: Sequence[Tuple[TensorMapping, xr.DataArray]]):
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.__getitem__.side_effect = lambda idx: data[idx]
    mock_dataset_instance.__len__.return_value = len(data)
    return mock_dataset_instance


def mock_xarray_dataset_selector(paths_to_mocks: Dict[str, MagicMock]):
    def select_mock_instance(
        params: fme.core.data_loading.params.DataLoaderParams,
        requirements: DataRequirements,
    ):
        del requirements  # unused
        try:
            return paths_to_mocks[params.data_path]
        except KeyError:
            raise ValueError(f"Unexpected path {params.data_path}")

    return select_mock_instance


@patch("fme.downscaling.datasets.XarrayDataset")
@pytest.mark.parametrize("n_examples,batch_size", [(1, 1), (4, 4), (4, 2), (5, 2)])
def test_downscaling_dataloader(mock_xarray_dataset, n_examples, batch_size):
    n_examples = 2
    batch_size = 1

    highres_shape = (1, 1, 8, 16)
    lowres_shape = (1, 1, 4, 8)

    highres_data = [
        (random_named_tensor(["x"], highres_shape), data_array(0))
        for _ in range(n_examples)
    ]
    lowres_data = [
        (random_named_tensor(["x"], lowres_shape), data_array(0))
        for _ in range(n_examples)
    ]

    mock_xarray_dataset.side_effect = mock_xarray_dataset_selector(
        {
            "/path/to/nowhere/highres": mock_xarray_dataset_instance(highres_data),
            "/path/to/nowhere/lowres": mock_xarray_dataset_instance(lowres_data),
        }
    )

    params = DataLoaderParams(
        "/path/to/nowhere/highres",
        "/path/to/nowhere/lowres",
        ["x"],
        "xarray",
        batch_size,
        1,
    )
    loader = params.build(True)

    assert len(loader.loader) == n_examples // batch_size
    assert len(loader.loader.dataset) == n_examples
    batch = next(iter(loader.loader))
    assert batch.highres["x"].shape == torch.Size((batch_size,) + highres_shape)
    assert batch.lowres["x"].shape == torch.Size((batch_size,) + lowres_shape)
