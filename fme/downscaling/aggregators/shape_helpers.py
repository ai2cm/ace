from typing import List

from fme.core.typing_ import TensorDict, TensorMapping

from ..models import ModelOutputs


def _check_all_datasets_compatible_sample_dim(
    tensor_datasets: List[TensorMapping], sample_dim: int = 1
) -> int:
    """
    Check that all fields in all datasets have a sample dimension of either 1 or the
    same number of samples.

    Returns:
        The number of samples in the sample dimension of the datasets with > 1 samples.
    """
    sample_length = 1
    for i, data in enumerate(tensor_datasets):
        for key, value in data.items():
            if len(value.shape) != 4:
                raise ValueError(
                    f"expected data item {i} to have a sample dimension, "
                    f"has shape {value.shape} for key {key}"
                )

            current_nsample = value.shape[sample_dim]
            if current_nsample > 1:
                if sample_length == 1:
                    sample_length = current_nsample
                elif current_nsample != sample_length:
                    raise ValueError(
                        "Expected all data items to have 1 or same number of samples, "
                        f"but found {current_nsample} for item{i}, key {key}"
                        f"that conflicts with previous sample length {sample_length}."
                    )

    return sample_length


def _check_batch_dims_for_recording(
    outputs: ModelOutputs, coarse: TensorMapping, num_dims: int
) -> None:
    fields = {
        "target": outputs.target,
        "prediction": outputs.prediction,
        "coarse": coarse,
    }
    expected_dims = {
        3: "[batch, height, width]",
        4: "[batch, sample, height, width]",
    }
    for batch_member, data in fields.items():
        for variable, tensor in data.items():
            if len(tensor.shape) != num_dims:
                raise ValueError(
                    f"{batch_member} {variable} has shape {tensor.shape}, "
                    f"expected {expected_dims[num_dims]}"
                )


def _fold_sample_dim(
    tensor_datasets: List[TensorMapping], sample_dim: int = 1
) -> List[TensorDict]:
    """
    Takes data with a [batch, sample, y, x] dimension and returns a list of
    dictionaries with a [batch, y, x] dimension.

    This is used to pass data to aggregators written to expect a single
    batch dimension, or which don't care whether two values come
    from the same input or not.

    Args:
        tensor_datasets: List of tensor mappings with values of shape
            [batch, sample, ...].
        sample_dim: The dimension number of the sample dimension.

    Returns:
        List of dictionaries with values of shape [batch, ...].
    """
    n_samples = _check_all_datasets_compatible_sample_dim(
        tensor_datasets, sample_dim=sample_dim
    )
    batch_size = None

    new_datasets = []
    for dataset in tensor_datasets:
        new_dataset = {}
        for key, field in dataset.items():
            batch_size = field.shape[0]
            if field.shape[sample_dim] == 1:
                field = field.repeat_interleave(n_samples, dim=sample_dim)
            folded = field.reshape(batch_size * n_samples, *field.shape[-2:])
            new_dataset[key] = folded
        new_datasets.append(new_dataset)

    return new_datasets
