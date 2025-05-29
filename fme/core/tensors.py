from typing import Any

import torch

from .typing_ import EnsembleTensorDict, TensorDict, TensorMapping


def assert_dict_allclose(a: dict[str, Any], b: dict[str, Any]):
    """
    Check if two (possibly nested) dictionaries which may contain tensors are close.

    Non-tensor values are checked for equality.
    """
    if a.keys() != b.keys():
        raise AssertionError(f"Keys do not match, got {a.keys()} and {b.keys()}")
    for k in a.keys():
        if isinstance(a[k], torch.Tensor) and isinstance(b[k], torch.Tensor):
            torch.testing.assert_close(
                a[k], b[k], msg=f"Tensors for key {k} are not close"
            )
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            assert_dict_allclose(a[k], b[k])
        elif a[k] != b[k]:
            raise AssertionError(f"Values for key {k} are not equal")


def add_ensemble_dim(d: TensorMapping, repeats: int = 1) -> EnsembleTensorDict:
    """
    Add an explicit ensemble dimension to a tensor dict.

    Args:
        d: The tensor dict to add the ensemble dimension to.
        repeats: The number of ensemble members. If greater than 1, the ensemble
            will consist of repeated copies of the input tensor.

    Returns:
        The tensor dict with an explicit ensemble dimension.
    """
    return EnsembleTensorDict(
        {
            k: v[:, None, ...].repeat(1, repeats, *([1] * (v.ndim - 1)))
            for k, v in d.items()
        }
    )


def fold_sized_ensemble_dim(d: EnsembleTensorDict, n_ensemble: int) -> TensorDict:
    """
    Take a tensor dict with an explicit ensemble dimension and fold it into a
    batch/sample dimension.

    If the sample dimension is 1, it may be broadcasted to a larger n_ensemble
    count before folding.

    Args:
        d: The tensor dict to fold.
        n_ensemble: The number of ensemble members.

    Returns:
        The folded tensor dict.
    """
    for v in d.values():
        input_n_ensemble = v.shape[1]
        break
    else:
        return d  # empty data
    if input_n_ensemble == 1 and n_ensemble != 1:
        d = EnsembleTensorDict(
            {k: v.repeat(1, n_ensemble, *([1] * (v.ndim - 2))) for k, v in d.items()}
        )
    elif input_n_ensemble != n_ensemble:
        raise ValueError(
            f"Input n_ensemble ({input_n_ensemble}) does not match "
            f"n_ensemble ({n_ensemble}) in given data"
        )
    try:
        reshaped = {
            k: v.reshape(v.shape[0] * n_ensemble, *v.shape[2:]) for k, v in d.items()
        }
    except RuntimeError as e:
        if "is invalid for input of size" in str(e):
            raise ValueError(
                f"some values in d have invalid ensemble member counts, "
                f"should be {n_ensemble}, got: "
                f"{', '.join(f'{k}: {str(v.shape[1])}' for k, v in d.items())}"
            ) from e
        raise
    return EnsembleTensorDict(reshaped)


def fold_ensemble_dim(d: EnsembleTensorDict) -> tuple[TensorDict, int]:
    """
    Take a tensor dict with an explicit ensemble dimension and fold it into a
    batch/sample dimension.

    Args:
        d: The ensemble tensor dict to fold.

    Returns:
        A tuple of the folded tensor dict and the ensemble dimension length.
    """
    for v in d.values():
        n_ensemble = v.shape[1]
        break
    else:
        raise ValueError("input is empty, ensemble member count is not defined")
    return fold_sized_ensemble_dim(d, n_ensemble), n_ensemble


def repeat_interleave_batch_dim(data: TensorMapping, repeats: int) -> TensorDict:
    """
    Repeats each sample in the batch a given number of times using repeat_interleave.

    If you were to "unfold" the result of this operation with the repeat count as
    the ensemble dimension, you would get an ensemble tensor dict with the data
    repeated across the ensemble dimension.
    """
    if repeats == 1:
        return dict(data)  # no-op
    return {k: v.repeat_interleave(repeats, dim=0) for k, v in data.items()}


def unfold_ensemble_dim(d: TensorDict, n_ensemble: int) -> EnsembleTensorDict:
    """
    Take a tensor dict with a batch/sample dimension and unfold it into an
    explicit ensemble dimension.

    If an ensemble tensor dict is folded, this will unfold it back to the
    original ensemble dimension.

    Args:
        d: The tensor dict to unfold.
        n_ensemble: The number of ensemble members.

    Returns:
        The unfolded ensemble tensor dict.
    """
    return EnsembleTensorDict(
        {
            k: v.reshape(v.shape[0] // n_ensemble, n_ensemble, *v.shape[1:])
            for k, v in d.items()
        }
    )
