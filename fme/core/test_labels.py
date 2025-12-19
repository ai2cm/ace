import pytest
import torch

from fme.core.device import get_device
from fme.core.labels import BatchLabels, InvalidLabelError, LabelEncoding


def test_label_encoder_encode():
    encoder = LabelEncoding(["a", "b", "c"])
    encoded = encoder.encode([{"a", "b"}, {"a", "c"}])
    assert encoded.names == ["a", "b", "c"]
    assert encoded.tensor.device == get_device()
    assert encoded.tensor.tolist() == [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    assert encoded.tensor.dtype == torch.float32


def test_label_encoder_encode_non_sorted():
    encoder = LabelEncoding(["c", "b", "a"])
    encoded = encoder.encode([{"a", "b"}, {"a", "c"}])
    assert encoded.names == ["c", "b", "a"]
    assert encoded.tensor.device == get_device()
    assert encoded.tensor.tolist() == [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
    assert encoded.tensor.dtype == torch.float32


def test_label_encoder_invalid_labels():
    encoder = LabelEncoding(["a"])
    with pytest.raises(InvalidLabelError):
        encoder.encode([{"a", "b"}])


def test_label_encoder_conform_to_encoding_with_more_labels():
    labels = BatchLabels(torch.ones(2, 3), ["a", "b", "c"]).to(get_device())
    encoding = LabelEncoding(["a", "b", "c", "d"])
    new_labels = labels.conform_to_encoding(encoding)
    assert new_labels.names == encoding.names
    assert new_labels.tensor.tolist() == [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]]
    assert new_labels.tensor.device == labels.tensor.device


def test_label_encoder_conform_to_encoding_with_less_labels():
    labels = BatchLabels(torch.ones(2, 4), ["a", "b", "c", "d"])
    encoding = LabelEncoding(["a", "b", "c"])
    new_labels = labels.conform_to_encoding(encoding)
    assert new_labels.names == encoding.names
    assert new_labels.tensor.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_label_encoder_conform_to_encoding_with_different_order():
    labels = BatchLabels(torch.zeros(2, 3), ["a", "b", "c"])
    labels.tensor[:, 0] = 0
    labels.tensor[:, 1] = 1
    labels.tensor[:, 2] = 2
    encoding = LabelEncoding(["c", "b", "a"])
    new_labels = labels.conform_to_encoding(encoding)
    assert new_labels.names == encoding.names
    assert new_labels.tensor.tolist() == [[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]]


def test_label_encoder_add_to_empty_labels():
    labels = BatchLabels(torch.zeros(2, 0), [])
    encoding = LabelEncoding(["a", "b", "c"])
    new_labels = labels.conform_to_encoding(encoding)
    assert new_labels.names == encoding.names
    assert new_labels.tensor.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_label_encoder_remove_all_labels():
    labels = BatchLabels(torch.ones(2, 3), ["a", "b", "c"])
    encoding = LabelEncoding([])
    new_labels = labels.conform_to_encoding(encoding)
    assert new_labels.names == encoding.names
    assert new_labels.tensor.shape == (2, 0)


@pytest.mark.parametrize(
    "labels1, labels2, expected",
    [
        (
            BatchLabels(torch.tensor([[1.0, 0.0]]), ["a", "b"]),
            BatchLabels(torch.tensor([[1.0, 0.0]]), ["a", "b"]),
            True,
        ),
        (
            BatchLabels(torch.tensor([[1.0, 0.0]]), ["a", "b"]),
            BatchLabels(torch.tensor([[0.0, 1.0]]), ["a", "b"]),
            False,
        ),
        (
            BatchLabels(torch.tensor([[1.0, 0.0]]), ["a", "b"]),
            BatchLabels(torch.tensor([[1.0, 0.0]]), ["b", "a"]),
            False,
        ),
        (
            BatchLabels(torch.tensor([[1.0, 0.0]]), ["a", "b"]),
            BatchLabels(torch.tensor([[0.0, 1.0]]), ["b", "a"]),
            False,  # names must be equal, even if data meaning is the same
        ),
        (BatchLabels(torch.tensor([[1.0, 0.0]]), ["a", "b"]), None, False),
    ],
)
def test_labels_equality(
    labels1: BatchLabels, labels2: BatchLabels | None, expected: bool
):
    assert (labels1 == labels2) == expected
