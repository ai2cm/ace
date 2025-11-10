import pytest
import torch

from fme.core.device import get_device
from fme.core.labels import InvalidLabelError, LabelEncoder


def test_label_encoder_encode():
    encoder = LabelEncoder({"a", "b", "c"})
    encoded = encoder.encode([{"a", "b"}, {"a", "c"}])
    assert encoded.device == get_device()
    assert encoded.tolist() == [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    assert encoded.dtype == torch.float32


def test_label_encoder_decode():
    encoder = LabelEncoder({"a", "b", "c"})
    decoded = encoder.decode(torch.tensor([[1, 1, 0], [1, 0, 1]], device=get_device()))
    assert decoded == [{"a", "b"}, {"a", "c"}]


@pytest.mark.parametrize(
    "batch_labels, all_labels",
    [
        ([{"a", "b"}, {"a", "c"}], {"a", "b", "c"}),
        ([{"a", "b"}, {"a", "c"}, {"b", "c"}], {"a", "b", "c"}),
        ([{"a", "b"}, {"a", "c"}, {"b", "c"}, {"a", "b", "c"}], {"a", "b", "c"}),
    ],
)
def test_label_encoder_roundtrip(batch_labels: list[set[str]], all_labels: set[str]):
    encoder = LabelEncoder(all_labels)
    encoded = encoder.encode(batch_labels)
    decoded = encoder.decode(encoded)
    assert decoded == batch_labels


def test_label_encoder_invalid_labels():
    encoder = LabelEncoder({"a"})
    with pytest.raises(InvalidLabelError):
        encoder.encode([{"a", "b"}])


def test_label_encoder_invalid_labels_decode():
    encoder = LabelEncoder({"a", "b"})
    with pytest.raises(InvalidLabelError):
        encoder.decode(
            torch.tensor([[1, 1, 0], [1, 0, 1], [1, 1, 1]], device=get_device())
        )
