import pytest
import torch

from fme.core.device import get_device
from fme.core.labels import BatchLabels, InvalidLabelError, LabelEncoding


def test_label_encoder_encode():
    encoder = LabelEncoding(["a", "b", "c"])
    encoded = encoder.encode([{"a", "b"}, {"a", "c"}], device=get_device())
    assert encoded.names == ["a", "b", "c"]
    assert encoded.tensor.device == get_device()
    assert encoded.tensor.tolist() == [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    assert encoded.tensor.dtype == torch.float32


def test_label_encoder_encode_non_sorted():
    encoder = LabelEncoding(["c", "b", "a"])
    encoded = encoder.encode([{"a", "b"}, {"a", "c"}], device=get_device())
    assert encoded.names == ["c", "b", "a"]
    assert encoded.tensor.device == get_device()
    assert encoded.tensor.tolist() == [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
    assert encoded.tensor.dtype == torch.float32


def test_label_encoder_invalid_labels():
    encoder = LabelEncoding(["a"])
    with pytest.raises(InvalidLabelError):
        encoder.encode([{"a", "b"}], device=get_device())


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


def test_semantically_equal_identical():
    a = BatchLabels(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), ["a", "b"])
    b = BatchLabels(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), ["a", "b"])
    assert a.semantically_equal(b)


def test_semantically_equal_reordered_names_same_meaning():
    a = BatchLabels(torch.tensor([[1.0, 0.0]]), ["a", "b"])
    b = BatchLabels(torch.tensor([[0.0, 1.0]]), ["b", "a"])
    assert a.semantically_equal(b)


def test_semantically_equal_disjoint_names_zero_padded():
    # "a"-active on left vs "a"-active on right with extra zero col on the right
    a = BatchLabels(torch.tensor([[1.0]]), ["a"])
    b = BatchLabels(torch.tensor([[1.0, 0.0]]), ["a", "b"])
    assert a.semantically_equal(b)


def test_semantically_equal_different_meaning_returns_false():
    a = BatchLabels(torch.tensor([[1.0]]), ["a"])
    b = BatchLabels(torch.tensor([[1.0]]), ["b"])
    assert not a.semantically_equal(b)


def test_semantically_equal_different_n_samples_returns_false():
    a = BatchLabels(torch.tensor([[1.0]]), ["a"])
    b = BatchLabels(torch.tensor([[1.0], [1.0]]), ["a"])
    assert not a.semantically_equal(b)


def test_batchlabels_cat_empty_raises():
    with pytest.raises(ValueError, match="empty sequence"):
        BatchLabels.cat([])


def test_batchlabels_cat_single_returns_input():
    a = BatchLabels(torch.tensor([[1.0, 0.0]]), ["a", "b"])
    assert BatchLabels.cat([a]) is a


def test_batchlabels_cat_same_names_stacks_tensors():
    a = BatchLabels(torch.tensor([[1.0, 0.0]]), ["a", "b"])
    b = BatchLabels(torch.tensor([[0.0, 1.0], [1.0, 1.0]]), ["a", "b"])
    out = BatchLabels.cat([a, b])
    assert out.names == ["a", "b"]
    assert out.tensor.tolist() == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def test_batchlabels_cat_disjoint_names_unions_with_zero_pad():
    a = BatchLabels(torch.tensor([[1.0, 2.0]]), ["a", "b"])
    b = BatchLabels(torch.tensor([[3.0, 4.0]]), ["c", "d"])
    out = BatchLabels.cat([a, b])
    assert out.names == ["a", "b", "c", "d"]
    assert out.tensor.tolist() == [
        [1.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 4.0],
    ]


def test_batchlabels_cat_overlapping_names_aligns_columns():
    a = BatchLabels(torch.tensor([[1.0, 2.0]]), ["a", "b"])
    b = BatchLabels(torch.tensor([[3.0, 4.0]]), ["b", "c"])
    out = BatchLabels.cat([a, b])
    assert out.names == ["a", "b", "c"]
    assert out.tensor.tolist() == [
        [1.0, 2.0, 0.0],
        [0.0, 3.0, 4.0],
    ]


def test_batchlabels_cat_different_orderings_aligns_columns():
    # Same name set but different orderings on inputs.
    a = BatchLabels(torch.tensor([[1.0, 2.0]]), ["a", "b"])
    b = BatchLabels(torch.tensor([[5.0, 4.0]]), ["b", "a"])
    out = BatchLabels.cat([a, b])
    assert out.names == ["a", "b"]
    assert out.tensor.tolist() == [
        [1.0, 2.0],
        [4.0, 5.0],
    ]


def test_batchlabels_cat_three_inputs_with_partial_overlap():
    a = BatchLabels(torch.tensor([[1.0, 0.0]]), ["a", "b"])
    b = BatchLabels(torch.tensor([[1.0, 0.0]]), ["b", "c"])
    c = BatchLabels(torch.tensor([[1.0]]), ["d"])
    out = BatchLabels.cat([a, b, c])
    assert out.names == ["a", "b", "c", "d"]
    assert out.tensor.tolist() == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_batchlabels_cat_does_not_duplicate_names():
    # Whether overlap is total, partial, or zero, the result has each name once.
    a = BatchLabels(torch.tensor([[1.0, 1.0]]), ["a", "b"])
    b = BatchLabels(torch.tensor([[1.0, 1.0]]), ["a", "b"])
    c = BatchLabels(torch.tensor([[1.0, 1.0]]), ["b", "c"])
    out = BatchLabels.cat([a, b, c])
    assert out.names == ["a", "b", "c"]
    assert len(set(out.names)) == len(out.names)


def test_batchlabels_cat_preserves_device():
    device = get_device()
    a = BatchLabels(torch.tensor([[1.0, 0.0]]), ["a", "b"]).to(device)
    b = BatchLabels(torch.tensor([[0.0, 1.0]]), ["b", "c"]).to(device)
    out = BatchLabels.cat([a, b])
    assert out.tensor.device == a.tensor.device


def test_batchlabels_cat_empty_label_lists_combine_to_empty():
    a = BatchLabels(torch.zeros(2, 0), [])
    b = BatchLabels(torch.zeros(3, 0), [])
    out = BatchLabels.cat([a, b])
    assert out.names == []
    assert out.tensor.shape == (5, 0)


def test_batchlabels_cat_empty_with_non_empty_zero_pads():
    a = BatchLabels(torch.zeros(2, 0), [])
    b = BatchLabels(torch.tensor([[1.0, 1.0]]), ["a", "b"])
    out = BatchLabels.cat([a, b])
    assert out.names == ["a", "b"]
    # The empty-labels rows get zeroed across the union.
    assert out.tensor.tolist() == [
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
    ]
