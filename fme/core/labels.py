from typing import Any

import torch

from fme.core.device import get_device

BatchLabels = list[set[str]]


class InvalidLabelError(ValueError):
    """
    Raised when a label is invalid.
    """

    pass


class LabelEncoder:
    """
    Transforms labels for each batch member into a tensor of one-hot encoded labels.
    """

    def __init__(self, labels: set[str]):
        if not isinstance(labels, set):
            raise ValueError("Labels must be a set of strings")
        self._labels = sorted(list(labels))

    def encode(self, labels: list[set[str]]) -> torch.Tensor:
        """
        Encodes a list of sets of labels into a tensor of one-hot encoded labels.

        Args:
            labels: List of sets of labels, where each set contains the labels
                for a single batch member.

        Returns:
            Tensor of one-hot encoded labels, of shape (batch_size, n_labels).
        """
        for batch_labels in labels:
            if not batch_labels.issubset(self._labels):
                raise InvalidLabelError(
                    f"Invalid labels: at least one of {batch_labels} "
                    f"is not in {self._labels}"
                )
        return torch.tensor(
            [
                [1 if label in labels else 0 for label in self._labels]
                for labels in labels
            ],
            dtype=torch.float32,
            device=get_device(),
        )

    def decode(self, labels: torch.Tensor) -> list[set[str]]:
        """
        Decodes a tensor of one-hot encoded labels into a list of sets of labels.

        Args:
            labels: Tensor of one-hot encoded labels, of shape (batch_size, n_labels).

        Returns:
            List of sets of labels, where each set contains the labels
                for a single batch member.
        """
        if labels.shape[1] != len(self._labels):
            raise InvalidLabelError(
                f"Invalid labels: expected {len(self._labels)} labels, "
                f"got {labels.shape[0]}"
            )
        return [
            {
                label
                for label, one_hot in zip(self._labels, one_hot_labels)
                if one_hot == 1
            }
            for one_hot_labels in labels
        ]

    def get_state(self) -> dict[str, Any]:
        return {"labels": self._labels}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "LabelEncoder":
        encoder = cls(set(state["labels"]))
        encoder.conform_to_state(state)
        return encoder

    def conform_to_state(self, state: dict[str, Any]) -> None:
        """
        Conform the labels of the encoder to the given state.

        This is required when loading weights from a checkpoint which have different
        labels or differently ordered labels than the current set. In this case,
        the loaded weights require the first N labels to be identical to the loaded
        labels. If we're using more labels than that, we need them to come after the
        loaded labels.

        Args:
            state: The state to conform to.
        """
        additional_labels = set(self._labels).difference(state["labels"])
        self._labels = state["labels"] + sorted(list(additional_labels))
