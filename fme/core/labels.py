import logging
from typing import Any

import torch

from fme.core.device import get_device


class BatchLabels:
    def __init__(self, tensor: torch.Tensor, names: list[str]):
        self.tensor = tensor
        self.names = names
        if len(names) != tensor.shape[1]:
            raise ValueError(
                f"Number of names ({len(names)}) must match number of "
                f"columns in tensor ({tensor.shape[1]})."
            )
        self._names_set = set(names)

    def to(self, device) -> "BatchLabels":
        """
        Move the BatchLabels tensor to the specified device.

        Args:
            device: The target device.

        Returns:
            A new BatchLabels instance on the specified device.
        """
        return BatchLabels(self.tensor.to(device), self.names)

    def __repr__(self) -> str:
        return f"BatchLabels(names={self.names}, tensor={self.tensor})"

    def conform_to_encoding(self, encoding: "LabelEncoding") -> "BatchLabels":
        """
        Conform the labels to the given encoding.

        Returns a new BatchLabels instance whose "names" come from
        the provided encoding, and whose tensor columns are determined
        from the existing data.

        Args:
            encoding: The target LabelEncoding.

        Returns:
            A new BatchLabels instance conforming to the given encoding.
        """
        if len(self.names) == 0:
            return BatchLabels(
                torch.zeros(
                    (self.tensor.shape[0], len(encoding.names)),
                    device=self.tensor.device,
                ),
                names=encoding.names,
            )
        old_index = {name: i for i, name in enumerate(self.names)}
        new_names = encoding.names

        # Build index list: existing label → idx; new label → -1
        idx = torch.tensor(
            [old_index.get(name, -1) for name in new_names],
            dtype=torch.long,
            device=self.tensor.device,
        )

        # Mask for new labels
        new_mask = idx == -1

        # Clamp missing indices to 0 so gather won't crash
        safe_idx = idx.clone()
        safe_idx[new_mask] = 0

        # Gather existing columns in one shot
        gathered = self.tensor[:, safe_idx]  # shape: (N, len(new_names))

        # Zero out the newly added columns
        if new_mask.any():
            gathered[:, new_mask] = 0

        # Warn on dropped labels
        dropped = self._names_set.difference(new_names)
        if dropped:
            logging.warning(f"Dropping labels not present in new encoding: {dropped}")

        return BatchLabels(gathered, new_names)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BatchLabels):
            return False
        if self.names != other.names:
            return False
        return torch.equal(self.tensor, other.tensor)

    @classmethod
    def new_from_set(cls, label_set: set[str], n_samples: int, device) -> "BatchLabels":
        """
        Create a BatchLabels instance from a set of labels and number of samples.

        Each sample will have all the labels in the set.
        """
        names = sorted(list(label_set))
        tensor = torch.ones((n_samples, len(names)), dtype=torch.float32).to(device)
        return cls(tensor=tensor, names=names)


class InvalidLabelError(ValueError):
    """
    Raised when a label is invalid.
    """

    pass


class LabelEncoding:
    """
    Transforms labels for each batch member into a tensor of one-hot encoded labels.
    """

    def __init__(self, labels: list[str]):
        if not isinstance(labels, list):
            raise ValueError("Labels must be an ordered list of strings")
        self.names = labels.copy()

    def encode(self, labels: list[set[str]]) -> BatchLabels:
        """
        Encodes a list of sets of labels into a tensor of one-hot encoded labels.

        Args:
            labels: List of sets of labels, where each set contains the labels
                for a single batch member.

        Returns:
            Tensor of one-hot encoded labels, of shape (batch_size, n_labels).
        """
        list_data = []
        for batch_labels in labels:
            if not batch_labels.issubset(self.names):
                raise InvalidLabelError(
                    f"Invalid labels: at least one of {batch_labels} "
                    f"is not in {self.names}"
                )
            list_data.append(
                [1 if label in batch_labels else 0 for label in self.names]
            )
        return BatchLabels(
            tensor=torch.tensor(
                list_data,
                dtype=torch.float32,
                device=get_device(),
            ),
            names=self.names,
        )

    def get_state(self) -> dict[str, Any]:
        return {"labels": self.names}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "LabelEncoding":
        encoder = cls(state["labels"])
        encoder.conform_to_state(state)
        return encoder

    def append_missing_labels(self, labels: list[str]) -> "LabelEncoding":
        """
        Append labels to the end of the encoding that are not already in the encoding.
        """
        missing_labels = set(labels).difference(self.names)
        if missing_labels:
            new_names = self.names + sorted(list(missing_labels))
            return LabelEncoding(new_names)
        return self

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
        state_labels: list[str] = state["labels"]
        additional_labels = set(self.names).difference(state_labels)
        self.names = state_labels + sorted(list(additional_labels))
