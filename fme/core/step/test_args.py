import torch

from fme.core.labels import BatchLabels
from fme.core.step.args import StepArgs


def test_apply_input_process_func_propagates_metadata():
    n_batch = 4
    input_data = {"a": torch.randn(n_batch, 8, 16), "b": torch.randn(n_batch, 8, 16)}
    next_step = {"a": torch.randn(n_batch, 8, 16), "b": torch.randn(n_batch, 8, 16)}
    labels = BatchLabels(torch.zeros(n_batch, 2), ["label_0", "label_1"])
    data_mask = {
        "a": torch.ones(n_batch, dtype=torch.bool),
        "b": torch.zeros(n_batch, dtype=torch.bool),
    }
    channel_mask = {
        "a": torch.ones(n_batch, dtype=torch.bool),
        "b": torch.zeros(n_batch, dtype=torch.bool),
    }
    args = StepArgs(
        input=input_data,
        next_step_input_data=next_step,
        labels=labels,
        data_mask=data_mask,
        channel_mask=channel_mask,
    )

    def double(tensors):
        return {k: v * 2 for k, v in tensors.items()}

    result = args.apply_input_process_func(double)

    for name in input_data:
        torch.testing.assert_close(result.input[name], input_data[name] * 2)
        torch.testing.assert_close(
            result.next_step_input_data[name], next_step[name] * 2
        )

    assert result.labels == labels
    assert result.data_mask is not None
    for name in data_mask:
        torch.testing.assert_close(result.data_mask[name], data_mask[name])
    assert result.channel_mask is not None
    for name in channel_mask:
        torch.testing.assert_close(result.channel_mask[name], channel_mask[name])

    known_attrs = {
        "input",
        "next_step_input_data",
        "labels",
        "data_mask",
        "channel_mask",
    }
    actual_attrs = {
        name
        for name in vars(result)
        if not name.startswith("_") and not callable(getattr(result, name))
    }
    unexpected = actual_attrs - known_attrs
    if unexpected:
        raise AssertionError(
            f"StepArgs has new public attributes {unexpected} not covered by this "
            f"test. Update the test to verify they are propagated by "
            f"apply_input_process_func."
        )
