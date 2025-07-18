import torch


def compare_restored_parameters(module1_params, module2_params, optimizer1, optimizer2):
    """
    This helper function is used to compare the model parameters and optimizer
    states after restoring from a checkpoint.

    This covered a tricky failure case where the optimizer state was not
    restored correctly, preventing the restored model from training any further.
    https://github.com/ai2cm/full-model/pull/1936

    Note that `optimizer.state` objects are defaultdicts where keys are the model
    parameter tensors and the values are optimizer state dictionaries.
    """
    for param1, param2 in zip(
        module1_params,
        module2_params,
    ):
        assert torch.equal(param1, param2)
        assert param1 in optimizer1.state
        assert param2 in optimizer2.state

        # optimizer state is a defaultdict with the parameter tensor as the key
        # this links the model to relevant optimizer state for training
        # https://docs.pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.load_state_dict # noqa: E501
        optimizer_state1 = optimizer1.state[param1]
        optimizer_state2 = optimizer2.state[param2]

        for key, value1 in optimizer_state1.items():
            assert key in optimizer_state2
            value2 = optimizer_state2[key]
            if key == "step":
                # step is not put on device, but state restore loads to GPU
                value2 = value2.cpu()
            assert torch.equal(value1, value2)
