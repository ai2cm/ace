import copy
from typing import Literal, Optional

import pytest
import torch
import torch.nn as nn
import torch.random
import yaml

import fme
from fme.core.optimization import Optimization
from fme.core.scheduler import SchedulerConfig


@pytest.mark.parametrize("scheduler", [None, "CosineAnnealingLR"])
@pytest.mark.parametrize("enable_amp", [False, True])
def test_optimization_reload(
    scheduler: Optional[Literal["CosineAnnealingLR"]],
    enable_amp: bool,
):
    """
    Test that when we save and reload the optimizer, we get an identical
    final training result as if we didn't save and reload.
    """
    torch.manual_seed(0)
    optimizer_type: Literal["Adam", "FusedAdam"] = "Adam"
    lr = 0.001
    max_epochs = 10  # final epoch
    checkpoint_epoch = 5  # epoch where we will save and later restart training
    # set up a toy single-layer model with random training data to optimize
    model = nn.Linear(1, 1).to(fme.get_device())
    x = torch.randn(100, 1).to(fme.get_device())
    if scheduler == "CosineAnnealingLR":
        scheduler_config = SchedulerConfig(
            type="CosineAnnealingLR",
            kwargs={"T_max": max_epochs},
        )
    elif scheduler is None:
        scheduler_config = SchedulerConfig()
    else:
        raise NotImplementedError()
    optimization = Optimization(
        parameters=model.parameters(),
        optimizer_type=optimizer_type,
        lr=lr,
        max_epochs=max_epochs,
        scheduler=scheduler_config,
        enable_automatic_mixed_precision=enable_amp,
        kwargs={},
    )
    # train the model
    optimization.set_mode(model)
    model_intermediate_state = None
    for i in range(max_epochs):
        if i == checkpoint_epoch:
            # save the state
            intermediate_state = yaml.dump(optimization.get_state())
            # save the model weights
            model_intermediate_state = copy.deepcopy(model.state_dict())
        loss = model(x).sum()
        optimization.step_weights(loss)
        if scheduler is not None:
            optimization.step_scheduler(loss.item())
    model_first_final_state = copy.deepcopy(model.state_dict())
    # reset the model
    model = nn.Linear(1, 1).to(fme.get_device())
    # reload the model weights
    model.load_state_dict(model_intermediate_state)
    # reload the state
    optimization = Optimization(
        parameters=model.parameters(),
        optimizer_type=optimizer_type,
        lr=lr,
        max_epochs=max_epochs,
        scheduler=scheduler_config,
        enable_automatic_mixed_precision=enable_amp,
        kwargs={},
    )
    optimization.load_state(yaml.load(intermediate_state, Loader=yaml.CLoader))
    # train the model again
    optimization.set_mode(model)
    for i in range(max_epochs - checkpoint_epoch):
        loss = model(x).sum()
        optimization.step_weights(loss)
        if scheduler is not None:
            optimization.step_scheduler(loss.item())
    model_second_final_state = model.state_dict()
    # check that the final weights are the same
    for k in model_first_final_state.keys():
        assert torch.allclose(model_first_final_state[k], model_second_final_state[k])


def test_adam_reload():
    """
    Test that when we save and reload the Adam optimizer, we get an identical
    final training result as if we didn't save and reload.

    This is an external dependency test only.
    """
    torch.manual_seed(0)
    lr = 0.001
    max_epochs = 6  # final epoch
    checkpoint_epoch = 3  # epoch where we will save and later restart training
    # set up a toy single-layer model with random training data to optimize
    model = nn.Linear(1, 1).to(fme.get_device())
    x = torch.randn(100, 1).to(fme.get_device())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_intermediate_state = None
    # train the model
    for i in range(max_epochs):
        if i == checkpoint_epoch:
            # save the state
            intermediate_state = yaml.dump(optimizer.state_dict())
            # save the model weights
            model_intermediate_state = copy.deepcopy(model.state_dict())
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model_first_final_state = copy.deepcopy(model.state_dict())
    # reset the model
    model = nn.Linear(1, 1).to(fme.get_device())
    # reload the model weights
    model.load_state_dict(model_intermediate_state)
    # reload the state
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(yaml.load(intermediate_state, Loader=yaml.CLoader))
    # train the model again
    for i in range(max_epochs - checkpoint_epoch):
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model_second_final_state = model.state_dict()
    # check that the final weights are the same
    for k in model_first_final_state.keys():
        assert torch.allclose(model_first_final_state[k], model_second_final_state[k])
