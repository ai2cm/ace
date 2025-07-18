import copy
from typing import Literal

import pytest
import torch
import torch.nn as nn
import torch.random
import yaml

import fme
from fme.core.optimization import (
    Checkpoint,
    CheckpointConfig,
    NoCheckpoint,
    NullOptimization,
    Optimization,
)
from fme.core.scheduler import SchedulerConfig


@pytest.mark.parametrize("scheduler", [None, "CosineAnnealingLR"])
@pytest.mark.parametrize("enable_amp", [False, True])
def test_optimization_reload(
    scheduler: Literal["CosineAnnealingLR"] | None,
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
    optimization.set_mode(nn.ModuleList([model]))
    model_intermediate_state = None
    for i in range(max_epochs):
        if i == checkpoint_epoch:
            # save the state
            intermediate_state = yaml.dump(optimization.get_state())
            # save the model weights
            model_intermediate_state = copy.deepcopy(model.state_dict())
        loss = model(x).sum()
        optimization.accumulate_loss(loss)
        optimization.step_weights()
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
    optimization.set_mode(nn.ModuleList([model]))
    for i in range(max_epochs - checkpoint_epoch):
        loss = model(x).sum()
        optimization.accumulate_loss(loss)
        optimization.step_weights()
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


def test_null_optimization_accumulates_loss():
    optimizer = NullOptimization()
    optimizer.accumulate_loss(torch.tensor(1.0))
    optimizer.accumulate_loss(torch.tensor(2.0))
    assert optimizer.get_accumulated_loss() == torch.tensor(3.0)
    optimizer.step_weights()
    assert optimizer.get_accumulated_loss() == torch.tensor(0.0)


def test_adam_optimization_accumulates_loss():
    model = nn.Linear(1, 1).to(fme.get_device())
    input = torch.randn(100, 1).to(fme.get_device())
    output = model(input)
    loss = output.sum()
    optimizer = Optimization(
        parameters=model.parameters(),
        optimizer_type="Adam",
        lr=0.001,
        max_epochs=10,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=False,
        kwargs={},
    )
    optimizer.accumulate_loss(loss)
    optimizer.accumulate_loss(loss)
    assert optimizer.get_accumulated_loss() == loss * 2
    optimizer.step_weights()
    assert optimizer.get_accumulated_loss() == torch.tensor(0.0)


@pytest.mark.parametrize("use_gradient_accumulation", [True, False])
def test_detach_if_using_gradient_accumulation(use_gradient_accumulation: bool):
    model = nn.Linear(1, 1).to(fme.get_device())
    a = torch.randn(10, 1).to(fme.get_device())
    data = {"a": model(a)}
    optimizer = Optimization(
        parameters=model.parameters(),
        optimizer_type="Adam",
        lr=0.001,
        max_epochs=10,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=False,
        use_gradient_accumulation=use_gradient_accumulation,
        kwargs={},
    )
    data = optimizer.detach_if_using_gradient_accumulation(data)
    assert data["a"].requires_grad == (not use_gradient_accumulation)


def test_change_identical_with_or_without_gradient_accumulation():
    def get_change(use_gradient_accumulation: bool):
        model = nn.Linear(1, 1).to(fme.get_device())
        model.load_state_dict(
            {
                "weight": torch.ones(1, 1).to(fme.get_device()),
                "bias": torch.zeros(1).to(fme.get_device()),
            }
        )
        optimizer = Optimization(
            parameters=model.parameters(),
            optimizer_type="Adam",
            lr=0.001,
            max_epochs=10,
            scheduler=SchedulerConfig(),
            enable_automatic_mixed_precision=False,
            use_gradient_accumulation=use_gradient_accumulation,
            kwargs={},
        )
        a = torch.ones(10, 1).to(fme.get_device())
        loss = model(a).sum()
        optimizer.accumulate_loss(loss)
        loss = model(a).sum()
        optimizer.accumulate_loss(loss)
        optimizer.step_weights()
        return model.state_dict()

    state_with_gradient_accumulation = get_change(True)
    state_without_gradient_accumulation = get_change(False)
    assert state_with_gradient_accumulation == state_without_gradient_accumulation


@pytest.mark.parametrize("after_n_forward_steps", [0, 1, 2])
@pytest.mark.parametrize("step", [0, 1, 2])
def test_checkpoint(after_n_forward_steps: int, step: int):
    config = CheckpointConfig(after_n_forward_steps=after_n_forward_steps)
    checkpoint = config.build(step)
    if step >= after_n_forward_steps:
        assert isinstance(checkpoint, Checkpoint)
    else:
        assert isinstance(checkpoint, NoCheckpoint)
