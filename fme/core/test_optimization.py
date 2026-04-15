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
from fme.core.scheduler import SchedulerConfig, SequentialSchedulerConfig


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


def test_sequential_scheduler_trapezoid():
    """
    Test SequentialSchedulerConfig implements a trapezoid learning rate schedule:
    linear warmup -> constant -> linear cooldown.

    This tests the schedule described in "Scaling Laws and Compute-Optimal
    Training Beyond Fixed Training Durations" (Hägele et al., 2024).
    """
    torch.manual_seed(0)
    max_epochs = 20
    warmup_steps = 5
    constant_steps = 10
    cooldown_steps = 5

    # Create trapezoid schedule: warmup -> constant -> cooldown
    warmup_scheduler = SchedulerConfig(
        type="LinearLR",
        kwargs={"start_factor": 0.1, "end_factor": 1.0, "total_iters": warmup_steps},
        step_each_iteration=True,
    )
    # ConstantLR keeps the LR constant at the current level
    constant_scheduler = SchedulerConfig(
        type="ConstantLR",
        kwargs={"factor": 1.0, "total_iters": constant_steps},
        step_each_iteration=True,
    )
    # LinearLR for cooldown from current LR to 0
    cooldown_scheduler = SchedulerConfig(
        type="LinearLR",
        kwargs={"start_factor": 1.0, "end_factor": 0.1, "total_iters": cooldown_steps},
        step_each_iteration=True,
    )

    sequential_scheduler = SequentialSchedulerConfig(
        schedulers=[warmup_scheduler, constant_scheduler, cooldown_scheduler],
        milestones=[warmup_steps, warmup_steps + constant_steps],
    )

    # Create model and optimizer to test the scheduler
    model = nn.Linear(1, 1).to(fme.get_device())
    optimization = Optimization(
        parameters=model.parameters(),
        optimizer_type="Adam",
        lr=0.01,  # base learning rate
        max_epochs=max_epochs,
        scheduler=sequential_scheduler,
        enable_automatic_mixed_precision=False,
        kwargs={},
    )

    learning_rates = []
    x = torch.randn(10, 1).to(fme.get_device())

    # Simulate training with per-step scheduling
    total_steps = warmup_steps + constant_steps + cooldown_steps
    for step in range(total_steps):
        lr_before = optimization.learning_rate
        learning_rates.append(lr_before)

        # Simulate training step
        loss = model(x).sum()
        optimization.accumulate_loss(loss)
        optimization.step_weights()
        # Step scheduler each iteration
        optimization.step_scheduler(is_iteration=True)

    # Verify trapezoid shape
    # Warmup phase: LR should increase
    for i in range(1, warmup_steps):
        assert (
            learning_rates[i] >= learning_rates[i - 1]
        ), f"LR should increase during warmup at step {i}"

    # Constant phase: LR should remain relatively stable
    constant_start = warmup_steps
    constant_end = warmup_steps + constant_steps
    constant_lr = learning_rates[constant_start]
    for i in range(constant_start + 1, min(constant_end, len(learning_rates))):
        # Allow small numerical differences due to floating point precision
        assert (
            abs(learning_rates[i] - constant_lr) < 1e-6
        ), f"LR should be constant during plateau at step {i}"

    # Cooldown phase: LR should decrease
    cooldown_start = warmup_steps + constant_steps
    for i in range(cooldown_start + 1, len(learning_rates)):
        assert (
            learning_rates[i] <= learning_rates[i - 1]
        ), f"LR should decrease during cooldown at step {i}"

    # Verify overall trapezoid shape properties
    assert (
        learning_rates[0] < learning_rates[warmup_steps - 1]
    ), "LR should be higher after warmup"
    assert (
        learning_rates[-1] < learning_rates[cooldown_start]
    ), "LR should be lower after cooldown"


def test_sequential_scheduler_reload():
    """
    Test that SequentialSchedulerConfig can be saved and reloaded correctly
    during training, similar to the existing scheduler reload test.
    """
    torch.manual_seed(0)
    max_epochs = 10
    checkpoint_epoch = 5

    # Create a simple 2-phase sequential scheduler
    phase1_scheduler = SchedulerConfig(
        type="LinearLR",
        kwargs={"start_factor": 0.5, "end_factor": 1.0, "total_iters": 5},
    )
    phase2_scheduler = SchedulerConfig(
        type="LinearLR",
        kwargs={"start_factor": 1.0, "end_factor": 0.1, "total_iters": 5},
    )

    sequential_scheduler = SequentialSchedulerConfig(
        schedulers=[phase1_scheduler, phase2_scheduler], milestones=[5]
    )

    # Set up model and training
    model = nn.Linear(1, 1).to(fme.get_device())
    x = torch.randn(100, 1).to(fme.get_device())

    optimization = Optimization(
        parameters=model.parameters(),
        optimizer_type="Adam",
        lr=0.001,
        max_epochs=max_epochs,
        scheduler=sequential_scheduler,
        enable_automatic_mixed_precision=False,
        kwargs={},
    )

    # Train with checkpoint
    optimization.set_mode(nn.ModuleList([model]))
    model_intermediate_state = None
    for i in range(max_epochs):
        if i == checkpoint_epoch:
            intermediate_state = yaml.dump(optimization.get_state())
            model_intermediate_state = copy.deepcopy(model.state_dict())
        loss = model(x).sum()
        optimization.accumulate_loss(loss)
        optimization.step_weights()
        optimization.step_scheduler(loss.item(), is_iteration=False)
    model_first_final_state = copy.deepcopy(model.state_dict())

    # Reset and reload
    model = nn.Linear(1, 1).to(fme.get_device())
    model.load_state_dict(model_intermediate_state)

    optimization = Optimization(
        parameters=model.parameters(),
        optimizer_type="Adam",
        lr=0.001,
        max_epochs=max_epochs,
        scheduler=sequential_scheduler,
        enable_automatic_mixed_precision=False,
        kwargs={},
    )
    optimization.load_state(yaml.load(intermediate_state, Loader=yaml.CLoader))

    # Continue training
    optimization.set_mode(nn.ModuleList([model]))
    for i in range(max_epochs - checkpoint_epoch):
        loss = model(x).sum()
        optimization.accumulate_loss(loss)
        optimization.step_weights()
        optimization.step_scheduler(loss.item(), is_iteration=False)
    model_second_final_state = model.state_dict()

    # Verify identical final states
    for k in model_first_final_state.keys():
        assert torch.allclose(model_first_final_state[k], model_second_final_state[k])


def _build_optimization(
    parameters, lr=0.001, optimizer_type="Adam", max_epochs=10
) -> Optimization:
    """Helper to construct an Optimization with common test defaults."""
    return Optimization(
        parameters=parameters,
        optimizer_type=optimizer_type,
        lr=lr,
        max_epochs=max_epochs,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=False,
        kwargs={},
    )


def test_set_learning_rate():
    model = nn.Linear(1, 1).to(fme.get_device())
    optimization = _build_optimization(model.parameters())
    assert optimization.learning_rate == 0.001
    optimization.set_learning_rate(0.01)
    assert optimization.learning_rate == 0.01


def test_set_learning_rate_null():
    optimization = NullOptimization()
    optimization.set_learning_rate(0.01)  # should not raise


def test_load_state_into_different_parameters():
    """
    Test that optimizer state (including momentum) can be loaded from one
    Optimization into another built with different parameter objects but
    the same structure. This is the pattern used by LR tuning trials,
    where we deepcopy a model and need the fork's optimizer to start
    with the original's momentum.
    """
    torch.manual_seed(0)
    model = nn.Linear(2, 2).to(fme.get_device())
    x = torch.randn(10, 2).to(fme.get_device())

    optimization = _build_optimization(model.parameters())

    # Train a few steps to build up momentum state
    for _ in range(3):
        loss = model(x).sum()
        optimization.accumulate_loss(loss)
        optimization.step_weights()

    saved_state = optimization.get_state()

    # Create a new model with the same structure but different parameter objects
    model2 = copy.deepcopy(model)
    optimization2 = _build_optimization(model2.parameters())
    optimization2.load_state(saved_state)

    # Train both for one more step on identical data and verify identical results
    x2 = x.clone()
    loss1 = model(x).sum()
    optimization.accumulate_loss(loss1)
    optimization.step_weights()

    loss2 = model2(x2).sum()
    optimization2.accumulate_loss(loss2)
    optimization2.step_weights()

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(
            p1, p2
        ), "Parameters should match after identical training"


def test_load_state_then_set_learning_rate():
    """
    Test that set_learning_rate works correctly after loading state,
    which is the pattern used to create a candidate fork at a different LR.
    """
    torch.manual_seed(0)
    model = nn.Linear(2, 2).to(fme.get_device())
    x = torch.randn(10, 2).to(fme.get_device())

    optimization = _build_optimization(model.parameters())

    # Train a few steps
    for _ in range(3):
        loss = model(x).sum()
        optimization.accumulate_loss(loss)
        optimization.step_weights()

    saved_state = optimization.get_state()

    # Build a new optimization, load state, then override LR
    model2 = copy.deepcopy(model)
    optimization2 = _build_optimization(model2.parameters())
    optimization2.load_state(saved_state)
    optimization2.set_learning_rate(0.0005)

    assert optimization2.learning_rate == 0.0005

    # Verify it actually trains at the new LR (different from original)
    x2 = x.clone()

    loss1 = model(x).sum()
    optimization.accumulate_loss(loss1)
    optimization.step_weights()

    loss2 = model2(x2).sum()
    optimization2.accumulate_loss(loss2)
    optimization2.step_weights()

    # With different LRs, parameters should diverge
    params_match = all(
        torch.allclose(p1, p2)
        for p1, p2 in zip(model.parameters(), model2.parameters())
    )
    assert (
        not params_match
    ), "Parameters should differ when trained at different learning rates"


def test_scheduler_step_timing():
    """
    Test that schedulers step at the correct timing based on
    step_each_iteration setting.

    """
    model = nn.Linear(1, 1).to(fme.get_device())

    # Test per-iteration stepping
    per_iteration_scheduler = SchedulerConfig(
        type="StepLR", kwargs={"step_size": 2, "gamma": 0.5}, step_each_iteration=True
    )

    optimization = Optimization(
        parameters=model.parameters(),
        optimizer_type="Adam",
        lr=0.1,
        max_epochs=5,
        scheduler=per_iteration_scheduler,
        enable_automatic_mixed_precision=False,
        kwargs={},
    )

    # Initial LR
    initial_lr = optimization.learning_rate

    # Step scheduler each iteration
    optimization.step_scheduler(valid_loss=None, is_iteration=True)
    optimization.step_scheduler(
        valid_loss=None, is_iteration=True
    )  # 2 steps, should trigger StepLR
    lr_after_iterations = optimization.learning_rate

    # Step scheduler per epoch (should not step since step_each_iteration=True)
    optimization.step_scheduler(valid_loss=0.5, is_iteration=False)
    lr_after_epoch = optimization.learning_rate

    # Verify per-iteration stepping worked but per-epoch didn't
    assert (
        lr_after_iterations < initial_lr
    ), "LR should decrease after per-iteration steps"
    assert (
        lr_after_epoch == lr_after_iterations
    ), "LR should not change on per-epoch step when step_each_iteration=True"

    # Test per-epoch stepping
    model = nn.Linear(1, 1).to(fme.get_device())
    per_epoch_scheduler = SchedulerConfig(
        type="StepLR", kwargs={"step_size": 1, "gamma": 0.5}, step_each_iteration=False
    )

    optimization = Optimization(
        parameters=model.parameters(),
        optimizer_type="Adam",
        lr=0.1,
        max_epochs=5,
        scheduler=per_epoch_scheduler,
        enable_automatic_mixed_precision=False,
        kwargs={},
    )

    initial_lr = optimization.learning_rate

    # Step scheduler each iteration (should not step since step_each_iteration=False)
    optimization.step_scheduler(is_iteration=True)
    lr_after_iteration = optimization.learning_rate

    # Step scheduler per epoch (should step)
    optimization.step_scheduler(is_iteration=False)
    lr_after_epoch = optimization.learning_rate

    # Verify per-epoch stepping worked but per-iteration didn't
    assert (
        lr_after_iteration == initial_lr
    ), "LR should not change on per-iteration step when step_each_iteration=False"
    assert lr_after_epoch < initial_lr, "LR should decrease after per-epoch step"
