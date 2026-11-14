import copy
import os
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
    OptimizationConfig,
    OptimizationDict,
    build_optimization_dict,
    validate_optimization_config_dict,
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
    optimization = OptimizationConfig(
        optimizer_type=optimizer_type,
        lr=lr,
        scheduler=scheduler_config,
        enable_automatic_mixed_precision=enable_amp,
        kwargs={},
    ).build(nn.ModuleList([model]), max_epochs=max_epochs)
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
    optimization = OptimizationConfig(
        optimizer_type=optimizer_type,
        lr=lr,
        scheduler=scheduler_config,
        enable_automatic_mixed_precision=enable_amp,
        kwargs={},
    ).build(nn.ModuleList([model]), max_epochs=max_epochs)
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
    optimizer = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.001,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=False,
        kwargs={},
    ).build(nn.ModuleList([model]), max_epochs=10)
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
    optimizer = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.001,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=False,
        use_gradient_accumulation=use_gradient_accumulation,
        kwargs={},
    ).build(nn.ModuleList([model]), max_epochs=10)
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
        optimizer = OptimizationConfig(
            optimizer_type="Adam",
            lr=0.001,
            scheduler=SchedulerConfig(),
            enable_automatic_mixed_precision=False,
            use_gradient_accumulation=use_gradient_accumulation,
            kwargs={},
        ).build(nn.ModuleList([model]), max_epochs=10)
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
    optimization = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.01,  # base learning rate
        scheduler=sequential_scheduler,
        enable_automatic_mixed_precision=False,
        kwargs={},
    ).build(nn.ModuleList([model]), max_epochs=max_epochs)

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

    optimization = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.001,
        scheduler=sequential_scheduler,
        enable_automatic_mixed_precision=False,
        kwargs={},
    ).build(nn.ModuleList([model]), max_epochs=max_epochs)

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

    optimization = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.001,
        scheduler=sequential_scheduler,
        enable_automatic_mixed_precision=False,
        kwargs={},
    ).build(nn.ModuleList([model]), max_epochs=max_epochs)
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
    modules, lr=0.001, optimizer_type="Adam", max_epochs=10, max_grad_norm=None
) -> Optimization:
    """Helper to construct an Optimization with common test defaults."""
    return OptimizationConfig(
        optimizer_type=optimizer_type,
        lr=lr,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=False,
        kwargs={},
        max_grad_norm=max_grad_norm,
    ).build(modules, max_epochs=max_epochs)


def test_set_learning_rate():
    model = nn.Linear(1, 1).to(fme.get_device())
    optimization = _build_optimization(nn.ModuleList([model]))
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

    optimization = _build_optimization(nn.ModuleList([model]))

    # Train a few steps to build up momentum state
    for _ in range(3):
        loss = model(x).sum()
        optimization.accumulate_loss(loss)
        optimization.step_weights()

    saved_state = optimization.get_state()

    # Create a new model with the same structure but different parameter objects
    model2 = copy.deepcopy(model)
    optimization2 = _build_optimization(nn.ModuleList([model2]))
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

    optimization = _build_optimization(nn.ModuleList([model]))

    # Train a few steps
    for _ in range(3):
        loss = model(x).sum()
        optimization.accumulate_loss(loss)
        optimization.step_weights()

    saved_state = optimization.get_state()

    # Build a new optimization, load state, then override LR
    model2 = copy.deepcopy(model)
    optimization2 = _build_optimization(nn.ModuleList([model2]))
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


def test_load_optimizer_state_for_finetuning():
    """load_optimizer_state_for_finetuning restores momentum buffers from the
    checkpoint while preserving the freshly-configured per-group hyperparameters
    (lr, weight_decay, betas) and leaves the scheduler in its initial state."""
    torch.manual_seed(0)
    model = nn.Linear(2, 2).to(fme.get_device())
    x = torch.randn(10, 2).to(fme.get_device())

    source_kwargs = {"weight_decay": 0.1, "betas": (0.85, 0.995)}
    optimization = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.001,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=False,
        kwargs=source_kwargs,
    ).build(nn.ModuleList([model]), max_epochs=10)
    assert optimization.optimizer.state_dict()["state"] == {}

    for _ in range(3):
        loss = model(x).sum()
        optimization.accumulate_loss(loss)
        optimization.step_weights()
    optimization.step_scheduler(is_iteration=False)

    assert optimization.optimizer.state_dict()["state"] != {}

    saved_state = optimization.get_state()

    model2 = copy.deepcopy(model)
    new_lr = 0.01
    new_weight_decay = 0.05
    new_betas = (0.9, 0.999)
    optimization2 = OptimizationConfig(
        optimizer_type="Adam",
        lr=new_lr,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=False,
        kwargs={"weight_decay": new_weight_decay, "betas": new_betas},
    ).build(nn.ModuleList([model2]), max_epochs=10)
    fresh_scheduler_state = optimization2.scheduler.state_dict()

    optimization2.load_optimizer_state_for_finetuning(saved_state)

    for group in optimization2.optimizer.param_groups:
        assert group["lr"] == new_lr
        assert group["weight_decay"] == new_weight_decay
        assert group["betas"] == new_betas

    orig_opt_state = optimization.optimizer.state_dict()["state"]
    loaded_opt_state = optimization2.optimizer.state_dict()["state"]
    for param_id in orig_opt_state:
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                loaded_opt_state[param_id][key], orig_opt_state[param_id][key]
            )

    assert optimization2.scheduler.state_dict() == fresh_scheduler_state


def test_load_optimizer_state_for_finetuning_param_group_mismatch():
    """load_optimizer_state_for_finetuning raises an informative ValueError
    when the checkpoint's parameter groups are incompatible with the
    freshly-built optimizer (e.g. different number of parameters)."""
    torch.manual_seed(0)
    source_model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)).to(fme.get_device())
    source = _build_optimization(nn.ModuleList([source_model]))
    saved_state = source.get_state()

    target_model = nn.Linear(2, 2).to(fme.get_device())
    target = _build_optimization(nn.ModuleList([target_model]))

    with pytest.raises(ValueError, match="incompatible"):
        target.load_optimizer_state_for_finetuning(saved_state)


def test_load_optimizer_state_for_finetuning_does_not_leak_hparams():
    """Per-group hyperparameters from the checkpoint do not leak into the
    freshly-built optimizer. Even keys absent from the finetune config (e.g.
    ``amsgrad`` set on the source, or optimizer-type-specific or
    scheduler-injected keys) are scrubbed: the finetune config is
    authoritative."""
    torch.manual_seed(0)
    source_model = nn.Linear(2, 2).to(fme.get_device())
    source = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.001,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=False,
        kwargs={"amsgrad": True},
    ).build(nn.ModuleList([source_model]), max_epochs=10)
    saved_state = source.get_state()
    # Simulate a stray key from the source side that the target should not
    # adopt (e.g. an optimizer-type-specific flag, or a scheduler-injected
    # key like ``initial_lr`` from a scheduler the target is not using).
    for group in saved_state["optimizer_state_dict"]["param_groups"]:
        group["bogus_source_only_flag"] = True

    target_model = nn.Linear(2, 2).to(fme.get_device())
    target = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.001,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=False,
        kwargs={"amsgrad": False},
    ).build(nn.ModuleList([target_model]), max_epochs=10)

    target.load_optimizer_state_for_finetuning(saved_state)

    for group in target.optimizer.param_groups:
        assert group["amsgrad"] is False
        assert "bogus_source_only_flag" not in group


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GradScaler requires CUDA")
def test_load_optimizer_state_for_finetuning_with_gscaler():
    """load_optimizer_state_for_finetuning restores grad scaler state when AMP
    is enabled on both the source and target Optimization."""
    torch.manual_seed(0)
    model = nn.Linear(2, 2).to(fme.get_device())
    x = torch.randn(10, 2).to(fme.get_device())

    optimization = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.001,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=True,
        kwargs={},
    ).build(nn.ModuleList([model]), max_epochs=10)

    for _ in range(3):
        with optimization.autocast():
            loss = model(x).sum()
        optimization.accumulate_loss(loss)
        optimization.step_weights()

    saved_state = optimization.get_state()
    assert saved_state["gscaler_state_dict"] != {}

    model2 = copy.deepcopy(model)
    optimization2 = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.01,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=True,
        kwargs={},
    ).build(nn.ModuleList([model2]), max_epochs=10)

    optimization2.load_optimizer_state_for_finetuning(saved_state)

    assert optimization2.gscaler is not None
    assert optimization2.gscaler.state_dict() == saved_state["gscaler_state_dict"]


def test_optimization_config_build_no_resume_path():
    """OptimizationConfig.build with resume_optimizer_ckpt_path=None leaves the
    optimizer state empty (no-op path)."""
    model = nn.Linear(2, 2).to(fme.get_device())
    optimization = OptimizationConfig().build(
        torch.nn.ModuleList([model]), max_epochs=1
    )
    assert optimization.optimizer.state_dict()["state"] == {}


def test_optimization_config_build_resumes_optimizer_state(tmp_path: str):
    """OptimizationConfig.build with resume_optimizer_ckpt_path loads
    per-parameter optimizer state from the checkpoint while preserving the
    fresh config's per-group hyperparameters."""
    torch.manual_seed(0)
    source_model = nn.Linear(2, 2).to(fme.get_device())
    x = torch.randn(10, 2).to(fme.get_device())

    source = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.001,
        kwargs={"weight_decay": 0.1, "betas": (0.85, 0.995)},
    ).build(torch.nn.ModuleList([source_model]), max_epochs=10)
    for _ in range(3):
        loss = source_model(x).sum()
        source.accumulate_loss(loss)
        source.step_weights()

    ckpt_path = os.path.join(tmp_path, "ckpt.tar")
    torch.save({"optimization": source.get_state()}, ckpt_path)

    new_lr = 0.01
    new_weight_decay = 0.05
    new_betas = (0.9, 0.999)
    target_model = copy.deepcopy(source_model)
    target = OptimizationConfig(
        optimizer_type="Adam",
        lr=new_lr,
        kwargs={"weight_decay": new_weight_decay, "betas": new_betas},
        resume_optimizer_ckpt_path=ckpt_path,
    ).build(torch.nn.ModuleList([target_model]), max_epochs=10)

    src_state = source.optimizer.state_dict()["state"]
    tgt_state = target.optimizer.state_dict()["state"]
    assert tgt_state != {}
    for param_id in src_state:
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                tgt_state[param_id][key], src_state[param_id][key]
            )

    for group in target.optimizer.param_groups:
        assert group["lr"] == new_lr
        assert group["weight_decay"] == new_weight_decay
        assert group["betas"] == new_betas


def test_optimization_config_build_missing_optimization_key(tmp_path: str):
    """OptimizationConfig.build raises ValueError when the checkpoint at
    resume_optimizer_ckpt_path does not contain an 'optimization' key (e.g.
    a best_ckpt.tar that omits optimizer state)."""
    ckpt_path = os.path.join(tmp_path, "bad_ckpt.tar")
    torch.save({"stepper": {}, "ema": {}}, ckpt_path)

    model = nn.Linear(1, 1).to(fme.get_device())
    config = OptimizationConfig(resume_optimizer_ckpt_path=ckpt_path)
    with pytest.raises(ValueError, match="does not contain optimization state"):
        config.build(torch.nn.ModuleList([model]), max_epochs=1)


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

    optimization = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.1,
        scheduler=per_iteration_scheduler,
        enable_automatic_mixed_precision=False,
        kwargs={},
    ).build(nn.ModuleList([model]), max_epochs=5)

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

    optimization = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.1,
        scheduler=per_epoch_scheduler,
        enable_automatic_mixed_precision=False,
        kwargs={},
    ).build(nn.ModuleList([model]), max_epochs=5)

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


@pytest.mark.parametrize("max_grad_norm,expect_clipped", [(0.01, True), (None, False)])
def test_gradient_clipping(max_grad_norm, expect_clipped):
    """step_weights clips the global gradient norm to max_grad_norm when set."""
    torch.manual_seed(0)
    model = nn.Linear(10, 1).to(fme.get_device())
    optimization = _build_optimization(
        nn.ModuleList([model]), max_grad_norm=max_grad_norm
    )

    x = torch.ones(1, 10).to(fme.get_device()) * 1e4  # large input → large gradients
    optimization.accumulate_loss(model(x).sum())
    optimization.step_weights()

    if expect_clipped:
        # _last_grad_norm is the pre-clip norm returned by clip_grad_norm_
        assert optimization._last_grad_norm is not None
        assert optimization._last_grad_norm > max_grad_norm
    else:
        assert optimization._last_grad_norm is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP requires CUDA")
def test_gradient_clipping_with_amp():
    """_clip_gradients unscales via gscaler before clipping when AMP is active."""
    torch.manual_seed(0)
    model = nn.Linear(10, 1).to("cuda")
    optimization = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.001,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=True,
        kwargs={},
        max_grad_norm=0.01,
    ).build(nn.ModuleList([model]), max_epochs=10)

    with optimization.autocast():
        x = torch.ones(1, 10, device="cuda") * 1e4
        optimization.accumulate_loss(model(x).sum())

    assert optimization.gscaler is not None
    unscale_calls = []
    original_unscale = optimization.gscaler.unscale_

    def spy_unscale(opt):
        unscale_calls.append(True)
        return original_unscale(opt)

    optimization.gscaler.unscale_ = spy_unscale
    optimization.step_weights()

    assert len(unscale_calls) == 1
    assert optimization._last_grad_norm is not None


def test_optimization_config_passes_max_grad_norm():
    """OptimizationConfig.max_grad_norm is wired through to Optimization."""
    model = nn.Linear(1, 1).to(fme.get_device())
    opt = OptimizationConfig(max_grad_norm=1.5).build(
        nn.ModuleList([model]), max_epochs=1
    )
    assert opt._max_grad_norm == 1.5


def _two_component_modules(seed: int = 0) -> dict[str, nn.ModuleList]:
    torch.manual_seed(seed)
    return {
        "a": nn.ModuleList([nn.Linear(3, 1).to(fme.get_device())]),
        "b": nn.ModuleList([nn.Linear(3, 1).to(fme.get_device())]),
    }


def _summed_loss(modules: dict[str, nn.ModuleList], x: torch.Tensor) -> torch.Tensor:
    return sum((component[0](x) ** 2).sum() for component in modules.values())  # type: ignore[return-value]


def test_optimization_dict_distinct_lr_and_schedules():
    max_epochs = 6
    modules = _two_component_modules()
    configs = {
        "a": OptimizationConfig(
            optimizer_type="AdamW",
            lr=0.01,
            scheduler=SchedulerConfig(type="CosineAnnealingLR"),
        ),
        "b": OptimizationConfig(
            optimizer_type="Adam",
            lr=0.1,
            scheduler=SchedulerConfig(
                type="StepLR", kwargs={"step_size": 2, "gamma": 0.5}
            ),
        ),
    }
    optimization = build_optimization_dict(configs, modules, max_epochs=max_epochs)
    reference_a = torch.optim.AdamW(modules["a"].parameters(), lr=0.01)
    reference_b = torch.optim.Adam(modules["b"].parameters(), lr=0.1)
    schedule_a = torch.optim.lr_scheduler.CosineAnnealingLR(
        reference_a, T_max=max_epochs
    )
    schedule_b = torch.optim.lr_scheduler.StepLR(reference_b, step_size=2, gamma=0.5)
    for _ in range(max_epochs):
        assert optimization.learning_rates == {
            "a": pytest.approx(reference_a.param_groups[0]["lr"]),
            "b": pytest.approx(reference_b.param_groups[0]["lr"]),
        }
        assert optimization.learning_rate == pytest.approx(
            reference_a.param_groups[0]["lr"]
        )
        optimization.step_scheduler(valid_loss=1.0)
        reference_a.step()
        reference_b.step()
        schedule_a.step()
        schedule_b.step()


def test_optimization_dict_matches_separate_optimizations():
    x = torch.randn(5, 3).to(fme.get_device())
    configs = {
        "a": OptimizationConfig(optimizer_type="AdamW", lr=0.01),
        "b": OptimizationConfig(optimizer_type="Adam", lr=0.1),
    }
    joint_modules = _two_component_modules()
    joint = build_optimization_dict(configs, joint_modules, max_epochs=1)
    separate_modules = _two_component_modules()
    for _ in range(3):
        joint.accumulate_loss(_summed_loss(joint_modules, x))
        joint.step_weights()
    separate = {
        key: configs[key].build(separate_modules[key], max_epochs=1)
        for key in separate_modules
    }
    for _ in range(3):
        for key, optimization in separate.items():
            optimization.accumulate_loss((separate_modules[key][0](x) ** 2).sum())
            optimization.step_weights()
    for key in joint_modules:
        for joint_param, separate_param in zip(
            joint_modules[key].parameters(), separate_modules[key].parameters()
        ):
            torch.testing.assert_close(joint_param, separate_param)


def _dict_configs() -> dict[str, OptimizationConfig]:
    return {
        "a": OptimizationConfig(
            optimizer_type="AdamW",
            lr=0.01,
            scheduler=SchedulerConfig(type="CosineAnnealingLR"),
        ),
        "b": OptimizationConfig(
            optimizer_type="Adam",
            lr=0.1,
            scheduler=SchedulerConfig(
                type="StepLR", kwargs={"step_size": 1, "gamma": 0.5}
            ),
        ),
    }


def test_optimization_dict_state_round_trip():
    x = torch.randn(5, 3).to(fme.get_device())
    modules = _two_component_modules()
    optimization = build_optimization_dict(_dict_configs(), modules, max_epochs=4)
    for _ in range(2):
        optimization.accumulate_loss(_summed_loss(modules, x))
        optimization.step_weights()
        optimization.step_scheduler(valid_loss=1.0)
    state = optimization.get_state()
    assert set(state) == {"components", "gscaler_state_dict"}
    assert set(state["components"]) == {"a", "b"}
    resumed_modules = _two_component_modules()
    for key in modules:
        resumed_modules[key].load_state_dict(modules[key].state_dict())
    resumed = build_optimization_dict(_dict_configs(), resumed_modules, max_epochs=4)
    resumed.load_state(copy.deepcopy(state))
    assert resumed.learning_rates == optimization.learning_rates
    for opt, mods in ((optimization, modules), (resumed, resumed_modules)):
        opt.accumulate_loss(_summed_loss(mods, x))
        opt.step_weights()
        opt.step_scheduler(valid_loss=1.0)
    assert resumed.learning_rates == optimization.learning_rates
    for key in modules:
        for param, resumed_param in zip(
            modules[key].parameters(), resumed_modules[key].parameters()
        ):
            torch.testing.assert_close(param, resumed_param)


def test_optimization_dict_load_state_rejects_single_form():
    modules = _two_component_modules()
    single = OptimizationConfig().build(
        nn.ModuleList([*modules["a"], *modules["b"]]), max_epochs=1
    )
    dict_form = build_optimization_dict(_dict_configs(), modules, max_epochs=1)
    with pytest.raises(ValueError, match="single optimizer"):
        dict_form.load_state(single.get_state())


def test_optimization_load_state_rejects_dict_form():
    modules = _two_component_modules()
    single = OptimizationConfig().build(
        nn.ModuleList([*modules["a"], *modules["b"]]), max_epochs=1
    )
    dict_form = build_optimization_dict(_dict_configs(), modules, max_epochs=1)
    with pytest.raises(ValueError, match="one optimizer per component"):
        single.load_state(dict_form.get_state())


def test_optimization_dict_load_state_rejects_other_components():
    modules = _two_component_modules()
    dict_form = build_optimization_dict(_dict_configs(), modules, max_epochs=1)
    state = dict_form.get_state()
    state["components"]["c"] = state["components"].pop("b")
    with pytest.raises(ValueError, match="differ from configured components"):
        dict_form.load_state(state)


def _cpu_grad_scaler() -> torch.amp.GradScaler:
    # GradScaler("cuda") disables itself without CUDA; a CPU scaler runs the
    # same per-optimizer inf-check bookkeeping that the skip guards against.
    return torch.amp.GradScaler("cpu", init_scale=2.0**4)


@pytest.mark.parametrize(
    "amp",
    [
        "off",
        "cpu_scaler",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="AMP requires CUDA"
            ),
        ),
    ],
)
def test_optimization_dict_skips_component_without_grad(amp: str):
    x = torch.randn(5, 3).to(fme.get_device())
    modules = _two_component_modules()
    for param in modules["b"].parameters():
        param.requires_grad_(False)
    configs = {
        key: OptimizationConfig(
            optimizer_type="Adam",
            lr=0.1,
            enable_automatic_mixed_precision=amp == "cuda",
        )
        for key in ("a", "b")
    }
    optimization = build_optimization_dict(configs, modules, max_epochs=1)
    if amp == "cpu_scaler":
        optimization.gscaler = _cpu_grad_scaler()
    b_before = copy.deepcopy(modules["b"].state_dict())
    a_before = copy.deepcopy(modules["a"].state_dict())
    with optimization.autocast():
        loss = (modules["a"][0](x) ** 2).sum() + (modules["b"][0](x) ** 2).sum()
    optimization.accumulate_loss(loss)
    optimization.step_weights()
    for name, value in modules["b"].state_dict().items():
        torch.testing.assert_close(value, b_before[name])
    assert any(
        not torch.equal(value, a_before[name])
        for name, value in modules["a"].state_dict().items()
    )


@pytest.mark.parametrize(
    "amp",
    [
        "cpu_scaler",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="AMP requires CUDA"
            ),
        ),
    ],
)
def test_optimization_dict_no_component_with_grad_does_not_raise(amp: str):
    x = torch.randn(5, 3).to(fme.get_device())
    modules = _two_component_modules()
    configs = {
        key: OptimizationConfig(
            optimizer_type="Adam", enable_automatic_mixed_precision=amp == "cuda"
        )
        for key in ("a", "b")
    }
    optimization = build_optimization_dict(configs, modules, max_epochs=1)
    if amp == "cpu_scaler":
        optimization.gscaler = _cpu_grad_scaler()
    assert optimization.gscaler is not None
    scale_before = optimization.gscaler.get_scale()
    with optimization.autocast():
        loss = (modules["a"][0](x) ** 2).sum().detach().requires_grad_()
    optimization.accumulate_loss(loss)
    optimization.step_weights()
    assert optimization.gscaler.get_scale() == scale_before


def test_optimization_dict_clips_per_component():
    # Adam's normalized step hides clipping in the weights; the first moment
    # holds (1 - beta1) * clipped grad, so it shows whether a's grad was shrunk.
    x = torch.ones(1, 3).to(fme.get_device())
    configs = {
        key: OptimizationConfig(optimizer_type="Adam", lr=0.1, max_grad_norm=1.0)
        for key in ("a", "b")
    }
    modules = _two_component_modules()
    optimization = build_optimization_dict(configs, modules, max_epochs=1)
    optimization.accumulate_loss(
        0.01 * modules["a"][0](x).sum() + 1e6 * modules["b"][0](x).sum()
    )
    optimization.step_weights()
    a_optimizer = optimization._entries["a"].optimizer
    weight = modules["a"][0].weight
    torch.testing.assert_close(
        a_optimizer.state[weight]["exp_avg"], 0.1 * 0.01 * torch.ones_like(weight)
    )


def test_optimization_dict_set_learning_rate_keeps_ratio():
    modules = _two_component_modules()
    optimization = build_optimization_dict(
        {
            "a": OptimizationConfig(lr=0.01),
            "b": OptimizationConfig(lr=0.1),
        },
        modules,
        max_epochs=1,
    )
    optimization.set_learning_rate(0.02)
    assert optimization.learning_rate == pytest.approx(0.02)
    assert optimization.learning_rates == {
        "a": pytest.approx(0.02),
        "b": pytest.approx(0.2),
    }


@pytest.mark.parametrize(
    "configs, match",
    [
        ({"a": OptimizationConfig()}, "missing"),
        (
            {
                "a": OptimizationConfig(),
                "b": OptimizationConfig(),
                "c": OptimizationConfig(),
            },
            "unknown",
        ),
        (
            {
                "a": OptimizationConfig(enable_automatic_mixed_precision=True),
                "b": OptimizationConfig(),
            },
            "enable_automatic_mixed_precision",
        ),
        (
            {
                "a": OptimizationConfig(use_gradient_accumulation=True),
                "b": OptimizationConfig(),
            },
            "use_gradient_accumulation",
        ),
        (
            {
                "a": OptimizationConfig(
                    checkpoint=CheckpointConfig(after_n_forward_steps=1)
                ),
                "b": OptimizationConfig(),
            },
            "checkpoint",
        ),
    ],
    ids=[
        "missing",
        "unknown",
        "amp_differs",
        "accumulation_differs",
        "checkpoint_differs",
    ],
)
def test_validate_optimization_config_dict_rejects(configs, match):
    with pytest.raises(ValueError, match=match):
        validate_optimization_config_dict(configs, keys=("a", "b"))


def test_build_optimization_dict_resume_from_uncoupled_ckpt(tmp_path: str):
    x = torch.randn(5, 3).to(fme.get_device())
    source_modules = _two_component_modules()
    source = OptimizationConfig(optimizer_type="Adam", lr=0.001).build(
        source_modules["b"], max_epochs=1
    )
    for _ in range(3):
        source.accumulate_loss((source_modules["b"][0](x) ** 2).sum())
        source.step_weights()
    ckpt_path = os.path.join(tmp_path, "ckpt.tar")
    torch.save({"optimization": source.get_state()}, ckpt_path)

    modules = _two_component_modules()
    optimization = build_optimization_dict(
        {
            "a": OptimizationConfig(optimizer_type="AdamW", lr=0.01),
            "b": OptimizationConfig(
                optimizer_type="Adam", lr=0.05, resume_optimizer_ckpt_path=ckpt_path
            ),
        },
        modules,
        max_epochs=1,
    )
    assert isinstance(optimization, OptimizationDict)
    assert optimization._entries["a"].optimizer.state_dict()["state"] == {}
    source_state = source.optimizer.state_dict()["state"]
    target_state = optimization._entries["b"].optimizer.state_dict()["state"]
    for param_id in source_state:
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                target_state[param_id][key], source_state[param_id][key]
            )
    assert optimization.learning_rates["b"] == 0.05
