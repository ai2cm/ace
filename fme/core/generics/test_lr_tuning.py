"""Tests for run_lr_tuning_trial and LRTuningConfig."""

import copy
import itertools
import unittest.mock
from typing import Any

import torch

from fme.core.device import get_device
from fme.core.ema import EMATracker
from fme.core.generics.aggregator import AggregatorABC
from fme.core.generics.data import DataLoader, GriddedDataABC
from fme.core.generics.lr_tuning import LRTuningConfig, run_lr_tuning_trial
from fme.core.generics.optimization import OptimizationABC
from fme.core.generics.train_stepper import TrainOutputABC, TrainStepperABC
from fme.core.optimization import Optimization
from fme.core.scheduler import SchedulerConfig
from fme.core.training_history import TrainingJob
from fme.core.typing_ import Slice, TensorDict


class _TrainOutput(TrainOutputABC):
    def get_metrics(self) -> TensorDict:
        return {}


class _BatchData:
    def __init__(self, i: int):
        self.i = i


class _TrainData(GriddedDataABC["_BatchData"]):
    """Minimal GriddedDataABC for testing."""

    def __init__(self, n_batches: int):
        self._n_batches = n_batches

    @property
    def loader(self) -> DataLoader["_BatchData"]:
        return [_BatchData(i) for i in range(self._n_batches)]

    @property
    def n_samples(self) -> int:
        return self._n_batches

    @property
    def n_batches(self) -> int:
        return self._n_batches

    @property
    def batch_size(self) -> int:
        return 1

    def set_epoch(self, epoch: int):
        pass

    def alternate_shuffle(self):
        pass

    def subset_loader(
        self, start_batch: int | None = None, stop_batch: int | None = None
    ) -> DataLoader["_BatchData"]:
        return [_BatchData(i) for i in range(self._n_batches)][
            slice(start_batch, stop_batch)
        ]

    def log_info(self, name: str):
        pass


class _Stepper(TrainStepperABC["None", "_BatchData", "None", "None", "_TrainOutput"]):
    def __init__(self):
        self._modules = torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False)]).to(
            get_device()
        )

    def train_on_batch(
        self,
        data: "_BatchData",
        optimization: OptimizationABC,
        compute_derived_variables: bool = False,
    ) -> _TrainOutput:
        # Produce a loss with a grad_fn so Optimization.step_weights can backward
        x = torch.ones(1, 1, device=get_device())
        loss = self._modules[0](x).sum()
        optimization.accumulate_loss(loss)
        optimization.step_weights()
        return _TrainOutput()

    def predict_paired(
        self, initial_condition, forcing, compute_derived_variables=False
    ):
        return None, None

    @property
    def modules(self) -> torch.nn.ModuleList:
        return self._modules

    def get_state(self) -> dict[str, Any]:
        return {"modules": self._modules.state_dict()}

    def load_state(self, state: dict[str, Any]) -> None:
        self._modules.load_state_dict(state["modules"])

    def set_eval(self) -> None:
        pass

    def set_train(self) -> None:
        pass

    def update_training_history(self, training_job: TrainingJob) -> None:
        pass


class _ValidationAggregator(AggregatorABC["_TrainOutput"]):
    def __init__(self, loss: float):
        self._loss = loss

    def record_batch(self, batch: "_TrainOutput") -> None:
        pass

    def get_logs(self, label: str) -> dict[str, float]:
        return {f"{label}/mean/loss": self._loss}

    def flush_diagnostics(self, subdir: str | None) -> None:
        pass


def _build_optimization(modules: torch.nn.ModuleList) -> Optimization:
    return Optimization(
        parameters=itertools.chain(*[m.parameters() for m in modules]),
        optimizer_type="Adam",
        lr=0.01,
        max_epochs=10,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=False,
        kwargs={},
    )


def _make_copy_ema(ema: EMATracker):
    """Return a callable that creates a copy of the EMA via state APIs,
    matching the real Trainer._copy_ema pattern."""

    def copy_ema(modules: torch.nn.ModuleList) -> EMATracker:
        return EMATracker.from_state(ema.get_state(), modules)

    return copy_ema


def _make_copy_stepper(stepper: _Stepper):
    """Return a callable that creates a copy of the stepper via state APIs,
    matching the real Trainer._copy_stepper pattern (deepcopy + load_state)."""

    def copy_stepper() -> _Stepper:
        new = copy.deepcopy(stepper)
        new.load_state(copy.deepcopy(stepper.get_state()))
        return new

    return copy_stepper


def _make_aggregator_factory(*losses: float):
    """Return a get_validation_aggregator callable that yields the given losses."""
    it = iter(losses)

    def factory():
        return _ValidationAggregator(next(it))

    return factory


def test_candidate_wins():
    """Candidate wins when its loss is below baseline - threshold * baseline."""
    stepper = _Stepper()
    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epochs=Slice(),
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.1,
    )

    # baseline=0.8, candidate=0.5
    # threshold = 0.8 - 0.1*0.8 = 0.72; candidate 0.5 < 0.72 → candidate wins
    result = run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_make_copy_ema(EMATracker(stepper.modules, decay=0.9999)),
        config=config,
        current_lr=0.01,
        get_validation_aggregator=_make_aggregator_factory(0.8, 0.5),
        validate_using_ema=False,
    )

    assert result == 0.01 * 0.5


def test_candidate_below_threshold():
    """Candidate improves but not enough to beat the threshold."""
    stepper = _Stepper()
    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epochs=Slice(),
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.5,
    )

    # baseline=0.8, candidate=0.75
    # threshold = 0.8 - 0.5*0.8 = 0.4; candidate 0.75 > 0.4 → baseline wins
    result = run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_make_copy_ema(EMATracker(stepper.modules, decay=0.9999)),
        config=config,
        current_lr=0.01,
        get_validation_aggregator=_make_aggregator_factory(0.8, 0.75),
        validate_using_ema=False,
    )

    assert result is None


def test_candidate_worsens():
    """Candidate worsens validation loss -> baseline wins."""
    stepper = _Stepper()
    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epochs=Slice(),
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.1,
    )

    # baseline=0.9, candidate=1.1
    # threshold = 0.9 - 0.1*0.9 = 0.81; candidate 1.1 > 0.81 → baseline wins
    result = run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_make_copy_ema(EMATracker(stepper.modules, decay=0.9999)),
        config=config,
        current_lr=0.01,
        get_validation_aggregator=_make_aggregator_factory(0.9, 1.1),
        validate_using_ema=False,
    )

    assert result is None


def test_both_worsen():
    """Both worsen -> baseline wins."""
    stepper = _Stepper()
    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epochs=Slice(),
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.1,
    )

    # baseline=1.2, candidate=1.3
    # threshold = 1.2 - 0.1*1.2 = 1.08; candidate 1.3 > 1.08 → baseline wins
    result = run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_make_copy_ema(EMATracker(stepper.modules, decay=0.9999)),
        config=config,
        current_lr=0.01,
        get_validation_aggregator=_make_aggregator_factory(1.2, 1.3),
        validate_using_ema=False,
    )

    assert result is None


def test_baseline_worsens_candidate_improves():
    """Baseline worsens but candidate improves enough -> candidate wins."""
    stepper = _Stepper()
    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epochs=Slice(),
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.1,
    )

    # baseline=1.1, candidate=0.5
    # threshold = 1.1 - 0.1*1.1 = 0.99; candidate 0.5 < 0.99 → candidate wins
    result = run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_make_copy_ema(EMATracker(stepper.modules, decay=0.9999)),
        config=config,
        current_lr=0.01,
        get_validation_aggregator=_make_aggregator_factory(1.1, 0.5),
        validate_using_ema=False,
    )

    assert result == 0.01 * 0.5


def test_does_not_mutate_original_stepper():
    """The original stepper's parameters must not be modified by the trial."""
    stepper = _Stepper()
    stepper.modules[0].weight.data.fill_(42.0)
    original_weight = stepper.modules[0].weight.data.clone()

    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epochs=Slice(),
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.1,
    )

    run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_make_copy_ema(EMATracker(stepper.modules, decay=0.9999)),
        config=config,
        current_lr=0.01,
        get_validation_aggregator=_make_aggregator_factory(0.8, 0.5),
        validate_using_ema=False,
    )

    assert torch.allclose(stepper.modules[0].weight.data, original_weight)


def test_uses_subset_loader_with_num_batches():
    """The trial should train on exactly config.num_batches batches."""
    stepper = _Stepper()
    train_data = _TrainData(n_batches=10)
    train_data.subset_loader = unittest.mock.MagicMock(  # type: ignore
        wraps=train_data.subset_loader
    )
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epochs=Slice(),
        lr_factor=0.5,
        num_batches=4,
        improvement_threshold=0.1,
    )

    run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_make_copy_ema(EMATracker(stepper.modules, decay=0.9999)),
        config=config,
        current_lr=0.01,
        get_validation_aggregator=_make_aggregator_factory(0.9, 0.8),
        validate_using_ema=False,
    )

    train_data.subset_loader.assert_called_once_with(stop_batch=4)


def test_with_ema_validation():
    """Trial should pass validate_using_ema through to run_validation."""
    stepper = _Stepper()
    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epochs=Slice(),
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.1,
    )

    # This should not raise even with validate_using_ema=True
    result = run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_make_copy_ema(EMATracker(stepper.modules, decay=0.9999)),
        config=config,
        current_lr=0.01,
        get_validation_aggregator=_make_aggregator_factory(0.8, 0.5),
        validate_using_ema=True,
    )

    assert result == 0.01 * 0.5


def test_trial_does_not_mutate_original_ema_num_updates():
    """The original EMA's num_updates must not be modified by the trial."""
    stepper = _Stepper()
    ema = EMATracker(stepper.modules, decay=0.9999)
    # Simulate some prior training updates
    for _ in range(5):
        ema(stepper.modules)
    original_num_updates = ema.num_updates.clone()

    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epochs=Slice(),
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.1,
    )

    run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_make_copy_ema(ema),
        config=config,
        current_lr=0.01,
        get_validation_aggregator=_make_aggregator_factory(0.8, 0.5),
        validate_using_ema=False,
    )

    assert torch.equal(ema.num_updates, original_num_updates)


def test_trial_does_not_mutate_original_ema_params():
    """The original EMA's tracked parameters must not be modified by the trial."""
    stepper = _Stepper()
    ema = EMATracker(stepper.modules, decay=0.9999)
    for _ in range(5):
        ema(stepper.modules)
    original_ema_params = {
        name: param.clone() for name, param in ema._ema_params.items()
    }

    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epochs=Slice(),
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.1,
    )

    run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_make_copy_ema(ema),
        config=config,
        current_lr=0.01,
        get_validation_aggregator=_make_aggregator_factory(0.8, 0.5),
        validate_using_ema=False,
    )

    for name in original_ema_params:
        assert torch.equal(ema._ema_params[name], original_ema_params[name])


def test_deepcopy_stepper_produces_independent_copy():
    """Training a deepcopied stepper must not change the original's predictions."""
    torch.manual_seed(0)
    stepper = _Stepper()
    # Set a known weight so we can detect any change
    stepper.modules[0].weight.data.fill_(3.0)

    # Record original prediction
    x = torch.ones(1, 1, device=get_device())
    original_pred = stepper.modules[0](x).detach().clone()

    # Copy the stepper using the same pattern as Trainer._copy_stepper
    copied = copy.deepcopy(stepper)
    copied.load_state(copy.deepcopy(stepper.get_state()))

    # Train the copy for several batches with a real optimizer
    opt = _build_optimization(copied.modules)
    train_data = _TrainData(n_batches=10)
    for batch in train_data.loader:
        copied.train_on_batch(batch, opt)

    # The copy's weight should have changed (sanity check that training did something)
    copied_pred = copied.modules[0](x).detach()
    assert not torch.allclose(
        copied_pred, original_pred
    ), "Copy's weight didn't change after training — test is vacuous"

    # The original's weight must be unchanged
    current_pred = stepper.modules[0](x).detach()
    assert torch.equal(current_pred, original_pred), (
        f"Original stepper prediction changed from {original_pred.item()} "
        f"to {current_pred.item()} after training the copy"
    )


def test_trial_does_not_mutate_original_optimizer_state():
    """The original optimizer's momentum buffers must not be modified by the trial."""
    stepper = _Stepper()
    ema = EMATracker(stepper.modules, decay=0.9999)
    optimization = _build_optimization(stepper.modules)

    # Train a few batches to populate optimizer momentum buffers
    train_data = _TrainData(n_batches=5)
    for batch in train_data.loader:
        stepper.train_on_batch(batch, optimization)

    original_state = copy.deepcopy(optimization.get_state())

    valid_data = _TrainData(n_batches=3)
    config = LRTuningConfig(
        epochs=Slice(),
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.1,
    )

    run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_make_copy_ema(ema),
        config=config,
        current_lr=0.01,
        get_validation_aggregator=_make_aggregator_factory(0.8, 0.5),
        validate_using_ema=False,
    )

    current_state = optimization.get_state()
    for key in original_state["optimizer_state_dict"]["state"]:
        for buf_name, buf in original_state["optimizer_state_dict"]["state"][
            key
        ].items():
            if isinstance(buf, torch.Tensor):
                assert torch.equal(
                    buf,
                    current_state["optimizer_state_dict"]["state"][key][buf_name],
                ), f"Optimizer state[{key}][{buf_name}] was mutated by the trial"
            else:
                assert (
                    buf == current_state["optimizer_state_dict"]["state"][key][buf_name]
                ), f"Optimizer state[{key}][{buf_name}] was mutated by the trial"
