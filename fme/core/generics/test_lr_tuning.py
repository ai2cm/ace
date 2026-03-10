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
from fme.core.typing_ import TensorDict


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


def _copy_ema(modules: torch.nn.ModuleList) -> EMATracker:
    """For tests, a fresh EMA is sufficient since we don't have prior state."""
    return EMATracker(modules, decay=0.9999)


def _make_copy_stepper(stepper: _Stepper):
    """Return a callable that creates a copy of the stepper via state APIs."""

    def copy_stepper() -> _Stepper:
        new = _Stepper()
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
    """Candidate wins when its improvement exceeds the threshold."""
    stepper = _Stepper()
    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epoch_frequency=1,
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.1,
    )

    # pre_trial=1.0, baseline->0.8 (improvement=0.2), candidate->0.5 (improvement=0.5)
    # candidate_improvement (0.5) > baseline_improvement * 1.1 (0.22) → candidate wins
    result = run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_copy_ema,
        config=config,
        current_lr=0.01,
        pre_trial_val_loss=1.0,
        get_validation_aggregator=_make_aggregator_factory(0.8, 0.5),
        validate_using_ema=False,
    )

    assert result == 0.01 * 0.5


def test_candidate_below_threshold():
    """Candidate improves but not enough to exceed the threshold."""
    stepper = _Stepper()
    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epoch_frequency=1,
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.5,
    )

    # pre_trial=1.0, baseline->0.8 (improvement=0.2), candidate->0.75 (improvement=0.25)
    # candidate_improvement (0.25) < baseline_improvement * 1.5 (0.3) → baseline wins
    result = run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_copy_ema,
        config=config,
        current_lr=0.01,
        pre_trial_val_loss=1.0,
        get_validation_aggregator=_make_aggregator_factory(0.8, 0.75),
        validate_using_ema=False,
    )

    assert result is None


def test_candidate_worsens():
    """Candidate worsens validation loss → baseline wins."""
    stepper = _Stepper()
    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epoch_frequency=1,
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.1,
    )

    # pre_trial=1.0, baseline->0.9 (improvement=0.1), candidate->1.1 (improvement=-0.1)
    result = run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_copy_ema,
        config=config,
        current_lr=0.01,
        pre_trial_val_loss=1.0,
        get_validation_aggregator=_make_aggregator_factory(0.9, 1.1),
        validate_using_ema=False,
    )

    assert result is None


def test_both_worsen():
    """Both worsen → baseline wins."""
    stepper = _Stepper()
    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epoch_frequency=1,
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.1,
    )

    # pre_trial=1.0, baseline->1.2, candidate->1.3
    result = run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_copy_ema,
        config=config,
        current_lr=0.01,
        pre_trial_val_loss=1.0,
        get_validation_aggregator=_make_aggregator_factory(1.2, 1.3),
        validate_using_ema=False,
    )

    assert result is None


def test_baseline_worsens_candidate_improves():
    """Baseline worsens but candidate improves → still returns None
    (requirement: both must improve)."""
    stepper = _Stepper()
    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epoch_frequency=1,
        lr_factor=0.5,
        num_batches=3,
        improvement_threshold=0.1,
    )

    # pre_trial=1.0, baseline->1.1 (worsens), candidate->0.5 (improves)
    result = run_lr_tuning_trial(
        train_data=train_data,
        valid_data=valid_data,
        optimization=optimization,
        copy_stepper=_make_copy_stepper(stepper),
        build_optimization=_build_optimization,
        copy_ema=_copy_ema,
        config=config,
        current_lr=0.01,
        pre_trial_val_loss=1.0,
        get_validation_aggregator=_make_aggregator_factory(1.1, 0.5),
        validate_using_ema=False,
    )

    assert result is None


def test_does_not_mutate_original_stepper():
    """The original stepper's parameters must not be modified by the trial."""
    stepper = _Stepper()
    stepper.modules[0].weight.data.fill_(42.0)
    original_weight = stepper.modules[0].weight.data.clone()

    train_data = _TrainData(n_batches=5)
    valid_data = _TrainData(n_batches=3)
    optimization = _build_optimization(stepper.modules)
    config = LRTuningConfig(
        epoch_frequency=1,
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
        copy_ema=_copy_ema,
        config=config,
        current_lr=0.01,
        pre_trial_val_loss=1.0,
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
        epoch_frequency=1,
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
        copy_ema=_copy_ema,
        config=config,
        current_lr=0.01,
        pre_trial_val_loss=1.0,
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
        epoch_frequency=1,
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
        copy_ema=_copy_ema,
        config=config,
        current_lr=0.01,
        pre_trial_val_loss=1.0,
        get_validation_aggregator=_make_aggregator_factory(0.8, 0.5),
        validate_using_ema=True,
    )

    assert result == 0.01 * 0.5
