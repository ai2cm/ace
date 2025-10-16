import contextlib
import dataclasses
import itertools
import os
import signal
import unittest.mock
from typing import Any, Literal, TypeVar, cast

import numpy as np
import pytest
import torch

from fme.ace.data_loading.gridded_data import DataLoader
from fme.core.ema import EMATracker
from fme.core.generics.aggregator import (
    AggregatorABC,
    InferenceAggregatorABC,
    InferenceLog,
    InferenceLogs,
)
from fme.core.generics.data import GriddedDataABC, InferenceDataABC
from fme.core.generics.optimization import OptimizationABC
from fme.core.generics.trainer import (
    AggregatorBuilderABC,
    CheckpointPaths,
    TrainConfigProtocol,
    Trainer,
    TrainOutputABC,
    TrainStepperABC,
)
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import NullOptimization, Optimization
from fme.core.scheduler import SchedulerConfig
from fme.core.testing.wandb import mock_wandb
from fme.core.typing_ import Slice, TensorDict, TensorMapping


class PSType:
    pass


class BDType:
    def __init__(self, i: int):
        self.i = i

    def __repr__(self) -> str:
        return f"BDType({self.i})"


class FDType:
    pass


class SDType:
    pass


class TrainOutput(TrainOutputABC):
    def get_metrics(self) -> dict[str, torch.Tensor]:
        return {}


class TrainData(GriddedDataABC[BDType]):
    def __init__(self, n_batches: int, shuffle: bool = False):
        self._set_epoch = unittest.mock.MagicMock()
        self._log_info = unittest.mock.MagicMock()
        self._n_batches = n_batches
        if shuffle:
            self._shuffle_seed: int | None = 0
        else:
            self._shuffle_seed = None

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def loader(self) -> DataLoader[BDType]:
        batches = [BDType(i) for i in range(self._n_batches)]
        if self._shuffle_seed is not None:
            generator = np.random.default_rng(self._shuffle_seed)
            generator.shuffle(batches)
        return batches

    @property
    def n_samples(self) -> int:
        return 3

    @property
    def n_batches(self) -> int:
        return self._n_batches

    @property
    def n_forward_steps(self) -> int:
        return 1

    def set_epoch(self, epoch: int) -> None:
        self._set_epoch(epoch)

    def log_info(self, name: str) -> None:
        self._log_info(name)

    @property
    def set_epoch_mock(self) -> unittest.mock.Mock:
        return self._set_epoch

    @property
    def log_info_mock(self) -> unittest.mock.Mock:
        return self._log_info

    def subset_loader(self, start_batch: int) -> DataLoader[BDType]:
        batches = [BDType(i) for i in range(self._n_batches)]
        if self._shuffle_seed is not None:
            generator = np.random.default_rng(self._shuffle_seed)
            generator.shuffle(batches)
        return batches[start_batch:]


class InferenceData(InferenceDataABC[PSType, FDType]):
    def __init__(self, n_time_windows: int = 1):
        self.n_time_windows = n_time_windows

    @property
    def initial_condition(self) -> PSType:
        return PSType()

    @property
    def loader(self) -> DataLoader[FDType]:
        return [FDType() for _ in range(self.n_time_windows)]

    @property
    def n_window_forward_steps(self) -> int:
        return 1


class TrainStepper(TrainStepperABC[PSType, BDType, FDType, SDType, TrainOutput]):
    SelfType = TypeVar("SelfType", bound="TrainStepper")

    def __init__(
        self,
        state: dict[str, Any] | None = None,
    ):
        self._modules = torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False)])
        self._modules[0].weight.data.fill_(0.0)
        if state is not None:
            self._state = state
        else:
            self._state = {}
        self.loaded_state: dict[str, Any] | None = None
        self.train_batches_seen: list[int] = []
        self.validation_batches_seen: list[int] = []

    def get_state(self) -> dict[str, Any]:
        return {**self._state, "modules": self._modules.state_dict()}

    def load_state(self, state: dict[str, Any]) -> None:
        self._state = state
        self.loaded_state = state
        self._modules.load_state_dict(state["modules"])

    @classmethod
    def from_state(cls: type[SelfType], state: dict[str, Any]) -> SelfType:
        ret = cls()
        ret.load_state(state)
        return ret

    @property
    def modules(self) -> torch.nn.ModuleList:
        return self._modules

    @property
    def n_ic_timesteps(self) -> int:
        return 1

    def normalize(self, data: TensorMapping) -> TensorDict:
        return dict(data)

    def predict_paired(
        self,
        initial_condition: PSType,
        forcing: FDType,
        compute_derived_variables: bool = False,
    ) -> tuple[SDType, PSType]:
        return SDType(), PSType()

    def train_on_batch(
        self,
        batch: BDType,
        optimization: OptimizationABC,
        compute_derived_variables: bool = False,
    ) -> TrainOutput:
        optimization.accumulate_loss(torch.tensor(float("inf")))
        optimization.step_weights()
        if isinstance(optimization, NullOptimization):
            self.validation_batches_seen.append(batch.i)
        else:
            self.train_batches_seen.append(batch.i)
        return TrainOutput()

    def set_train(self) -> None:
        pass

    def set_eval(self) -> None:
        pass

    def update_training_history(self, *args: Any, **kwargs: Any) -> None:
        pass


@dataclasses.dataclass
class Config:
    experiment_dir: str = "test_experiment_dir"
    checkpoint_dir: str = "test_checkpoint_dir"
    output_dir: str = "test_output_dir"
    max_epochs: int = 2
    save_checkpoint: bool = True
    validate_using_ema: bool = True
    log_train_every_n_batches: int = 1
    checkpoint_every_n_batches: int = 0
    inference_n_forward_steps: int = 1
    checkpoint_save_epochs: Slice | None = None
    ema_checkpoint_save_epochs: Slice | None = None
    segment_epochs: int | None = None
    evaluate_before_training: bool = False
    save_best_inference_epoch_checkpoints: bool = False

    def __post_init__(self):
        start_epoch = 0 if self.evaluate_before_training else 1
        self.get_inference_epochs = unittest.mock.MagicMock(
            return_value=[i for i in range(start_epoch, self.max_epochs + 1)]
        )


_: TrainConfigProtocol = Config()


class TrainAggregator(AggregatorABC[TrainOutput]):
    def __init__(self, train_loss: float):
        self.train_loss = train_loss

    def record_batch(self, batch: TrainOutput) -> None:
        pass

    def get_logs(self, label: str) -> dict[str, Any]:
        return {f"{label}/mean/loss": self.train_loss}

    def flush_diagnostics(self, subdir: str | None) -> None:
        pass


class ValidationAggregator(AggregatorABC[TrainOutput]):
    def __init__(self, validation_loss: float):
        self.validation_loss = validation_loss

    def record_batch(self, batch: TrainOutput) -> None:
        pass

    def get_logs(self, label: str) -> dict[str, Any]:
        return {f"{label}/mean/loss": self.validation_loss}

    def flush_diagnostics(self, subdir: str | None) -> None:
        pass


class InferenceAggregator(InferenceAggregatorABC[PSType, SDType]):
    def __init__(self, inference_loss: float):
        self.inference_loss = inference_loss

    def record_batch(self, data: SDType) -> InferenceLogs:
        return [{}]

    def record_initial_condition(self, initial_condition: PSType) -> InferenceLogs:
        return [{}]

    def get_summary_logs(self) -> InferenceLog:
        return {"time_mean_norm/rmse/channel_mean": self.inference_loss}

    def flush_diagnostics(self, subdir: str | None) -> None:
        pass


class AggregatorBuilder(AggregatorBuilderABC[PSType, TrainOutput, SDType]):
    def __init__(
        self,
        train_losses: np.ndarray,
        validation_losses: np.ndarray,
        inference_losses: np.ndarray,
    ):
        self.train_losses = train_losses
        self.validation_losses = validation_losses
        self.inference_losses = inference_losses
        self._train_calls = 0
        self._validation_calls = 0
        self._inference_calls = 0

    def get_train_aggregator(self) -> AggregatorABC[TrainOutput]:
        ret = TrainAggregator(self.train_losses[self._train_calls])
        self._train_calls += 1
        return ret

    def get_validation_aggregator(self) -> AggregatorABC[TrainOutput]:
        ret = ValidationAggregator(self.validation_losses[self._validation_calls])
        self._validation_calls += 1
        return ret

    def get_inference_aggregator(self) -> InferenceAggregatorABC[PSType, SDType]:
        ret = InferenceAggregator(self.inference_losses[self._inference_calls])
        self._inference_calls += 1
        return ret


def get_trainer(
    tmp_path: str,
    checkpoint_save_epochs: Slice | None = None,
    segment_epochs: int | None = None,
    max_epochs: int = 8,
    checkpoint_dir: str | None = None,
    stepper_state: dict[str, Any] | None = None,
    train_losses: np.ndarray | None = None,
    validation_losses: np.ndarray | None = None,
    inference_losses: np.ndarray | None = None,
    stepper_module_values: np.ndarray | None = None,
    ema_decay: float = 0.9999,
    validate_using_ema: bool = True,
    evaluate_before_training: bool = False,
    checkpoint_every_n_batches: int = 0,
    n_train_batches: int = 100,
    save_best_inference_epoch_checkpoints: bool = False,
    scheduler_config: SchedulerConfig | None = None,
    n_validation_batches: int = 5,
) -> tuple[TrainConfigProtocol, Trainer]:
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(tmp_path, "checkpoints")
    if train_losses is None:
        train_losses = np.zeros(max_epochs)
    if validation_losses is None:
        n_validation_steps = max_epochs + 1 if evaluate_before_training else max_epochs
        validation_losses = np.zeros(n_validation_steps)
    if inference_losses is None:
        n_inference_steps = max_epochs + 1 if evaluate_before_training else max_epochs
        inference_losses = np.zeros(n_inference_steps)
    if stepper_module_values is None:
        stepper_module_values = np.zeros(max_epochs)
    train_data = TrainData(n_batches=n_train_batches, shuffle=True)
    validation_data = TrainData(n_batches=n_validation_batches, shuffle=False)
    inference_data = InferenceData()
    stepper = TrainStepper(state=stepper_state)

    def build_optimization(modules: torch.nn.ModuleList) -> Optimization:
        if len(modules) != 1:
            raise ValueError("Expected 1 linear module with 1 weight")
        if not isinstance(modules[0], torch.nn.Linear):
            raise ValueError("Expected a linear module")
        module = modules[0]
        if module.weight.numel() != 1:
            raise ValueError("Expected a linear module with 1 weight")
        i = 0
        nonlocal scheduler_config
        if scheduler_config is None:
            scheduler_config = SchedulerConfig()
        opt = Optimization(
            parameters=itertools.chain(*[module.parameters() for module in modules]),
            optimizer_type="Adam",
            lr=0.01,
            max_epochs=max_epochs,
            scheduler=scheduler_config,
            enable_automatic_mixed_precision=False,
            kwargs={},
        )
        original_step_scheduler = opt.step_scheduler

        def step_scheduler_side_effect(*args, **kwargs):
            original_step_scheduler(*args, **kwargs)
            is_iteration = kwargs.get("is_iteration", False)
            if not is_iteration:
                # this is an "epoch" step
                nonlocal i
                i += 1

        opt.step_scheduler = unittest.mock.MagicMock(  # type: ignore
            side_effect=step_scheduler_side_effect
        )

        def step_weights_side_effect(*args, **kwargs):
            if stepper_module_values is None:
                raise ValueError("stepper_module_values is None")
            module.weight.data.fill_(stepper_module_values[i])

        opt.step_weights = unittest.mock.MagicMock(side_effect=step_weights_side_effect)  # type: ignore
        return opt

    def build_ema(modules: torch.nn.ModuleList) -> EMATracker:
        return EMATracker(modules, decay=ema_decay)

    config: TrainConfigProtocol = Config(
        experiment_dir=tmp_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_save_epochs=checkpoint_save_epochs,
        checkpoint_every_n_batches=checkpoint_every_n_batches,
        segment_epochs=segment_epochs,
        max_epochs=max_epochs,
        validate_using_ema=validate_using_ema,
        evaluate_before_training=evaluate_before_training,
        save_best_inference_epoch_checkpoints=save_best_inference_epoch_checkpoints,
    )
    aggregator_builder = AggregatorBuilder(
        train_losses=train_losses,
        validation_losses=validation_losses,
        inference_losses=inference_losses,
    )
    return config, Trainer(
        train_data=train_data,
        validation_data=validation_data,
        inference_data=inference_data,
        stepper=stepper,
        build_optimization=build_optimization,
        build_ema=build_ema,
        config=config,
        aggregator_builder=aggregator_builder,
        end_of_batch_callback=unittest.mock.MagicMock(),
        end_of_epoch_callback=unittest.mock.MagicMock(side_effect=lambda epoch: {}),
        do_gc_collect=False,  # for much faster tests
    )


@pytest.mark.parametrize(
    "checkpoint_save_epochs",
    [None, Slice(start=2, stop=3), Slice(start=1, step=2)],
)
def test_trainer(tmp_path: str, checkpoint_save_epochs: Slice | None):
    config, trainer = get_trainer(tmp_path, checkpoint_save_epochs, max_epochs=4)
    trainer.train()
    assert os.path.exists(config.experiment_dir)
    assert os.path.exists(config.checkpoint_dir)
    paths = CheckpointPaths(config.checkpoint_dir)
    assert os.path.exists(paths.latest_checkpoint_path)
    assert os.path.exists(paths.best_checkpoint_path)
    assert os.path.exists(paths.best_inference_checkpoint_path)
    assert os.path.exists(paths.ema_checkpoint_path)
    save_epochs = list(range(config.max_epochs))
    if checkpoint_save_epochs is not None:
        save_epochs = save_epochs[checkpoint_save_epochs.slice]
    else:
        save_epochs = []
    for i in range(config.max_epochs):
        if i in save_epochs:
            assert os.path.exists(paths.epoch_checkpoint_path(i))
        else:
            assert not os.path.exists(paths.epoch_checkpoint_path(i))
        assert not os.path.exists(paths.ema_epoch_checkpoint_path(i))
    train_data = cast(TrainData, trainer.train_data)
    valid_data = cast(TrainData, trainer.valid_data)
    assert train_data.set_epoch_mock.mock_calls == [
        unittest.mock.call(i) for i in range(1, config.max_epochs + 1)
    ]
    assert valid_data.set_epoch_mock.mock_calls == []  # no shuffling
    assert train_data.log_info_mock.called
    assert valid_data.log_info_mock.called
    assert trainer._end_of_epoch_callback.mock_calls == [  # type: ignore
        unittest.mock.call(i) for i in range(1, config.max_epochs + 1)
    ]


@pytest.mark.parametrize("segment_epochs", [1, 2, 3])
def test_segmented_trainer_runs_correct_epochs(tmp_path: str, segment_epochs: int):
    max_epochs = 4
    total_segments = max_epochs // segment_epochs
    for i in range(total_segments):
        config, trainer = get_trainer(
            tmp_path,
            checkpoint_dir=os.path.join(tmp_path, "checkpoint_dir"),  # same dir for all
            # for speed, don't save per-epoch checkpoints for this test
            checkpoint_save_epochs=Slice(start=0, stop=0),
            segment_epochs=segment_epochs,
            max_epochs=max_epochs,
        )
        trainer.train()
        paths = CheckpointPaths(config.checkpoint_dir)
        assert os.path.exists(paths.latest_checkpoint_path)
        train_data = cast(TrainData, trainer.train_data)
        assert train_data.set_epoch_mock.mock_calls == [
            unittest.mock.call(i)
            for i in range(
                i * segment_epochs + 1,
                min((i + 1) * segment_epochs, config.max_epochs) + 1,
            )
        ]


class TrainingInterrupted(Exception):
    pass


@contextlib.contextmanager
def fail_after_calls_patch(object, method: str, call_count: int):
    total_calls = 0
    original_method = getattr(object, method)

    def wrapper(*args, **kwargs):
        nonlocal total_calls
        total_calls += 1
        if total_calls >= call_count:
            raise TrainingInterrupted()
        return original_method(*args, **kwargs)

    with unittest.mock.patch.object(object, method) as mock:
        mock.side_effect = wrapper
        try:
            yield mock
        except TrainingInterrupted:
            pass


@contextlib.contextmanager
def preempt_after_calls_patch(object, method: str, call_count: int):
    total_calls = 0
    original_method = getattr(object, method)

    def wrapper(*args, **kwargs):
        nonlocal total_calls
        total_calls += 1
        if total_calls >= call_count:
            signal.raise_signal(signal.SIGTERM)
        return original_method(*args, **kwargs)

    with unittest.mock.patch.object(object, method) as mock:
        mock.side_effect = wrapper
        try:
            yield mock
        except SystemExit:
            pass


@pytest.mark.parametrize(
    "interrupt_method",
    ["train_one_epoch", "validate_one_epoch", "inference_one_epoch"],
)
def test_resume_after_interrupted_training(tmp_path: str, interrupt_method: str):
    max_epochs = 4
    calls_before_interrupt = 2
    stepper_state = {"foo": "bar"}
    config, trainer = get_trainer(
        tmp_path,
        stepper_state=stepper_state,
        checkpoint_save_epochs=Slice(start=0, stop=0),
        max_epochs=max_epochs,
    )
    with fail_after_calls_patch(trainer, interrupt_method, calls_before_interrupt):
        trainer.train()
    train_data = cast(TrainData, trainer.train_data)
    if interrupt_method == "train_one_epoch":
        assert train_data.set_epoch_mock.mock_calls == [
            # epoch gets set in train_one_epoch which we interrupt before
            unittest.mock.call(i)
            for i in range(1, calls_before_interrupt)
        ]
    else:
        assert train_data.set_epoch_mock.mock_calls == [
            unittest.mock.call(i) for i in range(1, calls_before_interrupt + 1)
        ]
    paths = CheckpointPaths(config.checkpoint_dir)
    assert os.path.exists(paths.latest_checkpoint_path)
    _, trainer = get_trainer(
        tmp_path,
        checkpoint_save_epochs=Slice(start=0, stop=0),
        max_epochs=max_epochs,
        stepper_state=stepper_state,
    )
    trainer.train()
    train_data = cast(TrainData, trainer.train_data)
    assert train_data.set_epoch_mock.mock_calls == [
        unittest.mock.call(i) for i in range(calls_before_interrupt, max_epochs + 1)
    ]
    stepper = cast(TrainStepper, trainer.stepper)
    assert stepper.loaded_state is not None
    assert stepper.loaded_state["foo"] == "bar"
    assert "modules" in stepper.loaded_state
    assert len(stepper.loaded_state) == 2


def get_batch_indices(batches) -> list[int]:
    return [batch.i for batch in batches]


@pytest.mark.parametrize(
    "interrupt_method",
    ["preempt", "fail"],
)
def test_resume_after_interrupted_training_during_epoch(
    tmp_path: str, interrupt_method: Literal["preempt", "fail"]
):
    if interrupt_method == "preempt":
        patch_func = preempt_after_calls_patch
    else:
        patch_func = fail_after_calls_patch
    checkpoint_every_n_batches = 20
    batches_before_interrupt = 25
    if interrupt_method == "preempt":
        # saves checkpoint gracefully during interrupt
        n_checkpointed_batches = batches_before_interrupt
    else:
        # exception leads to immediate termination without checkpointing
        n_checkpointed_batches = (
            batches_before_interrupt
            // checkpoint_every_n_batches
            * checkpoint_every_n_batches
        )
    n_train_batches = batches_before_interrupt * 2  # > batches_before_interrupt
    stepper_state = {"foo": "bar"}
    config, trainer = get_trainer(
        tmp_path,
        stepper_state=stepper_state,
        checkpoint_save_epochs=Slice(start=0, stop=0),
        max_epochs=1,
        n_train_batches=n_train_batches,
        checkpoint_every_n_batches=checkpoint_every_n_batches,
    )
    with (
        unittest.mock.patch.object(
            trainer, "_log_first_batch_metrics", return_value=None
        ),
    ):  # would throw off count for actual training batches seen
        with patch_func(
            trainer.stepper, "train_on_batch", batches_before_interrupt + 1
        ):
            trainer.train()
    assert isinstance(trainer.stepper, TrainStepper)
    stepper = cast(TrainStepper, trainer.stepper)
    pre_interrupt_batches = stepper.train_batches_seen
    assert (
        get_batch_indices(trainer.train_data.subset_loader(n_checkpointed_batches))
        == get_batch_indices(trainer.train_data.loader)[n_checkpointed_batches:]
    )  # check test subset_loader is implemented correctly
    assert len(pre_interrupt_batches) == batches_before_interrupt
    assert (
        pre_interrupt_batches
        == get_batch_indices(trainer.train_data.loader)[:batches_before_interrupt]
    )
    paths = CheckpointPaths(config.checkpoint_dir)
    assert os.path.exists(paths.latest_checkpoint_path)
    _, trainer = get_trainer(
        tmp_path,
        checkpoint_save_epochs=Slice(start=0, stop=0),
        max_epochs=1,
        n_train_batches=n_train_batches,
        stepper_state=stepper_state,
    )
    with (
        unittest.mock.patch.object(
            trainer, "validate_one_epoch", return_value={"val/mean/loss": 0.0}
        ),
    ):  # would throw off count for actual training batches seen
        trainer.train()
    stepper = cast(TrainStepper, trainer.stepper)
    assert len(stepper.train_batches_seen) == n_train_batches - n_checkpointed_batches
    expected_batches = get_batch_indices(trainer.train_data.loader)[
        n_checkpointed_batches:
    ]
    assert stepper.train_batches_seen == expected_batches
    repeated_batches = get_batch_indices(trainer.train_data.loader)[
        n_checkpointed_batches : batches_before_interrupt + 1
    ]
    assert set(stepper.train_batches_seen).intersection(repeated_batches) == set(
        repeated_batches
    )


@pytest.mark.parametrize("evaluate_before_training", [True, False])
def test_resume_after_preemption_during_validation(
    tmp_path: str, evaluate_before_training: bool
):
    checkpoint_every_n_batches = 20
    n_train_batches = checkpoint_every_n_batches * 2
    stepper_state = {"foo": "bar"}
    n_validation_batches = 4
    config, trainer = get_trainer(
        tmp_path,
        stepper_state=stepper_state,
        checkpoint_save_epochs=Slice(start=0, stop=0),
        max_epochs=1,
        n_train_batches=n_train_batches,
        checkpoint_every_n_batches=checkpoint_every_n_batches,
        evaluate_before_training=evaluate_before_training,
        n_validation_batches=n_validation_batches,
    )
    with (
        unittest.mock.patch.object(
            trainer, "_log_first_batch_metrics", return_value=None
        ),
    ):  # would throw off count for actual training batches seen
        with preempt_after_calls_patch(
            trainer,
            "validate_one_epoch",
            1 + int(evaluate_before_training),
        ):
            trainer.train()
    assert isinstance(trainer.stepper, TrainStepper)
    stepper = cast(TrainStepper, trainer.stepper)
    assert len(stepper.train_batches_seen) == n_train_batches
    assert (
        len(stepper.validation_batches_seen)
        == int(evaluate_before_training) * n_validation_batches
    )
    paths = CheckpointPaths(config.checkpoint_dir)
    assert os.path.exists(paths.latest_checkpoint_path)
    assert not os.path.exists(
        paths.best_checkpoint_path
    )  # requires end-of-epoch validation loss
    _, trainer = get_trainer(
        tmp_path,
        checkpoint_save_epochs=Slice(start=0, stop=0),
        max_epochs=1,
        n_train_batches=n_train_batches,
        stepper_state=stepper_state,
    )
    with (
        unittest.mock.patch.object(
            trainer, "validate_one_epoch", return_value={"val/mean/loss": 0.0}
        ) as validate_mock,
    ):
        assert trainer._epochs_trained == 0
        trainer.train()
        assert validate_mock.call_count == 1
        assert trainer._epochs_trained == 1
    stepper = cast(TrainStepper, trainer.stepper)
    assert len(stepper.train_batches_seen) == 0  # empty epoch after preemption
    assert (
        len(stepper.validation_batches_seen) == 0
    )  # already did evaluate_before_training before pre-emption
    assert os.path.exists(paths.best_checkpoint_path)


@pytest.mark.parametrize("ema_decay", [0.05, 0.99])
@pytest.mark.parametrize("validate_using_ema", [True, False])
def test_saves_correct_ema_checkpoints(
    tmp_path: str, ema_decay: float, validate_using_ema: bool
):
    config, trainer = get_trainer(
        tmp_path,
        checkpoint_dir=os.path.join(tmp_path, "checkpoint_dir"),  # same dir for all
        ema_decay=ema_decay,
        validate_using_ema=validate_using_ema,
    )
    valid_loss = 0.1
    inference_error = 0.2
    trainer.stepper.modules[0].weight.data.fill_(1.0)
    trainer._ema(model=trainer.stepper.modules)
    trainer.save_all_checkpoints(valid_loss=valid_loss, inference_error=inference_error)
    paths = CheckpointPaths(config.checkpoint_dir)
    assert os.path.exists(paths.ema_checkpoint_path)
    ema_checkpoint = torch.load(paths.ema_checkpoint_path)
    ema_weight = 1.0 - min(ema_decay, 2.0 / 11.0)
    np.testing.assert_allclose(
        ema_checkpoint["stepper"]["modules"]["0.weight"].cpu().numpy(),
        ema_weight,
        atol=1e-7,
    )
    assert ema_checkpoint["best_validation_loss"] == valid_loss
    assert ema_checkpoint["best_inference_error"] == inference_error
    assert os.path.exists(paths.latest_checkpoint_path)
    latest_checkpoint = torch.load(paths.latest_checkpoint_path)
    np.testing.assert_allclose(
        latest_checkpoint["stepper"]["modules"]["0.weight"].cpu().numpy(),
        1.0,
        atol=1e-7,
    )
    assert latest_checkpoint["best_validation_loss"] == valid_loss
    assert latest_checkpoint["best_inference_error"] == inference_error
    if validate_using_ema:
        best_weight = ema_weight
    else:
        best_weight = 1.0
    assert os.path.exists(paths.best_checkpoint_path)
    best_checkpoint = torch.load(paths.best_checkpoint_path)
    assert best_checkpoint["best_validation_loss"] == valid_loss
    assert best_checkpoint["best_inference_error"] == inference_error
    np.testing.assert_allclose(
        best_checkpoint["stepper"]["modules"]["0.weight"].cpu().numpy(),
        best_weight,
        atol=1e-7,
    )
    best_inference_checkpoint = torch.load(
        paths.best_inference_checkpoint_path, weights_only=False
    )
    assert best_inference_checkpoint["best_validation_loss"] == valid_loss
    assert best_inference_checkpoint["best_inference_error"] == inference_error
    np.testing.assert_allclose(
        best_inference_checkpoint["stepper"]["modules"]["0.weight"].cpu().numpy(),
        best_weight,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "segment_epochs, best_val_epoch, best_inference_epoch",
    [
        (None, 1, 1),
        (None, 3, 5),
        (2, 3, 5),
        (2, 5, 3),
        (2, 4, 6),
        (2, 6, 4),
    ],
)
def test_saves_correct_non_ema_epoch_checkpoints(
    tmp_path: str,
    segment_epochs: int | None,
    best_val_epoch: int,
    best_inference_epoch: int,
):
    max_epochs = 10
    if segment_epochs is None:
        total_segments = 1
        segment_epochs_value = max_epochs
    else:
        total_segments = max_epochs // segment_epochs
        segment_epochs_value = segment_epochs
    train_losses = np.random.rand(max_epochs) + 0.01
    val_losses = np.random.rand(max_epochs) + 0.01
    inference_losses = np.random.rand(max_epochs) + 0.01
    val_losses[best_val_epoch - 1] = 0.0
    inference_losses[best_inference_epoch - 1] = 0.0
    module_values = np.random.rand(max_epochs)
    for i in range(total_segments):
        config, trainer = get_trainer(
            tmp_path,
            checkpoint_dir=os.path.join(tmp_path, "checkpoint_dir"),  # same dir for all
            # for speed, don't save per-epoch checkpoints for this test
            checkpoint_save_epochs=Slice(start=0, stop=0),
            segment_epochs=segment_epochs,
            max_epochs=max_epochs,
            train_losses=train_losses[
                i * segment_epochs_value : (i + 1) * segment_epochs_value
            ],
            validation_losses=val_losses[
                i * segment_epochs_value : (i + 1) * segment_epochs_value
            ],
            inference_losses=inference_losses[
                i * segment_epochs_value : (i + 1) * segment_epochs_value
            ],
            stepper_module_values=module_values[
                i * segment_epochs_value : (i + 1) * segment_epochs_value
            ],
            validate_using_ema=False,
        )
        trainer.train()
        paths = CheckpointPaths(config.checkpoint_dir)
        assert os.path.exists(paths.latest_checkpoint_path)
        train_data = cast(TrainData, trainer.train_data)
        assert train_data.set_epoch_mock.mock_calls == [
            unittest.mock.call(i)
            for i in range(
                i * segment_epochs_value + 1,
                min((i + 1) * segment_epochs_value, config.max_epochs) + 1,
            )
        ]
        latest_checkpoint = torch.load(paths.latest_checkpoint_path, weights_only=False)
        assert latest_checkpoint["epoch"] == min(
            max_epochs, (i + 1) * segment_epochs_value
        )
        np.testing.assert_allclose(
            latest_checkpoint["stepper"]["modules"]["0.weight"].cpu().numpy(),
            module_values[min((i + 1) * segment_epochs_value - 1, max_epochs - 1)],
        )
    paths = CheckpointPaths(config.checkpoint_dir)
    assert os.path.exists(paths.latest_checkpoint_path)
    assert os.path.exists(paths.best_checkpoint_path)
    assert os.path.exists(paths.best_inference_checkpoint_path)
    assert os.path.exists(paths.ema_checkpoint_path)
    best_checkpoint = torch.load(paths.best_checkpoint_path, weights_only=False)
    assert best_checkpoint["epoch"] == best_val_epoch
    assert best_checkpoint["best_validation_loss"] == 0.0
    assert best_checkpoint["best_inference_error"] == np.min(
        inference_losses[:best_val_epoch]
    )
    np.testing.assert_allclose(
        best_checkpoint["stepper"]["modules"]["0.weight"].cpu().numpy(),
        module_values[best_val_epoch - 1],
    )
    best_inference_checkpoint = torch.load(
        paths.best_inference_checkpoint_path, weights_only=False
    )
    assert best_inference_checkpoint["epoch"] == best_inference_epoch
    assert best_inference_checkpoint["best_validation_loss"] == np.min(
        val_losses[:best_inference_epoch]
    )
    assert best_inference_checkpoint["best_inference_error"] == 0.0
    latest_checkpoint = torch.load(paths.latest_checkpoint_path, weights_only=False)
    assert latest_checkpoint["epoch"] == max_epochs
    np.testing.assert_allclose(
        latest_checkpoint["stepper"]["modules"]["0.weight"].cpu().numpy(),
        module_values[-1],
    )


def test_evaluate_before_training(tmp_path: str):
    max_epochs = 2
    n_train_batches = 5
    train_losses = np.random.rand(max_epochs)
    val_losses = np.random.rand(max_epochs + 1)
    inference_errors = np.random.rand(max_epochs + 1)
    train_loss_name = "train/mean/loss"
    val_loss_name = "val/mean/loss"
    inference_error_name = "inference/time_mean_norm/rmse/channel_mean"

    def _get_trainer(train_losses, val_losses, inference_errors):
        _, trainer = get_trainer(
            tmp_path,
            max_epochs=max_epochs,
            evaluate_before_training=True,
            train_losses=train_losses,
            validation_losses=val_losses,
            inference_losses=inference_errors,
            segment_epochs=1,
            n_train_batches=n_train_batches,
        )
        return trainer

    with mock_wandb() as wandb:
        LoggingConfig(log_to_wandb=True).configure_wandb({"experiment_dir": tmp_path})
        # run training in two segments to ensure coverage of check that extra validation
        # really only happens before any training is done.
        trainer = _get_trainer(train_losses[:1], val_losses[:2], inference_errors[:2])
        trainer.train()
        trainer = _get_trainer(train_losses[1:], val_losses[2:], inference_errors[2:])
        trainer.train()  # job is segmented, so need to call train twice to complete
        wandb_logs = wandb.get_logs()

        for i in range(max_epochs + 1):
            # only validate logs at end of each epoch, not per-batch logs
            logs = wandb_logs[i * n_train_batches]
            assert logs["epoch"] == i
            assert logs[val_loss_name] == val_losses[i]
            assert logs[inference_error_name] == inference_errors[i]
            if i == 0:
                assert train_loss_name not in logs
            else:
                assert logs[train_loss_name] == train_losses[i - 1]


def test_save_best_inference_epoch_ckpts(tmp_path: str):
    """Test that save_best_inference_epoch_checkpoints saves epoch-specific
    checkpoints."""
    max_epochs = 3
    n_train_batches = 5

    # set up losses where inference improves at epochs 1 and 3 but not 2
    train_losses = np.array([0.5, 0.4, 0.3])
    val_losses = np.array([0.6, 0.5, 0.4])
    inference_losses = np.array([0.3, 0.4, 0.2])

    config, trainer = get_trainer(
        tmp_path,
        max_epochs=max_epochs,
        train_losses=train_losses,
        validation_losses=val_losses,
        inference_losses=inference_losses,
        n_train_batches=n_train_batches,
        validate_using_ema=False,
        save_best_inference_epoch_checkpoints=True,
    )

    trainer.train()

    paths = CheckpointPaths(config.checkpoint_dir)

    # check that standard checkpoints exist
    assert os.path.exists(paths.latest_checkpoint_path)
    assert os.path.exists(paths.best_checkpoint_path)
    assert os.path.exists(paths.best_inference_checkpoint_path)

    # check for checkpoints
    assert os.path.exists(
        paths.best_inference_epoch_checkpoint_path(1)
    ), "Should save epoch 1"
    assert not os.path.exists(
        paths.best_inference_epoch_checkpoint_path(2)
    ), "Should not save epoch 2"
    assert os.path.exists(
        paths.best_inference_epoch_checkpoint_path(3)
    ), "Should save epoch 3"

    epoch1_checkpoint = torch.load(
        paths.best_inference_epoch_checkpoint_path(1), weights_only=False
    )
    assert epoch1_checkpoint["epoch"] == 1
    assert epoch1_checkpoint["best_inference_error"] == 0.3

    epoch3_checkpoint = torch.load(
        paths.best_inference_epoch_checkpoint_path(3), weights_only=False
    )
    assert epoch3_checkpoint["epoch"] == 3
    assert epoch3_checkpoint["best_inference_error"] == 0.2

    best_inference_checkpoint = torch.load(
        paths.best_inference_checkpoint_path, weights_only=False
    )
    assert best_inference_checkpoint["best_inference_error"] == 0.2
    assert best_inference_checkpoint["epoch"] == 3


def test_save_best_inference_epoch_ckpts_disabled(tmp_path: str):
    """Test that when save_best_inference_epoch_checkpoints is False, no
    epoch-specific checkpoints are saved."""
    max_epochs = 3
    n_train_batches = 5

    # set up losses
    train_losses = np.array([0.5, 0.4, 0.3])
    val_losses = np.array([0.6, 0.5, 0.4])
    inference_losses = np.array([0.3, 0.2, 0.1])

    config, trainer = get_trainer(
        tmp_path,
        max_epochs=max_epochs,
        train_losses=train_losses,
        validation_losses=val_losses,
        inference_losses=inference_losses,
        n_train_batches=n_train_batches,
        validate_using_ema=False,
        save_best_inference_epoch_checkpoints=False,
    )

    trainer.train()

    paths = CheckpointPaths(config.checkpoint_dir)

    # check that standard checkpoints exist
    assert os.path.exists(paths.latest_checkpoint_path)
    assert os.path.exists(paths.best_checkpoint_path)
    assert os.path.exists(paths.best_inference_checkpoint_path)

    # check that checkpoints weren't saved
    for epoch in range(1, max_epochs + 1):
        assert not os.path.exists(
            paths.best_inference_epoch_checkpoint_path(epoch)
        ), f"Should not save best_inference_ckpt_{epoch}.tar when disabled"


def test_lr_logging_by_epoch(tmp_path: str):
    max_epochs = 2
    n_train_batches = 5
    train_losses = np.random.rand(max_epochs)
    val_losses = np.random.rand(max_epochs + 1)
    inference_errors = np.random.rand(max_epochs + 1)

    def _get_trainer(train_losses, val_losses, inference_errors):
        _, trainer = get_trainer(
            tmp_path,
            max_epochs=max_epochs,
            train_losses=train_losses,
            validation_losses=val_losses,
            inference_losses=inference_errors,
            evaluate_before_training=True,
            n_train_batches=n_train_batches,
            scheduler_config=SchedulerConfig(
                type="ConstantLR",
                step_each_iteration=False,
            ),
        )
        return trainer

    with mock_wandb() as wandb:
        LoggingConfig(log_to_wandb=True).configure_wandb({"experiment_dir": tmp_path})
        trainer = _get_trainer(train_losses, val_losses, inference_errors)
        trainer.train()
        wandb_logs = wandb.get_logs()

        epoch_iters = [n_train_batches * i for i in range(max_epochs + 1)]
        for i, logs in enumerate(wandb_logs):
            if i > 0:
                assert "lr" in logs
            else:
                assert "lr" not in logs
            if i in epoch_iters:
                assert "epoch" in logs


def test_lr_logging_by_iter(tmp_path: str):
    max_epochs = 2
    n_train_batches = 5
    train_losses = np.random.rand(max_epochs)
    val_losses = np.random.rand(max_epochs + 1)
    inference_errors = np.random.rand(max_epochs + 1)

    def _get_trainer(train_losses, val_losses, inference_errors):
        _, trainer = get_trainer(
            tmp_path,
            max_epochs=max_epochs,
            train_losses=train_losses,
            validation_losses=val_losses,
            inference_losses=inference_errors,
            evaluate_before_training=True,
            n_train_batches=n_train_batches,
            scheduler_config=SchedulerConfig(
                type="ConstantLR",
                step_each_iteration=True,
            ),
        )
        return trainer

    with mock_wandb() as wandb:
        LoggingConfig(log_to_wandb=True).configure_wandb({"experiment_dir": tmp_path})
        trainer = _get_trainer(train_losses, val_losses, inference_errors)
        trainer.train()
        wandb_logs = wandb.get_logs()

        for i, logs in enumerate(wandb_logs):
            if i > 0:
                # epoch 0 doesn't log the LR
                assert "lr" in logs
            else:
                assert "lr" not in logs
