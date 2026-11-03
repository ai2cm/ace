import os
import pathlib
import unittest.mock
from collections.abc import Sequence
from typing import Any

import pytest
import torch

from fme.core.checkpoint_soup import SOUP_CHECKPOINT_KEY
from fme.core.ema import EMAConfig, EMATracker
from fme.core.generics.aggregator import AggregatorABC, InferenceSummary
from fme.core.generics.checkpoint_selection import (
    STATUS_FALLBACK,
    STATUS_PARTIAL,
    STATUS_PASSED,
    STATUS_UNGATED,
    CheckpointSelectionConfig,
    CheckpointSelector,
    max_abs_drift_t,
)
from fme.core.generics.test_trainer import (
    BDType,
    TrainAggregator,
    TrainData,
    TrainOutput,
    TrainStepper,
)
from fme.core.generics.trainer import (
    CHECKPOINT_SELECTION_KEY,
    AggregatorBuilderABC,
    CheckpointPaths,
    InferenceCallbackResult,
    InferenceTask,
    Trainer,
    TrainerParams,
    build_inference_callback,
)
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import Optimization, OptimizationConfig
from fme.core.testing.wandb import mock_wandb
from fme.core.typing_ import Slice

N_SAMPLES = 8
# zero-mean noise, t-statistic 0
CLEAN = torch.tensor([0.1, -0.1] * (N_SAMPLES // 2), dtype=torch.float64)
# large consistent drift, t-statistic far above any cutoff
DRIFT = 1.0 + 0.01 * torch.arange(N_SAMPLES, dtype=torch.float64)
NO_DRIFT_METRIC = None


class SoupStepper(TrainStepper):
    """Stepper with the ``{"config", "step"}`` state layout of real steppers,
    whose weight changes every training batch so EMA checkpoints differ."""

    def __init__(self):
        super().__init__()
        self._n_batches = 0

    def get_state(self) -> dict[str, Any]:
        return {"config": {"name": "soup"}, "step": self._modules.state_dict()}

    def load_state(self, state: dict[str, Any]) -> None:
        self.loaded_state = state
        self._modules.load_state_dict(state["step"])

    def train_on_batch(
        self,
        batch: BDType,
        optimization,
        compute_derived_variables: bool = False,
        evaluate_all_steps: bool = False,
    ) -> TrainOutput:
        if not evaluate_all_steps:  # training, not evaluation
            self._n_batches += 1
            self._modules[0].weight.data.fill_(float(self._n_batches))
        return TrainOutput()


class _AggregatorBuilder(AggregatorBuilderABC[TrainOutput]):
    def get_train_aggregator(self) -> AggregatorABC[TrainOutput]:
        return TrainAggregator(0.0)


def _get_trainer(
    tmp_path: str,
    scores: Sequence[float],
    slopes: Sequence[torch.Tensor | None],
    checkpoint_selection: CheckpointSelectionConfig | None,
    segment_epochs: int | None = None,
) -> Trainer:
    """Build a trainer whose inline inference at epoch ``e`` (1-based) returns
    error ``scores[e - 1]`` and drift slopes ``slopes[e - 1]`` for variable
    ``"a"`` of a weighted entry ``"inf"``."""
    max_epochs = len(scores)
    stepper = SoupStepper()

    def build_optimization(modules: torch.nn.ModuleList) -> Optimization:
        return OptimizationConfig(optimizer_type="Adam", lr=0.01).build(
            modules, max_epochs=max_epochs
        )

    def build_ema(modules: torch.nn.ModuleList) -> EMATracker:
        return EMAConfig(decay=0.5).build(modules)

    def inference_callback(epoch: int) -> InferenceCallbackResult:
        score = scores[epoch - 1]
        epoch_slopes = slopes[epoch - 1]
        drift = {} if epoch_slopes is None else {"inf": {"a": epoch_slopes}}
        return {"inf/error": score}, score, drift

    params = TrainerParams(
        experiment_dir=tmp_path,
        checkpoint_dir=os.path.join(tmp_path, "checkpoints"),
        max_epochs=max_epochs,
        save_checkpoint=True,
        validate_using_ema=False,
        log_train_every_n_batches=0,
        train_evaluation_batches=1,
        checkpoint_every_n_batches=0,
        segment_epochs=segment_epochs,
        checkpoint_save_epochs=None,
        ema_checkpoint_save_epochs=Slice(start=1),
        evaluate_before_training=False,
        save_best_inference_epoch_checkpoints=False,
        lr_tuning=None,
        checkpoint_selection=checkpoint_selection,
    )
    return Trainer(
        train_data=TrainData(n_batches=2),
        stepper=stepper,
        build_optimization=build_optimization,
        build_ema=build_ema,
        params=params,
        aggregator_builder=_AggregatorBuilder(),
        validation_callback=lambda epoch: ({}, 0.0),
        inference_callback=inference_callback,
        do_gc_collect=False,
    )


def _train(tmp_path: str, *args, **kwargs) -> tuple[Trainer, list[dict[str, Any]]]:
    with mock_wandb() as wandb:
        LoggingConfig(log_to_wandb=True)._configure_wandb(
            experiment_dir=tmp_path, config={}, resumable=True
        )
        trainer = _get_trainer(tmp_path, *args, **kwargs)
        trainer.train()
        return trainer, wandb.get_logs()


def _soup_logs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-epoch soup logs, in epoch order."""
    epoch_logs = [log for log in logs if "epoch" in log]
    return [
        {k: v for k, v in log.items() if k.startswith("soup/")} for log in epoch_logs
    ]


def _load(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_soup_of_epochs(paths: CheckpointPaths, epochs: Sequence[int]):
    soup = _load(paths.soup_checkpoint_path)
    sources = [paths.ema_epoch_checkpoint_path(epoch) for epoch in epochs]
    assert soup[SOUP_CHECKPOINT_KEY]["sources"] == sources
    ema_weights = [_load(path)["stepper"]["step"]["0.weight"].cpu() for path in sources]
    if len(ema_weights) > 1:  # the window's EMA weights differ
        assert not torch.equal(ema_weights[0], ema_weights[-1])
    torch.testing.assert_close(
        soup["stepper"]["step"]["0.weight"], torch.stack(ema_weights).mean(dim=0)
    )


def _assert_states_equal(actual: dict[str, Any], expected: dict[str, Any]):
    assert actual["logs"] == expected["logs"]
    torch.testing.assert_close(
        {k: v for k, v in actual.items() if k != "logs"},
        {k: v for k, v in expected.items() if k != "logs"},
    )


def _selection_config(**kwargs) -> CheckpointSelectionConfig:
    return CheckpointSelectionConfig(**{"window": 2, "t_cutoff": 3.1, **kwargs})


def test_feature_off_writes_no_soup_or_state(tmp_path: pathlib.Path):
    trainer, logs = _train(
        str(tmp_path), [0.3, 0.2, 0.1], [CLEAN] * 3, checkpoint_selection=None
    )
    assert not os.path.exists(trainer.paths.soup_checkpoint_path)
    assert CHECKPOINT_SELECTION_KEY not in _load(trainer.paths.latest_checkpoint_path)
    assert all(log == {} for log in _soup_logs(logs))


def test_drifting_epochs_excluded_from_passing_windows(tmp_path: pathlib.Path):
    # epochs 1-2 have the lowest error but drift; windows containing them fail
    scores = [0.1, 0.15, 0.5, 0.4, 0.4, 0.6]
    slopes = [DRIFT, DRIFT, CLEAN, CLEAN, CLEAN, CLEAN]
    trainer, logs = _train(str(tmp_path), scores, slopes, _selection_config())
    _assert_soup_of_epochs(trainer.paths, [4, 5])
    final = _soup_logs(logs)[-1]
    assert final["soup/status"] == STATUS_PASSED
    assert final["soup/epochs"] == "4-5"
    assert final["soup/score"] == pytest.approx(0.4)
    assert final["soup/max_abs_t"] <= 3.1
    # the existing best-inference checkpoint still follows the raw error
    assert _load(trainer.paths.best_inference_checkpoint_path)["epoch"] == 1


def test_fallback_when_no_window_passes(tmp_path: pathlib.Path):
    scores = [0.1, 0.2, 0.3, 0.4]
    # epochs 3-4 drift least relative to their spread, so have the lowest |t|
    noisy_drift = DRIFT + 3.0 * CLEAN
    slopes = [DRIFT, DRIFT, noisy_drift, noisy_drift]
    trainer, logs = _train(str(tmp_path), scores, slopes, _selection_config())
    _assert_soup_of_epochs(trainer.paths, [3, 4])
    final = _soup_logs(logs)[-1]
    assert final["soup/status"] == STATUS_FALLBACK
    assert final["soup/max_abs_t"] > 3.1


def test_partial_soup_written_after_every_inference_epoch(tmp_path: pathlib.Path):
    config = _selection_config(window=5)
    trainer, logs = _train(str(tmp_path), [0.3], [CLEAN], config)
    _assert_soup_of_epochs(trainer.paths, [1])
    trainer, logs = _train(
        str(tmp_path / "three"), [0.3, 0.2, 0.1], [CLEAN] * 3, config
    )
    _assert_soup_of_epochs(trainer.paths, [1, 2, 3])
    assert [log["soup/status"] for log in _soup_logs(logs)] == [STATUS_PARTIAL] * 3
    assert [log["soup/epochs"] for log in _soup_logs(logs)] == ["1-1", "1-2", "1-3"]


def test_ungated_without_drift_metric(tmp_path: pathlib.Path):
    scores = [0.3, 0.1, 0.1, 0.3]
    trainer, logs = _train(
        str(tmp_path), scores, [NO_DRIFT_METRIC] * 4, _selection_config()
    )
    _assert_soup_of_epochs(trainer.paths, [2, 3])
    assert _soup_logs(logs)[-1]["soup/status"] == STATUS_UNGATED


def test_final_window_writes_only_at_end(tmp_path: pathlib.Path):
    scores = [0.1, 0.1, 0.5, 0.6]
    config = _selection_config(mode="final_window")
    # first segment ends before the final epoch, and must not write a soup
    with mock_wandb():
        _get_trainer(
            str(tmp_path), scores, [CLEAN] * 4, config, segment_epochs=3
        ).train()
    paths = CheckpointPaths(os.path.join(str(tmp_path), "checkpoints"))
    assert not os.path.exists(paths.soup_checkpoint_path)
    trainer, logs = _train(str(tmp_path), scores, [CLEAN] * 4, config)
    # epoch 3 was recorded by the first segment and restored on resume
    _assert_soup_of_epochs(trainer.paths, [3, 4])
    assert _soup_logs(logs)[-1]["soup/status"] == STATUS_PASSED


def test_resume_restores_selection_state(tmp_path: pathlib.Path):
    scores = [0.5, 0.2, 0.2, 0.4, 0.3, 0.25]
    slopes = [CLEAN, DRIFT, CLEAN, CLEAN, CLEAN, CLEAN]
    config = _selection_config()
    uninterrupted, _ = _train(str(tmp_path / "full"), scores, slopes, config)

    resumed_dir = str(tmp_path / "resumed")
    with mock_wandb():
        first = _get_trainer(resumed_dir, scores, slopes, config, segment_epochs=3)
        first.train()
    assert first._checkpoint_selector is not None
    saved_state = first._checkpoint_selector.get_state()
    with mock_wandb():
        restored = _get_trainer(resumed_dir, scores, slopes, config)
    assert restored._checkpoint_selector is not None
    restored_state = restored._checkpoint_selector.get_state()
    assert [r["epoch"] for r in restored_state["records"]] == [2, 3]
    _assert_states_equal(restored_state, saved_state)
    with mock_wandb():
        restored.train()
    assert uninterrupted._checkpoint_selector is not None
    assert restored._checkpoint_selector is not None
    _assert_states_equal(
        restored._checkpoint_selector.get_state(),
        uninterrupted._checkpoint_selector.get_state(),
    )
    _assert_soup_of_epochs(restored.paths, [5, 6])


@pytest.mark.parametrize(
    "ema_checkpoint_save_epochs", [None, Slice(start=1, step=2), Slice(start=2)]
)
def test_startup_error_when_ema_checkpoints_not_saved_every_epoch(
    tmp_path: str, ema_checkpoint_save_epochs: Slice | None
):
    with pytest.raises(ValueError, match="ema_checkpoint_save_epochs"):
        TrainerParams(
            experiment_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path),
            max_epochs=4,
            save_checkpoint=True,
            validate_using_ema=False,
            log_train_every_n_batches=0,
            train_evaluation_batches=1,
            checkpoint_every_n_batches=0,
            segment_epochs=None,
            checkpoint_save_epochs=None,
            ema_checkpoint_save_epochs=ema_checkpoint_save_epochs,
            evaluate_before_training=False,
            save_best_inference_epoch_checkpoints=False,
            lr_tuning=None,
            checkpoint_selection=CheckpointSelectionConfig(),
        )


def _record_epochs(
    selector: CheckpointSelector, scores: Sequence[float], slopes: Sequence[Any]
):
    for epoch, (score, epoch_slopes) in enumerate(zip(scores, slopes), start=1):
        selector.record(
            epoch=epoch,
            score=score,
            valid_loss=0.0,
            drift_slopes={"inf": {"a": epoch_slopes}},
        )
        selector.evaluate_current_window()


def test_selector_state_round_trip():
    selector = CheckpointSelector(_selection_config())
    _record_epochs(selector, [0.1, 0.3, 0.2, 0.4], [DRIFT, CLEAN, CLEAN, CLEAN])
    restored = CheckpointSelector(_selection_config())
    restored.load_state(selector.get_state())
    _assert_states_equal(restored.get_state(), selector.get_state())
    # continuing after restore matches continuing the original
    _record_epochs(selector, [0.01], [CLEAN])
    _record_epochs(restored, [0.01], [CLEAN])
    assert restored.get_logs() == selector.get_logs()


def test_max_abs_drift_t_pools_epochs_and_ignores_nan():
    records_selector = CheckpointSelector(_selection_config(window=3))
    with_nan = CLEAN.clone()
    with_nan[0] = float("nan")
    _record_epochs(records_selector, [0.0] * 2, [CLEAN, with_nan])
    assert max_abs_drift_t(list(records_selector._records)) < 1.0
    all_nan = torch.full((N_SAMPLES,), float("nan"), dtype=torch.float64)
    selector = CheckpointSelector(_selection_config(window=1))
    _record_epochs(selector, [0.0], [all_nan])
    assert selector.status == STATUS_UNGATED


@pytest.mark.parametrize(
    "kwargs", [{"window": 0}, {"t_cutoff": 0.0}, {"mode": "best_epoch"}]
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        CheckpointSelectionConfig(**kwargs)


def test_inference_callback_returns_slopes_of_weighted_entries():
    slopes = {"a": CLEAN}
    tasks: list[InferenceTask] = [
        InferenceTask(
            name=name,
            data=unittest.mock.MagicMock(),
            aggregator_factory=unittest.mock.MagicMock(),
            epoch_set=frozenset({1}),
            weight=weight,
        )
        for name, weight in [("weighted", 1.0), ("unweighted", 0.0), ("no_drift", 1.0)]
    ]
    summaries = [
        InferenceSummary(logs={"weighted/x": 1.0}, loss=1.0, drift_slopes=slopes),
        InferenceSummary(logs={"unweighted/x": 1.0}, loss=1.0, drift_slopes=slopes),
        InferenceSummary(logs={"no_drift/x": 1.0}, loss=1.0, drift_slopes=None),
    ]
    with unittest.mock.patch(
        "fme.core.generics.trainer.inference_one_epoch", side_effect=summaries
    ):
        callback = build_inference_callback(
            tasks=tasks, inference_epochs=[1], stepper=unittest.mock.MagicMock()
        )
        _, error, drift_slopes = callback(epoch=1)
    assert error == 2.0
    assert drift_slopes == {"weighted": slopes}
    _, _, empty = callback(epoch=2)
    assert empty == {}
