import dataclasses
import datetime
import random

import pytest
import torch
from torch import nn

import fme
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.step.args import StepArgs
from fme.core.step.infill_prediction import (
    InferenceSchemeConfig,
    InfillPredictionStep,
    InfillPredictionStepConfig,
    SampledTasks,
    TaskSampler,
    TaskSamplingConfig,
    TaskType,
    TaskWeights,
    _get_task_loss_scale,
)
from fme.core.step.step import StepSelector


def _only_weight(task_type: TaskType, loss_scale: float = 1.0) -> TaskWeights:
    """Return TaskWeights with only the given task enabled."""
    kwargs = {t.value: 0.0 for t in TaskType}
    kwargs[task_type.value] = 1.0
    kwargs[f"{task_type.value}_loss_scale"] = loss_scale
    return TaskWeights(**kwargs)


def _make_sampler(
    task_type: TaskType,
    all_names: list[str] | None = None,
    forcing_names: list[str] | None = None,
    min_in: int = 1,
    min_out: int = 1,
    loss_scale: float = 1.0,
) -> TaskSampler:
    if all_names is None:
        all_names = ["a", "b", "c", "d", "e"]
    if forcing_names is None:
        forcing_names = []
    config = TaskSamplingConfig(
        task_weights=_only_weight(task_type, loss_scale),
        min_input_variables=min_in,
        min_output_variables=min_out,
    )
    return TaskSampler(config, all_names, forcing_names)


def _extract_sample(
    tasks: SampledTasks, idx: int, all_names: list[str]
) -> tuple[set[str], set[str], set[str]]:
    """Extract (prev_inputs, curr_inputs, outputs) for a single sample."""
    prev = {n for n in all_names if tasks.previous_step_input_mask[n][idx].item()}
    curr = {n for n in all_names if tasks.current_step_input_mask[n][idx].item()}
    out_names = set(tasks.output_data_mask.keys())
    out = {n for n in out_names if tasks.output_data_mask[n][idx].item() > 0}
    return prev, curr, out


# ---------------------------------------------------------------------------
# Config / error tests
# ---------------------------------------------------------------------------


class TestTaskSamplingConfig:
    def test_min_output_variables_must_be_positive(self):
        with pytest.raises(ValueError, match="min_output_variables"):
            TaskSamplingConfig(min_output_variables=0)

    def test_min_input_variables_must_be_positive(self):
        with pytest.raises(ValueError, match="min_input_variables"):
            TaskSamplingConfig(min_input_variables=0)

    def test_defaults_are_valid(self):
        config = TaskSamplingConfig()
        assert config.min_input_variables == 1
        assert config.min_output_variables == 1


class TestTaskSamplerErrors:
    def test_negative_weight_raises(self):
        weights = TaskWeights(auto_encode=-1.0)
        config = TaskSamplingConfig(task_weights=weights)
        with pytest.raises(ValueError, match="must be >= 0"):
            TaskSampler(config, ["a", "b"], [])

    def test_all_zero_weights_raises(self):
        weights = TaskWeights(**{t.value: 0.0 for t in TaskType})
        config = TaskSamplingConfig(task_weights=weights)
        with pytest.raises(ValueError, match="At least one task"):
            TaskSampler(config, ["a", "b"], [])

    def test_infeasible_auto_encode_raises(self):
        with pytest.raises(ValueError, match="auto_encode"):
            _make_sampler(TaskType.AUTO_ENCODE, all_names=["f"], forcing_names=["f"])

    def test_infeasible_infill_raises(self):
        with pytest.raises(ValueError, match="infill requires"):
            _make_sampler(
                TaskType.INFILL,
                all_names=["a"],
                forcing_names=[],
                min_in=1,
                min_out=1,
            )

    def test_infeasible_prediction_too_few_outputs_raises(self):
        with pytest.raises(ValueError, match="prediction requires"):
            _make_sampler(
                TaskType.PREDICTION,
                all_names=["f"],
                forcing_names=["f"],
                min_out=1,
            )

    def test_infeasible_infill_prediction_raises(self):
        with pytest.raises(ValueError, match="infill_prediction requires"):
            _make_sampler(
                TaskType.INFILL_PREDICTION,
                all_names=["a"],
                forcing_names=[],
                min_in=1,
                min_out=1,
            )

    def test_infeasible_combined_all_raises(self):
        with pytest.raises(ValueError, match="combined_all requires"):
            _make_sampler(
                TaskType.COMBINED_ALL,
                all_names=["f"],
                forcing_names=["f"],
                min_out=1,
            )

    def test_zero_weight_task_skips_feasibility_check(self):
        weights = TaskWeights(
            auto_encode=0.0,
            infill=0.0,
            prediction=1.0,
            infill_prediction=0.0,
            combined_all=0.0,
        )
        config = TaskSamplingConfig(task_weights=weights)
        TaskSampler(config, ["a"], [])


# ---------------------------------------------------------------------------
# Constraint validation tests (deterministic checks per task)
# ---------------------------------------------------------------------------

ALL_NAMES = ["a", "b", "c", "d", "e"]
FORCING_NAMES = ["d", "e"]
NON_FORCING = ["a", "b", "c"]
BATCH_SIZE = 8


def _validate_basic_constraints(
    tasks: SampledTasks,
    all_names: list[str],
    non_forcing: list[str],
    forcing: list[str],
    min_in: int,
    min_out: int,
    batch_size: int,
):
    """Check constraints common to all task types."""
    for i in range(batch_size):
        prev, curr, out = _extract_sample(tasks, i, all_names)
        assert out.issubset(
            set(non_forcing)
        ), f"Sample {i}: outputs {out} include forcing variables"
        assert (
            len(out) >= min_out
        ), f"Sample {i}: only {len(out)} outputs, need {min_out}"
        total_inputs = prev | curr
        assert (
            len(total_inputs) >= min_in
        ), f"Sample {i}: only {len(total_inputs)} inputs, need {min_in}"
    assert set(tasks.output_data_mask.keys()) == set(non_forcing)


class TestAutoEncodeConstraints:
    @pytest.fixture()
    def sampler(self):
        return _make_sampler(
            TaskType.AUTO_ENCODE,
            all_names=ALL_NAMES,
            forcing_names=FORCING_NAMES,
        )

    def test_outputs_equal_current_inputs(self, sampler):
        random.seed(42)
        tasks = sampler.sample(None, BATCH_SIZE)
        for i in range(BATCH_SIZE):
            prev, curr, out = _extract_sample(tasks, i, ALL_NAMES)
            assert prev == set()
            nf_curr = curr - set(FORCING_NAMES)
            assert nf_curr == out

    def test_basic_constraints(self, sampler):
        random.seed(42)
        tasks = sampler.sample(None, BATCH_SIZE)
        _validate_basic_constraints(
            tasks, ALL_NAMES, NON_FORCING, FORCING_NAMES, 1, 1, BATCH_SIZE
        )


class TestInfillConstraints:
    @pytest.fixture()
    def sampler(self):
        return _make_sampler(
            TaskType.INFILL,
            all_names=ALL_NAMES,
            forcing_names=FORCING_NAMES,
        )

    def test_no_previous_step_inputs(self, sampler):
        random.seed(42)
        tasks = sampler.sample(None, BATCH_SIZE)
        for i in range(BATCH_SIZE):
            prev, _, _ = _extract_sample(tasks, i, ALL_NAMES)
            assert prev == set()

    def test_current_inputs_disjoint_from_outputs(self, sampler):
        random.seed(42)
        tasks = sampler.sample(None, BATCH_SIZE)
        for i in range(BATCH_SIZE):
            _, curr, out = _extract_sample(tasks, i, ALL_NAMES)
            assert (
                curr & out == set()
            ), f"Sample {i}: overlap between inputs {curr} and outputs {out}"

    def test_basic_constraints(self, sampler):
        random.seed(42)
        tasks = sampler.sample(None, BATCH_SIZE)
        _validate_basic_constraints(
            tasks, ALL_NAMES, NON_FORCING, FORCING_NAMES, 1, 1, BATCH_SIZE
        )


class TestPredictionConstraints:
    @pytest.fixture()
    def sampler(self):
        return _make_sampler(
            TaskType.PREDICTION,
            all_names=ALL_NAMES,
            forcing_names=FORCING_NAMES,
        )

    def test_no_current_step_inputs(self, sampler):
        random.seed(42)
        tasks = sampler.sample(None, BATCH_SIZE)
        for i in range(BATCH_SIZE):
            _, curr, _ = _extract_sample(tasks, i, ALL_NAMES)
            assert curr == set()

    def test_basic_constraints(self, sampler):
        random.seed(42)
        tasks = sampler.sample(None, BATCH_SIZE)
        _validate_basic_constraints(
            tasks, ALL_NAMES, NON_FORCING, FORCING_NAMES, 1, 1, BATCH_SIZE
        )


class TestInfillPredictionConstraints:
    @pytest.fixture()
    def sampler(self):
        return _make_sampler(
            TaskType.INFILL_PREDICTION,
            all_names=ALL_NAMES,
            forcing_names=FORCING_NAMES,
        )

    def test_has_previous_and_current_inputs(self, sampler):
        random.seed(42)
        tasks = sampler.sample(None, BATCH_SIZE)
        for i in range(BATCH_SIZE):
            prev, curr, _ = _extract_sample(tasks, i, ALL_NAMES)
            assert len(prev) >= 1
            assert len(curr) >= 1

    def test_current_inputs_disjoint_from_outputs(self, sampler):
        random.seed(42)
        tasks = sampler.sample(None, BATCH_SIZE)
        for i in range(BATCH_SIZE):
            _, curr, out = _extract_sample(tasks, i, ALL_NAMES)
            assert curr & out == set()

    def test_basic_constraints(self, sampler):
        random.seed(42)
        tasks = sampler.sample(None, BATCH_SIZE)
        _validate_basic_constraints(
            tasks, ALL_NAMES, NON_FORCING, FORCING_NAMES, 1, 1, BATCH_SIZE
        )


class TestCombinedAllConstraints:
    @pytest.fixture()
    def sampler(self):
        return _make_sampler(
            TaskType.COMBINED_ALL,
            all_names=ALL_NAMES,
            forcing_names=FORCING_NAMES,
        )

    def test_has_at_least_one_input(self, sampler):
        random.seed(42)
        tasks = sampler.sample(None, BATCH_SIZE)
        for i in range(BATCH_SIZE):
            prev, curr, _ = _extract_sample(tasks, i, ALL_NAMES)
            assert len(prev | curr) >= 1

    def test_basic_constraints(self, sampler):
        random.seed(42)
        tasks = sampler.sample(None, BATCH_SIZE)
        _validate_basic_constraints(
            tasks, ALL_NAMES, NON_FORCING, FORCING_NAMES, 1, 1, BATCH_SIZE
        )


# ---------------------------------------------------------------------------
# Loss scale tests
# ---------------------------------------------------------------------------


class TestLossScale:
    def test_output_data_mask_uses_loss_scale(self):
        loss_scale = 2.5
        sampler = _make_sampler(TaskType.PREDICTION, loss_scale=loss_scale)
        random.seed(42)
        tasks = sampler.sample(None, 4)
        for name in tasks.output_data_mask:
            for i in range(4):
                val = tasks.output_data_mask[name][i].item()
                assert val == 0.0 or val == loss_scale

    def test_default_loss_scale_is_one(self):
        sampler = _make_sampler(TaskType.PREDICTION)
        random.seed(42)
        tasks = sampler.sample(None, 4)
        for name in tasks.output_data_mask:
            for i in range(4):
                val = tasks.output_data_mask[name][i].item()
                assert val == 0.0 or val == 1.0


# ---------------------------------------------------------------------------
# data_mask filtering tests
# ---------------------------------------------------------------------------


class TestDataMaskFiltering:
    def test_absent_variables_excluded_from_inputs_and_outputs(self):
        all_names = ["a", "b", "c", "d"]
        sampler = _make_sampler(
            TaskType.PREDICTION,
            all_names=all_names,
            forcing_names=[],
            min_in=1,
            min_out=1,
        )
        data_mask = {
            "a": torch.tensor([True, False]),
            "b": torch.tensor([True, True]),
            "c": torch.tensor([True, True]),
            "d": torch.tensor([False, True]),
        }
        random.seed(42)
        tasks = sampler.sample(data_mask, 2)
        _, _, out_0 = _extract_sample(tasks, 0, all_names)
        prev_0, _, _ = _extract_sample(tasks, 0, all_names)
        _, _, out_1 = _extract_sample(tasks, 1, all_names)
        prev_1, _, _ = _extract_sample(tasks, 1, all_names)

        assert "d" not in out_0 and "d" not in prev_0
        assert "a" not in out_1 and "a" not in prev_1

    def test_none_data_mask_uses_all_variables(self):
        sampler = _make_sampler(TaskType.PREDICTION, all_names=["a", "b", "c"])
        random.seed(42)
        tasks = sampler.sample(None, 16)
        seen = set()
        for name in tasks.output_data_mask:
            for i in range(16):
                if tasks.output_data_mask[name][i].item() > 0:
                    seen.add(name)
        assert seen == {"a", "b", "c"}

    def test_variable_not_in_data_mask_treated_as_available(self):
        sampler = _make_sampler(
            TaskType.PREDICTION, all_names=["a", "b"], forcing_names=[]
        )
        data_mask = {"a": torch.tensor([True] * 64)}
        random.seed(42)
        tasks = sampler.sample(data_mask, 64)
        seen = set()
        for i in range(64):
            prev, _, out = _extract_sample(tasks, i, ["a", "b"])
            seen.update(prev | out)
        assert "b" in seen

    def test_device_propagated_from_data_mask(self):
        sampler = _make_sampler(TaskType.PREDICTION, all_names=["a", "b"])
        data_mask = {"a": torch.tensor([True], device="cpu")}
        tasks = sampler.sample(data_mask, 1)
        for tensor in tasks.previous_step_input_mask.values():
            assert tensor.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Zero-weight task exclusion
# ---------------------------------------------------------------------------


class TestZeroWeightExclusion:
    def test_zero_weight_task_never_sampled(self):
        weights = TaskWeights(
            auto_encode=0.0,
            infill=0.0,
            prediction=1.0,
            infill_prediction=0.0,
            combined_all=0.0,
        )
        config = TaskSamplingConfig(task_weights=weights)
        sampler = TaskSampler(config, ALL_NAMES, FORCING_NAMES)
        random.seed(42)
        for _ in range(100):
            tasks = sampler.sample(None, BATCH_SIZE)
            for i in range(BATCH_SIZE):
                _, curr, _ = _extract_sample(tasks, i, ALL_NAMES)
                assert curr == set(), "prediction should have no current-step inputs"


# ---------------------------------------------------------------------------
# Per-sample independence
# ---------------------------------------------------------------------------


class TestPerSampleIndependence:
    def test_different_samples_can_get_different_outputs(self):
        sampler = _make_sampler(TaskType.PREDICTION, all_names=ALL_NAMES)
        random.seed(42)
        tasks = sampler.sample(None, 32)
        outputs_per_sample = []
        for i in range(32):
            _, _, out = _extract_sample(tasks, i, ALL_NAMES)
            outputs_per_sample.append(frozenset(out))
        assert len(set(outputs_per_sample)) > 1

    def test_different_data_masks_give_different_pools(self):
        all_names = ["a", "b", "c"]
        sampler = _make_sampler(
            TaskType.PREDICTION, all_names=all_names, forcing_names=[]
        )
        data_mask = {
            "a": torch.tensor([True, False]),
            "b": torch.tensor([True, True]),
            "c": torch.tensor([True, True]),
        }
        random.seed(42)
        tasks = sampler.sample(data_mask, 2)
        prev_0, _, _ = _extract_sample(tasks, 0, all_names)
        prev_1, _, _ = _extract_sample(tasks, 1, all_names)
        assert "a" not in prev_1


# ---------------------------------------------------------------------------
# Statistical variable selection tests (run many samples)
# ---------------------------------------------------------------------------

N_STAT_BATCHES = 200
STAT_BATCH_SIZE = 16


def _collect_stats(
    sampler: TaskSampler,
    all_names: list[str],
    n_batches: int = N_STAT_BATCHES,
    batch_size: int = STAT_BATCH_SIZE,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Count how often each variable appears as prev input, curr input, output."""
    prev_counts: dict[str, int] = {n: 0 for n in all_names}
    curr_counts: dict[str, int] = {n: 0 for n in all_names}
    out_counts: dict[str, int] = {n: 0 for n in all_names}
    for _ in range(n_batches):
        tasks = sampler.sample(None, batch_size)
        for i in range(batch_size):
            prev, curr, out = _extract_sample(tasks, i, all_names)
            for n in prev:
                prev_counts[n] += 1
            for n in curr:
                curr_counts[n] += 1
            for n in out:
                out_counts[n] += 1
    return prev_counts, curr_counts, out_counts


class TestStatisticalSymmetry:
    def test_infill_output_uniformity(self):
        """Each non-forcing variable should appear as output with ~equal frequency."""
        names = ["a", "b", "c", "d", "e"]
        forcing = ["d", "e"]
        non_forcing = ["a", "b", "c"]
        sampler = _make_sampler(TaskType.INFILL, all_names=names, forcing_names=forcing)
        random.seed(0)
        _, _, out_counts = _collect_stats(sampler, names)
        nf_counts = [out_counts[n] for n in non_forcing]
        mean = sum(nf_counts) / len(nf_counts)
        for n in non_forcing:
            assert abs(out_counts[n] - mean) / mean < 0.3, (
                f"Variable {n} appears as output {out_counts[n]} times "
                f"vs mean {mean:.0f}"
            )

    def test_infill_input_output_count_symmetry(self):
        """For disjoint task with min_in=min_out, E[n_out] ~ E[n_in_contested]."""
        names = ["a", "b", "c", "d", "e", "f"]
        sampler = _make_sampler(
            TaskType.INFILL,
            all_names=names,
            forcing_names=[],
            min_in=1,
            min_out=1,
        )
        random.seed(0)
        total_in = 0
        total_out = 0
        n = N_STAT_BATCHES * STAT_BATCH_SIZE
        for _ in range(N_STAT_BATCHES):
            tasks = sampler.sample(None, STAT_BATCH_SIZE)
            for i in range(STAT_BATCH_SIZE):
                _, curr, out = _extract_sample(tasks, i, names)
                total_in += len(curr)
                total_out += len(out)
        mean_in = total_in / n
        mean_out = total_out / n
        assert abs(mean_in - mean_out) / max(mean_in, mean_out) < 0.15, (
            f"Mean inputs {mean_in:.2f} vs mean outputs {mean_out:.2f} "
            f"differ by more than 15%"
        )

    def test_prediction_output_uniformity(self):
        names = ["a", "b", "c", "d"]
        sampler = _make_sampler(TaskType.PREDICTION, all_names=names, forcing_names=[])
        random.seed(0)
        _, _, out_counts = _collect_stats(sampler, names)
        counts = list(out_counts.values())
        mean = sum(counts) / len(counts)
        for n in names:
            assert abs(out_counts[n] - mean) / mean < 0.3

    def test_prediction_input_uniformity(self):
        names = ["a", "b", "c", "d"]
        sampler = _make_sampler(TaskType.PREDICTION, all_names=names, forcing_names=[])
        random.seed(0)
        prev_counts, _, _ = _collect_stats(sampler, names)
        counts = list(prev_counts.values())
        mean = sum(counts) / len(counts)
        for n in names:
            assert abs(prev_counts[n] - mean) / mean < 0.3


class TestForcingNeverInOutputs:
    @pytest.mark.parametrize("task_type", list(TaskType))
    def test_forcing_never_output(self, task_type):
        names = ["a", "b", "c", "d", "e"]
        forcing = ["d", "e"]
        non_forcing = ["a", "b", "c"]
        if task_type == TaskType.INFILL:
            min_in, min_out = 1, 1
        elif task_type == TaskType.INFILL_PREDICTION:
            min_in, min_out = 1, 1
        else:
            min_in, min_out = 1, 1

        sampler = _make_sampler(
            task_type,
            all_names=names,
            forcing_names=forcing,
            min_in=min_in,
            min_out=min_out,
        )
        random.seed(42)
        for _ in range(50):
            tasks = sampler.sample(None, 4)
            assert set(tasks.output_data_mask.keys()) == set(non_forcing)
            for i in range(4):
                _, _, out = _extract_sample(tasks, i, names)
                assert out.issubset(set(non_forcing))


class TestForcingAsInputOnly:
    def test_forcing_appears_in_inputs(self):
        """Forcing variables should appear as inputs in at least some samples."""
        names = ["a", "b", "c", "d", "e"]
        forcing = ["d", "e"]
        sampler = _make_sampler(
            TaskType.INFILL,
            all_names=names,
            forcing_names=forcing,
        )
        random.seed(42)
        prev_counts, curr_counts, _ = _collect_stats(sampler, names)
        for f in forcing:
            assert curr_counts[f] > 0, f"Forcing variable {f} never used as input"


class TestFullCoverage:
    def test_all_non_forcing_can_be_outputs(self):
        """With enough samples, there's a sample where all non-forcing are outputs."""
        names = ["a", "b", "c"]
        sampler = _make_sampler(TaskType.PREDICTION, all_names=names, forcing_names=[])
        random.seed(42)
        found = False
        for _ in range(500):
            tasks = sampler.sample(None, 1)
            _, _, out = _extract_sample(tasks, 0, names)
            if out == set(names):
                found = True
                break
        assert found, "Never found sample with all variables as outputs"

    def test_all_can_be_inputs(self):
        """With enough samples, there's a sample where all variables are inputs."""
        names = ["a", "b", "c"]
        sampler = _make_sampler(TaskType.PREDICTION, all_names=names, forcing_names=[])
        random.seed(42)
        found = False
        for _ in range(500):
            tasks = sampler.sample(None, 1)
            prev, _, _ = _extract_sample(tasks, 0, names)
            if prev == set(names):
                found = True
                break
        assert found, "Never found sample with all variables as inputs"


class TestMultipleTaskWeights:
    def test_mixed_weights_sample_multiple_task_types(self):
        """With multiple tasks enabled, different task patterns should appear."""
        weights = TaskWeights(
            auto_encode=1.0,
            infill=0.0,
            prediction=1.0,
            infill_prediction=0.0,
            combined_all=0.0,
        )
        config = TaskSamplingConfig(task_weights=weights)
        sampler = TaskSampler(config, ALL_NAMES, FORCING_NAMES)
        random.seed(42)
        has_prev_only = False
        has_curr_only = False
        for _ in range(200):
            tasks = sampler.sample(None, 1)
            prev, curr, _ = _extract_sample(tasks, 0, ALL_NAMES)
            if len(prev) > 0 and len(curr) == 0:
                has_prev_only = True
            if len(prev) == 0 and len(curr) > 0:
                has_curr_only = True
        assert has_prev_only, "Expected prediction tasks (prev only)"
        assert has_curr_only, "Expected auto_encode tasks (curr only)"


class TestGetTaskLossScale:
    def test_returns_correct_scale(self):
        weights = TaskWeights(prediction_loss_scale=3.0)
        assert _get_task_loss_scale(weights, TaskType.PREDICTION) == 3.0

    def test_default_scale_is_one(self):
        weights = TaskWeights()
        for task_type in TaskType:
            assert _get_task_loss_scale(weights, task_type) == 1.0


# ---------------------------------------------------------------------------
# InfillPredictionStepConfig tests
# ---------------------------------------------------------------------------

IMG_SHAPE = (16, 32)
TIMESTEP = datetime.timedelta(hours=6)
STEP_ALL_NAMES = ["a", "b", "c", "forcing_x"]
STEP_FORCING = ["forcing_x"]
STEP_NON_FORCING = ["a", "b", "c"]


def _norm_config(names: list[str]) -> NetworkAndLossNormalizationConfig:
    return NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={n: 0.0 for n in names},
            stds={n: 1.0 for n in names},
        ),
    )


def _make_step_config(
    all_names: list[str] | None = None,
    forcing_names: list[str] | None = None,
    in_names: list[str] | None = None,
    out_names: list[str] | None = None,
) -> InfillPredictionStepConfig:
    if all_names is None:
        all_names = STEP_ALL_NAMES
    if forcing_names is None:
        forcing_names = STEP_FORCING
    if in_names is None:
        in_names = ["a", "b", "forcing_x"]
    if out_names is None:
        out_names = ["a", "b"]
    return InfillPredictionStepConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet",
            config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
        ),
        all_names=all_names,
        forcing_names=forcing_names,
        normalization=_norm_config(all_names),
        inference_scheme=InferenceSchemeConfig(
            in_names=in_names,
            out_names=out_names,
        ),
    )


def _make_dataset_info() -> DatasetInfo:
    device = fme.get_device()
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(IMG_SHAPE[0], device=device),
            lon=torch.zeros(IMG_SHAPE[1], device=device),
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(7, device=device),
            bk=torch.arange(7, device=device),
        ),
        timestep=TIMESTEP,
    )


def _make_step(
    config: InfillPredictionStepConfig | None = None,
) -> InfillPredictionStep:
    if config is None:
        config = _make_step_config()
    dataset_info = _make_dataset_info()
    return config.get_step(dataset_info, lambda _: None)


def _tensor_dict(names: list[str], batch: int = 2) -> dict[str, torch.Tensor]:
    device = fme.get_device()
    return {n: torch.randn(batch, *IMG_SHAPE, device=device) for n in names}


class TestInfillPredictionStepConfigValidation:
    def test_valid_construction(self):
        config = _make_step_config()
        assert config.all_names == STEP_ALL_NAMES

    def test_in_name_not_in_all_names_raises(self):
        with pytest.raises(ValueError, match="inference_scheme.in_names"):
            _make_step_config(in_names=["a", "b", "not_there"])

    def test_out_name_not_in_all_names_raises(self):
        with pytest.raises(ValueError, match="inference_scheme.out_names"):
            _make_step_config(out_names=["a", "not_there"])

    def test_forcing_not_in_all_names_raises(self):
        with pytest.raises(ValueError, match="forcing_names"):
            _make_step_config(forcing_names=["not_there"])

    def test_forcing_in_out_names_raises(self):
        with pytest.raises(ValueError, match="forcing variable"):
            _make_step_config(
                all_names=["a", "b", "c", "f"],
                forcing_names=["f"],
                in_names=["a", "f"],
                out_names=["a", "f"],
            )

    def test_include_channel_mask_inputs_false_raises(self):
        with pytest.raises(ValueError, match="include_channel_mask_inputs"):
            InfillPredictionStepConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet",
                    config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
                ),
                all_names=["a", "b"],
                forcing_names=[],
                normalization=_norm_config(["a", "b"]),
                inference_scheme=InferenceSchemeConfig(in_names=["a"], out_names=["a"]),
                include_channel_mask_inputs=False,
            )


class TestInfillPredictionStepConfigProperties:
    @pytest.fixture()
    def config(self):
        return _make_step_config()

    def test_input_names(self, config):
        assert set(config.input_names) == {"a", "b", "forcing_x"}

    def test_output_names(self, config):
        assert set(config.output_names) == {"a", "b"}

    def test_loss_names(self, config):
        assert set(config.loss_names) == set(STEP_NON_FORCING)

    def test_all_training_names(self, config):
        assert config.all_training_names == STEP_ALL_NAMES

    def test_allow_missing_variables(self, config):
        assert config.allow_missing_variables is True

    def test_n_ic_timesteps(self, config):
        assert config.n_ic_timesteps == 1

    def test_non_forcing_names(self, config):
        assert config.non_forcing_names == STEP_NON_FORCING

    def test_next_step_forcing_names(self, config):
        assert config.get_next_step_forcing_names() == []

    def test_next_step_forcing_names_with_config(self):
        config = InfillPredictionStepConfig(
            builder=ModuleSelector(
                type="SphericalFourierNeuralOperatorNet",
                config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
            ),
            all_names=["a", "b", "forcing_x"],
            forcing_names=["forcing_x"],
            normalization=_norm_config(["a", "b", "forcing_x"]),
            inference_scheme=InferenceSchemeConfig(
                in_names=["a", "b", "forcing_x"],
                out_names=["a", "b"],
                next_step_forcing_names=["forcing_x"],
            ),
        )
        assert config.get_next_step_forcing_names() == ["forcing_x"]

    def test_prognostic_names(self, config):
        assert set(config.prognostic_names) == {"a", "b"}


class TestInfillPredictionStepRegistry:
    def test_selector_construction(self):
        config = _make_step_config()
        selector = StepSelector(
            type="infill_prediction",
            config=dataclasses.asdict(config),
        )
        assert selector.all_training_names == STEP_ALL_NAMES
        assert set(selector.input_names) == {"a", "b", "forcing_x"}
        assert set(selector.output_names) == {"a", "b"}

    def test_selector_get_step(self):
        config = _make_step_config()
        selector = StepSelector(
            type="infill_prediction",
            config=dataclasses.asdict(config),
        )
        dataset_info = _make_dataset_info()
        step = selector.get_step(dataset_info)
        assert isinstance(step, InfillPredictionStep)


class TestInfillPredictionStep:
    def test_forward_with_full_input(self):
        step = _make_step()
        input_data = _tensor_dict(STEP_ALL_NAMES)
        next_step = _tensor_dict(step.next_step_input_names)
        data_mask = {
            n: torch.ones(2, dtype=torch.bool, device=fme.get_device())
            for n in STEP_ALL_NAMES
        }
        output = step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step,
                data_mask=data_mask,
            )
        )
        assert set(output.keys()) == set(STEP_NON_FORCING)
        for v in output.values():
            assert v.shape == (2, *IMG_SHAPE)

    def test_forward_with_partial_input(self):
        step = _make_step()
        inference_names = ["a", "b", "forcing_x"]
        input_data = _tensor_dict(inference_names)
        next_step = _tensor_dict(step.next_step_input_names)
        output = step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step,
            )
        )
        assert set(output.keys()) == set(STEP_NON_FORCING)
        for v in output.values():
            assert v.shape == (2, *IMG_SHAPE)

    def test_output_excludes_forcing(self):
        step = _make_step()
        input_data = _tensor_dict(STEP_ALL_NAMES)
        next_step = _tensor_dict(step.next_step_input_names)
        output = step.step(
            StepArgs(
                input=input_data,
                next_step_input_data=next_step,
            )
        )
        for name in STEP_FORCING:
            assert name not in output

    def test_modules_list(self):
        step = _make_step()
        assert isinstance(step.modules, nn.ModuleList)
        assert len(step.modules) >= 1

    def test_get_and_load_state(self):
        step = _make_step()
        state = step.get_state()
        assert "module" in state
        step.load_state(state)
