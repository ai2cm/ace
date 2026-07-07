<!-- Planning artifact for early review. Deleted before merge. -->

# Add normalized correction metrics to the inference aggregators

Corrector-equipped runs now log how much the model leans on its corrector: normalized-space
correction-magnitude metrics, computed from the `StepDiagnostics.delta` carried on prediction
data (#1335), merged into the existing `time_mean_norm` / `mean_norm` metric groups in both the
evaluator and no-target inference aggregators. Metrics are uniform over all corrected variables
(no per-variable category filtering).

---

## `fme/ace/aggregator/inference/correction.py` (new)

```python
def compute_correction_norm(
    prediction: TensorMapping,
    delta: TensorMapping,
    normalize: Callable[[TensorMapping], TensorDict],
) -> TensorDict:
    """The normalized correction, over the delta's keys.

    Computed as normalize(prediction) - normalize(prediction - delta) —
    reconstructing the uncorrected output rather than normalizing the raw
    delta, which would wrongly subtract the per-variable mean offset. Off-mask
    cells stay NaN (NaN - NaN), consistent with the masked prediction.
    Raises ValueError if a delta key is dropped by the normalizer (the
    StandardNormalizer silently filters to known names; a silently vanishing
    metric would be worse than a loud failure)."""


class CorrectionDeltaTimeMeanAggregator:
    """Time-mean maps and scalars of the normalized correction. Silent (empty
    logs/dataset) until non-empty data is recorded."""

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ): ...

    def record_batch(self, correction_norm: TensorMapping) -> None: ...

    def get_logs(self, label: str) -> dict[str, float | Image]:
        # correction_magnitude/{name}: area-weighted global mean of the
        #   time-mean |normalized correction| (mask-aware: reduction goes
        #   through ops.area_weighted_mean(..., name=name), so NaN off-mask
        #   cells drop out of the weights)
        # correction_magnitude/channel_mean: arithmetic mean of those scalars
        #   over the corrected variables (mirrors time_mean_norm/rmse/channel_mean)
        # correction_map/{name}: signed time-mean map image (diverging palette)
        ...

    def get_dataset(self) -> xr.Dataset:
        # correction_map-{name}, dims ("lat", "lon") — signed time-mean map
        ...


class CorrectionDeltaMeanAggregator:
    """Per-forecast-step area-weighted series of the normalized correction,
    mirroring the mean_norm MeanAggregator. Built only when time series are
    enabled (inline training drops it via enable_time_series)."""

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        n_timesteps: int,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        # Two AreaWeightedSingleTargetReducedMetric instances (reuse from .reduced):
        #   weighted_correction_magnitude -> ops.area_weighted_mean_dict(|corr|)
        #   weighted_correction_std      -> ops.area_weighted_std_dict(corr)
        ...

    def record_batch(self, correction_norm: TensorMapping, i_time_start: int) -> None: ...

    def get_logs(self, label: str, step_slice: slice | None = None) -> dict[str, Any]:
        # One wandb Table under "{label}/correction_series" — a key distinct
        # from the MeanAggregator's "{label}/series" because to_inference_logs
        # strips after the final "/" and the two tables would collide.
        ...

    def get_dataset(self) -> xr.Dataset:
        # weighted_correction_{magnitude,std}-{name} over forecast_step
        ...


class CorrectionDeltaRecorder:
    """Facade owned by each inference aggregator: reads the delta off the
    batch's StepDiagnostics, computes the normalized correction once, and
    dispatches to both sub-aggregators.

    A facade rather than a registered sub-aggregator because (a) the metrics
    consume data.step_diagnostics, which InferenceBatchData does not carry,
    and (b) the logs merge into the existing time_mean_norm / mean_norm label
    groups, which the one-name-per-aggregator registry cannot express."""

    TIME_MEAN_LABEL = "time_mean_norm"
    TIME_SERIES_LABEL = "mean_norm"

    def __init__(
        self,
        normalize: Callable[[TensorMapping], TensorDict],
        time_mean: CorrectionDeltaTimeMeanAggregator,
        time_series: CorrectionDeltaMeanAggregator | None,
    ): ...

    def record(
        self,
        prediction: TensorMapping,
        step_diagnostics: StepDiagnostics | None,
        i_time_start: int,
    ) -> None:
        # Silent no-op when step_diagnostics is None (no corrector ran, or
        # coupled inference, which never attaches diagnostics).
        # Raises ValueError if the delta's time dim does not match the
        # prediction window's — the alignment is guaranteed by construction
        # in Stepper.predict, but asserted here rather than trusted.
        ...

    def summary_logs(self) -> dict[str, Any]: ...          # time_mean under "time_mean_norm"
    def time_series_logs(self, step_slice: slice) -> dict[str, Any]: ...  # under "mean_norm"
    def diagnostics(self) -> dict[str, Any]:
        # {"time_mean_norm_correction": ..., "mean_norm_correction": ...} —
        # extra entries for flush_diagnostics; suffixed names because the
        # plain "time_mean_norm"/"mean_norm" diagnostics files belong to the
        # existing aggregators.
        ...


def build_correction_delta_recorder(
    gridded_operations: GriddedOperations,
    n_timesteps: int,
    variable_metadata: Mapping[str, VariableMetadata] | None,
    enable_time_series: bool,
    normalize: Callable[[TensorMapping], TensorDict],
) -> CorrectionDeltaRecorder: ...
```

## `fme/ace/aggregator/inference/main.py` (modified)

```python
def build_inference_evaluator_aggregator(
    ...,
    log_correction_metrics: bool = True,  # NEW
) -> "InferenceEvaluatorAggregator":
    # NEW: when enabled, builds a CorrectionDeltaRecorder from
    # dataset_info.gridded_operations / variable_metadata, n_timesteps,
    # enable_time_series, and the existing normalize argument.
    ...


class InferenceEvaluatorAggregatorConfig:
    log_correction_metrics: bool = True  # NEW — also on the Legacy flag config

    def build(self, ...) -> "InferenceEvaluatorAggregator":  # CHANGED — forwards the flag
        ...


class InferenceEvaluatorAggregator:
    def __init__(
        self,
        ...,
        correction: CorrectionDeltaRecorder | None = None,  # NEW
    ): ...

    def record_batch(self, data: PairedData) -> InferenceLogs:
        # CHANGED — self._correction.record(prediction=data.prediction,
        #   step_diagnostics=data.step_diagnostics,
        #   i_time_start=self._n_timesteps_seen) before slicing logs
        ...

    def _get_logs(self): ...                    # CHANGED — merge correction.summary_logs()
    def get_summary(self): ...                  # CHANGED — same merge (loss key untouched)
    def _get_inference_logs_slice(self, ...): ...  # CHANGED — merge correction.time_series_logs(step_slice)
    def flush_diagnostics(self, subdir=None): ...  # CHANGED — merge correction.diagnostics()


class InferenceAggregatorConfig:
    log_correction_metrics: bool = True  # NEW

    def build(
        self,
        dataset_info: DatasetInfo,
        n_timesteps: int,
        output_dir: str | None = None,
        save_diagnostics: bool = True,
        normalize: Callable[[TensorMapping], TensorDict] | None = None,  # NEW
    ) -> "InferenceAggregator":
        # The no-target aggregator has no normalizer today; the recorder is
        # built only when the flag is on AND normalize is provided — callers
        # that do not pass one keep current behavior (metrics silently skipped).
        ...


class InferenceAggregator:
    def __init__(
        self,
        ...,
        correction: CorrectionDeltaRecorder | None = None,  # NEW
    ):
        # CHANGED — _log_time_series also true when the recorder carries a
        # time-series aggregator (mean_norm has no other member here)
        ...

    # record_batch / _get_logs / get_summary / _get_inference_logs_slice /
    # flush_diagnostics: CHANGED — same correction merges as the evaluator
```

## `fme/ace/inference/inference.py` (modified)

```python
# CHANGED — the config.aggregator.build(...) call gains
#   normalize=stepper.normalizer.normalize
# (evaluator.py already passes it; no change there)
```

---

## Tests

## `fme/ace/aggregator/inference/test_correction.py` (new)

```python
# Identity normalizer (lambda x: dict(x)) where hand-computable exactness matters.

def test_compute_correction_norm_uses_normalized_difference():
    # GOAL: with a scaling normalizer, result equals
    # normalize(pred) - normalize(pred - delta), not normalize(delta).

def test_compute_correction_norm_raises_on_unknown_delta_key():
    # GOAL: a delta key the normalizer drops raises ValueError, not a silent skip.

def test_time_mean_aggregator_constant_offset():
    # GOAL: exact values — constant delta per variable yields
    # correction_magnitude/{var} == |delta|, channel_mean == mean of those,
    # and a constant signed correction_map.

def test_time_mean_aggregator_masked_cells_do_not_poison_scalars():
    # GOAL: NaN off-mask delta cells + a mask-aware GriddedOperations yield
    # finite scalars matching a hand computation over on-mask cells only.

def test_time_mean_aggregator_null_masking_matches_unmasked_hand_computation():
    # GOAL: with NullSpatialMasking (no NaNs), values are identical to the
    # unmasked hand-computed case.

def test_mean_aggregator_constant_offset_series():
    # GOAL: per-step weighted_correction_magnitude == |delta| and
    # weighted_correction_std == 0 for a spatially constant correction;
    # a spatially varying case matches ops.area_weighted_std_dict.

def test_aggregators_silent_without_data():
    # GOAL: empty logs and empty dataset before any recorded batch.
    # PARAMETERIZE: both aggregator classes.

def test_recorder_raises_on_time_dim_mismatch():
    # GOAL: a delta whose time dim differs from the prediction window's raises.

def test_evaluator_aggregator_logs_correction_metrics():
    # GOAL: integration — record_batch with a PairedData carrying a real
    # StepDiagnostics (constant-offset delta) logs
    # time_mean_norm/correction_magnitude/{var,channel_mean},
    # mean_norm/weighted_correction_{magnitude,std}/{var}; only corrected
    # variables appear; existing time_mean_norm/rmse metrics are unaffected;
    # flush_diagnostics writes the correction datasets.

def test_evaluator_aggregator_silent_paths():
    # GOAL: no correction logs and no diagnostics files.
    # PARAMETERIZE: log_correction_metrics=False; step_diagnostics=None.

def test_inline_training_drops_correction_series():
    # GOAL: enable_time_series=False keeps time_mean_norm correction metrics
    # but drops the mean_norm correction series.

def test_no_target_aggregator_logs_correction_metrics():
    # GOAL: InferenceAggregatorConfig.build(normalize=...) logs the same
    # metric keys from prediction-only PairedData.

def test_no_target_aggregator_skips_without_normalizer():
    # GOAL: build without normalize logs nothing correction-related
    # (backward compatibility for callers that do not pass one).

def test_config_defaults_enable_correction_metrics():
    # GOAL: both configs default log_correction_metrics=True.
```
