import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Generic, Protocol, TypeVar

from fme.core.generics.aggregator import InferenceAggregatorABC, InferenceLogs
from fme.core.generics.data import InferenceDataABC
from fme.core.generics.writer import NullDataWriter, WriterABC
from fme.core.timing import GlobalTimer
from fme.core.wandb import WandB

PS = TypeVar("PS")  # prognostic state
FD = TypeVar("FD", contravariant=True)  # forcing data
SD = TypeVar("SD", covariant=True)  # stepped data

# Invariant TypeVars used where Generic appears in both input and output position.
PS_I = TypeVar("PS_I")
FD_I = TypeVar("FD_I")
SD_I = TypeVar("SD_I")


class PredictFunction(Protocol, Generic[PS, FD, SD]):
    def __call__(
        self,
        initial_condition: PS,
        forcing: FD,
        compute_derived_variables: bool = False,
    ) -> tuple[SD, PS]: ...


class Looper(Generic[PS, FD, SD]):
    """
    Class for stepping a model forward arbitarily many times.
    """

    def __init__(
        self,
        predict: PredictFunction[PS, FD, SD],
        data: InferenceDataABC[PS, FD],
    ):
        """
        Args:
            predict: The prediction function to use.
            data: The data to use.
        """
        self._predict = predict
        self._prognostic_state = data.initial_condition
        self._len = len(data.loader)
        self._loader = iter(data.loader)

    def __iter__(self) -> Iterator[SD]:
        return self

    def __len__(self) -> int:
        return self._len

    def __next__(self) -> SD:
        """Return predictions for the time period corresponding to the next batch
        of forcing data. Also returns the forcing data.
        """
        timer = GlobalTimer.get_instance()
        with timer.context("data_loading"):
            try:
                forcing_data = next(self._loader)
            except StopIteration:
                raise StopIteration
        output_data, self._prognostic_state = self._predict(
            self._prognostic_state,
            forcing=forcing_data,
            compute_derived_variables=True,
        )
        return output_data

    def get_prognostic_state(self) -> PS:
        return self._prognostic_state


class PredictionPromise(Generic[PS_I, SD_I]):
    """Deferred handle for one slot of a batched ``BatchedPredictor`` call.

    Created by ``BatchedPredictor.submit``. The first call to ``resolve()``
    on any sibling promise triggers the single underlying forward pass; all
    promises then return their own slice of the split outputs.
    """

    def __init__(self, predictor: "BatchedPredictor[PS_I, Any, SD_I]", slot_id: int):
        self._predictor = predictor
        self._slot_id = slot_id

    def resolve(self) -> tuple[SD_I, PS_I]:
        self._predictor._ensure_flushed()
        return self._predictor._get_result(self._slot_id)


class BatchedPredictor(Generic[PS_I, FD_I, SD_I]):
    """Coordinates several ``PredictFunction`` calls into a single forward pass.

    Submitters call ``submit(ic, forcing)`` and receive a ``PredictionPromise``.
    The first ``promise.resolve()`` invokes the underlying predict function on
    inputs concatenated along the sample dimension and splits the outputs back
    into per-submission pieces. After all promises are resolved, call
    ``reset()`` to clear the queue for the next step.

    The concat / split callables make the abstraction independent of any
    concrete batch container, so the same coordinator can drive ACE-style
    ``BatchData`` runs and coupled-style ``CoupledBatchData`` runs.
    """

    def __init__(
        self,
        base_predict: "PredictFunction[PS_I, FD_I, SD_I]",
        concat_ic: Callable[[Sequence[PS_I]], PS_I],
        concat_forcing: Callable[[Sequence[FD_I]], FD_I],
        split_output: Callable[[SD_I, Sequence[int]], list[SD_I]],
        split_state: Callable[[PS_I, Sequence[int]], list[PS_I]],
        sample_size_of_state: Callable[[PS_I], int],
    ):
        """``sample_size_of_state`` returns the per-call sample count from the
        initial condition. Take it from the IC, not the forcing, because some
        steppers broadcast a small forcing up to a larger IC (e.g. per-IC
        ensemble members) before stepping, and the split must match the
        post-broadcast output.
        """
        self._base_predict = base_predict
        self._concat_ic = concat_ic
        self._concat_forcing = concat_forcing
        self._split_output = split_output
        self._split_state = split_state
        self._sample_size_of_state = sample_size_of_state
        self._pending_ic: list[PS_I] = []
        self._pending_forcing: list[FD_I] = []
        self._sample_sizes: list[int] = []
        self._compute_derived_variables: bool | None = None
        self._results: list[tuple[SD_I, PS_I]] | None = None

    @property
    def num_pending(self) -> int:
        return len(self._pending_ic)

    def submit(
        self,
        initial_condition: PS_I,
        forcing: FD_I,
        compute_derived_variables: bool = True,
    ) -> PredictionPromise[PS_I, SD_I]:
        if self._results is not None:
            raise RuntimeError(
                "Cannot submit after flush. Call reset() before reusing the "
                "predictor."
            )
        if (
            self._compute_derived_variables is not None
            and self._compute_derived_variables != compute_derived_variables
        ):
            raise ValueError(
                "All submissions in a batched call must share "
                "compute_derived_variables."
            )
        self._compute_derived_variables = compute_derived_variables
        slot_id = len(self._pending_ic)
        self._pending_ic.append(initial_condition)
        self._pending_forcing.append(forcing)
        self._sample_sizes.append(self._sample_size_of_state(initial_condition))
        return PredictionPromise(self, slot_id)

    def _ensure_flushed(self) -> None:
        if self._results is not None:
            return
        if not self._pending_ic:
            raise RuntimeError(
                "Cannot flush BatchedPredictor with no pending submissions."
            )
        if len(self._pending_ic) == 1:
            sd, new_state = self._base_predict(
                self._pending_ic[0],
                self._pending_forcing[0],
                compute_derived_variables=bool(self._compute_derived_variables),
            )
            self._results = [(sd, new_state)]
            return
        merged_ic = self._concat_ic(self._pending_ic)
        merged_forcing = self._concat_forcing(self._pending_forcing)
        merged_sd, merged_state = self._base_predict(
            merged_ic,
            merged_forcing,
            compute_derived_variables=bool(self._compute_derived_variables),
        )
        outputs = self._split_output(merged_sd, self._sample_sizes)
        states = self._split_state(merged_state, self._sample_sizes)
        self._results = list(zip(outputs, states))

    def _get_result(self, slot_id: int) -> tuple[SD_I, PS_I]:
        if self._results is None:
            raise RuntimeError("Predictor has not been flushed.")
        return self._results[slot_id]

    def reset(self) -> None:
        self._pending_ic = []
        self._pending_forcing = []
        self._sample_sizes = []
        self._compute_derived_variables = None
        self._results = None


class LazyLooper(Generic[PS_I, FD_I, SD_I]):
    """Same iteration contract as ``Looper`` but defers prediction to a shared
    ``BatchedPredictor``.

    A driver loops over several ``LazyLooper`` instances in lock-step: it calls
    ``submit()`` on each (registering the next forcing window with the shared
    predictor) and then ``commit()`` on each (resolving the promise and
    updating internal prognostic state). One batched forward pass per round.
    """

    def __init__(
        self,
        predictor: BatchedPredictor[PS_I, FD_I, SD_I],
        data: InferenceDataABC[PS_I, FD_I],
    ):
        self._predictor = predictor
        self._prognostic_state = data.initial_condition
        self._loader = iter(data.loader)
        self._len = len(data.loader)
        self._pending: PredictionPromise[PS_I, SD_I] | None = None

    def __len__(self) -> int:
        return self._len

    def get_prognostic_state(self) -> PS_I:
        return self._prognostic_state

    def submit(self) -> PredictionPromise[PS_I, SD_I]:
        """Load the next forcing batch and submit it to the predictor.

        Raises ``StopIteration`` when the data loader is exhausted.
        """
        if self._pending is not None:
            raise RuntimeError("commit() must be called before submit() again.")
        timer = GlobalTimer.get_instance()
        with timer.context("data_loading"):
            forcing = next(self._loader)
        self._pending = self._predictor.submit(
            self._prognostic_state,
            forcing,
            compute_derived_variables=True,
        )
        return self._pending

    def commit(self) -> SD_I:
        """Resolve the pending promise, update internal state, return the output."""
        if self._pending is None:
            raise RuntimeError("submit() must be called before commit().")
        sd, new_state = self._pending.resolve()
        self._prognostic_state = new_state
        self._pending = None
        return sd


class WandBStepLogger:
    """Logs inference metrics to wandb with step tracking and optional key prefixing.

    The ``log`` method additionally accepts a per-call *label* override so that
    callers can mix prefixed and unprefixed keys through the same step counter.

    """

    def __init__(self, label: str = ""):
        self._wandb = WandB.get_instance()
        self._label = label
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def _prefix_label(
        self, log_dict: dict[str, Any], label: str | None = None
    ) -> dict[str, Any]:
        if label is None:
            label = self._label
        if label:  # not None and not ""
            log_dict = {f"{label}/{k}": v for k, v in log_dict.items()}
        return log_dict

    def log(self, logs: InferenceLogs, label: str | None = None) -> None:
        """Log each step in a sequence of logs."""
        for log_dict in logs:
            self.log_to_current_step(log_dict, label)
            self._step += 1

    def log_to_current_step(
        self, log_dict: dict[str, Any], label: str | None = None
    ) -> None:
        """Log to the current step without incrementing."""
        if len(log_dict) > 0:
            log_dict = self._prefix_label(log_dict, label)
            self._wandb.log(log_dict, step=self._step)


def get_record_to_wandb(label: str = "") -> WandBStepLogger:
    return WandBStepLogger(label=label)


class ConcurrentInferenceEntry(Generic[PS_I, FD_I, SD_I]):
    """One inference run participating in a concurrent group.

    Bundles the data, aggregator, writer, and log recorder for a single
    inline-inference configuration so several can be passed to
    ``run_concurrent_inference``.
    """

    def __init__(
        self,
        name: str,
        data: InferenceDataABC[PS_I, FD_I],
        aggregator: InferenceAggregatorABC[PS_I, SD_I],
        writer: WriterABC[PS_I, SD_I] | None = None,
        record_logs: Callable[[InferenceLogs], None] | None = None,
    ):
        self.name = name
        self.data = data
        self.aggregator = aggregator
        self.writer: WriterABC[PS_I, SD_I] = (
            writer if writer is not None else NullDataWriter()
        )
        self.record_logs: Callable[[InferenceLogs], None] = (
            record_logs
            if record_logs is not None
            else get_record_to_wandb(label=name).log
        )


def run_concurrent_inference(
    predictor: BatchedPredictor[PS_I, FD_I, SD_I],
    entries: Sequence[ConcurrentInferenceEntry[PS_I, FD_I, SD_I]],
):
    """Run several inference data loaders concurrently through a shared predictor.

    Each entry is independent in its data, aggregator, and writer, but they
    share the predictor's underlying model. At each step, every still-active
    entry submits its next forcing window and a single batched forward pass
    produces all outputs. Entries whose loaders exhaust are dropped from
    subsequent rounds, so this works for heterogeneous run lengths.
    """
    timer = GlobalTimer.get_instance()
    loopers: dict[str, LazyLooper[PS_I, FD_I, SD_I]] = {}
    n_windows: dict[str, int] = {}
    windows_done: dict[str, int] = {}
    entries_by_name: dict[str, ConcurrentInferenceEntry[PS_I, FD_I, SD_I]] = {}
    for entry in entries:
        entries_by_name[entry.name] = entry
        looper = LazyLooper(predictor, entry.data)
        loopers[entry.name] = looper
        n_windows[entry.name] = len(looper)
        windows_done[entry.name] = 0
        with timer.context("aggregator"):
            logs = entry.aggregator.record_initial_condition(
                entry.data.initial_condition
            )
        with timer.context("wandb_logging"):
            entry.record_logs(logs)
        with timer.context("data_writer"):
            entry.writer.write(entry.data.initial_condition, "initial_condition.nc")

    active = dict(loopers)
    while active:
        submitted: list[str] = []
        for name in list(active.keys()):
            looper = active[name]
            try:
                looper.submit()
            except StopIteration:
                with timer.context("data_writer"):
                    entries_by_name[name].writer.write(
                        looper.get_prognostic_state(), "restart.nc"
                    )
                del active[name]
                continue
            submitted.append(name)
        if not submitted:
            break
        for name in submitted:
            looper = active[name]
            sd = looper.commit()
            windows_done[name] += 1
            logging.info(
                f"Concurrent inference {name!r}: processing output from "
                f"window {windows_done[name]} of {n_windows[name]}."
            )
            entry = entries_by_name[name]
            with timer.context("data_writer"):
                entry.writer.append_batch(batch=sd)
            with timer.context("aggregator"):
                logs = entry.aggregator.record_batch(data=sd)
            with timer.context("wandb_logging"):
                entry.record_logs(logs)
        predictor.reset()


def run_inference(
    predict: PredictFunction[PS, FD, SD],
    data: InferenceDataABC[PS, FD],
    aggregator: InferenceAggregatorABC[PS, SD],
    writer: WriterABC[PS, SD] | None = None,
    record_logs: Callable[[InferenceLogs], None] | None = None,
):
    """Run extended inference loop given initial condition and forcing data.

    Args:
        predict: The prediction function to use.
        data: Provides an initial condition and appropriately aligned windows of
            forcing data.
        aggregator: Aggregator for collecting and reducing metrics.
        writer: Data writer for saving the inference results to disk.
        record_logs: Function for recording logs. By default, logs are recorded to
            wandb.
    """
    if record_logs is None:
        record_logs = get_record_to_wandb(label="inference").log
    if writer is None:
        writer = NullDataWriter()
    timer = GlobalTimer.get_instance()
    looper = Looper(predict=predict, data=data)
    with timer.context("aggregator"):
        logs = aggregator.record_initial_condition(
            initial_condition=data.initial_condition,
        )
    with timer.context("wandb_logging"):
        record_logs(logs)
    with timer.context("data_writer"):
        writer.write(data.initial_condition, "initial_condition.nc")
    n_windows = len(looper)
    for i, batch in enumerate(looper):
        logging.info(
            f"Inference: processing output from window {i + 1} of {n_windows}."
        )
        with timer.context("data_writer"):
            writer.append_batch(
                batch=batch,
            )
        with timer.context("aggregator"):
            logs = aggregator.record_batch(
                data=batch,
            )
        with timer.context("wandb_logging"):
            record_logs(logs)
    with timer.context("data_writer"):
        prognostic_state = looper.get_prognostic_state()
        writer.write(prognostic_state, "restart.nc")
