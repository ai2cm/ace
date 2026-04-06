import logging
import time as time_module
from collections.abc import Callable, Iterator
from typing import Any, Generic, Protocol, TypeVar

from fme.core.distributed import Distributed
from fme.core.generics.aggregator import InferenceAggregatorABC, InferenceLogs
from fme.core.generics.data import InferenceDataABC
from fme.core.generics.writer import NullDataWriter, WriterABC
from fme.core.timing import GlobalTimer
from fme.core.wandb import WandB

PS = TypeVar("PS")  # prognostic state
FD = TypeVar("FD", contravariant=True)  # forcing data
SD = TypeVar("SD", covariant=True)  # stepped data


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
        self.last_data_loading_s: float = 0.0
        self.last_forward_pass_s: float = 0.0

    def __iter__(self) -> Iterator[SD]:
        return self

    def __len__(self) -> int:
        return self._len

    def __next__(self) -> SD:
        """Return predictions for the time period corresponding to the next batch
        of forcing data. Also returns the forcing data.
        """
        t0 = time_module.monotonic()
        timer = GlobalTimer.get_instance()
        with timer.context("data_loading"):
            try:
                forcing_data = next(self._loader)
            except StopIteration:
                raise StopIteration
        t1 = time_module.monotonic()
        self.last_data_loading_s = t1 - t0
        output_data, self._prognostic_state = self._predict(
            self._prognostic_state,
            forcing=forcing_data,
            compute_derived_variables=True,
        )
        t2 = time_module.monotonic()
        self.last_forward_pass_s = t2 - t1
        return output_data

    def get_prognostic_state(self) -> PS:
        return self._prognostic_state


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


class _TimingLogger:
    """Logs per-window timing dicts via WandB with an auto-incrementing step.

    Unlike ``WandBStepLogger``, this always calls ``WandB.log`` even when the
    dict is empty.  This ensures non-root ranks (which pass ``{}``) still hit
    the ``dist.barrier()`` inside ``WandB.log``, preventing rank divergence.
    """

    def __init__(self, label: str = "timing"):
        self._wandb = WandB.get_instance()
        self._label = label
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def log(self, data: dict[str, Any]) -> None:
        prefixed = (
            {f"{self._label}/{k}": v for k, v in data.items()} if self._label else data
        )
        self._wandb.log(prefixed, step=self._step)
        self._step += 1


def run_timed_inference(
    predict: PredictFunction[PS, FD, SD],
    data: InferenceDataABC[PS, FD],
    aggregator: InferenceAggregatorABC[PS, SD],
):
    """Run inference with per-window timing instrumentation.

    Like ``run_inference`` but omits writer/record_logs and instead collects
    wall-clock timing for data loading, forward pass, and aggregator phases.
    After each window, timing is gathered across ranks via ``gather_object``
    (which doubles as a sync point) and logged to wandb on root.

    Args:
        predict: The prediction function to use.
        data: Provides an initial condition and appropriately aligned windows of
            forcing data.
        aggregator: Aggregator for collecting and reducing metrics.
    """
    dist = Distributed.get_instance()
    timer = GlobalTimer.get_instance()
    logger = _TimingLogger(label="timing")
    looper = Looper(predict=predict, data=data)

    with timer.context("aggregator"):
        aggregator.record_initial_condition(
            initial_condition=data.initial_condition,
        )

    n_windows = len(looper)
    cumulative: dict[str, float] = {
        "data_loading_s": 0.0,
        "forward_pass_s": 0.0,
        "aggregator_s": 0.0,
        "total_s": 0.0,
    }

    for i, batch in enumerate(looper):
        t_agg_start = time_module.monotonic()
        with timer.context("aggregator"):
            aggregator.record_batch(data=batch)
        aggregator_s = time_module.monotonic() - t_agg_start

        window_timing: dict[str, float] = {
            "data_loading_s": looper.last_data_loading_s,
            "forward_pass_s": looper.last_forward_pass_s,
            "aggregator_s": aggregator_s,
            "total_s": (
                looper.last_data_loading_s + looper.last_forward_pass_s + aggregator_s
            ),
        }

        sub_timings = getattr(aggregator, "last_batch_sub_timings", None)
        if sub_timings is not None:
            for name, elapsed in sub_timings.items():
                window_timing[f"aggregator/{name}_s"] = elapsed

        for k, v in window_timing.items():
            if k in cumulative:
                cumulative[k] += v

        all_timings = dist.gather_object(window_timing)
        if dist.is_root() and all_timings is not None:
            wandb_data: dict[str, Any] = {}
            for r, rt in enumerate(all_timings):
                for key, val in rt.items():
                    wandb_data[f"{key}/rank_{r}"] = val
            totals = [rt["total_s"] for rt in all_timings]
            wandb_data["spread_s"] = max(totals) - min(totals)
            wandb_data["max_total_s"] = max(totals)
            wandb_data["min_total_s"] = min(totals)
            wandb_data["window_index"] = i
        else:
            wandb_data = {}
        logger.log(wandb_data)

        logging.info(
            f"Window {i + 1}/{n_windows}: "
            f"data={window_timing['data_loading_s']:.2f}s "
            f"fwd={window_timing['forward_pass_s']:.2f}s "
            f"agg={window_timing['aggregator_s']:.2f}s "
            f"total={window_timing['total_s']:.2f}s"
        )

    all_cumulative = dist.gather_object(cumulative)
    if dist.is_root() and all_cumulative is not None:
        summary: dict[str, Any] = {}
        for r, rc in enumerate(all_cumulative):
            for key, val in rc.items():
                summary[f"summary/{key}/rank_{r}"] = val
        rank_totals = [rc["total_s"] for rc in all_cumulative]
        summary["summary/max_total_s"] = max(rank_totals)
        summary["summary/min_total_s"] = min(rank_totals)
        summary["summary/spread_s"] = max(rank_totals) - min(rank_totals)
        summary["summary/n_windows"] = n_windows
    else:
        summary = {}
    logger.log(summary)
