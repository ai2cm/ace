import dataclasses
import datetime
import logging
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar

import cftime

from fme.core.cloud import exists
from fme.core.distributed import Distributed
from fme.core.generics.aggregator import InferenceAggregatorABC, InferenceLogs
from fme.core.generics.data import InferenceDataABC
from fme.core.generics.writer import NullDataWriter, WriterABC
from fme.core.logging_utils import LoggingConfig
from fme.core.timing import GlobalTimer
from fme.core.wandb import WandB

# Truncated to hour precision: segment start times in existing runs have always
# been at least 6h apart, so finer precision would just add visual noise. We can
# reconsider if this changes.
SEGMENT_LABEL_FORMAT = "segment_%Y%m%dT%H"

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
    dist = Distributed.get_instance()
    for i, batch in enumerate(looper):
        dist.park_if_terminating()
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


def get_segment_label(
    initialization_time: cftime.datetime,
    timestep: datetime.timedelta,
    segment: int,
    n_steps: int,
) -> str:
    """Label a segment by the start time of its first (or only) ensemble member."""
    segment_length = n_steps * timestep
    current_start_time = initialization_time + segment * segment_length
    current_label = current_start_time.strftime(SEGMENT_LABEL_FORMAT)

    if segment > 0:
        previous_start_time = initialization_time + (segment - 1) * segment_length
        previous_label = previous_start_time.strftime(SEGMENT_LABEL_FORMAT)
        if previous_label == current_label:
            raise ValueError(
                f"Consecutive segments have the same label ({previous_label!r} "
                f"and {current_label!r}), meaning the current segment would "
                f"overwrite the previous segment. Please open an issue on "
                f"GitHub if having greater temporal precision in segmented run "
                f"directory labels is an important use-case for you."
            )

    return current_label


def run_segments(
    segments: int,
    experiment_dir: str,
    logging_config: LoggingConfig,
    logging_config_dict: Mapping[str, Any],
    n_ensemble_per_ic: int,
    n_steps_per_segment: int,
    get_initialization: Callable[[], tuple[cftime.datetime, datetime.timedelta]],
    get_restart_paths: Callable[[str], Sequence[str]],
    run_segment: Callable[[str], None],
    set_initial_condition: Callable[[Sequence[str]], None],
    description: str = "segmented inference",
) -> None:
    """Run inference as a sequence of resumable segments.

    Each segment runs ``n_steps_per_segment`` steps into a subdirectory of
    ``experiment_dir`` labeled by its start time, and gets its own wandb run
    named ``WANDB_NAME`` with the segment label appended. A segment counts as
    complete once its restart files exist, so re-running the same configuration
    skips finished segments. Restart files are written before the data writer's
    final flush, so a segment interrupted in that window counts as complete
    despite having incomplete diagnostics.

    Args:
        segments: Total number of segments; only missing ones are run.
        experiment_dir: Directory holding the per-segment subdirectories.
        logging_config: Logging configuration. Applied at the top level with
            wandb disabled, since each segment opens its own run.
        logging_config_dict: Full run configuration, logged to wandb.
        n_ensemble_per_ic: Ensemble size, which must be 1. A segment's restart
            already carries the broadcast ensemble as its sample dimension, so
            later segments cannot re-broadcast it consistently.
        n_steps_per_segment: Steps per segment, used to compute segment labels.
        get_initialization: Returns the run's start time and timestep. Called
            once, since it may be expensive.
        get_restart_paths: Maps a segment directory to the restart files that
            segment writes on completion.
        run_segment: Runs one segment into the given directory.
        set_initial_condition: Points the next segment at the given restarts.
        description: Run description for the opening log message.
    """
    if n_ensemble_per_ic > 1:
        raise ValueError(
            "Ensemble inference (n_ensemble_per_ic > 1) is not supported with "
            "segmented inference. A segment's restart already carries the "
            "broadcasted ensemble as its sample dimension, so later segments "
            "cannot re-broadcast it consistently. Run with n_ensemble_per_ic=1, "
            "or run a single non-segmented inference for ensemble runs."
        )
    # Top-level logging has no wandb run; each segment owns its own.
    top_level_logging = dataclasses.replace(logging_config, log_to_wandb=False)
    top_level_logging.configure_logging(
        experiment_dir,
        "inference_out.log",
        config=logging_config_dict,
        resumable=False,
    )
    logging.info(
        f"Starting {description} with {segments} segments. "
        f"Saving to {experiment_dir}."
    )
    original_wandb_name = os.environ.get("WANDB_NAME")
    initialization_time, timestep = get_initialization()

    for segment in range(segments):
        segment_label = get_segment_label(
            initialization_time,
            timestep,
            segment,
            n_steps_per_segment,
        )
        segment_dir = os.path.join(experiment_dir, segment_label)
        restart_paths = get_restart_paths(segment_dir)
        if all(exists(path) for path in restart_paths):
            logging.info(f"Skipping segment {segment} because it has already been run.")
        else:
            logging.info(f"Running segment {segment}.")
            if original_wandb_name is not None:
                os.environ["WANDB_NAME"] = f"{original_wandb_name}-{segment_label}"
            with GlobalTimer():
                run_segment(segment_dir)
            # Finish so the next segment starts a fresh wandb run.
            WandB.get_instance().finish()
        set_initial_condition(restart_paths)
