"""Benchmark the inline-inference suite: sequential vs concurrent code paths,
with per-phase (forward / rollout loop vs aggregator) timers.

Motivation
----------
The concurrent-inline-inference change batches several inline-inference runs
(e.g. the 10-year and 46-year suites) through a single ``BatchedPredictor`` so
each rollout step is one batched forward pass instead of one per run. A 2-seed
A/B showed it is faster, but a log-timestamp reanalysis could not cleanly
attribute the speed-up to the forward/rollout loop versus the (unparallelized)
aggregator, because (a) GPU calls are async so log markers don't bound GPU work
and (b) the two A/B jobs shared a host, confounding the per-phase split with
CPU/IO contention (see the research-repo investigation
``2026-06-11-concurrent-inference-throughput``).

This script makes a clean measurement: it runs *both* code paths back-to-back in
one process on one (isolated) GPU, with explicit in-code timers that
``torch.cuda.synchronize()`` at every phase boundary and record CUDA-event GPU
time alongside wall-clock. That removes the async-dispatch ambiguity and (one
job per node) removes the shared-host contention.

It is config-driven: it reuses the production training config (model
architecture, ERA5 4-degree daily data, normalization, and the inline-inference
task definitions) so the workload replicates production. Model weights are
random by default (forward/aggregator cost is shape-dependent, not
value-dependent); pass ``--checkpoint`` to load trained weights.

The sequential and concurrent loops below mirror ``run_inference`` and
``run_concurrent_inference`` in ``fme.core.generics.inference`` and
``inference_one_epoch`` in ``fme.core.generics.trainer``; they are re-implemented
here only so each phase can be wrapped in a synchronizing CUDA-event timer
without changing the production timing semantics.

Usage::

    python -m fme.ace.train.benchmark_inline_inference CONFIG.yaml \
        [--repeats 3] [--warmup 1] [--max-windows 0] [--checkpoint CKPT] \
        [--output-json results.json] [--modes sequential,concurrent]
"""

import argparse
import contextlib
import dataclasses
import json
import logging
import os
import time
from collections.abc import Callable, Hashable, Sequence

import dacite
import torch

import fme
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.train.train_config import InlineInferenceConfig, TrainBuilders, TrainConfig
from fme.core.cli import prepare_config
from fme.core.derived_variables import get_derived_variable_metadata
from fme.core.generics.aggregator import InferenceAggregatorABC
from fme.core.generics.data import InferenceDataABC
from fme.core.generics.inference import BatchedPredictor, LazyLooper
from fme.core.timing import GlobalTimer

# Phases that make up an inline-inference run. "forward" is the rollout / model
# forward pass (the part the concurrent path batches); the remaining phases are
# the aggregator work the concurrent path does *not* parallelize.
FORWARD = "forward"
DATA_LOADING = "data_loading"
AGGREGATOR_RECORD = "aggregator_record"
FLUSH = "flush"
SUMMARY = "summary"
PHASE_ORDER = [DATA_LOADING, FORWARD, AGGREGATOR_RECORD, FLUSH, SUMMARY]


class PhaseTimer:
    """Accumulate wall-clock and (optionally) CUDA-event GPU time for one phase.

    Each ``measure()`` block ``synchronize()``s before starting and after
    finishing so the recorded time is this phase's own GPU+CPU work, not the
    async tail of a previous phase. CUDA-event time is the pure GPU-stream time
    for the block.
    """

    def __init__(self, name: str, use_cuda: bool):
        self.name = name
        self.use_cuda = use_cuda
        self.wall = 0.0  # seconds
        self.gpu = 0.0  # seconds (CUDA-event time; 0 on CPU)
        self.count = 0
        self.values: list[float] = []  # per-call time (gpu if cuda else wall)

    @contextlib.contextmanager
    def measure(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_gpu = 0.0
            if self.use_cuda:
                end_evt.record()
                torch.cuda.synchronize()
                dt_gpu = start_evt.elapsed_time(end_evt) / 1000.0
                self.gpu += dt_gpu
            dt_wall = time.perf_counter() - t0
            self.wall += dt_wall
            self.count += 1
            self.values.append(dt_gpu if self.use_cuda else dt_wall)

    def as_dict(self) -> dict:
        return {"wall": self.wall, "gpu": self.gpu, "count": self.count}


def _make_timers(use_cuda: bool) -> dict[str, PhaseTimer]:
    return {phase: PhaseTimer(phase, use_cuda) for phase in PHASE_ORDER}


@dataclasses.dataclass
class BenchTask:
    """One inline-inference run, packaged for the benchmark loops."""

    name: str
    data: InferenceDataABC
    aggregator_factory: Callable[[int], InferenceAggregatorABC]
    forward_steps_in_memory: int
    n_ensemble_per_ic: int

    @property
    def concurrent_group(self) -> Hashable:
        # Matches train.get_inference_callback: tasks share a batched forward
        # pass iff their window length and per-IC ensemble size match.
        return (self.forward_steps_in_memory, self.n_ensemble_per_ic)


def _build_batched_predictor(stepper) -> BatchedPredictor:
    # Identical to train.get_inference_callback.build_predictor.
    return BatchedPredictor(
        base_predict=stepper.predict_paired,
        concat_ic=PrognosticState.cat,
        concat_forcing=BatchData.cat,
        split_output=lambda sd, sizes: sd.split(sizes),
        split_state=lambda ps, sizes: ps.split(sizes),
        sample_size_of_state=lambda ic: ic.as_batch_data().time.shape[0],
        batch_key_of_forcing=lambda forcing: forcing.n_timesteps,
    )


def _flush_and_summarize(
    name: str,
    aggregator: InferenceAggregatorABC,
    timers: dict[str, PhaseTimer],
    epoch: int,
) -> None:
    # Mirrors the tail of inference_one_epoch: flush reduced diagnostics to disk,
    # then build the summary logs.
    with timers[FLUSH].measure():
        aggregator.flush_diagnostics(subdir=f"epoch_{epoch:04d}")
    with timers[SUMMARY].measure():
        aggregator.get_summary_logs()


def run_sequential(
    tasks: Sequence[BenchTask],
    stepper,
    timers: dict[str, PhaseTimer],
    max_windows: int,
    epoch: int = 0,
) -> dict:
    """Run each task on its own (mirrors run_inference, one task at a time)."""
    per_task_windows: dict[str, int] = {}
    per_task_samples: dict[str, int] = {}
    per_task_forward_windows: dict[str, list[float]] = {}
    for task in tasks:
        aggregator = task.aggregator_factory(epoch)
        data = task.data
        with timers[AGGREGATOR_RECORD].measure():
            aggregator.record_initial_condition(
                initial_condition=data.initial_condition
            )
        per_task_samples[task.name] = data.initial_condition.as_batch_data().time.shape[
            0
        ]
        loader = iter(data.loader)
        state = data.initial_condition
        fwd_start = len(timers[FORWARD].values)
        i = 0
        while max_windows <= 0 or i < max_windows:
            with timers[DATA_LOADING].measure():
                try:
                    forcing = next(loader)
                except StopIteration:
                    break
            with timers[FORWARD].measure():
                output, state = stepper.predict_paired(
                    state, forcing, compute_derived_variables=True
                )
            with timers[AGGREGATOR_RECORD].measure():
                aggregator.record_batch(data=output)
            i += 1
        per_task_windows[task.name] = i
        per_task_forward_windows[task.name] = list(timers[FORWARD].values[fwd_start:])
        _flush_and_summarize(task.name, aggregator, timers, epoch)
    return {
        "windows": per_task_windows,
        "samples_per_window": per_task_samples,
        "forward_windows": per_task_forward_windows,
    }


def run_concurrent(
    tasks: Sequence[BenchTask],
    stepper,
    timers: dict[str, PhaseTimer],
    max_windows: int,
    epoch: int = 0,
    build_predictor=_build_batched_predictor,
) -> dict:
    """Run tasks sharing a concurrent group through a shared BatchedPredictor
    (mirrors run_concurrent_inference). Tasks in distinct groups are run in
    separate concurrent passes, matching the production callback.
    """
    groups: dict[Hashable, list[BenchTask]] = {}
    for task in tasks:
        groups.setdefault(task.concurrent_group, []).append(task)

    per_task_windows: dict[str, int] = {}
    per_task_samples: dict[str, int] = {}
    max_batch_samples = 0
    forward_rounds: list[dict] = []
    for group_tasks in groups.values():
        predictor = build_predictor(stepper)
        fwd_start = len(timers[FORWARD].values)
        round_meta: list[dict] = []
        aggregators = {t.name: t.aggregator_factory(epoch) for t in group_tasks}
        loopers = {t.name: LazyLooper(predictor, t.data) for t in group_tasks}
        for task in group_tasks:
            with timers[AGGREGATOR_RECORD].measure():
                aggregators[task.name].record_initial_condition(
                    initial_condition=task.data.initial_condition
                )
            per_task_samples[task.name] = (
                task.data.initial_condition.as_batch_data().time.shape[0]
            )
            per_task_windows[task.name] = 0

        active = dict(loopers)
        i = 0
        while active and (max_windows <= 0 or i < max_windows):
            submitted: list[str] = []
            with timers[DATA_LOADING].measure():
                for name in list(active.keys()):
                    try:
                        active[name].submit()
                    except StopIteration:
                        del active[name]
                        continue
                    submitted.append(name)
            if not submitted:
                break
            batch_samples = sum(per_task_samples[name] for name in submitted)
            max_batch_samples = max(max_batch_samples, batch_samples)
            round_meta.append(
                {"n_active": len(submitted), "batch_samples": batch_samples}
            )
            # The first commit triggers the single batched forward pass for the
            # whole round; the rest just fetch their slice.
            outputs: dict[str, object] = {}
            with timers[FORWARD].measure():
                for name in submitted:
                    outputs[name] = active[name].commit()
            for name in submitted:
                with timers[AGGREGATOR_RECORD].measure():
                    aggregators[name].record_batch(data=outputs[name])
                per_task_windows[name] += 1
            predictor.reset()
            i += 1

        for task in group_tasks:
            _flush_and_summarize(task.name, aggregators[task.name], timers, epoch)
        for meta_r, t in zip(round_meta, timers[FORWARD].values[fwd_start:]):
            forward_rounds.append({**meta_r, "gpu": t})
    return {
        "windows": per_task_windows,
        "samples_per_window": per_task_samples,
        "max_concurrent_batch_samples": max_batch_samples,
        "forward_rounds": forward_rounds,
    }


def _make_aggregator_factory(
    entry_config: InlineInferenceConfig,
    data,
    dataset_info,
    name,
    stepper,
    output_dir,
    save_diagnostics,
):
    # Mirrors train._make_ace_aggregator_factory.
    def factory(epoch: int):
        return entry_config.aggregator.build(
            dataset_info=dataset_info,
            n_ic_steps=stepper.n_ic_timesteps,
            n_forward_steps=entry_config.n_forward_steps,
            initial_time=data.initial_time,
            normalize=stepper.normalizer.normalize,
            output_dir=os.path.join(output_dir, name),
            channel_mean_names=stepper.loss_names,
            save_diagnostics=save_diagnostics,
            n_ensemble_per_ic=entry_config.n_ensemble_per_ic,
            enable_time_series=False,
        )

    return factory


def build_tasks_and_stepper(
    config: TrainConfig, output_dir: str, save_diagnostics: bool
):
    """Build the stepper and the inline-inference tasks from a train config,
    mirroring the relevant parts of train.build_trainer.
    """
    builder = TrainBuilders(config)
    logging.info("Building train data (for dataset_info)")
    train_data = builder.get_train_data()
    variable_metadata = get_derived_variable_metadata() | train_data.variable_metadata
    logging.info("Building inline-inference data loaders")
    inference_entries = builder.get_inference_data(variable_metadata)
    logging.info("Building stepper")
    stepper = builder.get_stepper(dataset_info=train_data.dataset_info)

    tasks: list[BenchTask] = []
    for entry_config, data, entry_dataset_info, name in inference_entries:
        tasks.append(
            BenchTask(
                name=name,
                data=data,
                aggregator_factory=_make_aggregator_factory(
                    entry_config,
                    data,
                    entry_dataset_info,
                    name,
                    stepper,
                    output_dir,
                    save_diagnostics,
                ),
                forward_steps_in_memory=entry_config.forward_steps_in_memory,
                n_ensemble_per_ic=entry_config.n_ensemble_per_ic,
            )
        )
    return stepper, tasks


def _warmup(stepper, tasks: Sequence[BenchTask], use_cuda: bool):
    """Run a couple of forward passes to trigger CUDA kernel/cudnn autotune so it
    doesn't pollute the first timed pass.
    """
    if not tasks:
        return
    logging.info("Warmup: 2 forward windows on the first task")
    task = tasks[0]
    loader = iter(task.data.loader)
    state = task.data.initial_condition
    with torch.no_grad():
        for _ in range(2):
            try:
                forcing = next(loader)
            except StopIteration:
                break
            _, state = stepper.predict_paired(
                state, forcing, compute_derived_variables=True
            )
    if use_cuda:
        torch.cuda.synchronize()


def _summarize_run(timers: dict[str, PhaseTimer]) -> dict:
    out = {phase: timers[phase].as_dict() for phase in PHASE_ORDER}
    out["forward_total"] = {
        "wall": timers[FORWARD].wall,
        "gpu": timers[FORWARD].gpu,
    }
    out["aggregator_total"] = {  # all the work the concurrent path does NOT batch
        "wall": sum(timers[p].wall for p in (AGGREGATOR_RECORD, FLUSH, SUMMARY)),
        "gpu": sum(timers[p].gpu for p in (AGGREGATOR_RECORD, FLUSH, SUMMARY)),
    }
    return out


def main(
    yaml_config: str,
    repeats: int,
    warmup: int,
    max_windows: int,
    modes: Sequence[str],
    checkpoint: str | None,
    output_json: str | None,
    output_dir: str,
    save_diagnostics: bool,
):
    config_data = prepare_config(yaml_config, override=None)
    config = dacite.from_dict(
        data_class=TrainConfig, data=config_data, config=dacite.Config(strict=True)
    )
    config.set_random_seed()
    use_cuda = fme.using_gpu()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    os.makedirs(output_dir, exist_ok=True)
    stepper, tasks = build_tasks_and_stepper(config, output_dir, save_diagnostics)
    if checkpoint is not None:
        logging.info(f"Loading stepper weights from checkpoint {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        stepper.load_state(ckpt["stepper"])
    stepper.set_eval()

    logging.info(
        "Benchmark tasks: "
        + ", ".join(
            f"{t.name}(fsim={t.forward_steps_in_memory}, "
            f"n_ens={t.n_ensemble_per_ic}, group={t.concurrent_group})"
            for t in tasks
        )
    )

    runners = {"sequential": run_sequential, "concurrent": run_concurrent}
    for m in modes:
        if m not in runners:
            raise ValueError(f"Unknown mode {m!r}; choose from {list(runners)}")

    if warmup > 0:
        for _ in range(warmup):
            _warmup(stepper, tasks, use_cuda)

    results: dict[str, list[dict]] = {m: [] for m in modes}
    meta: dict[str, dict] = {}
    for r in range(repeats):
        # Alternate which mode goes first to cancel any ordering effect.
        ordered = list(modes) if r % 2 == 0 else list(reversed(modes))
        for m in ordered:
            logging.info(f"=== repeat {r + 1}/{repeats}  mode={m} ===")
            timers = _make_timers(use_cuda)
            with torch.no_grad(), GlobalTimer():
                info = runners[m](tasks, stepper, timers, max_windows)
            run_summary = _summarize_run(timers)
            run_summary["info"] = info
            results[m].append(run_summary)
            meta[m] = info
            fwd = run_summary["forward_total"]
            agg = run_summary["aggregator_total"]
            logging.info(
                f"[{m}] forward gpu={fwd['gpu']:.2f}s wall={fwd['wall']:.2f}s | "
                f"aggregator gpu={agg['gpu']:.2f}s wall={agg['wall']:.2f}s"
            )

    report = _build_report(
        results, modes, meta, config, max_windows, use_cuda, checkpoint, repeats
    )
    print("\n" + json.dumps(report, indent=2))
    if output_json is not None:
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        logging.info(f"Wrote results to {output_json}")
    return report


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _agg_phase(results: dict[str, list[dict]], mode: str, phase: str, field: str):
    vals = [run[phase][field] for run in results[mode]]
    return {"mean": _mean(vals), "values": vals}


def _build_report(
    results, modes, meta, config, max_windows, use_cuda, checkpoint, repeats
) -> dict:
    report = {
        "config": {
            "max_windows": max_windows,
            "repeats": repeats,
            "use_cuda": use_cuda,
            "checkpoint": checkpoint,
            "device": str(fme.get_device()),
            "tasks": meta,
        },
        "per_mode": {},
        "per_window": {m: [r["info"] for r in results[m]] for m in modes},
    }
    phases = PHASE_ORDER + ["forward_total", "aggregator_total"]
    for m in modes:
        report["per_mode"][m] = {
            phase: {
                "wall": _agg_phase(results, m, phase, "wall"),
                "gpu": _agg_phase(results, m, phase, "gpu"),
            }
            for phase in phases
        }
    if "sequential" in modes and "concurrent" in modes:
        report["speedup"] = {}
        for field in ("gpu", "wall"):
            seq_fwd = _mean([r["forward_total"][field] for r in results["sequential"]])
            con_fwd = _mean([r["forward_total"][field] for r in results["concurrent"]])
            seq_agg = _mean(
                [r["aggregator_total"][field] for r in results["sequential"]]
            )
            con_agg = _mean(
                [r["aggregator_total"][field] for r in results["concurrent"]]
            )
            report["speedup"][field] = {
                "forward_seq": seq_fwd,
                "forward_concurrent": con_fwd,
                "forward_speedup_pct": (
                    100.0 * (seq_fwd - con_fwd) / seq_fwd if seq_fwd else float("nan")
                ),
                "aggregator_seq": seq_agg,
                "aggregator_concurrent": con_agg,
                # Should be ~0 on an isolated node: the aggregator is identical
                # work in both paths. A large delta signals contention/noise.
                "aggregator_delta_pct": (
                    100.0 * (seq_agg - con_agg) / seq_agg if seq_agg else float("nan")
                ),
            }
    return report


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml_config", type=str, help="train config yaml")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="cap forward windows per task (0 = full production length). "
        "Capping gives stable per-window stats faster; flush/summary cost is "
        "then computed over a truncated rollout (a minor undercount).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="sequential,concurrent",
        help="comma-separated subset of {sequential,concurrent}",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("BENCH_OUTPUT_DIR", "/results/benchmark"),
        help="where aggregators flush diagnostics",
    )
    parser.add_argument(
        "--no-save-diagnostics",
        action="store_true",
        help="skip writing reduced diagnostics to disk during flush",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from fme.core.distributed import Distributed

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args()
    with Distributed.context():
        main(
            yaml_config=args.yaml_config,
            repeats=args.repeats,
            warmup=args.warmup,
            max_windows=args.max_windows,
            modes=[m.strip() for m in args.modes.split(",") if m.strip()],
            checkpoint=args.checkpoint,
            output_json=args.output_json,
            output_dir=args.output_dir,
            save_diagnostics=not args.no_save_diagnostics,
        )
