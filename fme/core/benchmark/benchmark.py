import abc
import dataclasses
import pathlib
import time
from collections.abc import Callable
from typing import Self

import dacite
import matplotlib.pyplot as plt
import torch

from fme.core.benchmark.memory import MemoryResult, benchmark_memory
from fme.core.benchmark.timer import CUDATimer, NullTimer, Timer, TimerResult
from fme.core.typing_ import TensorDict


class CPUEventPair:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def record_start(self):
        if self.start_time is not None:
            raise RuntimeError(
                "record_start has already been called on this CPUEventPair."
            )
        self.start_time = time.time()

    def record_end(self):
        if self.start_time is None:
            raise RuntimeError("record_start must be called before record_end")
        if self.end_time is not None:
            raise RuntimeError(
                "record_end has already been called on this CPUEventPair."
            )
        self.end_time = time.time()

    def elapsed_time_ms(self) -> float:
        if self.start_time is None or self.end_time is None:
            raise RuntimeError(
                "Both record_start and record_end must be called "
                "before elapsed_time_ms can be called."
            )
        return (self.end_time - self.start_time) * 1000


@dataclasses.dataclass
class BenchmarkResult:
    memory: MemoryResult
    timer: TimerResult
    cpu_time: float
    diagnostics: dict = dataclasses.field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(memory={self.memory}, timer={self.timer}, "
            f"diagnostics={self.diagnostics})"
            f"cpu_time={self.cpu_time})"
        )

    def asdict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkResult":
        return dacite.from_dict(cls, d, config=dacite.Config(strict=True))

    def assert_close(
        self, other: "BenchmarkResult", rtol=0.02, children_rtol=0.02
    ) -> None:
        try:
            self.timer.assert_close(other.timer, rtol=rtol, children_rtol=children_rtol)
        except AssertionError as e:
            raise AssertionError(f"Timer results differ: {e}") from e
        try:
            self.memory.assert_close(other.memory, rtol=rtol)
        except AssertionError as e:
            raise AssertionError(f"Memory results differ: {e}") from e
        if not torch.isclose(
            torch.tensor(self.cpu_time), torch.tensor(other.cpu_time), rtol=rtol
        ):
            raise AssertionError(
                f"CPU times differ: {self.cpu_time} vs {other.cpu_time} "
                f"given rtol={rtol}"
            )

    def get_logs(self, max_depth: int) -> dict[str, float]:
        logs = {
            "max_alloc_mb": self.memory.max_alloc / (1024.0 * 1024.0),
        }
        logs.update(self.timer.get_logs(max_depth=max_depth))
        return logs

    def to_png(
        self, path: str | pathlib.Path, label: str, child: str | None = None
    ) -> None:
        # note this function was generated with AI
        def avg_time(t: TimerResult) -> float:
            return float(t.avg_time)

        def self_time(t: TimerResult) -> float:
            t_avg = avg_time(t)
            c_avg = sum(avg_time(c) for c in t.children.values())
            return max(t_avg - c_avg, 0.0)

        def fmt_time(ms: float) -> str:
            if ms >= 1000.0:
                return f"{ms/1000.0:.2f}s"
            if ms >= 10.0:
                return f"{ms:.1f}ms"
            return f"{ms:.2f}ms"

        def label_ok(name: str, ms: float, frac_of_root: float) -> bool:
            if not name:
                return False
            return frac_of_root >= 0.05

        def ordered_children(t: TimerResult) -> list[tuple[str, TimerResult]]:
            return list(t.children.items())  # maintain dict order (insertion order)

        def blend_with_white(
            rgb: tuple[float, float, float], amount: float
        ) -> tuple[float, float, float]:
            # amount in [0,1]: 0 -> original, 1 -> white
            return (
                rgb[0] + (1.0 - rgb[0]) * amount,
                rgb[1] + (1.0 - rgb[1]) * amount,
                rgb[2] + (1.0 - rgb[2]) * amount,
            )

        root = self.timer
        if child is not None:
            for part in child.split("."):
                if part not in root.children:
                    raise ValueError(f"Child '{child}' not found in timer results.")
                root = root.children[part]
        root_avg = avg_time(root)

        max_alloc_mb = self.memory.max_alloc / (1024.0 * 1024.0)

        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        if root_avg <= 0.0:
            fig.suptitle(
                f"Benchmark for {label}\ntotal=0.00s, max_alloc={max_alloc_mb:.1f} MB",
                fontsize=14,
            )
            ax0 = fig.add_subplot(1, 1, 1)
            ax0.text(0.5, 0.5, "No timing data", ha="center", va="center")
            ax0.axis("off")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            return

        fig.suptitle(
            f"Benchmark for {label}\ntotal={fmt_time(root_avg)}, "
            f"max_alloc={max_alloc_mb:.1f} MB",
            fontsize=14,
        )

        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, root_avg)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["Level 1", "Level 2"])
        ax.set_ylabel("Avg time")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        gray = (0.85, 0.85, 0.85, 1.0)
        cmap = plt.get_cmap("tab20")

        lvl1 = ordered_children(root)
        lvl1_names = [n for n, _ in lvl1]
        lvl1_index = {n: i for i, n in enumerate(lvl1_names)}

        # Level 1 stack (root children + root self in gray, unlabeled)
        lvl1_segments: list[tuple[str, float, tuple[float, float, float, float]]] = []
        for n1, t1 in lvl1:
            base = cmap(lvl1_index[n1] % cmap.N)
            lvl1_segments.append((n1, avg_time(t1), base))
        r_self = self_time(root)
        if r_self > 0.0:
            lvl1_segments.append(("", r_self, gray))

        def draw_stack(
            x_center: float,
            segments: list[tuple[str, float, tuple[float, float, float, float]]],
        ) -> None:
            width = 0.86
            y = 0.0
            for name, sec, color in segments:
                if sec <= 0.0:
                    continue
                ax.bar(
                    x_center,
                    sec,
                    bottom=y,
                    width=width,
                    align="center",
                    color=color,
                    edgecolor="white",
                    linewidth=1.0,
                )
                frac = sec / root_avg
                if label_ok(name, sec, frac):
                    ax.text(
                        x_center,
                        y + sec / 2.0,
                        f"{name}\n{fmt_time(sec)}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        rotation=0,  # keep horizontal to avoid cross-column overlap
                        clip_on=True,
                    )
                y += sec
            if y < root_avg:
                ax.bar(
                    x_center,
                    root_avg - y,
                    bottom=y,
                    width=width,
                    align="center",
                    color=gray,
                    edgecolor="white",
                    linewidth=1.0,
                )

        draw_stack(0.5, lvl1_segments)

        # Level 2 stack:
        # For each level-1 slice, stack its children
        # (colored as parent hue variants) + self in gray.
        lvl2_segments: list[tuple[str, float, tuple[float, float, float, float]]] = []
        for n1, t1 in lvl1:
            parent_rgba = cmap(lvl1_index[n1] % cmap.N)
            parent_rgb = (parent_rgba[0], parent_rgba[1], parent_rgba[2])

            children = ordered_children(t1)
            k = len(children)
            for i, (n2, t2) in enumerate(children):
                # Same “type” of color as parent: lighten progressively per child.
                # First child is closest to parent; later children are lighter.
                lighten = 0.10 + (0.55 * (i / max(k - 1, 1)))
                rgb = blend_with_white(parent_rgb, lighten)
                lvl2_segments.append((n2, avg_time(t2), (rgb[0], rgb[1], rgb[2], 1.0)))

            s1 = self_time(t1)
            if s1 > 0.0:
                lvl2_segments.append(("", s1, gray))

        draw_stack(1.5, lvl2_segments)

        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)


class BenchmarkABC(abc.ABC):
    @classmethod
    def new_from_fn(
        cls,
        fn: Callable[[Timer], TensorDict],
    ) -> "BenchmarkABC":
        class FnBenchmark(BenchmarkABC):
            @classmethod
            def new(cls) -> "FnBenchmark":
                return FnBenchmark()

            def run_instance(self, timer: Timer) -> TensorDict:
                return fn(timer)

        return FnBenchmark()

    @classmethod
    @abc.abstractmethod
    def new(cls: type[Self]) -> Self:
        """
        Initialize any state needed for the benchmark.
        This will be called once before the benchmark is run.
        """
        pass

    @classmethod
    def new_for_regression(cls: type[Self]) -> Self | None:
        """
        Initialize any state needed for regression testing.
        This will be called once before regression tests are run.

        If regression testing is not needed, this can return None,
        and regression testing will not be run.

        This exists as a separate method from new so that it can
        use small data sizes more conducive to storing regression targets in git.
        """
        return None

    @classmethod
    def run_benchmark(cls, iters=10, warmup=1) -> BenchmarkResult:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot run benchmark.")
        null_timer = NullTimer()
        benchmark = cls.new()
        for _ in range(warmup):
            benchmark.run_instance(null_timer)
        timer = CUDATimer()
        cpu_timer = CPUEventPair()
        with benchmark_memory() as bm:
            cpu_timer.record_start()
            for _ in range(iters):
                with timer:
                    benchmark_result = benchmark.run_instance(timer)
            torch.cuda.synchronize()
            cpu_timer.record_end()
        return BenchmarkResult(
            timer=timer.result,
            cpu_time=cpu_timer.elapsed_time_ms(),
            memory=bm.result,
            diagnostics=benchmark_result.get("diagnostics", {}),
        )

    @classmethod
    def run_regression(cls) -> TensorDict | None:
        benchmark = cls.new_for_regression()
        if benchmark is None:
            return None
        null_timer = NullTimer()
        return benchmark.run_instance(null_timer)

    @abc.abstractmethod
    def run_instance(self: Self, timer: Timer) -> TensorDict:
        """
        Run the benchmark. This will be called multiple times,
        and should return a TensorDict of results.

        This must not mutate any state on self, since the same instance may be
        used across multiple iterations.
        """
        pass


_BENCHMARKS: dict[str, type[BenchmarkABC]] = {}


def register_benchmark(name: str) -> Callable[[type[BenchmarkABC]], type[BenchmarkABC]]:
    def _register(fn: type[BenchmarkABC]) -> type[BenchmarkABC]:
        if name in _BENCHMARKS:
            raise ValueError(f"Benchmark with name '{name}' is already registered.")
        _BENCHMARKS[name] = fn
        return fn

    return _register


def get_benchmarks() -> dict[str, type[BenchmarkABC]]:
    return _BENCHMARKS.copy()
