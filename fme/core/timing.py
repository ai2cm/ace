import contextlib
import logging
import time
import warnings

import numpy as np

INACTIVE_WARNING_MESSAGE = (
    "The GlobalTimer is currently inactive; therefore no timing information "
    "will be recorded. To activate it, wrap your code within the GlobalTimer() "
    "context."
)


class CumulativeTimer:
    def __init__(self, category):
        self._duration = 0.0
        self._start_time = None
        self._category = category

    def start(self):
        if self._start_time is not None:
            raise RuntimeError(f"timer {self._category!r} is already running")
        self._start_time = time.time()

    def stop(self):
        if self._start_time is None:
            raise RuntimeError(
                f"must call start for timer {self._category!r} before stop"
            )
        self._duration += time.time() - self._start_time
        self._start_time = None

    @property
    def duration(self) -> float:
        if self._start_time is not None:
            raise RuntimeError(f"timer {self._category!r} is still running")
        return self._duration


class GlobalTimer:
    """
    A singleton class to make timing inference code easier.

    Supports a two-level hierarchy of timers:

    - **Outer timer**: at most one active at a time. Tracks total wall time for
      a top-level phase (e.g. ``"inference"``).
    - **Inner timers**: multiple can be active simultaneously.
      When an outer timer is active, inner timer keys are automatically
      qualified as ``"{outer}/{inner}"``; without an outer timer they remain
      flat.
    """

    @classmethod
    def get_instance(cls) -> "GlobalTimer":
        """
        Get the singleton instance of the GlobalTimer class.
        """
        global singleton
        if singleton is None:
            singleton = cls()
            warnings.warn(INACTIVE_WARNING_MESSAGE)
        return singleton

    @classmethod
    def __enter__(cls):
        global singleton
        if singleton is None:
            singleton = cls()
        if singleton._active:
            raise RuntimeError("GlobalTimer is currently in use in another context")
        singleton._active = True

    @classmethod
    def __exit__(cls, type, value, traceback):
        global singleton
        singleton = None

    def __init__(self):
        self._timers: dict[str, CumulativeTimer] = {}
        self._active = False
        self._current_outer: str | None = None
        self._active_inners: set[str] = set()

    def _inner_key(self, category: str) -> str:
        if self._current_outer is not None:
            return f"{self._current_outer}/{category}"
        return category

    def _start_timer(self, key: str):
        if key not in self._timers:
            self._timers[key] = CumulativeTimer(key)
        self._timers[key].start()

    def _stop_timer(self, key: str):
        self._timers[key].stop()

    def outer_context(self, category: str) -> contextlib.AbstractContextManager:
        """
        Context manager for timing a top-level phase.

        Only one outer timer can be active at a time.  Inner timers started
        within this context are automatically namespaced under *category*.
        """

        @contextlib.contextmanager
        def timer_context():
            self.start_outer(category)
            try:
                yield
            finally:
                self.stop_outer(category)

        return timer_context()

    def context(self, category: str) -> contextlib.AbstractContextManager:
        """
        Context manager for timing a block of code.

        If an outer timer is active, the key is qualified as
        ``"{outer}/{category}"``.
        """

        @contextlib.contextmanager
        def timer_context():
            self.start(category)
            try:
                yield
            finally:
                self.stop(category)

        return timer_context()

    def start(self, category: str):
        """
        Start an inner timer for the given category.

        Multiple inner timers can be active simultaneously.  If an outer timer
        is active, the stored key is ``"{outer}/{category}"``.
        """
        if not self._active:
            return
        key = self._inner_key(category)
        self._start_timer(key)
        self._active_inners.add(key)

    def start_outer(self, category: str):
        """
        Start an outer timer for the given category.

        Only one outer timer can be active at a time.
        """
        if not self._active:
            return
        if self._current_outer is not None:
            raise RuntimeError(
                "GlobalTimer already has an active outer timer, "
                f"{self._current_outer!r}"
            )
        self._current_outer = category
        self._start_timer(category)

    def stop(self, category: str):
        """
        Stop the inner timer for the given category.
        """
        if not self._active:
            return
        key = self._inner_key(category)
        if key not in self._active_inners:
            raise RuntimeError(
                f"GlobalTimer does not have a running inner timer {key!r}"
            )
        self._stop_timer(key)
        self._active_inners.discard(key)

    def stop_outer(self, category: str):
        """
        Stop the outer timer for the given category.
        """
        if not self._active:
            return
        if self._current_outer != category:
            raise RuntimeError(
                f"Cannot stop outer timer {category!r}; "
                f"active outer timer is {self._current_outer!r}"
            )
        self._stop_timer(category)
        self._current_outer = None

    def get_duration(self, category: str) -> float:
        if self._active:
            return self._timers[category].duration
        else:
            return np.nan

    def get_durations(self) -> dict[str, float]:
        return {category: timer.duration for category, timer in self._timers.items()}

    def log_durations(self):
        for name, duration in self.get_durations().items():
            logging.info(f"{name} duration: {duration:.2f}s")


singleton: GlobalTimer | None = None
