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
        self._current_category: str | None = None

    def outer_context(self, category: str) -> contextlib.AbstractContextManager:
        """
        Context manager for timing a block of code.

        May be active at the same time as other timers.
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

        Only one inner timer can be active at a time.
        """

        @contextlib.contextmanager
        def timer_context():
            self.start(category)
            try:
                yield
            finally:
                self.stop()

        return timer_context()

    def start(self, category: str):
        """
        Start an inner timer for the given category.

        Only one inner timer can be active at a time.
        """
        if self._current_category is not None:
            raise RuntimeError(
                "GlobalTimer already has an active inner timer, "
                f"{self._current_category}"
            )
        self.start_outer(category)
        self._current_category = category

    def start_outer(self, category: str):
        """
        Start a timer for the given category.

        May be active at the same time as other timers.
        """
        if self._active:
            if category not in self._timers:
                self._timers[category] = CumulativeTimer(category)
            self._timers[category].start()

    def stop(self):
        """
        Stop the currently active inner timer.
        """
        if self._current_category is None:
            raise RuntimeError("GlobalTimer does not have a running timer")
        self.stop_outer(self._current_category)
        self._current_category = None

    def stop_outer(self, category: str):
        """
        Stop the timer for the given category.

        Does not change the currently active inner timer.
        """
        if self._active:
            self._timers[category].stop()

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
