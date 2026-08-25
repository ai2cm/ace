from contextlib import nullcontext

try:
    from dask.diagnostics import ProgressBar
except ImportError:
    # dask only drives progress output here, and is absent from the repo-wide test
    # environment, so degrade to a no-op rather than making it a hard import.
    ProgressBar = nullcontext
