# These scripts run in their own conda environment (see README) and require dask,
# both for progress output and because set_shards_chunks calls DataArray.chunk.
# The repo-wide test environment has no dask, so skip rather than fail there.
try:
    import dask.diagnostics  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]
