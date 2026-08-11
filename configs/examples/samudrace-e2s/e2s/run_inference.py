#!/usr/bin/env python
"""earth2studio SamudrACE validation inference.

Runs the earth2studio ``SamudrACE`` prognostic wrapper through
``earth2studio.run.deterministic`` with the SamudrACE CM4 piControl
checkpoint, and writes the comparison fields to a zarr store.

All model inputs (checkpoint tar, forcing NetCDF, initial conditions) are read
from a local artifact directory that mirrors the
``allenai/SamudrACE-CM4-piControl`` HuggingFace repository layout; nothing is
fetched from the network at run time.

Configuration comes from the environment:

``SAMUDRACE_ARTIFACTS``
    Artifact root directory, default ``/samudrace-artifacts``.
``SAMUDRACE_OUTPUT_DIR``
    Output directory, default ``/results``.
``SAMUDRACE_N_COUPLED_CYCLES``
    Number of coupled (ocean) cycles to run, default ``24``.
``SAMUDRACE_SCENARIO``
    Forcing scenario, default ``0311``.
``SAMUDRACE_IC_TIME``
    Initial-condition timestamp, default ``0311-01-01T00:00:00``.
``SAMUDRACE_DEVICE``
    Torch device, default ``cuda``. Running on CPU is refused unless
    ``SAMUDRACE_ALLOW_CPU=1``.
``SAMUDRACE_ALLOW_CPU``
    Set to ``1`` to permit a CPU run (smoke tests). Without it, a missing or
    unusable CUDA device is a hard error rather than a silent degradation to
    CPU, which would turn a minutes-long GPU job into an hours-long one.
"""

import os
from collections import OrderedDict
from inspect import signature

import earth2studio.run as run
import numpy as np
import torch
import xarray as xr
from earth2studio.data.samudrace import SamudrACEData, SamudrACEForcingData
from earth2studio.data.utils import prep_data_array
from earth2studio.io import ZarrBackend
from earth2studio.models.auto import Package
from earth2studio.models.px.samudrace import SamudrACE
from loguru import logger

CHECKPOINT_FILE = "samudrACE_CM4_piControl_ckpt.tar"

# Comparison fields, in earth2studio names (see SamudrACELexicon):
#   atmosphere t2m/u10m/v10m/skt -> FME TMP2m/UGRD10m/VGRD10m/surface_temperature
#   ocean      sst/siconc        -> FME sst/ocean_sea_ice_fraction
COMPARISON_VARIABLES = ["t2m", "u10m", "v10m", "skt", "sst", "siconc"]


class LocalSamudrACEData(SamudrACEData):
    """Initial-condition data source backed by a local artifact directory.

    Parameters
    ----------
    root : str
        Directory mirroring the SamudrACE HuggingFace repository layout.
    verbose : bool, optional
        Log file access, by default True.
    """

    def __init__(self, root: str, verbose: bool = True):
        super().__init__(cache=True, verbose=verbose)
        self._root = root

    def _fetch_file(self, filename: str) -> str:
        """Resolve a repository-relative filename against the local root.

        Parameters
        ----------
        filename : str
            Path of the file within the repository layout.

        Returns:
        -------
        str
            Local filesystem path.
        """
        return _resolve_local(self._root, filename, self._verbose)


class LocalSamudrACEForcingData(SamudrACEForcingData):
    """Forcing data source backed by a local artifact directory.

    Parameters
    ----------
    root : str
        Directory mirroring the SamudrACE HuggingFace repository layout.
    scenario : str, optional
        Forcing scenario, by default "0311".
    verbose : bool, optional
        Log file access, by default True.
    """

    def __init__(self, root: str, scenario: str = "0311", verbose: bool = True):
        super().__init__(scenario=scenario, cache=True, verbose=verbose)
        self._root = root

    def _fetch_file(self, filename: str) -> str:
        """Resolve a repository-relative filename against the local root.

        Parameters
        ----------
        filename : str
            Path of the file within the repository layout.

        Returns:
        -------
        str
            Local filesystem path.
        """
        return _resolve_local(self._root, filename, self._verbose)


def _resolve_local(root: str, filename: str, verbose: bool) -> str:
    """Resolve a repository-relative filename inside a local artifact root.

    Parameters
    ----------
    root : str
        Artifact root directory.
    filename : str
        Path of the file within the repository layout.
    verbose : bool
        Log the resolved path.

    Returns:
    -------
    str
        Local filesystem path.

    Raises:
    ------
    FileNotFoundError
        If the file is absent from the artifact directory.
    """
    path = os.path.join(root, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SamudrACE artifact '{filename}' not found at '{path}'; the "
            f"artifact directory must mirror the HuggingFace repository layout"
        )
    if verbose:
        logger.info("Using local SamudrACE artifact: {}", path)
    return path


def _to_time_array_seconds(time: list) -> np.ndarray:
    """Convert a time iterable to a second-precision datetime64 array.

    Replaces ``earth2studio.utils.time.to_time_array``, which casts to
    nanosecond precision. SamudrACE times are CM4 model years (e.g. year 311),
    which silently wrap around when cast to ``datetime64[ns]``.

    Parameters
    ----------
    time : list
        Iterable of strings, datetimes, or np.datetime64 values.

    Returns:
    -------
    np.ndarray
        Array of np.datetime64 with second precision.
    """
    return np.array([np.datetime64(ts) for ts in time], dtype="datetime64[s]")


def _fetch_data_seconds(
    source: object,
    time: np.ndarray,
    variable: np.ndarray,
    lead_time: np.ndarray = np.array([np.timedelta64(0, "h")]),
    device: torch.device = "cpu",
    interp_to: OrderedDict | None = None,
    interp_method: str = "nearest",
) -> tuple[torch.Tensor, OrderedDict]:
    """Fetch data from a data source at second time precision.

    Replaces ``earth2studio.data.utils.fetch_data``, which casts requested
    times to nanosecond precision and so cannot address CM4 model years.

    Parameters
    ----------
    source : object
        Data source to fetch from.
    time : np.ndarray
        Timestamps to fetch, second precision.
    variable : np.ndarray
        Variable names to fetch.
    lead_time : np.ndarray, optional
        Lead times to fetch for each time, by default a single zero lead time.
    device : torch.device, optional
        Device to load the data tensor onto, by default "cpu".
    interp_to : OrderedDict | None, optional
        Unsupported here; must be None, by default None.
    interp_method : str, optional
        Unused, kept for signature compatibility, by default "nearest".

    Returns:
    -------
    tuple[torch.Tensor, OrderedDict]
        Data tensor and coordinate system.

    Raises:
    ------
    ValueError
        If regridding is requested, or the source is a forecast source.
    """
    if interp_to is not None:
        raise ValueError("Regridding is not supported for SamudrACE data sources")
    if "lead_time" in signature(source.__call__).parameters:
        raise ValueError("Forecast data sources are not supported")

    time = time.astype("datetime64[s]")
    arrays = []
    for lead in lead_time:
        adjusted = time + lead.astype("timedelta64[s]")
        da = source(adjusted, variable)
        da = da.expand_dims(dim={"lead_time": 1}, axis=1)
        da = da.assign_coords(
            lead_time=np.array([lead], dtype="timedelta64[s]"), time=time
        )
        arrays.append(da)
    da = xr.concat(arrays, "lead_time") if len(arrays) > 1 else arrays[0]
    return prep_data_array(da, device=device)


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : int
        Value used when the variable is unset.

    Returns:
    -------
    int
        Parsed value.
    """
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _resolve_device() -> torch.device:
    """Resolve the torch device, refusing to silently fall back to CPU.

    The device is ``SAMUDRACE_DEVICE`` if set, else ``cuda``. Running on CPU
    -- either because ``SAMUDRACE_DEVICE`` asked for it or because CUDA is
    unavailable -- requires the explicit opt-in ``SAMUDRACE_ALLOW_CPU=1``.

    Returns:
    -------
    torch.device
        Device to run inference on.

    Raises:
    ------
    RuntimeError
        If the run would land on CPU without ``SAMUDRACE_ALLOW_CPU=1``.
    """
    allow_cpu = os.environ.get("SAMUDRACE_ALLOW_CPU") == "1"
    requested = os.environ.get("SAMUDRACE_DEVICE", "cuda")
    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        message = (
            f"CUDA was requested (SAMUDRACE_DEVICE={requested}) but "
            f"torch.cuda.is_available() is False. torch {torch.__version__} "
            f"is built for CUDA {torch.version.cuda}; check that the torch "
            f"wheel's CUDA version is supported by the node's driver "
            f"(`nvidia-smi`) and that the job actually requested a GPU."
        )
        if not allow_cpu:
            raise RuntimeError(
                f"{message} Refusing to fall back to CPU: this run takes "
                f"minutes on a GPU and hours on CPU. Set "
                f"SAMUDRACE_ALLOW_CPU=1 to run on CPU anyway."
            )
        logger.warning("{} Falling back to CPU (SAMUDRACE_ALLOW_CPU=1).", message)
        device = torch.device("cpu")

    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            f"SAMUDRACE_DEVICE={requested} would run this job on CPU, which "
            f"is orders of magnitude slower. Set SAMUDRACE_ALLOW_CPU=1 to "
            f"confirm."
        )

    return device


def main() -> None:
    """Run the SamudrACE validation forecast and write the outputs."""
    artifacts = os.environ.get("SAMUDRACE_ARTIFACTS", "/samudrace-artifacts")
    output_dir = os.environ.get("SAMUDRACE_OUTPUT_DIR", "/results")
    n_cycles = _env_int("SAMUDRACE_N_COUPLED_CYCLES", 24)
    scenario = os.environ.get("SAMUDRACE_SCENARIO", "0311")
    ic_time = os.environ.get("SAMUDRACE_IC_TIME", "0311-01-01T00:00:00")
    device = _resolve_device()

    # Match the fme inference entrypoint (fme/coupled/inference/inference.py),
    # which enables cuDNN autotuning on GPU. earth2studio calls the stepper
    # directly and never runs that entrypoint.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Second-precision time handling: CM4 model years overflow the
    # nanosecond-precision timestamps that the stock workflow helpers use
    run.to_time_array = _to_time_array_seconds
    run.fetch_data = _fetch_data_seconds

    # Local package: Package resolves a plain directory through the local
    # filesystem, so no HuggingFace fetch happens. The forcing and
    # initial-condition sources read the same artifact directory.
    package = Package(artifacts)
    _resolve_local(artifacts, CHECKPOINT_FILE, verbose=True)
    forcing = LocalSamudrACEForcingData(artifacts, scenario=scenario)
    model = SamudrACE.load_model(package, forcing_data_source=forcing)
    data = LocalSamudrACEData(artifacts)

    n_inner_steps = model.stepper.n_inner_steps
    nsteps = n_cycles * n_inner_steps
    logger.info(
        "SamudrACE forecast: IC {}, scenario {}, {} coupled cycles = {} "
        "atmosphere steps ({} steps per cycle), device {}",
        ic_time,
        scenario,
        n_cycles,
        nsteps,
        n_inner_steps,
        device,
    )

    os.makedirs(output_dir, exist_ok=True)
    store_path = os.path.join(output_dir, "samudrace_forecast.zarr")
    io = ZarrBackend(
        file_name=store_path,
        chunks={"time": 1, "lead_time": 1},
        backend_kwargs={"overwrite": True},
    )

    run.deterministic(
        [np.datetime64(ic_time, "s")],
        nsteps,
        model,
        data,
        io,
        output_coords=OrderedDict(
            {"variable": np.array(COMPARISON_VARIABLES, dtype=object)}
        ),
        device=device,
    )

    logger.info("Wrote {}", store_path)
    logger.info("Store arrays: {}", sorted(io.root.array_keys()))


if __name__ == "__main__":
    main()
