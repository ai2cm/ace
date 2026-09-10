"""Post-run check that an output store opens with the expected variables.

Opens the store a pipeline invocation wrote and asserts its variable set
equals the set implied by the config (renamed stream outputs, full-cell
twins, postprocess additions, and the static ocean fraction), and optionally
the timestep count. Used by the Makefile smoke-test targets after each
DirectRunner subset run, and by hand after a production launch.
"""

import argparse
import logging

import xarray as xr

from .config import load_config
from .run import (
    TIME_DIM,
    _expected_output_names,
    _make_zarr_store,
    land_nan_exempt_names,
    open_stream,
)

logger = logging.getLogger(__name__)


def check_output(
    config_path: str,
    member: str | None = None,
    output_path: str | None = None,
    expected_timesteps: int | None = None,
) -> None:
    config = load_config(config_path, member)
    path = output_path if output_path is not None else config.output.path
    stream_dataset = open_stream(config.stream, config)
    expected = _expected_output_names(config, stream_dataset) | set(
        land_nan_exempt_names()
    )
    ds = xr.open_zarr(_make_zarr_store(path), chunks=None, decode_timedelta=False)
    actual = set(ds.data_vars)
    if actual != expected:
        raise AssertionError(
            f"output store {path} variables disagree with the set implied by "
            f"{config_path}; missing={sorted(expected - actual)} "
            f"unexpected={sorted(actual - expected)}"
        )
    if expected_timesteps is not None and ds.sizes[TIME_DIM] != expected_timesteps:
        raise AssertionError(
            f"output store {path} has {ds.sizes[TIME_DIM]} timesteps; "
            f"expected {expected_timesteps}"
        )
    logger.info(
        "output store %s: variables %s, %d timesteps — as expected",
        path,
        sorted(actual),
        ds.sizes[TIME_DIM],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that an output store opens with the variable set "
        "implied by its config."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the pipeline YAML config"
    )
    parser.add_argument(
        "--member", help="Ensemble-member name substituted into the config's URLs"
    )
    parser.add_argument(
        "--output-path",
        help="Store to check (defaults to the config's output path)",
    )
    parser.add_argument(
        "--expected-timesteps",
        type=int,
        help="Assert the store has exactly this many timesteps",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # apache_beam may have already configured the root logger on import,
    # making basicConfig a no-op; raise the level explicitly.
    logging.getLogger().setLevel(logging.INFO)
    check_output(args.config, args.member, args.output_path, args.expected_timesteps)


if __name__ == "__main__":
    main()
