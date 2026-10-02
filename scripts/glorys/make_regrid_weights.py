"""Precompute the 1/12 degree -> output grid conservative regridding weights.

The pipeline accepts ``--regrid_weights <file>`` and passes it to xESMF as
``reuse_weights=True``, but nothing wrote such a file, so every run -- and on
Dataflow, every worker -- rebuilt the weights for 8.8 M source cells from
scratch. That costs minutes and several GB each time, and it is the largest
single term in the pipeline's peak memory.

Generate once, then pass the file to every subsequent run:

    python make_regrid_weights.py /tmp/glorys-F90-weights.nc
    gsutil cp /tmp/glorys-F90-weights.nc gs://vcm-ml-intermediate/<prefix>/

    make glorys_local_debug \
        LOCAL_EXTRA_FLAGS="--regrid_weights /tmp/glorys-F90-weights.nc"

The weights depend only on the source and target grids, so one file serves the
whole record as long as ``--output_grid`` does not change.
"""

import argparse
import importlib.util
import logging
import os

import xesmf as xe

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "glorys_pipeline", os.path.join(_HERE, "pipeline", "glorys-pipeline.py")
)
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)


def main():
    logging.basicConfig(level=logging.INFO, force=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_path", help="netCDF file to write the weights to")
    p.add_argument("--output_grid", default=gp.DEFAULT_OUTPUT_GRID)
    args = p.parse_args()

    # The source grid comes from the bathymetry store's coordinates, which is
    # what build_invariants regrids from; taking it from the same place keeps
    # the weights valid for every field the pipeline regrids.
    bathy = gp.open_cmems(gp.URL_BATHY).rename(
        {"latitude": "lat", "longitude": "lon"}
    )
    src = gp._make_source_grid(bathy["lat"].values, bathy["lon"].values)
    dst = gp._make_target_grid(args.output_grid)
    logging.info(
        "building conservative weights: %d x %d -> %s",
        len(src["lat"]),
        len(src["lon"]),
        args.output_grid,
    )
    regridder = xe.Regridder(src, dst, "conservative", periodic=True)
    regridder.to_netcdf(args.output_path)
    size_mb = os.path.getsize(args.output_path) / 1e6
    logging.info("wrote %s (%.1f MB)", args.output_path, size_mb)


if __name__ == "__main__":
    main()
