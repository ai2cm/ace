"""Static staggered-velocity face masks as a versioned GCS artifact.

Some source stores carry coastal staggered velocity faces that are *valid
with value exactly 0.0* where the native vertical grid masks them as land:
MOM6's online z*-remap fills land columns with zeros instead of NaN. The
valid-neighbor C-grid -> tracer-center interpolation would average these
fake zeros into adjacent wet centers, biasing coastal speeds low. This
module precomputes, per level, which faces to treat as invalid, plus the
resulting tracer-center footprint the interpolated velocity pair can
actually cover.

A face is masked iff it satisfies the conjunction

    exactly 0.0 at every timestep of the scanned window (structural)
    AND at least one dry tracer neighbor at that level,

fold-aware for the v-face row on the tripolar seam. The structural leg
protects genuine deep faces whose native velocity mask is wider than the
tracer mask (real nonzero values never fire); the dry-neighbor leg is a
physical-plausibility gate that makes the detector robust to preprocessing
variants, and a safe no-op on properly-masked sources (nothing to flag).
False negatives are impossible by construction (a remap-born zero is 0.0 at
every step); false positives require a real velocity to be exactly 0.0 at
every scanned step next to land.

An artifact is a GCS prefix containing ``face_masks.nc`` with:

- ``u_face_mask`` (z_l, yh, xq) — True where a u face must be treated as
  invalid before center interpolation.
- ``v_face_mask`` (z_l, yq, xh) — likewise for v faces.
- ``center_mask`` (z_l, yh, xh) — the static velocity footprint: tracer
  wetmask AND at least one surviving valid face on each axis. Because
  rotation couples the components, a center missing either component drops
  from both; interpolated pairs are restricted and regrid-normalized to
  this footprint instead of the tracer wetmask (see run._rotate_pairs).

Artifacts are published alongside the regridding weights and treated as
immutable; generation is self-verifying (see generate_face_masks) and
refuses to clobber. Example invocation (see also the Makefile):

    python -m pipeline.face_masks \\
        --config configs/om4-picontrol-1deg.yaml \\
        --stream snapshot_ocean \\
        --output-url \\
          gs://vcm-ml-scratch/jamesd/gfdl-om4-face-masks/v1/om4-picontrol-2026-06-24 \\
        --expected-surface-count-u 16960 \\
        --expected-surface-count-v 16694
"""

import argparse
import logging
import os
import tempfile

import fsspec
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

FACE_MASKS_FILENAME = "face_masks.nc"
TIME_BATCH = 8

# Right/north-edge staggering (see run.STAGGERED_TO_TRACER_DIM): u face i
# sits on the east edge of tracer cell i, between centers i and i+1
# (periodic in x); v face j sits on the north edge of tracer cell j,
# between centers j and j+1. The last v-face row is the tripolar fold seam,
# where the northern neighbor of face (j_last, i) is the same row folded
# back onto itself: center (j_last, nx-1-i).


def u_face_tracer_neighbors(wet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two tracer-cell wet flags flanking each u face.

    ``wet`` has trailing dims (yh, xh); returns (west, east) arrays on the
    face grid (..., yh, xq), with the x-periodic wrap applied.
    """
    return wet, np.roll(wet, -1, axis=-1)


def v_face_tracer_neighbors(wet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two tracer-cell wet flags flanking each v face.

    ``wet`` has trailing dims (yh, xh); returns (south, north) arrays on the
    face grid (..., yq, xh). The northern neighbor of the last (tripolar
    fold seam) row is the same tracer row reflected in x.
    """
    north = np.empty_like(wet)
    north[..., :-1, :] = wet[..., 1:, :]
    north[..., -1, :] = wet[..., -1, ::-1]
    return wet, north


def count_masked_candidate_faces(
    values: np.ndarray, wet: np.ndarray, component: str
) -> int:
    """Count valid faces that are exactly 0.0 with >= 1 dry tracer neighbor.

    ``values`` has trailing face dims ((yh, xq) for "u", (yq, xh) for "v")
    plus any leading (time, level) dims broadcastable against ``wet``'s
    trailing (yh, xh). Used at run time to assert that, after the face
    masks are applied, no face matching the mask-candidate fingerprint
    survives into the center interpolation (a nonzero count means the mask
    artifact is stale relative to the source).
    """
    neighbors = {"u": u_face_tracer_neighbors, "v": v_face_tracer_neighbors}[component]
    side_a, side_b = neighbors(wet)
    dry_neighbor = ~(side_a & side_b)
    return int((np.isfinite(values) & (values == 0.0) & dry_neighbor).sum())


def _center_has_valid_u_face(valid: np.ndarray) -> np.ndarray:
    """Centers with >= 1 valid u face; ``valid`` (..., yh, xq) -> (..., yh, xh).

    Center i averages faces i-1 and i, wrapping in x (the interpolation's
    periodic roll).
    """
    return valid | np.roll(valid, 1, axis=-1)


def _center_has_valid_v_face(valid: np.ndarray) -> np.ndarray:
    """Centers with >= 1 valid v face; ``valid`` (..., yq, xh) -> (..., yh, xh).

    Center j averages faces j-1 and j; the first row has only its own north
    face (the interpolation's non-periodic shift).
    """
    has = valid.copy()
    has[..., 1:, :] |= valid[..., :-1, :]
    return has


def _scan_structural(da: xr.DataArray, time_dim: str) -> tuple[np.ndarray, np.ndarray]:
    """One pass over ``da``'s full time axis, in batches.

    Returns (valid, structural_zero) boolean arrays over the non-time dims:
    faces finite at every step, and faces exactly 0.0 at every step. Errors
    if the finite pattern varies in time (the masks and the center footprint
    are static, so a time-varying face validity would make them wrong).
    """
    finite_all = None
    finite_any = None
    zero_all = None
    n_steps = da.sizes[time_dim]
    for start in range(0, n_steps, TIME_BATCH):
        batch = da.isel({time_dim: slice(start, start + TIME_BATCH)}).load().values
        finite = np.isfinite(batch)
        zero = finite & (batch == 0.0)
        finite_all = (
            finite.all(axis=0)
            if finite_all is None
            else (finite_all & finite.all(axis=0))
        )
        finite_any = (
            finite.any(axis=0)
            if finite_any is None
            else (finite_any | finite.any(axis=0))
        )
        zero_all = (
            zero.all(axis=0) if zero_all is None else (zero_all & zero.all(axis=0))
        )
        logger.info(
            "  %s: scanned steps %d..%d of %d",
            da.name,
            start,
            min(start + TIME_BATCH, n_steps) - 1,
            n_steps,
        )
    assert finite_all is not None and finite_any is not None and zero_all is not None
    varying = int((finite_any & ~finite_all).sum())
    if varying:
        raise AssertionError(
            f"{da.name}: valid-face pattern varies in time at {varying} faces; "
            "a static face mask artifact cannot represent this source"
        )
    return finite_all, zero_all


def generate_face_masks(
    config_path: str,
    stream_name: str,
    output_url: str,
    expected_surface_count_u: int | None = None,
    expected_surface_count_v: int | None = None,
    overwrite: bool = False,
) -> None:
    """Scan a stream's C-grid velocity pair over the config's full time
    window, compute the per-level face masks and center footprint, verify
    them, and write the artifact under the ``output_url`` prefix.

    Self-verification, before anything is written:

    - the flagged surface counts equal the expected counts, when given
      (an independent census of the source's remap-born zeros);
    - at the surface, the flagged u faces are exactly the faces with at
      least one wet and at least one dry tracer neighbor (any-neighbor-wet
      minus both-neighbors-wet) — the geometric signature of a coastal
      remap-unmasked face — and likewise for v with fold-aware neighbors;
    - every flagged face borders at least one wet tracer cell (it must, to
      contaminate anything) and is structurally zero by construction.

    Refuses to clobber an existing artifact unless ``overwrite`` is set:
    published artifacts are immutable, so a changed detector or source
    belongs under a new version prefix.
    """
    # Imported here: run.py imports this module for the artifact loader.
    from .config import load_config
    from .run import LEVEL_DIM, TIME_DIM, load_wetmask, open_stream

    fs, _ = fsspec.url_to_fs(output_url)
    url = f"{output_url.rstrip('/')}/{FACE_MASKS_FILENAME}"
    if fs.exists(url) and not overwrite:
        raise FileExistsError(
            f"Face-mask artifact already exists at {url}. Published artifacts "
            "are immutable: bump the version in the output URL (e.g. "
            "FACE_MASKS_VERSION in the Makefile), or pass --overwrite if you "
            "really mean to replace it."
        )

    config = load_config(config_path)
    streams = [s for s in config.streams if s.name == stream_name]
    if not streams:
        raise ValueError(
            f"stream {stream_name!r} not found in {config_path}; available: "
            f"{[s.name for s in config.streams]}"
        )
    stream = streams[0]
    if not stream.rotated_pairs:
        raise ValueError(f"stream {stream_name!r} has no rotated pairs")
    u_name, v_name = stream.rotated_pairs[0]

    logger.info("loading tracer wetmask")
    wetmask = load_wetmask(config)
    wet = wetmask.transpose(LEVEL_DIM, "yh", "xh").values

    ds = open_stream(stream, config)
    logger.info(
        "scanning %s/%s over %d timesteps (%s .. %s)",
        u_name,
        v_name,
        ds.sizes[TIME_DIM],
        ds[TIME_DIM].values[0],
        ds[TIME_DIM].values[-1],
    )
    u = ds[u_name].transpose(TIME_DIM, LEVEL_DIM, "yh", "xq")
    v = ds[v_name].transpose(TIME_DIM, LEVEL_DIM, "yq", "xh")
    u_valid, u_zero = _scan_structural(u, TIME_DIM)
    v_valid, v_zero = _scan_structural(v, TIME_DIM)

    u_west, u_east = u_face_tracer_neighbors(wet)
    v_south, v_north = v_face_tracer_neighbors(wet)
    u_mask = u_zero & ~(u_west & u_east)
    v_mask = v_zero & ~(v_south & v_north)

    # Verification: the flagged surface faces must be exactly the valid
    # faces straddling the coastline (one wet, one dry tracer neighbor).
    for name, mask, valid, side_a, side_b in [
        (u_name, u_mask[0], u_valid[0], u_west[0], u_east[0]),
        (v_name, v_mask[0], v_valid[0], v_south[0], v_north[0]),
    ]:
        coastline = (side_a | side_b) & ~(side_a & side_b)
        mismatches = int((mask != coastline).sum())
        if mismatches:
            raise AssertionError(
                f"{name}: flagged surface faces differ from the coastal "
                f"(any-neighbor-wet minus both-neighbors-wet) faces at "
                f"{mismatches} faces; the source does not match the "
                "remap-unmasked-zero-face model this detector assumes"
            )
        if int((mask & ~valid).sum()):
            raise AssertionError(f"{name}: flagged faces outside the valid set")
        orphaned = int((mask & ~(side_a | side_b)).sum())
        if orphaned:
            raise AssertionError(
                f"{name}: {orphaned} flagged surface faces have no wet "
                "tracer neighbor and cannot contaminate any center"
            )
    for name, mask, expected in [
        (u_name, u_mask, expected_surface_count_u),
        (v_name, v_mask, expected_surface_count_v),
    ]:
        count = int(mask[0].sum())
        logger.info("%s: %d flagged surface faces, %d total", name, count, mask.sum())
        if expected is not None and count != expected:
            raise AssertionError(
                f"{name}: {count} flagged surface faces; expected {expected}"
            )

    center_mask = (
        wet
        & _center_has_valid_u_face(u_valid & ~u_mask)
        & _center_has_valid_v_face(v_valid & ~v_mask)
    )
    dropped = wet & ~center_mask
    logger.info(
        "center footprint: %d wet centers lose all valid faces on some axis "
        "(%d at the surface)",
        int(dropped.sum()),
        int(dropped[0].sum()),
    )

    artifact = xr.Dataset(
        {
            "u_face_mask": ((LEVEL_DIM, "yh", "xq"), u_mask),
            "v_face_mask": ((LEVEL_DIM, "yq", "xh"), v_mask),
            "center_mask": ((LEVEL_DIM, "yh", "xh"), center_mask),
        },
        coords={LEVEL_DIM: ds[LEVEL_DIM].values},
        attrs={
            "history": (
                "Structurally-zero coastal staggered face masks and the "
                "resulting tracer-center velocity footprint, computed by "
                "scripts/gfdl_om4/pipeline/face_masks.py from "
                f"{u_name}/{v_name} of {stream.store} over "
                f"{ds.sizes[TIME_DIM]} timesteps "
                f"({ds[TIME_DIM].values[0]} .. {ds[TIME_DIM].values[-1]}), "
                f"with the tracer wetmask from {config.wetmask.store}."
            ),
            "detector": (
                "face masked iff exactly 0.0 at every scanned timestep AND "
                "at least one dry tracer neighbor at that level (fold-aware "
                "on the tripolar seam row)"
            ),
        },
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, FACE_MASKS_FILENAME)
        artifact.to_netcdf(local_path)
        logger.info("uploading %s", url)
        with open(local_path, "rb") as src, fsspec.open(url, "wb") as dst:
            dst.write(src.read())
    logger.info("face-mask artifact complete")


# One loaded artifact per URL per worker process.
_FACE_MASK_CACHE: dict[str, xr.Dataset] = {}


def get_face_masks(face_mask_url: str) -> xr.Dataset:
    """Load (and cache) the face-mask artifact at the ``face_mask_url``
    prefix: boolean ``u_face_mask``/``v_face_mask``/``center_mask``."""
    if face_mask_url not in _FACE_MASK_CACHE:
        logger.info("loading face masks from %s", face_mask_url)
        url = f"{face_mask_url.rstrip('/')}/{FACE_MASKS_FILENAME}"
        with fsspec.open(url) as f:
            _FACE_MASK_CACHE[face_mask_url] = xr.open_dataset(f).load()
    return _FACE_MASK_CACHE[face_mask_url]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute static staggered-velocity face masks and "
        "write them to GCS as a versioned artifact."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the pipeline YAML config"
    )
    parser.add_argument(
        "--stream",
        required=True,
        help="Name of the config stream whose first rotated pair is scanned",
    )
    parser.add_argument(
        "--output-url", required=True, help="GCS prefix for the artifact"
    )
    parser.add_argument(
        "--expected-surface-count-u",
        type=int,
        help="Assert this many flagged surface u faces (independent census)",
    )
    parser.add_argument(
        "--expected-surface-count-v",
        type=int,
        help="Assert this many flagged surface v faces (independent census)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing artifact at the output URL "
        "(default: refuse; bump the artifact version instead)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_face_masks(
        args.config,
        args.stream,
        args.output_url,
        args.expected_surface_count_u,
        args.expected_surface_count_v,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
