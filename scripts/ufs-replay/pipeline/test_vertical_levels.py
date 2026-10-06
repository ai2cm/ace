"""Unit tests for the CM4-matched vertical coarsening of the UFS replay pipeline.

Runs in the pipeline's own environment (``make create_environment``); the
module imports apache_beam and xesmf, so the tests are skipped where those
are absent.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("apache_beam")
pytest.importorskip("xesmf")

_SPEC = importlib.util.spec_from_file_location(
    "ufs_replay_pipeline", pathlib.Path(__file__).with_name("ufs-replay-pipeline.py")
)
assert _SPEC is not None and _SPEC.loader is not None
pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pipeline)

# MOM6 z* 75-layer centres of the GEFSv13 replay ocean (metres), as stored in
# gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/06h-freq/zarr/mom6.zarr
# ("z_l"). The first layer is 1.03 m thick and thicknesses grow geometrically.
REPLAY_Z_L = np.array(
    [
        0.5154,
        1.5713,
        2.6869,
        3.8802,
        5.1700,
        6.5797,
        8.1377,
        9.8779,
        11.8403,
        14.0703,
        16.6179,
        19.5356,
        22.8758,
        26.6872,
        31.0119,
        35.8837,
        41.3280,
        47.3636,
        54.0065,
        61.2745,
        69.1918,
        77.7934,
        87.1278,
        97.2606,
        108.2755,
        120.2765,
        133.3895,
        147.7632,
        163.5713,
        181.0136,
        200.3178,
        221.7407,
        245.5697,
        272.1227,
        301.7486,
        334.8254,
        371.7584,
        412.9755,
        458.9213,
        510.0494,
        566.8126,
        629.6508,
        698.9779,
        775.1686,
        858.5435,
        949.3572,
        1047.7867,
        1153.9242,
        1267.7725,
        1389.2447,
        1518.1682,
        1654.2928,
        1797.3015,
        1946.8225,
        2102.4431,
        2263.7241,
        2430.2130,
        2601.4559,
        2777.0070,
        2956.4360,
        3139.3353,
        3325.3233,
        3514.0475,
        3705.1855,
        3898.4449,
        4093.5632,
        4290.3050,
        4488.4609,
        4687.8467,
        4888.2999,
        5089.6777,
        5291.8551,
        5494.7236,
        5698.1888,
        5902.0581,
    ]
)


def test_interfaces_are_contiguous_and_start_at_zero():
    zi = pipeline.native_interfaces_from_layer_centers(REPLAY_Z_L)
    assert zi[0] == 0.0
    assert len(zi) == len(REPLAY_Z_L) + 1
    # centres are the midpoints of the reconstructed interfaces
    np.testing.assert_allclose(0.5 * (zi[1:] + zi[:-1]), REPLAY_Z_L, rtol=0, atol=1e-6)
    assert np.all(np.diff(zi) > 0)


def test_default_indices_match_cm4_interfaces_on_the_replay_grid():
    groups = pipeline.cm4_matched_coarsening_indices(REPLAY_Z_L)
    assert groups == pipeline.DEFAULT_VERTICAL_COARSENING_INDICES
    zi = pipeline.native_interfaces_from_layer_centers(REPLAY_Z_L)
    bottoms = np.array([zi[end] for _, end in groups])
    targets = np.array(pipeline.CM4_INTERFACE_DEPTHS)
    # every band bottom is the nearest native interface to its CM4 target...
    for end, d in zip((end for _, end in groups[:-1]), targets[:-1]):
        assert end == int(np.argmin(np.abs(zi - d)))
    # ...within 16 m down to 1600 m and 100 m below, except the native bottom
    assert np.all(np.abs(bottoms[:13] - targets[:13]) < 16)
    assert np.all(np.abs(bottoms[13:-1] - targets[13:-1]) < 100)
    assert bottoms[-1] == pytest.approx(zi[-1])
    # bands tile the column
    assert groups[0][0] == 0 and groups[-1][1] == len(REPLAY_Z_L)
    assert all(a[1] == b[0] for a, b in zip(groups[:-1], groups[1:]))


def test_check_rejects_the_pre_fix_grouping():
    old = [
        [0, 3],
        [3, 8],
        [8, 13],
        [13, 17],
        [17, 20],
        [20, 25],
        [25, 29],
        [29, 33],
        [33, 37],
        [37, 41],
        [41, 44],
        [44, 47],
        [47, 50],
        [50, 53],
        [53, 57],
        [57, 61],
        [61, 66],
        [66, 71],
        [71, 75],
    ]
    with pytest.raises(ValueError, match="do not match the CM4 interfaces"):
        pipeline.check_coarsening_indices(REPLAY_Z_L, old)
    pipeline.check_coarsening_indices(
        REPLAY_Z_L, pipeline.DEFAULT_VERTICAL_COARSENING_INDICES
    )


def test_rejects_non_contiguous_centres():
    with pytest.raises(ValueError):
        pipeline.native_interfaces_from_layer_centers([1.0, 1.5, 1.6])


def test_stress_fields_follow_cm4_conventions():
    # atmosphere-side wind stress from FV3 (CM4's sign convention, defined over
    # land); ocean-side stress from MOM6 under CM4's tauuo/tauvo names
    assert pipeline.ATMO_FORCING_VARS["uflx_ave"] == "eastward_surface_wind_stress"
    assert pipeline.ATMO_FORCING_VARS["vflx_ave"] == "northward_surface_wind_stress"
    assert pipeline.STRESS_RENAME == {"taux": "tauuo", "tauy": "tauvo"}
