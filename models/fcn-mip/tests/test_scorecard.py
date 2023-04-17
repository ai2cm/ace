import glob
from fcn_mip.report import scorecard
import pytest
import datetime
import numpy


def test_scorecard():
    """Not a great test and will only work if .nc files are in 34vars/acc"""
    files = glob.glob("34Vars/acc/*.nc")
    if not files:
        pytest.skip()

    scorecard.read(
        files, channels=["z500", "z200"], lead_times=[datetime.timedelta(days=3)]
    )


def test_get_map():
    cmap = scorecard.get_cmap()
    white = (1, 1, 1, 1)
    numpy.testing.assert_allclose(cmap(0.5), white, atol=1 / 255)
