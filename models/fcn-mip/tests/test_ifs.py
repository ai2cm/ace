from fcn_mip.initial_conditions.ifs import _get_filename, get
from fcn_mip import schema
import datetime

import pytest


def test__get_filename():
    expected = "20230310/00z/0p4-beta/oper/20230310000000-0h-oper-fc.grib2"
    time = datetime.datetime(2023, 3, 10, 0)
    assert _get_filename(time, "0h") == expected


@pytest.mark.xfail
def test_get():
    # uses I/O and old ICs are not available forever.
    time = datetime.datetime(2023, 3, 10, 0)
    ds = get(time, schema.ChannelSet.var34)
    print(ds)
