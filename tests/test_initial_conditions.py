import fcn_mip.initial_conditions.era5 as initial_conditions
import datetime
import pytest


def test__get_path():
    assert "./out_of_sample/2018.h5" == initial_conditions._get_path(
        ".", datetime.datetime(2018, 1, 2)
    )
    assert "./train/1980.h5" == initial_conditions._get_path(
        ".", datetime.datetime(1980, 1, 2)
    )
    assert "./test/2017.h5" == initial_conditions._get_path(
        ".", datetime.datetime(2017, 1, 2)
    )

    with pytest.raises(KeyError):
        initial_conditions._get_path(".", datetime.datetime(2040, 1, 2))
