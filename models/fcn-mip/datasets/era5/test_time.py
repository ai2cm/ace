import datetime
from datasets.era5 import time


def test_datetime_range():
    times = time.datetime_range(2018, datetime.timedelta(hours=6), 2)
    assert times == [datetime.datetime(2018, 1, 1, 0), datetime.datetime(2018, 1, 1, 6)]


def test_filename_to_year():
    assert 2018 == time.filename_to_year("some/long/path/2018.h5")
