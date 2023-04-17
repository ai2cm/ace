from typing import List
import os
import datetime


def filename_to_year(path: str) -> int:
    filename = os.path.basename(path)
    return int(filename[:4])


def datetime_range(
    year: int, time_step: datetime.timedelta, n: int
) -> List[datetime.datetime]:
    initial_time = datetime.datetime(year=year, month=1, day=1)
    return [initial_time + time_step * i for i in range(n)]
