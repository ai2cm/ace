"""This script downloads specified variables of the ERA5 dataset from
the NCAR Research Data Archive (RDA) and uploads them to Google Cloud Storage."""

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import click
import fsspec
import pandas as pd

MEAN_FLUX_CATEGORY_NAME = "e5.oper.fc.sfc.meanflux"
INVARIANT_CATEGORY_NAME = "e5.oper.invariant"
SURFACE_ANALYSIS_CATEGORY_NAME = "e5.oper.an.sfc"

NCAR_DATA = "https://data.rda.ucar.edu"
NCAR_STRATUS = "https://stratus.rda.ucar.edu"


@dataclass
class VariableFile:
    category: str
    variable_name: str
    start_time: datetime
    end_time: datetime

    @property
    def filename(self) -> str:
        start_str = self.start_time.strftime("%Y%m%d%H")
        end_str = self.end_time.strftime("%Y%m%d%H")
        return f"{self.category}.{self.variable_name}.ll025sc.{start_str}_{end_str}.nc"

    def get_rda_url(self, server=NCAR_DATA) -> str:
        year_month = self.start_time.strftime("%Y%m")
        prefix = f"{server}/ds633.0"
        return f"{prefix}/{self.category}/{year_month}/{self.filename}"


def assert_valid_start_time(category: str, start_time: datetime):
    """
    Assert that the category and times are valid.
    """
    if category == MEAN_FLUX_CATEGORY_NAME:
        assert start_time.hour == 6, "All files start at 06Z"
        assert start_time.day in (1, 16), "Files start on 1st or 16th of month"
    elif category == SURFACE_ANALYSIS_CATEGORY_NAME:
        assert start_time.hour == 0, "All files start at 00Z"
        assert start_time.day == 1, "Files start on 1st of month"
    elif category == INVARIANT_CATEGORY_NAME:
        assert start_time == datetime(1979, 1, 1, 0)


def get_date_list(
    category: str, start_time: datetime, n_files: int
) -> List[Tuple[datetime, datetime]]:
    """
    Get a list of dates for a given category and time range.
    """
    assert_valid_start_time(category, start_time)

    if category == INVARIANT_CATEGORY_NAME:
        date_list = [(start_time, start_time)]
    elif category == MEAN_FLUX_CATEGORY_NAME:
        times = pd.date_range(start_time, freq="SMS-16", periods=n_files + 1)
        date_list = list(zip(times[:-1], times[1:]))
    elif category == SURFACE_ANALYSIS_CATEGORY_NAME:
        start_times = pd.date_range(start_time, periods=n_files, freq="MS")
        end_times = pd.date_range(
            start_time.strftime("%Y-%m"), periods=n_files, freq="M"
        ) + pd.Timedelta("23h")
        date_list = list(zip(start_times, end_times))
    return date_list


def get_file_list(
    category: str, variable_name: str, start_time: datetime, n_files: int
) -> List[VariableFile]:
    """
    Get a list of files for a given category, variable, and time range.
    """
    date_list = get_date_list(category, start_time, n_files)
    file_list = []
    for start_date, end_date in date_list:
        file_list.append(VariableFile(category, variable_name, start_date, end_date))
    return file_list


def wget_file(url: str, local_path: str) -> None:
    subprocess.check_call(["wget", "-N", "-O", local_path, url])


def upload_files_to_gcs(local_dir: str, gcs_dir: str) -> None:
    subprocess.check_call(["gsutil", "-m", "cp", f"{local_dir}/*.nc", gcs_dir])


@click.command()
@click.argument("category")
@click.argument("variable_name")
@click.argument("start_time")
@click.argument("n_files", type=int)
@click.option(
    "--gcs-dir",
    default="gs://vcm-ml-scratch/oliwm/era5",
    help="Output path in GCS.",
    show_default=True,
)
@click.option(
    "--n-upload",
    default=8,
    help="Number of files over which to parallelize gsutil upload.",
    show_default=True,
)
def main(category, variable_name, start_time, n_files, gcs_dir, n_upload):
    """Ingest ERA5 data from NCAR RDA database to Google Cloud Storage.

    \b
    CATEGORY: one of 'e5.oper.fc.sfc.meanflux', 'e5.oper.invariant', 'e5.oper.an.sfc'
    NAME: variable name including code, e.g. '235_040_mtnlwrf'
    START_TIME: format YYYYMMDDHH
    N_FILES: the number of files to download starting at START_TIME
    """
    start_time = datetime.strptime(start_time, "%Y%m%d%H")
    file_list = get_file_list(category, variable_name, start_time, n_files)
    fs = fsspec.filesystem("gs")
    full_gcs_dir = f"{gcs_dir}/{category}/{variable_name}"

    with tempfile.TemporaryDirectory() as local_dir:
        scratch_dir = os.path.join(local_dir, "scratch")
        os.makedirs(scratch_dir)
        for i, file in enumerate(file_list):
            if fs.exists(f"{full_gcs_dir}/{file.filename}"):
                print(f"{file.filename} already exists on GCS, skipping download.")
            else:
                print(f"Downloading {file.filename} to local")
                local_file_path = os.path.join(scratch_dir, file.filename)
                # hacky retry logic to handle NCAR server issues
                try:
                    wget_file(file.get_rda_url(), local_file_path)
                except subprocess.CalledProcessError:
                    print("Trying again with different NCAR server")
                    time.sleep(15)
                    try:
                        wget_file(file.get_rda_url(NCAR_STRATUS), local_file_path)
                    except subprocess.CalledProcessError:
                        print("Trying original NCAR server again")
                        time.sleep(60)
                        wget_file(file.get_rda_url(), local_file_path)

            if i % n_upload == n_upload - 1 or i == len(file_list) - 1:
                # parallelize upload to GCS across n_upload files using gsutil -m
                if len(os.listdir(scratch_dir)) > 1:
                    print(f"Uploading the following files to GCS:")
                    print(os.listdir(scratch_dir))
                    upload_files_to_gcs(scratch_dir, full_gcs_dir)
                    shutil.rmtree(scratch_dir)
                    os.makedirs(scratch_dir)
                elif len(os.listdir(scratch_dir)) == 1:
                    # gsutil has different behavior for single file uploads
                    print(f"Uploading {os.listdir(scratch_dir)[0]} to GCS")
                    upload_path = f"{full_gcs_dir}/{os.listdir(scratch_dir)[0]}"
                    upload_files_to_gcs(scratch_dir, upload_path)
                    shutil.rmtree(scratch_dir)
                    os.makedirs(scratch_dir)
                else:
                    print("No files to upload to GCS")


def test_get_date_list():
    start_time = datetime(1979, 1, 1, 0)
    date_list = get_date_list(INVARIANT_CATEGORY_NAME, start_time, 1)
    assert len(date_list) == 1
    assert date_list[0] == (start_time, start_time)

    start_time = datetime(2020, 1, 1, 0)
    date_list = get_date_list(SURFACE_ANALYSIS_CATEGORY_NAME, start_time, 2)
    assert len(date_list) == 2
    assert date_list[0] == (start_time, datetime(2020, 1, 31, 23))
    assert date_list[1] == (datetime(2020, 2, 1, 0), datetime(2020, 2, 29, 23))

    start_time = datetime(2020, 1, 1, 6)
    date_list = get_date_list(MEAN_FLUX_CATEGORY_NAME, start_time, 2)
    assert len(date_list) == 2
    assert date_list[0] == (start_time, datetime(2020, 1, 16, 6))
    assert date_list[1] == (datetime(2020, 1, 16, 6), datetime(2020, 2, 1, 6))


if __name__ == "__main__":
    main()
