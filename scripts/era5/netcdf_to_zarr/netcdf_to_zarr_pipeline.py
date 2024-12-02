import argparse
import itertools
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Generator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import apache_beam as beam
import fsspec
import pandas as pd
import xarray as xr
import xarray_beam as xbeam
import zarr
from apache_beam.pipeline import PipelineOptions

GCS_ROOT = (
    "gs://vcm-ml-raw-flexible-retention/2024-03-11-era5-025deg-2D-variables-from-NCAR"
)

TIME_FORMAT = "%Y%m%d%H"

AN_SFC = "e5.oper.an.sfc"
FC_SFC_MEANFLUX = "e5.oper.fc.sfc.meanflux"
INVARIANT = "e5.oper.invariant"

TARGET_CHUNKS = {
    FC_SFC_MEANFLUX: {
        "forecast_initial_time": 1,
        "forecast_hour": 12,
        "latitude": 721,
        "longitude": 1440,
    },
    AN_SFC: {"time": 12, "latitude": 721, "longitude": 1440},
    INVARIANT: {"time": 1, "latitude": 721, "longitude": 1440},
}

TARGET_SPLIT_CHUNKS = {
    FC_SFC_MEANFLUX: {
        "forecast_initial_time": 1,
        "forecast_hour": 1,
        "latitude": 721,
        "longitude": 1440,
    },
    AN_SFC: {"time": 1, "latitude": 721, "longitude": 1440},
    INVARIANT: {"time": 1, "latitude": 721, "longitude": 1440},
}

VARIABLES = {
    FC_SFC_MEANFLUX: [
        "235_040_mtnlwrf",
        "235_035_msdwswrf",
        "235_037_msnswrf",
        "235_036_msdwlwrf",
        "235_038_msnlwrf",
        "235_033_msshf",
        "235_034_mslhf",
        "235_043_mer",
        "235_055_mtpr",
        "235_031_msr",
        "235_054_mvimd",
        "235_053_mtdwswrf",
        "235_039_mtnswrf",
    ],
    AN_SFC: [
        "128_031_ci",
        "128_039_swvl1",
        "128_040_swvl2",
        "128_041_swvl3",
        "128_042_swvl4",
    ],
    INVARIANT: ["128_172_lsm", "128_129_z"],
}

# Throughout this script, "initial_time" refers to the global initial time of
# the dataset, while "start_time" refers to the first time in an individual
# file.
INITIAL_TIME = {
    FC_SFC_MEANFLUX: "1940-01-01T06",
    AN_SFC: "1940-01",
    INVARIANT: "1979-01-01",
}

PERIODS = {
    FC_SFC_MEANFLUX: 2 * 12 * 84,  # two files per month, 84 years
    AN_SFC: 12 * 84,  # one file per month, 84 years
    INVARIANT: 1,
}

INCLUSIVE = {
    FC_SFC_MEANFLUX: "left",
    AN_SFC: "both",
    INVARIANT: "both",  # Just must not be "neither"
}

TIME_DIM = {
    FC_SFC_MEANFLUX: "forecast_initial_time",
    AN_SFC: "time",
    INVARIANT: "time",
}

FREQUENCY = {
    FC_SFC_MEANFLUX: "12h",
    AN_SFC: "1h",
    INVARIANT: None,
}


def get_time_bounds(
    category: str,
    periods: int,
    initial_time: str,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if category == FC_SFC_MEANFLUX:
        times = pd.date_range(initial_time, freq="SMS-16", periods=periods + 1)
        return list(zip(times[:-1], times[1:]))
    elif category == AN_SFC:
        start_times = pd.date_range(initial_time, freq="MS", periods=periods)
        end_times = pd.date_range(
            initial_time, periods=periods, freq="ME"
        ) + pd.Timedelta("23h")
        return list(zip(start_times, end_times))
    elif category == INVARIANT:
        return [(pd.Timestamp(initial_time), pd.Timestamp(initial_time))]
    else:
        raise ValueError(f"Unknown category {category}")


def get_path(
    variable: str, start_time: pd.Timestamp, end_time: pd.Timestamp, category: str
) -> str:
    start_label = start_time.strftime(TIME_FORMAT)
    end_label = end_time.strftime(TIME_FORMAT)
    return (
        f"{GCS_ROOT}/{category}/{variable}/"
        f"{category}.{variable}.ll025sc.{start_label}_{end_label}.nc"
    )


def get_file_metadata(
    time_bounds: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
    variables: Sequence[str],
    category: str,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, str, str, str]]:
    items = []
    for (start_time, end_time), variable in itertools.product(time_bounds, variables):
        path = get_path(variable, start_time, end_time, category)
        item = (start_time, end_time, variable, category, path)
        items.append(item)
    return items


def validate_time(
    ds: xr.Dataset,
    time_dim: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    category: str,
) -> None:
    freq = FREQUENCY[category]
    inclusive = INCLUSIVE[category]
    expected = pd.date_range(start_time, end_time, freq=freq, inclusive=inclusive)
    message = "time coordinate of dataset does not match expected time coordinate"
    assert ds.indexes[time_dim].equals(expected), message


def _get_fs(path):
    protocol = urlparse(path).scheme
    if not protocol:
        protocol = "file"

    fs = fsspec.filesystem(protocol)

    return fs


def _get_key_and_ds(start_time, end_time, variable, category, path):
    """Adjust ds and return key, ds tuple"""

    initial_time = INITIAL_TIME[category]
    time_dim = TIME_DIM[category]

    # Open files as in pangeo-forge and as recommended by xarray-beam (i.e.
    # with the default chunks=None to prevent using dask). This makes Dataflow
    # jobs much less expensive.
    ds = xr.open_dataset(path, engine="h5netcdf", cache=False)

    # Drop utc_date variable, which is present in every Dataset, but is
    # redundant with the time coordinate.
    ds = ds.drop_vars(["utc_date"])

    # Check to make sure the times in the file match what is expected, i.e. a
    # time series with a regular frequency starting at the start time written
    # in the filename and ending either on the end time written in the filename
    # or the time interval before depending on the value of inclusive. This is
    # not strictly necessary, but is a sanity check. If it holds for each file,
    # then our calculation of the offset for each file and the total length of
    # the dataset will be correct.
    validate_time(ds, time_dim, start_time, end_time, category)

    # Infer the offset based on the frequency of the files and the initial
    # date. If the frequency is None, since the time dimension only contains
    # one element, set the offset to 0.
    freq = FREQUENCY[category]
    if freq is None:
        offset = 0
    else:
        offset = (start_time - pd.Timestamp(initial_time)) // pd.Timedelta(freq)

    key = xbeam.Key({time_dim: offset}, vars={variable})
    return key, ds


def create_record(
    item: Tuple[pd.Timestamp, pd.Timestamp, str, str, str], local=True
) -> Tuple[xbeam.Key, xr.Dataset]:
    start_time, end_time, variable, category, path = item

    file = fsspec.open(path, "rb").open()
    key, ds = _get_key_and_ds(start_time, end_time, variable, category, file)
    return key, ds


def create_record_local(
    item: Tuple[pd.Timestamp, pd.Timestamp, str, str, str],
) -> Generator[Tuple[xbeam.Key, xr.Dataset], None, None]:
    start_time, end_time, variable, category, path = item
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = _get_fs(path)
        file = os.path.join(tmpdir, os.path.basename(path))
        fs.get(path, file)
        key, ds = _get_key_and_ds(start_time, end_time, variable, category, file)
        yield key, ds


def get_full_time_coordinate(
    time_bounds: List[Tuple[pd.Timestamp, pd.Timestamp]],
    category: str,
    freq: Optional[str],
) -> pd.DatetimeIndex:
    initial_time, _ = time_bounds[0]
    _, final_time = time_bounds[-1]
    inclusive = INCLUSIVE[category]
    return pd.date_range(initial_time, final_time, inclusive=inclusive, freq=freq)


def get_template(
    time_bounds: List[Tuple[pd.Timestamp, pd.Timestamp]],
    variables: Sequence[str],
    category: str,
) -> xr.Dataset:
    # To construct a template we grab the first file for each variable in the
    # category.
    template_files = get_file_metadata(time_bounds[:1], variables, category)
    datasets = []
    for template_file in template_files:
        _, ds = create_record(template_file)
        datasets.append(ds)
    merged = xr.merge(datasets)

    freq = FREQUENCY[category]
    time = get_full_time_coordinate(time_bounds, category, freq)

    time_dim = TIME_DIM[category]
    merged = merged.isel({time_dim: 0}).drop_vars([time_dim])
    chunks = TARGET_CHUNKS[category]
    return xbeam.make_template(merged).expand_dims({time_dim: time}).chunk(chunks)


def test_ds(store: zarr.storage.FSStore) -> zarr.storage.FSStore:
    ds = xr.open_dataset(store, engine="zarr", chunks={})
    print(ds)
    return store


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_path", type=str, help="Output path for outputting zarr store"
    )
    parser.add_argument(
        "category",
        help=f"One of {FC_SFC_MEANFLUX!r}, {AN_SFC!r}, {INVARIANT!r}",
    )
    return parser


@dataclass
class OpenNetCDFandSplit(beam.PTransform):
    """Load a single netcdf locally and split into chunks"""

    category: str

    def _load_and_split(self, item):
        # creating record as generator ensures context with downloaded file
        # stays intact it is completely processed in the split loop
        record_generator = create_record_local(item)
        for key, ds in record_generator:
            split_gen = xbeam.split_chunks(
                key, ds, target_chunks=TARGET_SPLIT_CHUNKS[self.category]
            )
            for key, ds in split_gen:
                yield key, ds.load()

    def expand(self, pcoll):
        return pcoll | beam.FlatMap(self._load_and_split)


def main():
    parser = _get_parser()
    args, pipeline_args = parser.parse_known_args()
    category = args.category
    store = os.path.join(args.output_path, f"{category}.zarr")

    time_bounds = get_time_bounds(category, PERIODS[category], INITIAL_TIME[category])
    file_metadata = get_file_metadata(time_bounds, VARIABLES[category], category)
    template = get_template(time_bounds, VARIABLES[category], category)

    logging.basicConfig(level=logging.DEBUG)

    recipe = (
        beam.Create(file_metadata)
        | OpenNetCDFandSplit(category)
        | xbeam.ConsolidateChunks(TARGET_CHUNKS[category])
        | xbeam.ChunksToZarr(store, template, TARGET_CHUNKS[category])
    )

    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        p | recipe

    test_ds(store)


if __name__ == "__main__":
    main()
