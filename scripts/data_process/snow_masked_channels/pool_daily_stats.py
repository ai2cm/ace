"""Pool the per-store daily stats of a scenario-training source set.

The daily stats of each parent were computed by the data pipeline over that
parent's training window (``stats.start_date``/``end_date`` of the daily
configs) into ``Parent.stats_url``. Each parent's stats files are copied to a
local directory first (reading the 50 MB time-mean maps over the network piece
by piece takes long enough to outlive a gcloud access token), then pooled with
``combine_stats.combine_stats`` (sample-weighted means; residual standard
deviations as pooled variances; full-field standard deviations including the
spread of the per-store means) into one ``combined/`` directory, which is the
stats set of the control arm and the starting point of the treatment arm's
masked-snow fit (``fit_masked_snow_stats.py --pool``).

Usage:
  python pool_daily_stats.py pic-1pct
  python pool_daily_stats.py pic-1pct-randco2

Writes gs://vcm-ml-intermediate/<date>-cm4-<source-set>-daily-stats/combined/
and prints the URL.
"""

import argparse
import datetime
import os
import subprocess
import sys
import tempfile

import fsspec
from masked_snow import PARENTS, SOURCE_SETS, STATS_FILENAMES, gcs_credentials

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from combine_stats import combine_stats  # noqa: E402

BUCKET = "gs://vcm-ml-intermediate"


def output_directory(source_set: str, date: str) -> str:
    return f"{BUCKET}/{date}-cm4-{source_set}-daily-stats"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_set", choices=sorted(SOURCE_SETS))
    parser.add_argument(
        "--date", default=datetime.date.today().isoformat(), help="output name prefix"
    )
    args = parser.parse_args()
    keys = SOURCE_SETS[args.source_set]
    sources = [PARENTS[key].stats_url for key in keys]
    out = output_directory(args.source_set, args.date)
    with tempfile.TemporaryDirectory() as tmp:
        roots = []
        for key, source in zip(keys, sources):
            local = os.path.join(tmp, key)
            os.makedirs(local)
            subprocess.run(
                ["gsutil", "-m", "-q", "cp"]
                + [f"{source}/{name}" for name in STATS_FILENAMES]
                + [local],
                check=True,
            )
            roots.append(local + "/")
        # combine_stats writes the pooled files to GCS through fsspec's defaults;
        # give them the same fresh gcloud token the store builder uses, since
        # application-default credentials may be stale.
        token = {"token": gcs_credentials()}
        fsspec.config.conf["gs"] = token
        fsspec.config.conf["gcs"] = token
        combine_stats(
            stats_roots=roots,
            output_directory=out,
            history=f"pool_daily_stats.py {args.source_set}: " + " ".join(sources),
        )
    print(f"{out}/combined")


if __name__ == "__main__":
    main()
