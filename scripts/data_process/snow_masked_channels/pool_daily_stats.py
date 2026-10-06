"""Pool the per-store daily stats of a scenario-training source set.

The daily stats of each parent were computed by the data pipeline over that
parent's training window (``stats.start_date``/``end_date`` of the daily
configs) into ``Parent.stats_url``. This pools them with
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
import sys

from masked_snow import PARENTS, SOURCE_SETS

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
    roots = [PARENTS[key].stats_url + "/" for key in SOURCE_SETS[args.source_set]]
    out = output_directory(args.source_set, args.date)
    combine_stats(
        stats_roots=roots,
        output_directory=out,
        history=f"pool_daily_stats.py {args.source_set}: " + " ".join(roots),
    )
    print(f"{out}/combined")


if __name__ == "__main__":
    main()
