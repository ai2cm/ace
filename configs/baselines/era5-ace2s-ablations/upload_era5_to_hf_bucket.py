#!/usr/bin/env python
"""Upload ERA5 zarr stores from weka to the HuggingFace bucket.

Run as a beaker job with weka mounted at /climate-default/ and the
hugging-face-token secret set as HF_TOKEN.

Usage:
    python upload_era5_to_hf_bucket.py STORE [STORE ...]
    python upload_era5_to_hf_bucket.py --list
    python upload_era5_to_hf_bucket.py --dry-run STORE

Available stores: 1deg-daily, 1deg-6hourly, 4deg-daily,
  stats-1deg, stats-1deg-6h, stats-4deg,
  clim-1deg-daily, clim-1deg-6hourly, clim-4deg-daily
"""
import argparse
import sys

from huggingface_hub import HfApi

BUCKET_ID = "allenai/ACE2S-ERA5-reference-data"

STORES = {
    "1deg-daily": {
        "source": "/climate-default/2026-08-13-era5-1deg-8layer-daily-1940-2025.zarr",
        "dest": "1deg-8layer-daily/era5-1deg-8layer-daily-1940-2025.zarr",
        "desc": "1° daily ERA5 store (~597 GB)",
    },
    "1deg-6hourly": {
        "source": "/climate-default/2026-03-19-era5-1deg-8layer-1940-2025.zarr",
        "dest": "1deg-8layer-6hourly/era5-1deg-8layer-6hourly-1940-2025.zarr",
        "desc": "1° 6-hourly ERA5 store (~3.2 TB)",
    },
    "4deg-daily": {
        "source": "/climate-default/2026-09-08-era5-4deg-8layer-daily-1940-2025.zarr",
        "dest": "4deg-8layer-daily/era5-4deg-8layer-daily-1940-2025.zarr",
        "desc": "4° daily ERA5 store (~55 GB)",
    },
    "stats-1deg": {
        "source": "/statsdata",
        "dest": "normalization-stats/1deg",
        "desc": "1° normalization stats (centering + scaling, ~25 MB)",
    },
    "stats-4deg": {
        "source": "/climate-default/2026-09-08-era5-4deg-8layer-daily-stats-1990-2019",
        "dest": "normalization-stats/4deg",
        "desc": "4° normalization stats (~5 MB, on weka)",
    },
    "clim-1deg-daily": {
        "source": "/clim-forcing-1deg-daily",
        "dest": "climatological-forcing/1deg-daily",
        "desc": "1° daily climatological forcing (1991-2020)",
    },
    "clim-1deg-6hourly": {
        "source": "/clim-forcing-1deg-6h",
        "dest": "climatological-forcing/1deg-6hourly",
        "desc": "1° 6-hourly climatological forcing (1991-2020)",
    },
    "clim-4deg-daily": {
        "source": "/clim-forcing-4deg-daily",
        "dest": "climatological-forcing/4deg-daily",
        "desc": "4° daily climatological forcing (1991-2020)",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stores", nargs="*", help="Store names to upload")
    parser.add_argument("--list", action="store_true", help="List available stores")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, info in STORES.items():
            print(f"  {name:25s} {info['desc']}")
            print(f"    {'source:':10s} {info['source']}")
            print(f"    {'dest:':10s} {info['dest']}")
        return

    if not args.stores:
        parser.print_help()
        sys.exit(1)

    api = HfApi()
    print(f"Authenticated as: {api.whoami()['name']}")

    for name in args.stores:
        if name not in STORES:
            print(f"Unknown store: {name}. Use --list to see available stores.")
            sys.exit(1)
        info = STORES[name]
        source = info["source"]
        dest = f"hf://buckets/{BUCKET_ID}/{info['dest']}"
        print(f"\n{'=' * 60}")
        print(f"Uploading: {name} — {info['desc']}")
        print(f"  {source} -> {dest}")

        if args.dry_run:
            plan = api.sync_bucket(source=source, dest=dest, dry_run=True)
            print(f"  Dry-run plan: {plan}")
            continue

        api.sync_bucket(source=source, dest=dest)
        print(f"  Done: {name}")

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
