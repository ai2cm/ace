# HiRO demo zarrs → `hf://buckets/allenai/ai2cm-downscaling`

Beaker jobs that copy the HiRO demo zarr stores from weka to the HF bucket used by
the ace-viz paper-deploy catalog (ai2cm/ace-viz#71). Destination folders:

- `hirov2/`: HiRO v2 stores
- `hirov1/`: HiRO v1 stores
- `xshield/`: X-SHiELD reference stores (shared by v1 and v2)

The sync map is in [`stores.sh`](stores.sh). Each store is copied
whole and unchanged with `hf buckets sync <source>.zarr hf://buckets/allenai/ai2cm-downscaling/<folder>/<store>.zarr`,
without `--delete`. `xshield_100km_full.zarr` is intentionally not synced.

`USRHOME` (default `/climate-default/home/andrep`) is the weka path the sync
instructions call `/usrhome`. Override it in the environment if the stores live
elsewhere under `/climate-default`.

## Scripts

All scripts take optional row numbers or store names; with none they act on every row.
Gantry clones the repo at the current commit, so push the branch before launching.

| Script | What it launches |
|---|---|
| `sync.sh [rows]` | One job per store that runs `hf buckets sync`. It refuses to start if `<source>/zarr.json` is missing. |
| `verify.sh [rows]` | One job that checks, for each store: the source `zarr.json` has `consolidated_metadata`, and the HF file count and total bytes match the source. With `PUBLIC=1`, it also checks that the public `zarr.json` URL returns 200 and contains `consolidated_metadata`. |

`sync.sh` options, matching `scripts/upload_to_hugging_face/example.sh`:

- `IGNORE_EXISTING=1` resumes a failed sync.
- `HF_XET_HIGH_PERFORMANCE=0` helps if stores with large (>4 GB) shards time out.

## Procedure

1. Check the allenai org's HF storage quota. The source folders total about
   3.8 TiB, but that includes `xshield_100km_full.zarr` and two regional ensemble
   stores that aren't synced.
2. Layout gate: run `./sync.sh 6`, the smallest store. Then confirm that
   `hf buckets list allenai/ai2cm-downscaling/xshield/xshield_100km_2023.zarr -R -q | head`
   shows `xshield/xshield_100km_2023.zarr/zarr.json`, with no doubled
   `.zarr/.zarr/`.
3. `./sync.sh 1 2 3 4 5 7 8`
4. `./verify.sh`. Once the bucket is public, run `PUBLIC=1 ./verify.sh`.

`hf buckets sync` and `hf sync` are aliases in huggingface_hub 1.32. This
directory uses the `hf buckets sync` spelling from the sync instructions.
If the `hf` CLI baked into `spencerc/hf-cli-gantry` is too old to know `buckets`,
use `hf sync` or rebuild the image from `scripts/upload_to_hugging_face/`.
