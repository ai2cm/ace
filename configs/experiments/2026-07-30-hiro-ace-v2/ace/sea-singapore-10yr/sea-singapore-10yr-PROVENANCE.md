# SE Asia / Singapore 10-year downscaling: provenance and launch notes

Configs:

- `downscale-perfect-pred-sea-singapore-10yr.yaml`: HiRO perfect prediction from X-SHiELD 1deg
- `downscale-ace2s-sea-singapore-10yr-ic000{0..4}.yaml`: HiRO-ACE, one per existing ACE2S IC

Region 96–112E, 6S–10N (single 16x16 deg patch), 2014-01-01 to 2023-12-31T18,
saving `PRATEsfc`, `UGRD10m`, `VGRD10m`, `PRMSL`, `n_ens: 1`.

Output root: `/climate-default/2026-08-03-hiro-ace-v2-outputs/2026-10-18-sea-singapore-10yr/`,
with one subdirectory per job (`perfect-pred/`, `ace2s-ic000N/`). fme writes
`config.yaml` and a log with fixed names into `experiment_dir`, so concurrent
jobs must not share one.

No new ACE2S simulation is needed; the ACE configs read the existing 10-year run.

## Templates

These configs are byte-identical to the launched Bora/Croatia 10-year configs
except for `experiment_dir`, output `name`, and `lat_extent`/`lon_extent`:

| SE Asia config | Template (ai2cm/ace, `configs/experiments/2026-07-30-hiro-ace-v2/ace/`) | Launched at | Beaker |
|---|---|---|---|
| `downscale-perfect-pred-sea-singapore-10yr.yaml` | `downscale-perfect-pred-croatia-10yr.yaml` | `8e8ba332a6e428555cab3f8e6012daa9fd69c3d5` | `01M3WTNF53KB021KY15D8JTH44` (hiro-perfect-pred-inference-bora-2014-2023) |
| `downscale-ace2s-sea-singapore-10yr-ic000N.yaml` | `downscale-ace2s-croatia-10yr.yaml` | `71ca4779e5b7293a5ad068140a48d913f15494e7` | `01M3WRG5QK2EA5AS5J6ZPNWB0V` (hiro-ace2s-inference-bora-2014-2023) |

Both commits are on branch `config/hiro-v2` of `ai2cm/ace` (71ca477 is an
ancestor of 8e8ba33, so either works for both). Both jobs finished with exit 0.

## Example runscript

`configs/experiments/2026-07-30-hiro-ace-v2/ace/downscale.sh` at `8e8ba332a6e4`
launched the perfect-prediction Bora job. Settings that job used:

- gantry 3.7.0, `--workspace ai2/ace`, `--cluster ai2/titan`, `--budget ai2/atec-climate`, `--priority urgent`
- `--beaker-image 01KZPQX9R904MYQVT6N44EDA40` (from `latest_deps_only_image.txt` at that commit)
- `--gpus 4`, `--shared-memory 400GiB`, `--no-python`, `--install "pip install --no-deps ."`
- `--weka climate-default:/climate-default`
- `--dataset 01KRGZT4X2QCW2RFH7WN7X8BYA:checkpoints:/checkpoints_high_sigma`
- `--dataset 01KRBYGNYJ6FD7PGNF3VVHQ5V1:checkpoints:/checkpoints_low_sigma`
- `--dataset-secret google-credentials:/tmp/google_application_credentials.json`
- `-- torchrun --nproc_per_node 4 -m fme.downscaling.inference <config path relative to ace repo root>`

The config path is resolved inside the ace repo, so these YAMLs need to be
copied into the ace checkout (for example next to the Croatia configs) before
launching. The example script uses `wandb-api-key-annak`; use your own wandb
secret. These configs set `log_to_wandb: false`.

## Upstream sources

| Input | Beaker | Notes |
|---|---|---|
| ACE2S 10-year, 5 ICs (`/climate-default/2026-08-03-hiro-ace-v2-outputs/2026-08-03-coarse-ace2s-10yr-inference`) | experiment `01KZ4TJ22TW67JGM9DQ9PDH6GG` | `fme.ace.inference` with `inference-ace2s-10yr.yaml` at `bd2eef07d7270441eb13c057ffdc585a45137820` |
| ACE2S checkpoint | dataset `01KWMYV98Q79G2FNY3CE95N2NG` | written by experiment `01KWJ1ZPN2TS7S6P5S2CHQ5HKT` (ace2s-shieldplus-tune-xshield-10yr-control-correct-weights-seed0-14a1) |
| HiRO low-sigma checkpoint (`/checkpoints_low_sigma`) | dataset `01KRBYGNYJ6FD7PGNF3VVHQ5V1` | written by experiment `01KRBXAA4BNGS8TJ7WV1Y2BTW6` (xshield-downscaling-100km-to-3km-low-sigma-global-oversample-tropics-mult8, ai2/climate-titan) |
| HiRO high-sigma checkpoint (`/checkpoints_high_sigma`) | dataset `01KRGZT4X2QCW2RFH7WN7X8BYA` | written by experiment `01KRGZT4PGD6RF76C6492AXD28` (xshield-downscaling-100km-to-3km-high-sigma-global-oversample-tropics-mult8, ai2/climate-titan) |
| X-SHiELD coarse input for perfect prediction | — | `/climate-default/2026-07-01-X-SHiELD-AMIP-1deg-8layer-11yr.zarr` |
