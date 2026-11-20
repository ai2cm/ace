"""Write the evaluator configs for the global-single-patch-inf experiment.

Question: for near-global snapshots (lon 0-360, -65 < lat < 65), is it better to
generate the whole domain in as few pieces as fit (here 3 x 3), as independent
16 deg x 16 deg tiles (the model's training patch), or with MultiDiffusion?

One config per (model, method). Each config holds four events, one per snapshot
time, each covering the whole domain; event output files (<event>.nc) hold the
generated sample and the X-SHiELD 3 km target, so statistics can be computed over
grid points afterwards. The default (non-event) evaluation is reduced to a single
16 x 16 coarse patch at one time, since only the events are used.

Run from this directory:  python make_configs.py
"""

import pathlib
import random
from datetime import datetime, timedelta

import yaml

HERE = pathlib.Path(__file__).resolve().parent

# experiment_dir is replaced by $AGENT_OUTPUT_DIR in run_in_job.sh.
EXPERIMENT_DIR_PLACEHOLDER = "/climate-default/agents/annak/REPLACED_AT_RUNTIME"

SEED = 20261118
N_TIMES = 4

MODELS = {
    # HiROv2 retrained with align_corners=False (ace acbf45f; saved under
    # model.interpolation, read by the loader fix on this branch).
    "no_tmp2m_acf": {
        "high_sigma_dataset": "01M5JP35DFARTGRGBPAPXS6RK1",
        "low_sigma_dataset": "01M5JP3AHKQ5BE48EKH2SN1ZE2",
    },
    # HiROv2 + TMP2m, align_corners=False, "matched" retrain (ace e2d79d6).
    "tmp2m_acf_matched": {
        "high_sigma_dataset": "01M71BAY8SFQZW5VH6832K1C9P",
        "low_sigma_dataset": "01M72ASTXQX1CFVE5ZQTR5MYH8",
    },
}

# This branch (from feature/allow-larger-patch-gen) runs only the 3 x 3 split;
# single pass, 16 deg tiles and MultiDiffusion are on
# agent/annak/global-single-patch-inf. 44 x 121 coarse patches with a 1-cell
# overlap (averaged) tile lat -65..65 x lon 0..360 (130 x 360) exactly 3 x 3,
# and keep the largest U-Net tensor (384 channels at full resolution) at 97.5%
# of PyTorch's 2**31 - 1 element limit.
METHODS = {
    "patches_3x3": {
        "divide_generation": True,
        "composite_prediction": True,
        "coarse_horizontal_overlap": 1,
        "coarse_patch_extent_lat": 44,
        "coarse_patch_extent_lon": 121,
    },
}

LAT_EXTENT = {"start": -65.0, "stop": 65.0}
LON_EXTENT = {"start": 0.0, "stop": 360.0}

XSHIELD = "/climate-default/2025-09-25-downscaling-data-X-SHiELD-AMIP-downscaling"
PRESSURE = "/climate-default/2026-02-03-downscaling-data-X-SHiELD-AMIP-pressure-data"
COARSE = [(XSHIELD, "100km.zarr"), (PRESSURE, "prmsl_100km.zarr")]
FINE = [
    (XSHIELD, "3km.zarr"),
    (PRESSURE, "instantaneous_surface_and_sea_level_pressure_3km.zarr"),
]


def snapshot_times() -> list[str]:
    """N_TIMES random 6-hourly times in 2023, fixed by SEED."""
    rng = random.Random(SEED)
    t0 = datetime(2023, 1, 1)
    idx = sorted(rng.sample(range(365 * 4), N_TIMES))
    return [(t0 + timedelta(hours=6 * i)).strftime("%Y-%m-%dT%H:%M") for i in idx]


def _merge(sources):
    # The default evaluation only needs one time step.
    return [
        {
            "merge": [
                {
                    "data_path": path,
                    "file_pattern": pattern,
                    "engine": "zarr",
                    "subset": {
                        "start_time": "2023-01-01T00:00:00",
                        "stop_time": "2023-01-01T00:00:00",
                    },
                }
                for path, pattern in sources
            ]
        }
    ]


def _expert(sigma: str, sigma_min: float, sigma_max: float) -> dict:
    return {
        "checkpoint_config": {
            "checkpoint_path": f"/checkpoints_{sigma}_sigma/best.ckpt"
        },
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
    }


def make_config(model: str, method: str) -> dict:
    return {
        "experiment_dir": EXPERIMENT_DIR_PLACEHOLDER,
        "n_samples": 1,
        "patch": METHODS[method],
        "model": {
            "denoising_expert_configs": [
                _expert("low", 0.005, 200.0),
                _expert("high", 200.0, 2000.0),
            ],
            "num_diffusion_generation_steps": 18,
            "churn": 0.0,
        },
        "data": {
            "coarse": _merge(COARSE),
            "fine": _merge(FINE),
            "lat_extent": {"start": 0, "stop": 16},
            "lon_extent": {"start": 0, "stop": 16},
            "batch_size": 1,
            "num_data_workers": 2,
            "strict_ensemble": False,
        },
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": False,
            "log_to_file": True,
            "project": "ace-downscaling",
            "entity": "ai2cm",
        },
        "events": [
            {
                "name": f"{method}_{t.replace('-', '').replace(':', '')}",
                "date": t,
                "n_samples": 1,
                "save_generated_samples": True,
                "lat_extent": LAT_EXTENT,
                "lon_extent": LON_EXTENT,
            }
            for t in snapshot_times()
        ],
    }


def main() -> None:
    for model in MODELS:
        for method in METHODS:
            path = HERE / f"eval-{model}-{method}.yaml".replace("_", "-")
            header = (
                f"# Written by make_configs.py: {model}, {method}.\n"
                "# experiment_dir is replaced by $AGENT_OUTPUT_DIR in run_in_job.sh.\n"
            )
            path.write_text(
                header + yaml.safe_dump(make_config(model, method), sort_keys=False)
            )
            print(path.name)


if __name__ == "__main__":
    main()
