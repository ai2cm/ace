"""Generate ocean-only rollout probe configs for the checkpoint-ensemble
experiment: blend the residfix residual ocean (best_ckpt of
samudra-enso-w1-residfix, the scored "residfixbest") with the corrected
full-field pretrain (best_inference_ckpt of
cm4-samudra-1pct-ocean-train-using-ufs-var-subset-ohc-hdfs-correctors,
aka "pretrain0") at several weights, per the checkpoint-ensemble evaluator
feature (upstream feature/checkpoint-ensemble-evaluator, ai2cm/ace#1492).

Protocol matches the residfix mode probes (truth-forced ocean-only rollout,
ICs 0251-01-03 and 0262-06-22) but extended from 400 to 1460 steps (20 yr)
so stability is answered outright: residfixbest alone goes unstable at
~step 150-220, the full-field pretrain is stable.

Arms: ff00 (pure residual control), ff10/ff20/ff50 (full-field fraction
0.1/0.2/0.5), ff100 (pure full-field reference).
"""

import argparse
import pathlib

import yaml

RESID_CKPT = "/ckpt_resid.tar"
FF_CKPT = "/ckpt_ff.tar"

# full-field fraction per arm
ARMS = {
    "ff00": 0.0,
    "ff10": 0.1,
    "ff20": 0.2,
    "ff50": 0.5,
    "ff100": 1.0,
}

WRITER_NAMES = [
    "sst",
    "zos",
    "ssu",
    "ssv",
    "thetao_0",
    "thetao_4",
    "thetao_8",
    "thetao_12",
    "thetao_18",
    "so_0",
    "so_18",
    "uo_0",
    "uo_8",
    "vo_8",
    "ocean_sea_ice_fraction",
    "sea_ice_volume",
    "hfds_total_area",
]


def checkpoint_path(ff_fraction: float):
    if ff_fraction == 0.0:
        return RESID_CKPT
    if ff_fraction == 1.0:
        return FF_CKPT
    return [
        {"path": RESID_CKPT, "weight": round(1.0 - ff_fraction, 4)},
        {"path": FF_CKPT, "weight": ff_fraction},
    ]


def make_config(arm: str, ff_fraction: float) -> dict:
    return {
        "experiment_dir": "/results",
        "checkpoint_path": checkpoint_path(ff_fraction),
        "n_forward_steps": 1460,
        "forward_steps_in_memory": 40,
        "loader": {
            "num_data_workers": 2,
            "dataset": {
                "merge": [
                    {
                        "data_path": "/climate-default",
                        "file_pattern": (
                            "2025-10-21-cm4-1pctCO2-140yr-no-smoothing"
                            "-coupled-ocean.zarr"
                        ),
                        "engine": "zarr",
                    },
                    {
                        "data_path": "/climate-default",
                        "file_pattern": (
                            "2025-10-16-cm4-1pctCO2-140yr-ocean-no-smoothing.zarr"
                        ),
                        "engine": "zarr",
                    },
                ]
            },
            "start_indices": {"times": ["0251-01-03T12:00:00", "0262-06-22T12:00:00"]},
        },
        "data_writer": {
            "save_prediction_files": True,
            "names": WRITER_NAMES,
        },
        "aggregator": {"log_histograms": False},
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": True,
            "log_to_file": True,
            "project": "ace-samudra-cm4",
            "entity": "ai2cm",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=sorted(ARMS), choices=sorted(ARMS))
    args = parser.parse_args()
    out_dir = pathlib.Path(__file__).parent
    for arm in args.arms:
        path = out_dir / f"probe-{arm}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(make_config(arm, ARMS[arm]), f, sort_keys=False)
        print(path)


if __name__ == "__main__":
    main()
