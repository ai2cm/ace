"""Generate evaluator suite configs for the VarMasking4 training runs.

Each suite config contains all inline inference entries from the corresponding
training config.  submit_eval_jobs.py submits one job per checkpoint, and that
job runs all entries in the suite under one WandB run.
"""

import argparse
import copy
import pathlib

import yaml
from generate_masking_configs import CONFIG_PREFIX, WANDB_PREFIX, WANDB_SUFFIX

HERE = pathlib.Path(__file__).parent
EVAL_SUITE_CONFIG_PREFIX = "ace-eval-suite-config-4deg-AIMIP-"
DEFAULT_CHECKPOINT_PATH = "/ckpt.tar"

TRAINING_RESULT_DATASETS = {
    "ace2-var-mask-nc-sfno-mask0.11-bernoulli-co2-0.4-v4": (
        "01KV6QX2NW2PVGJ1SYCT992WWZ"
    ),
    "ace2-var-mask-nc-sfno-mask0-uniform-co2-default-v4": (
        "01KV6QWTMDNX0GGD4XY0CSYTX1"
    ),
    "ace2-var-mask-nc-sfno-mask0.11-bernoulli-co2-default-v4": (
        "01KV6RGHGET37DCC83SJPNNS16"
    ),
    "ace2-var-mask-nc-sfno-mask0.11-bernoulli-co2-0.8-v4": (
        "01KV6RG9W0JFGWB6Q1ZE3A7Z5R"
    ),
    "ace2-var-mask-nc-sfno-mask10-uniform-co2-0.8-v4": ("01KV6RGZJPZ3PE2V016ZA398WV"),
    "ace2-var-mask-nc-sfno-mask10-uniform-co2-0.4-v4": ("01KV6RGRKSNJ5PAVACP89MQGZH"),
    "ace2-var-mask-nc-sfno-mask20-uniform-co2-default-v4": (
        "01KV6RHDFZDCHBHD0522KKGFV7"
    ),
    "ace2-var-mask-nc-sfno-mask10-uniform-co2-default-v4": (
        "01KV6RH6BAFGVHA9V5H9VX5A0E"
    ),
    "ace2-var-mask-nc-sfno-mask5-uniform-co2-default-v4": (
        "01KV6RHMHN3ATB1HY2V7YFZD03"
    ),
    "ace2-var-mask-sfno-mask0-uniform-co2-default-v4": ("01KV6RHVGGHX21N9NJK3Y42CJR"),
}


def source_config_to_run_name(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}{WANDB_SUFFIX}"


def eval_suite_config_to_run_name(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix(EVAL_SUITE_CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}{WANDB_SUFFIX}"


def source_config_to_eval_suite_config(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix(CONFIG_PREFIX)
    return f"{EVAL_SUITE_CONFIG_PREFIX}{suffix}.yaml"


def _inference_entries(train_cfg: dict) -> list[dict]:
    entries = train_cfg.get("inference", [])
    if isinstance(entries, list):
        return entries
    return [entries]


def _resolve_inference_entries(
    train_cfg: dict,
    inference_names: list[str] | None,
) -> list[tuple[str, dict]]:
    entries = _inference_entries(train_cfg)
    resolved_entries = [
        (entry.get("name", f"inference_{i}"), entry) for i, entry in enumerate(entries)
    ]
    if inference_names is None:
        return resolved_entries

    entry_by_name = {name: entry for name, entry in resolved_entries}
    missing_names = sorted(set(inference_names) - set(entry_by_name))
    if missing_names:
        raise ValueError(
            f"Inference entries {missing_names!r} not found; "
            f"available entries: {list(entry_by_name)}"
        )
    return [(name, entry_by_name[name]) for name in inference_names]


def _build_eval_config(
    train_cfg: dict,
    inference_cfg: dict,
    inference_name: str,
    checkpoint_path: str,
) -> dict:
    return {
        "experiment_dir": f"/results/{inference_name}",
        "n_forward_steps": inference_cfg["n_forward_steps"],
        "forward_steps_in_memory": inference_cfg["forward_steps_in_memory"],
        "checkpoint_path": checkpoint_path,
        "logging": copy.deepcopy(train_cfg["logging"]),
        "loader": inference_cfg["loader"],
        "aggregator": inference_cfg.get("aggregator", {}),
        "data_writer": {
            "save_prediction_files": False,
            "save_monthly_files": False,
        },
        "n_ensemble_per_ic": inference_cfg.get("n_ensemble_per_ic", 1),
    }


def _build_eval_suite_config(
    train_cfg: dict,
    inference_names: list[str] | None,
    checkpoint_path: str,
) -> dict:
    inference_entries = _resolve_inference_entries(train_cfg, inference_names)
    return {
        "experiment_dir": "/results",
        "logging": copy.deepcopy(train_cfg["logging"]),
        "inferences": [
            {
                "name": inference_name,
                "config": _build_eval_config(
                    train_cfg=train_cfg,
                    inference_cfg=copy.deepcopy(inference_cfg),
                    inference_name=inference_name,
                    checkpoint_path=checkpoint_path,
                ),
            }
            for inference_name, inference_cfg in inference_entries
        ],
    }


def _write_config(
    cfg: dict,
    out_path: pathlib.Path,
    source_run_name: str,
    source_dataset_id: str,
    existing_only: bool,
) -> None:
    if existing_only and not out_path.exists():
        print(f"Skipped {out_path.name}")
        return
    header = (
        f"# source_run: {source_run_name}\n" f"# source_dataset: {source_dataset_id}\n"
    )
    with out_path.open("w") as f:
        f.write(header)
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path.name}")


def generate_eval_config(
    source_path: pathlib.Path,
    inference_names: list[str] | None,
    checkpoint_path: str,
    existing_only: bool,
) -> None:
    source_run_name = source_config_to_run_name(source_path.name)
    source_dataset_id = TRAINING_RESULT_DATASETS.get(source_run_name)
    if source_dataset_id is None:
        raise KeyError(
            f"No training result dataset ID configured for run {source_run_name!r}"
        )

    with source_path.open() as f:
        train_cfg = yaml.safe_load(f)

    cfg = _build_eval_suite_config(
        train_cfg=train_cfg,
        inference_names=inference_names,
        checkpoint_path=checkpoint_path,
    )
    out_path = HERE / source_config_to_eval_suite_config(source_path.name)
    _write_config(cfg, out_path, source_run_name, source_dataset_id, existing_only)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-name",
        nargs="+",
        default=None,
        help="Inline inference entry name(s) to export (default: all entries).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to the mounted checkpoint (default: {DEFAULT_CHECKPOINT_PATH}).",
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only rewrite evaluator configs that already exist.",
    )
    args = parser.parse_args()

    source_configs = sorted(
        p
        for p in HERE.glob("*-mask*.yaml")
        if p.name.startswith(CONFIG_PREFIX) and not p.name.endswith("-finetune.yaml")
    )

    for source_path in source_configs:
        generate_eval_config(
            source_path=source_path,
            inference_names=args.inference_name,
            checkpoint_path=args.checkpoint_path,
            existing_only=args.existing_only,
        )


if __name__ == "__main__":
    main()
