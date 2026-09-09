"""Generate training configs for the Swin efficiency experiment.

Starting from the two 4-degree AIMIP foundation-model training configs that
were run in August 2026 (``base_configs/``), this writes variants to
``run_configs/`` that enable the speed-ups added to ``fme`` in September 2026:

* ``optimization.float32_matmul_precision: high`` (TensorFloat-32 matmul),
* ``stepper.step.config.compile: true`` (``torch.compile`` of the network),
* ``skip_projection: true`` on the Swin builder (decoder at ``embed_dim``),
* fewer in-training inference runs (zero-weight inference sets removed, the
  remaining ones run every 25 epochs instead of every 10).

The SFNO variant only receives the TensorFloat-32 and inference changes so
that it can serve as a like-for-like baseline.

Run ``python generate_configs.py`` from this directory to regenerate.
"""

import copy
import pathlib
from collections.abc import Callable
from typing import Any

import yaml

HERE = pathlib.Path(__file__).parent
BASE_DIR = HERE / "base_configs"
RUN_DIR = HERE / "run_configs"

SWIN_BASE = "ace-train-config-4deg-AIMIP-nc-swin-v2-fm-a1.yaml"
SFNO_BASE = "ace-train-config-4deg-AIMIP-nc-sfno-fm-a1.yaml"

INFERENCE_EPOCH_STEP = 25

Modification = Callable[[dict[str, Any]], None]


def _trim_inference(config: dict[str, Any]) -> None:
    """Keep only inference sets that contribute to the validation score and
    run them less often.
    """
    kept = [entry for entry in config["inference"] if entry.get("weight", 0.0) > 0.0]
    for entry in kept:
        entry["epochs"]["step"] = INFERENCE_EPOCH_STEP
    config["inference"] = kept


def _enable_tf32(config: dict[str, Any]) -> None:
    config["optimization"]["float32_matmul_precision"] = "high"


def _step_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["stepper"]["step"]["config"]


def _enable_compile(config: dict[str, Any]) -> None:
    _step_config(config)["compile"] = True


def _enable_skip_projection(config: dict[str, Any]) -> None:
    _step_config(config)["builder"]["config"]["skip_projection"] = True


def _set_builder_options(**options: Any) -> Modification:
    def apply(config: dict[str, Any]) -> None:
        _step_config(config)["builder"]["config"].update(options)

    return apply


SWIN_FAST: list[Modification] = [
    _enable_tf32,
    _trim_inference,
    _enable_compile,
    _enable_skip_projection,
]
SFNO_FAST: list[Modification] = [_enable_tf32, _trim_inference]

# name -> (base file, list of modifications applied in order)
VARIANTS: dict[str, tuple[str, list[Modification]]] = {
    "ace-train-config-4deg-AIMIP-nc-swin-v2-fm-a1-fast.yaml": (SWIN_BASE, SWIN_FAST),
    "ace-train-config-4deg-AIMIP-nc-sfno-fm-a1-fast.yaml": (SFNO_BASE, SFNO_FAST),
}


def generate(name: str) -> dict[str, Any]:
    base_name, modifications = VARIANTS[name]
    with open(BASE_DIR / base_name) as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    for modify in modifications:
        modify(config)
    return config


def write_all(run_dir: pathlib.Path = RUN_DIR) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        with open(run_dir / name, "w") as f:
            yaml.safe_dump(generate(name), f, sort_keys=False)


if __name__ == "__main__":
    write_all()
