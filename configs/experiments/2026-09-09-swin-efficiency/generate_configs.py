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

Three further Swin variants scale the architecture down from the 265M-parameter
base towards the 14M-parameter, ~100 GFLOP/sample SFNO:

* ``compute-matched``: ``embed_dim=192, depth_multiplier=1`` (about 34M
  parameters, about 100 GFLOP/sample, the same FLOPs as SFNO),
* ``param-matched``: ``embed_dim=128, depth_multiplier=1, mlp_ratio=8/3``
  (about 12M parameters, about 37 GFLOP/sample),
* ``mid``: ``embed_dim=256, depth_multiplier=1, mlp_ratio=8/3`` (about 47M
  parameters, about 137 GFLOP/sample).

``mlp_ratio=8/3`` is the usual SwiGLU convention; the base config's SwiGLU with
``mlp_ratio=4`` carries 1.5x the MLP parameters of a GELU MLP at ratio 4.

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

SWIGLU_MLP_RATIO = 8 / 3

SWIN_COMPUTE_MATCHED: list[Modification] = [
    *SWIN_FAST,
    _set_builder_options(embed_dim=192, depth_multiplier=1, num_heads=[3, 6, 6, 3]),
]
SWIN_PARAM_MATCHED: list[Modification] = [
    *SWIN_FAST,
    _set_builder_options(embed_dim=128, depth_multiplier=1, mlp_ratio=SWIGLU_MLP_RATIO),
]
SWIN_MID: list[Modification] = [
    *SWIN_FAST,
    _set_builder_options(embed_dim=256, depth_multiplier=1, mlp_ratio=SWIGLU_MLP_RATIO),
]

# name -> (base file, list of modifications applied in order)
VARIANTS: dict[str, tuple[str, list[Modification]]] = {
    "ace-train-config-4deg-AIMIP-nc-swin-v2-fm-a1-fast.yaml": (SWIN_BASE, SWIN_FAST),
    "ace-train-config-4deg-AIMIP-nc-sfno-fm-a1-fast.yaml": (SFNO_BASE, SFNO_FAST),
    "ace-train-config-4deg-AIMIP-nc-swin-v2-fm-a1-fast-compute-matched.yaml": (
        SWIN_BASE,
        SWIN_COMPUTE_MATCHED,
    ),
    "ace-train-config-4deg-AIMIP-nc-swin-v2-fm-a1-fast-param-matched.yaml": (
        SWIN_BASE,
        SWIN_PARAM_MATCHED,
    ),
    "ace-train-config-4deg-AIMIP-nc-swin-v2-fm-a1-fast-mid.yaml": (
        SWIN_BASE,
        SWIN_MID,
    ),
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
