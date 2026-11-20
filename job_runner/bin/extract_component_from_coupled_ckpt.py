#!/usr/bin/env python
r"""Lift one component's stepper out of a coupled training checkpoint into a
single-component checkpoint that ``fme.ace.evaluator`` loads.

A coupled checkpoint holds each component's full Stepper state under
``stepper.atmosphere_state`` and ``stepper.ocean_state``; the lifted file holds
that state under ``stepper``, unchanged.

EMA: the per-epoch ``ckpt.tar`` holds the raw weights in ``stepper`` and the
EMA weights in ``ema.ema_params``, named ``"<i>.<parameter>"`` with ``i`` the
index in the coupled module list ``[*atmosphere.modules, *ocean.modules]``.
The lifted file keeps the component's EMA entries, re-indexed to its own
module list, so the evaluator applies them (``use_ema_if_available``, default
true). Files saved with EMA already applied (``ema_ckpt_*.tar``,
``best_*ckpt.tar``) hold no EMA params; their stepper weights pass through.

Config keys are never stripped. The component must load with the running fme,
or the script exits 1, naming the git sha the component was trained at.

    python extract_component_from_coupled_ckpt.py coupled_ckpt.tar out.tar \
        --component {atmosphere,ocean}
"""

import argparse
import os
import sys

import torch

COMPONENTS = ("atmosphere", "ocean")  # order of the coupled module list
EMA_PARAMS_KEY = "ema_params"
EMA_NAME_MAP_KEY = "module_name_to_ema_name"


def component_state(checkpoint, component: str, path: str) -> dict:
    key = f"{component}_state"
    stepper = checkpoint.get("stepper") if isinstance(checkpoint, dict) else None
    if not isinstance(stepper, dict) or key not in stepper:
        found = sorted(stepper) if isinstance(stepper, dict) else type(stepper).__name__
        sys.exit(
            f"error: {path} is not a coupled checkpoint: no stepper.{key} "
            f"(stepper: {found})"
        )
    return stepper[key]


def training_git_sha(state: dict) -> str:
    history = state.get("training_history") or [{}]
    last = history[-1] if isinstance(history[-1], dict) else {}
    return str(last.get("git_sha", "unknown"))


def build_stepper(state: dict, component: str, path: str):
    import fme
    from fme.ace.stepper.single_module import Stepper

    try:
        return Stepper.from_state(state)
    except Exception as e:  # any load failure ends the job with the same message
        sys.exit(
            f"error: the {component} stepper of {path} does not load with the fme at "
            f"{os.path.dirname(fme.__file__)}: {type(e).__name__}: {e}\n"
            f"It was trained at git sha {training_git_sha(state)}. Evaluate it from a "
            "branch whose fme loads it."
        )


def component_ema(ema: dict, offset: int, n_modules: int) -> dict:
    """Coupled EMA entries of modules offset..offset+n_modules-1, re-indexed from 0."""
    name_map = {}
    for name, ema_name in ema[EMA_NAME_MAP_KEY].items():
        index, parameter = name.split(".", 1)
        if offset <= int(index) < offset + n_modules:
            name_map[f"{int(index) - offset}.{parameter}"] = ema_name
    lifted = {
        k: v for k, v in ema.items() if k not in (EMA_PARAMS_KEY, EMA_NAME_MAP_KEY)
    }
    lifted[EMA_NAME_MAP_KEY] = name_map
    lifted[EMA_PARAMS_KEY] = {n: ema[EMA_PARAMS_KEY][n] for n in name_map.values()}
    return lifted


def check_ema_applies(stepper, ema: dict) -> None:
    parameters = dict(stepper.modules.named_parameters())
    for name, ema_name in ema[EMA_NAME_MAP_KEY].items():
        if name not in parameters:
            sys.exit(
                f"error: re-indexed EMA name {name} is not a parameter of the "
                "lifted stepper"
            )
        if parameters[name].shape != ema[EMA_PARAMS_KEY][ema_name].shape:
            sys.exit(f"error: EMA shape of {name} differs from the lifted stepper's")


def check_evaluator_load(path: str, ema: dict | None) -> int:
    """Load as fme.ace.evaluator does; return the number of EMA tensors it applied."""
    from fme.ace.stepper.single_module import load_stepper

    stepper = load_stepper(path, use_ema_if_available=True)
    if ema is None:
        return 0
    parameters = dict(stepper.modules.named_parameters())
    for name, ema_name in ema[EMA_NAME_MAP_KEY].items():
        if not torch.equal(parameters[name], ema[EMA_PARAMS_KEY][ema_name]):
            sys.exit(f"error: {path} loads without the EMA weight of {name}")
    return len(ema[EMA_NAME_MAP_KEY])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("src", help="coupled training checkpoint")
    ap.add_argument("dst", help="single-component checkpoint to write")
    ap.add_argument("--component", required=True, choices=COMPONENTS)
    args = ap.parse_args()

    # mmap: tensors stay on disk until read; ckpt.tar is mostly optimizer state
    checkpoint = torch.load(args.src, map_location="cpu", weights_only=False, mmap=True)
    state = component_state(checkpoint, args.component, args.src)
    stepper = build_stepper(state, args.component, args.src)

    out = {
        "stepper": state,
        "epoch": checkpoint.get("epoch"),
        "num_batches_seen": checkpoint.get("num_batches_seen"),
        "current_epoch_num_batches_seen": checkpoint.get(
            "current_epoch_num_batches_seen"
        ),
        "source": f"stepper.{args.component}_state of {args.src}",
    }
    ema = checkpoint.get("ema") or {}
    lifted_ema = None
    if EMA_PARAMS_KEY in ema:
        offset = 0
        for preceding in COMPONENTS[: COMPONENTS.index(args.component)]:
            preceding_state = component_state(checkpoint, preceding, args.src)
            offset += len(build_stepper(preceding_state, preceding, args.src).modules)
        lifted_ema = component_ema(ema, offset, len(stepper.modules))
        check_ema_applies(stepper, lifted_ema)
        out["ema"] = lifted_ema
    del stepper

    torch.save(out, args.dst)
    n_ema = check_evaluator_load(args.dst, lifted_ema)
    weights = (
        f"stepper weights raw, {n_ema} EMA tensors applied by the evaluator"
        if lifted_ema is not None
        else "stepper weights as saved, no EMA params in the source"
    )
    print(
        f"wrote {args.dst}: {args.component} of {args.src}, "
        f"num_batches_seen {out['num_batches_seen']}; {weights}"
    )


if __name__ == "__main__":
    main()
