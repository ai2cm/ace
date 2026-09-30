"""Add the total salt content correction to a checkpoint trained without one.

The stepper config lives in the checkpoint, so the correction is switched on at
inference by editing it and saving a new checkpoint (the network never saw the
correction in training). The values match
configs/experiments/2026-09-28-test-sal-total-correct/train-config.yaml.

Usage: python add_total_salt_corrector.py <in.tar> <out.tar> [--float32]
"""

import argparse

import torch

from fme.ace.stepper.single_module import load_stepper
from fme.core.corrector.ocean import OceanSaltContentCorrection


def main(in_path: str, out_path: str, use_float64: bool) -> None:
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    corrector = ckpt["stepper"]["config"]["step"]["config"]["corrector"]
    assert corrector["type"] == "ocean_corrector", corrector["type"]
    config = corrector["config"]
    existing = [key for key in config if key.startswith("ocean_salt_content")]
    assert not existing, f"checkpoint already has a salt correction: {existing}"
    config["ocean_salt_content_correction"] = {
        "method": "scaled_salinity",
        "ice_volume_salt_slope_psu": 39.617,
        "constant_unaccounted_salting": 5.16e-11,
        "use_float64": use_float64,
    }
    torch.save(ckpt, out_path)

    # the edited checkpoint builds the correction with the current code
    step = load_stepper(out_path)._step_obj
    corrections = getattr(step, "_corrector")._corrections  # noqa: B009
    salt = [c for c in corrections if isinstance(c, OceanSaltContentCorrection)]
    assert len(salt) == 1 and salt[0].use_float64 == use_float64
    print(f"wrote {out_path}; corrections: {[type(c).__name__ for c in corrections]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    parser.add_argument(
        "--float32",
        action="store_true",
        help="compute the correction in float32 instead of float64",
    )
    args = parser.parse_args()
    main(args.in_path, args.out_path, use_float64=not args.float32)
