"""Rename the total salt content correction in a checkpoint to its #1533 name.

Checkpoints from this experiment were trained on
exp/2026-09-23-test-salt-corrector, where the correction was configured as
``ocean_salt_content_total_correction``. ai2cm/ace#1533 merged the same
correction as ``ocean_salt_content_correction``, so the old checkpoints no
longer load on this branch. The stepper config lives in the checkpoint, so the
key is renamed there and a new checkpoint saved; the correction's values are
unchanged.

Usage: python rename_total_salt_corrector.py <in.tar> <out.tar>
"""

import argparse

import torch

from fme.ace.stepper.single_module import load_stepper
from fme.core.corrector.ocean import OceanSaltContentCorrection

OLD_KEY = "ocean_salt_content_total_correction"
NEW_KEY = "ocean_salt_content_correction"


def main(in_path: str, out_path: str) -> None:
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    corrector = ckpt["stepper"]["config"]["step"]["config"]["corrector"]
    assert corrector["type"] == "ocean_corrector", corrector["type"]
    config = corrector["config"]
    assert OLD_KEY in config, f"checkpoint has no {OLD_KEY}"
    assert NEW_KEY not in config, f"checkpoint already has {NEW_KEY}"
    salt_config = config.pop(OLD_KEY)
    config[NEW_KEY] = salt_config
    torch.save(ckpt, out_path)

    # the edited checkpoint builds the correction with the current code
    step = load_stepper(out_path)._step_obj
    corrections = getattr(step, "_corrector")._corrections  # noqa: B009
    salt = [c for c in corrections if isinstance(c, OceanSaltContentCorrection)]
    assert len(salt) == 1
    assert salt[0].ice_volume_salt_slope_psu == salt_config.get(
        "ice_volume_salt_slope_psu", 0.0
    )
    assert salt[0].unaccounted_salting == salt_config.get(
        "constant_unaccounted_salting", 0.0
    )
    assert salt[0].use_float64 == salt_config.get("use_float64", True)
    print(f"wrote {out_path}; corrections: {[type(c).__name__ for c in corrections]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    args = parser.parse_args()
    main(args.in_path, args.out_path)
