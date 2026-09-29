"""Add the total salt content correction to a checkpoint trained without one.

The stepper config lives in the checkpoint, so the correction is switched on at
inference by editing it and saving a new checkpoint (the network never saw the
correction in training). The values match
configs/experiments/2026-09-28-test-sal-total-correct/train-config.yaml.

Usage: python add_total_salt_corrector.py <in.tar> <out.tar>
"""

import sys

import torch

from fme.ace.stepper.single_module import load_stepper
from fme.core.corrector.ocean import OceanSaltContentTotalCorrection

SALT_CORRECTION = {
    "method": "scaled_salinity",
    "ice_volume_salt_slope_psu": 39.617,
    "constant_unaccounted_salting": 5.16e-11,
    "use_float64": True,
}


def main(in_path: str, out_path: str) -> None:
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    corrector = ckpt["stepper"]["config"]["step"]["config"]["corrector"]
    assert corrector["type"] == "ocean_corrector", corrector["type"]
    config = corrector["config"]
    existing = [key for key in config if key.startswith("ocean_salt_content")]
    assert not existing, f"checkpoint already has a salt correction: {existing}"
    config["ocean_salt_content_total_correction"] = SALT_CORRECTION
    torch.save(ckpt, out_path)

    # the edited checkpoint builds the correction with the current code
    corrections = load_stepper(out_path)._step_obj._corrector._corrections
    assert any(isinstance(c, OceanSaltContentTotalCorrection) for c in corrections)
    print(f"wrote {out_path}; corrections: {[type(c).__name__ for c in corrections]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
