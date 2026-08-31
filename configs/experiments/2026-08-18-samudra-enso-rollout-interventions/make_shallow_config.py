#!/usr/bin/env python3
"""Generate the shallow arm: Samudra truncated to the upper ocean.

Transform of the corrected pretrain run config: drop every 3-D ocean level
deeper than 1000 m — CM4's levels 11-18 (level 10 bottoms at 900 m, level 11
spans 900-1200 m) — from thetao/so/uo/vo, keeping the ORCA-DL-like vertical
domain where ENSO's recharge dynamics live and discarding the levels whose
tendency-to-field ratios are most pathological (thetao_18: 1/427).

The ocean heat content correction is OFF (Troy's call): a surface-flux budget
cannot close a truncated column — heat actually exported below 900 m would be
forced back into the upper ocean. Precedent for stability: the original
lineage pretrain ran with no OHC corrector and rolled out stably; noohc's
NaNs came from removing the corrector at FT after pretraining with it. The
surface energy flux correction and sea-ice/positivity corrections stay.

Everything else (forcings, loss, optimizer, data) matches the lineage: the
vertical truncation is the arm's single change.
"""

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE / "corrected-pretrain-run-config.yaml"
OUT = HERE / "wave1_configs" / "shallow-pretrain.yaml"

KEEP_LEVELS = range(11)  # levels 0-10: 0-900 m
DROP = [f"{v}_{k}" for v in ("thetao", "so", "uo", "vo") for k in range(11, 19)]


def main() -> None:
    with open(BASE) as f:
        c = yaml.safe_load(f)

    cfg = c["stepper"]["step"]["config"]
    for key in ("in_names", "out_names"):
        before = len(cfg[key])
        cfg[key] = [n for n in cfg[key] if n not in DROP]
        assert before - len(cfg[key]) == 32, (key, before, len(cfg[key]))

    corrector = cfg["corrector"]["config"]
    corrector["force_positive_names"] = [
        n for n in corrector["force_positive_names"] if n not in DROP
    ]
    del corrector["ocean_heat_content_correction"]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(yaml.safe_dump(c, sort_keys=False))
    print(f"wrote {OUT}")
    print(f"in={len(cfg['in_names'])} out={len(cfg['out_names'])}")


if __name__ == "__main__":
    main()
