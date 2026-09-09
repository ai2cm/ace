"""Parse hybridufsft.yaml field by field to surface the swallowed dacite error."""

import sys
import traceback

import dacite
import yaml

from fme.ace.train.train_config import InlineInferenceConfig, TrainConfig

c = yaml.safe_load(open(sys.argv[1]))
print("python:", sys.version)
try:
    dacite.from_dict(TrainConfig, c, config=dacite.Config(strict=True))
    print("FULL PARSE OK")
except Exception as e:
    print("full parse failed:", type(e).__name__, str(e)[:200])
try:
    dacite.from_dict(
        InlineInferenceConfig, c["inference"], config=dacite.Config(strict=True)
    )
    print("inference-only parse OK")
except Exception:
    traceback.print_exc()
