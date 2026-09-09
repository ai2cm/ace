"""Parse hybridufsft.yaml exactly as fme.ace.train.main does, and surface
the inner exception dacite swallows into UnionMatchError.
"""

import sys
import traceback

import dacite

from fme.ace.train.train_config import InlineInferenceConfig, TrainConfig
from fme.core.cli import prepare_config

path = sys.argv[1]
data = prepare_config(path, override=None)
print("python:", sys.version)
print("inference dict repr:")
print(repr(data.get("inference"))[:1500])
try:
    dacite.from_dict(TrainConfig, data, config=dacite.Config(strict=True))
    print("FULL PARSE via prepare_config: OK")
except Exception as e:
    print("full parse failed:", type(e).__name__, str(e)[:200])
try:
    dacite.from_dict(
        InlineInferenceConfig, data["inference"], config=dacite.Config(strict=True)
    )
    print("inference-only parse OK")
except Exception:
    traceback.print_exc()
