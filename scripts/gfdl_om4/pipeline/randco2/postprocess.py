"""randco2's named post-regrid transforms, selected in the YAML config.

Both are the shared ones in pipeline/postprocess.py; this module registers
their factories in POSTPROCESS (the contract is
pipeline.postprocess.Postprocess).
"""

from ..postprocess import (
    PostprocessFactory,
    kelvin_sst_postprocess,
    sea_ice_fraction_consistency_postprocess,
)

POSTPROCESS: dict[str, PostprocessFactory] = {
    "kelvin_sst": kelvin_sst_postprocess,
    "sea_ice_fraction_consistency": sea_ice_fraction_consistency_postprocess,
}
