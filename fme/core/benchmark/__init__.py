# Trigger benchmark registration when the benchmark package is loaded
from fme.core.models import conditional_sfno  # noqa: F401
from fme.downscaling.modules.physicsnemo_unets_v2 import benchmark  # noqa: F401
from fme.downscaling.modules.physicsnemo_unets_v3 import (
    benchmark as benchmark_v3,  # noqa: F401, E501
)
