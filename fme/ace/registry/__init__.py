# import modules so they are registered
from fme.core.models import mlp as _mlp

from . import land_net as _landnet
from . import local_net as _localnet
from . import m2lines as _m2lines
from . import patch_discriminator as _patch_disc
from . import prebuilt as _prebuilt
from . import sfno as _sfno
from . import stochastic_sfno as _sfno_crps
from . import swin_transformer as _swin
from .registry import ModuleSelector as ModuleSelector

del (
    _prebuilt,
    _sfno,
    _m2lines,
    _landnet,
    _localnet,
    _sfno_crps,
    _mlp,
    _swin,
    _patch_disc,
)
