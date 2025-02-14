# import modules so they are registered
from . import m2lines as _m2lines
from . import prebuilt as _prebuilt
from . import sfno as _sfno
from . import stochastic_sfno as _sfno_crps
from .registry import ModuleSelector

del _prebuilt, _sfno, _m2lines
