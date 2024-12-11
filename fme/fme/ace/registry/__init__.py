# import modules so they are registered
from . import prebuilt as _prebuilt
from . import sfno as _sfno
from .registry import ModuleSelector

del _prebuilt, _sfno
