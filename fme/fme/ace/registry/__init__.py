# import modules so they are registered
from . import prebuilt as _prebuilt
from . import sfno as _sfno
from .registry import ModuleSelector, get_from_registry, register

del _prebuilt, _sfno
