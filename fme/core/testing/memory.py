import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class PeakTensorMemory(TorchDispatchMode):
    """Tracks peak live tensor-storage bytes, on any device.

    Reports the same quantity as ``torch.cuda.max_memory_allocated`` (modulo the
    caching allocator's rounding), but works on CPU, so a memory regression in a
    data path can be asserted in CPU tests. Use
    :meth:`track` to account for tensors created before the context was entered.

    Storages are counted once each, so views are free and a
    ``reshape``-that-copies is not.

        with PeakTensorMemory() as mem:
            mem.track(inputs)
            run()
        assert mem.peak < budget
    """

    def __init__(self) -> None:
        super().__init__()
        self.current = 0
        self.peak = 0
        self._refs: dict[int, weakref.ref] = {}
        self._sizes: dict[int, int] = {}

    def track(self, *objs) -> None:
        """Count tensors nested anywhere in ``objs`` toward the live total."""
        flat, _ = tree_flatten(objs)
        for tensor in flat:
            if isinstance(tensor, torch.Tensor):
                self._add(tensor)

    def _add(self, tensor: torch.Tensor) -> None:
        if tensor.device.type == "meta":
            return
        # Relies on PyTorch's PyObject preservation for storages: the same
        # Python object comes back for a given C++ storage, so the weakref
        # callback fires exactly when that storage is freed.
        storage = tensor.untyped_storage()
        key = id(storage)
        if key in self._refs:
            return
        nbytes = storage.nbytes()

        def _on_free(_ref, key=key):
            self.current -= self._sizes.pop(key, 0)
            self._refs.pop(key, None)

        self._refs[key] = weakref.ref(storage, _on_free)
        self._sizes[key] = nbytes
        self.current += nbytes
        self.peak = max(self.peak, self.current)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.track(out)
        return out
