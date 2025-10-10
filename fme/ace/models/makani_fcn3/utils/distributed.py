# parallel helpers
from typing import Any, List, Tuple


class NotDistributed:
    def compute_split_shapes(self, size: int, num_chunks: int) -> List[int]:
        return [size]

    def reduce_from_parallel_region(self, input: Any, group: Any) -> Any | None:
        return input

    def scatter_to_parallel_region(
        self, input: Any, dim: Any, group: Any
    ) -> Any | None:
        return input

    def gather_from_parallel_region(
        self, input: Any, dim: Any, shapes: Any, group: Any
    ) -> Any | None:
        return input

    def copy_to_parallel_region(self, input: Any, group: Any) -> Any | None:
        return input

    def split_tensor_along_dim(
        self, tensor: Any, dim: Any, num_chunks: Any
    ) -> Tuple[Any, ...]:
        return tensor


dist = NotDistributed()
