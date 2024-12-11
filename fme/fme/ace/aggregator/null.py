from fme.core.typing_ import TensorMapping


class NullAggregator:
    """
    An aggregator that does nothing. Null object pattern.
    """

    def __init__(self):
        pass

    def record_batch(
        self,
        loss: float,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
        i_time_start: int = 0,
    ):
        pass

    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        return {}
