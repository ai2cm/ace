class MissingDatasetInfo(ValueError):
    def __init__(self, info: str):
        super().__init__(
            f"Dataset used for initialization is missing required information: {info}"
        )


class IncompatibleDatasetInfo(ValueError):
    pass
