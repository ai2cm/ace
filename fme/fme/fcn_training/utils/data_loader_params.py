import dataclasses


@dataclasses.dataclass
class DataLoaderParams:
    data_path: str
    data_type: str
    batch_size: int
    num_data_workers: int
    dt: float
