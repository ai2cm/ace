import dataclasses


@dataclasses.dataclass
class DataRequirements:
    fine_names: list[str]
    coarse_names: list[str]
    n_timesteps: int
    use_fine_topography: bool = False
