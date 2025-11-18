import dataclasses


@dataclasses.dataclass
class DataRequirements:
    fine_names: list[str]
    coarse_names: list[str]
    n_timesteps: int
    static_input_names: list[str] = dataclasses.field(default_factory=list)
