"""Seedable random state carried by the Stepper across predict calls.

A ``RandomState`` wraps a single CPU ``torch.Generator``. The stepper threads
it through ``StepperState`` (alongside the corrector state) and activates it
around each network call via ``fme.core.rand.use_generator`` so that stochastic
modules (e.g. ``NoiseConditionedSFNO``) draw their noise from it. The generator
advances in place as it is consumed, so threading the same object from one step
to the next - and from one ``predict`` call to the next via the returned
``PrognosticState`` - yields a deterministic noise sequence that does not depend
on how the rollout is chunked into ``forward_steps_in_memory`` windows.

Unlike the corrector state, the generator is not per-sample: a single generator
drives the noise for the whole batch (each sample/ensemble member receives a
distinct slice of the draw). The device/ensemble helpers are therefore no-ops
that preserve the same advancing generator.
"""

import dataclasses

import torch


@dataclasses.dataclass
class RandomState:
    """A seedable random source threaded through the Stepper.

    Parameters:
        generator: A CPU ``torch.Generator``. Draws are made on the CPU and
            moved to the compute device by ``fme.core.rand``, so the same seed
            reproduces results across devices.
    """

    generator: torch.Generator

    def __post_init__(self):
        if self.generator.device.type != "cpu":
            raise ValueError(
                "RandomState requires a CPU torch.Generator, got device "
                f"{self.generator.device}."
            )

    @classmethod
    def from_seed(cls, seed: int) -> "RandomState":
        """Create a ``RandomState`` from an integer seed."""
        generator = torch.Generator()
        generator.manual_seed(seed)
        return cls(generator=generator)

    def to_state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize the (advanced) generator state for a restart.

        ``get_state`` returns the generator's full Mersenne-Twister state as a
        CPU ``uint8`` ByteTensor, not the original seed, so restoring it
        continues the exact draw sequence rather than reseeding.
        """
        return {"generator_state": self.generator.get_state()}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "RandomState":
        """Rebuild a CPU generator from a serialized state (see ``to_state_dict``)."""
        generator = torch.Generator()
        generator.set_state(state["generator_state"])
        return cls(generator=generator)

    # The generator lives on the CPU and is consumed in place; device and
    # ensemble transforms must preserve the same advancing object rather than
    # reset or copy it, so they return self unchanged.
    def to_device(self) -> "RandomState":
        return self

    def to_cpu(self) -> "RandomState":
        return self

    def pin_memory(self) -> "RandomState":
        return self

    def broadcast_ensemble(self, n_ensemble: int) -> "RandomState":
        return self

    def sample_dim_size(self) -> int | None:
        """A single generator has no per-sample leading dimension."""
        return None
