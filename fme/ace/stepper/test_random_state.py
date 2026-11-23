"""Integration tests for seedable random-state propagation through the Stepper.

These exercise the full ``Stepper.predict`` path with a minimal
``NoiseConditionedSFNO`` (non-zero noise dimension), verifying that:

* a seeded rollout is independent of how it is chunked into
  ``forward_steps_in_memory`` windows (i.e. one ``predict`` of N steps equals N
  threaded ``predict`` calls), and
* two rollouts seeded identically give the same answer, while different seeds
  (and the unseeded default) do not.

These use a corrector that does not seed state, so they do not exercise the
``StepperState`` rebuild that ``Stepper.step`` must survive to keep propagating
the random_state; that path is covered by
``test_predict_threads_random_state_alongside_corrector_state`` in
``test_single_module.py``.
"""

import dataclasses

import numpy as np
import torch
import xarray as xr

import fme
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.ace.stepper.derived_forcings import DerivedForcingsConfig
from fme.ace.stepper.single_module import Stepper, StepperConfig
from fme.core.random_state import RandomState
from fme.core.registry.module import ModuleSelector
from fme.core.step import SingleModuleStepConfig, StepSelector
from fme.core.testing import get_dataset_info, trivial_network_and_loss_normalization

DEVICE = fme.get_device()


def _get_noise_conditioned_stepper(
    noise_embed_dim: int = 4,
    img_shape: tuple[int, int] = (8, 16),
) -> Stepper:
    """Build a minimal prognostic ``NoiseConditionedSFNO`` stepper on var "a"."""
    in_names = ["a"]
    out_names = ["a"]
    builder = ModuleSelector(
        type="NoiseConditionedSFNO",
        config=dataclasses.asdict(
            NoiseConditionedSFNOBuilder(
                embed_dim=4,
                noise_embed_dim=noise_embed_dim,
                noise_type="gaussian",
                num_layers=2,
                pos_embed=False,
                filter_type="linear",
                filter_num_groups=1,
            )
        ),
    )
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=builder,
                    in_names=in_names,
                    out_names=out_names,
                    normalization=trivial_network_and_loss_normalization(
                        set(in_names + out_names)
                    ),
                )
            ),
        ),
        derived_forcings=DerivedForcingsConfig(),
    )
    stepper = config.get_stepper(get_dataset_info(img_shape=img_shape))
    _activate_noise_conditioning(stepper)
    stepper.set_eval()
    return stepper


def _activate_noise_conditioning(stepper: Stepper) -> None:
    """Make the noise actually affect the output.

    ``ConditionalLayerNorm`` zero-initializes its noise-conditioning weights
    (``W_scale_2d``/``W_bias_2d``), so an untrained model ignores the noise
    entirely. Filling those weights with a fixed non-zero value mimics a trained
    model in which the noise modulates the output, so that seeding the noise has
    an observable effect.
    """
    with torch.no_grad():
        for name, param in stepper._step_obj.modules.named_parameters():
            if "W_scale_2d" in name or "W_bias_2d" in name:
                param.fill_(0.1)


def _get_ic_and_forcing(
    n_steps: int, img_shape: tuple[int, int], n_samples: int = 2
) -> tuple[PrognosticState, BatchData]:
    """Initial condition (var "a") and empty (time-only) forcing for n_steps."""
    index = xr.date_range("2000", freq="6h", periods=n_steps + 1, use_cftime=True)
    forcing_time = xr.DataArray(np.stack(n_samples * [index]), dims=["sample", "time"])
    input_time = forcing_time.isel(time=[0])
    ic = BatchData.new_on_device(
        data={"a": torch.rand(n_samples, 1, *img_shape).to(DEVICE)},
        time=input_time,
        labels=None,
    ).get_start(prognostic_names=["a"], n_ic_timesteps=1)
    forcing = BatchData.new_on_device(data={}, time=forcing_time, labels=None)
    return ic, forcing


def _seed_initial_condition(
    ic: PrognosticState, random_state: RandomState | None
) -> PrognosticState:
    """Attach a RandomState to the initial condition's stepper_state (or leave it
    unseeded when None, falling back to the global RNG)."""
    if random_state is None:
        return ic
    return ic.with_random_state(random_state)


def _forcing_window(forcing: BatchData, start: int, length: int) -> BatchData:
    """Slice a contiguous forcing window of ``length`` forward steps (length + 1
    timestamps), sharing the boundary timestamp with the neighbouring windows."""
    time = forcing.time.isel(time=slice(start, start + length + 1))
    return BatchData.new_on_device(data={}, time=time, labels=None)


def _rollout(
    stepper: Stepper,
    ic: PrognosticState,
    forcing: BatchData,
    window_sizes: list[int],
) -> torch.Tensor:
    """Run a rollout split into windows of the given sizes, threading the
    prognostic state (and its stepper_state) between ``predict`` calls. Returns
    the concatenated prediction for var "a" with shape [sample, time, lat, lon].
    """
    outputs: list[torch.Tensor] = []
    state = ic
    start = 0
    for length in window_sizes:
        window = _forcing_window(forcing, start, length)
        output, state = stepper.predict(state, window)
        outputs.append(output.data["a"])
        start += length
    return torch.cat(outputs, dim=1)


def test_seeded_rollout_independent_of_forward_steps_in_memory():
    """A seeded rollout gives identical results whether run in one window or
    split into smaller windows (the forward_steps_in_memory invariant)."""
    img_shape = (8, 16)
    n_steps = 6
    stepper = _get_noise_conditioned_stepper(img_shape=img_shape)
    ic, forcing = _get_ic_and_forcing(n_steps, img_shape)

    single = _rollout(
        stepper, _seed_initial_condition(ic, RandomState.from_seed(0)), forcing, [6]
    )
    chunked = _rollout(
        stepper,
        _seed_initial_condition(ic, RandomState.from_seed(0)),
        forcing,
        [1, 2, 3],
    )
    assert single.shape[1] == n_steps
    torch.testing.assert_close(single, chunked, rtol=0, atol=0)


def test_two_seeded_rollouts_match_and_differ_by_seed():
    """Same seed -> identical rollout; different seed -> different; and the
    unseeded default is not reproducible across runs."""
    img_shape = (8, 16)
    n_steps = 4
    stepper = _get_noise_conditioned_stepper(img_shape=img_shape)
    ic, forcing = _get_ic_and_forcing(n_steps, img_shape)

    run_a = _rollout(
        stepper, _seed_initial_condition(ic, RandomState.from_seed(42)), forcing, [4]
    )
    run_b = _rollout(
        stepper, _seed_initial_condition(ic, RandomState.from_seed(42)), forcing, [4]
    )
    run_c = _rollout(
        stepper, _seed_initial_condition(ic, RandomState.from_seed(123)), forcing, [4]
    )
    torch.testing.assert_close(run_a, run_b, rtol=0, atol=0)
    assert not torch.allclose(run_a, run_c)

    # Without a random_state the rollout draws from the global RNG and is not
    # reproducible run-to-run, confirming the noise is genuinely active.
    run_unseeded_1 = _rollout(stepper, _seed_initial_condition(ic, None), forcing, [4])
    run_unseeded_2 = _rollout(stepper, _seed_initial_condition(ic, None), forcing, [4])
    assert not torch.allclose(run_unseeded_1, run_unseeded_2)


def test_zero_noise_dimension_is_deterministic_without_seed():
    """With no noise channels the model is deterministic, so the noise (not some
    other source) is what the seed is controlling above."""
    img_shape = (8, 16)
    n_steps = 3
    stepper = _get_noise_conditioned_stepper(noise_embed_dim=0, img_shape=img_shape)
    ic, forcing = _get_ic_and_forcing(n_steps, img_shape)
    run_1 = _rollout(stepper, _seed_initial_condition(ic, None), forcing, [3])
    run_2 = _rollout(stepper, _seed_initial_condition(ic, None), forcing, [3])
    torch.testing.assert_close(run_1, run_2, rtol=0, atol=0)
