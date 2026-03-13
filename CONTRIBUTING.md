
# Contributing

Thank you for your interest in contributing to this project! This is a research codebase
for the [Ai2 Climate Modeling](https://allenai.org/climate-modeling) group. While we welcome
external users, please note that we do not guarantee backwards compatibility or supporting
features that don't align with our research priorities.

## Reporting Issues

We welcome bug reports and feature requests. Please open an issue to:
- Report bugs you encounter
- Suggest new features or improvements

## Submitting Changes

### Bug Fixes

We welcome pull requests that fix bugs. Please include a clear description of the
issue and how your fix addresses it. Ideally, your pull request should include a
unit test that would fail without your bug fix.

### New Features

If you're an external user and want to implement a new feature, **please open an
issue first to discuss it with us**. This helps ensure the feature aligns with
the project's direction and prevents wasted effort.

## Code Guidelines

### Design Principles

#### Isolate responsibilities to as few abstraction levels as possible

When designing a change, think first about what absolutely must change, then
about what level of abstraction that change could be handled in. Choose the
option that splits the concern across the fewest levels of abstraction. When a
decision changes in the future, as little code as possible should need to be
touched. This does not mean minimizing the number of lines changed — you may
need to modify APIs at several levels to properly isolate a feature into one
level of abstraction.

- **Prefer polymorphism over type-checking.** If you see `if isinstance(x, A)
  ... elif isinstance(x, B) ...` chains, the behavior should be a method on
  the types being checked. A single `isinstance` check can be acceptable, but
  multiple branches are a sign that logic belongs on the objects themselves.
- **Keep functions at one level of abstraction.** Don't mix high-level
  orchestration with low-level implementation details in the same function.
  Extract helpers when a function operates at mixed levels.
- **Centralize cross-cutting concerns.** Only `Distributed` should access
  distributed-aware code. Other modules call `dist.method()` rather than
  importing `torch.distributed` or backend-specific code directly. Prefer
  guard methods like `dist.require_no_model_parallelism(msg)` over if-else
  checks scattered throughout the codebase.
- **Training-specific concerns belong in training code.** The distributed
  module wrapper (DDP), weight freezing, and loss configuration are
  training concerns and should not be coupled into inference-capable code.

#### Refactoring pattern: facade first, swap later

When refactoring configs or APIs across multiple PRs, implement the new
internal class alongside the old one. The old class becomes a facade that
constructs the new classes. The final PR deletes the facade and promotes the
new class. This avoids strange intermediate states where partially-implemented
features block on breaking YAML changes.

### Naming

- Names should accurately describe behavior. "scatter" implies inter-process
  communication; use "localize" when each rank computes its own local view.
- Config class names: append `Config` to the name of the thing being built
  (e.g. `TrainStepperConfig` builds `TrainStepper`).
- Prefer descriptive names over abbreviations (`noise_distribution` not
  `distribution`). Names should only include information available in the
  present scope — avoid naming based on the caller's context. For example,
  a function that normalizes any tensor should be `normalize(x: Tensor)`,
  not `normalize_loss(loss: Tensor)`.
- Mark functions as private (prefix `_`) when they are only used internally.

### Configuration

- Validate configs eagerly in `__post_init__`, not at runtime. Catch
  misconfigurations before jobs are submitted.
- Don't add config options that duplicate existing ones.
- Remove unused fields rather than leaving them around.
- Use deprecation warnings rather than hard errors when removing config
  options, unless a breaking change has been communicated to the team.

### Testing

- **Test helpers over copy-paste.** Create helper functions to build common
  test fixtures. If the same setup appears 3+ times, extract a helper.
  Prefer explicit helper functions over pytest fixtures, which can become
  unwieldy; use fixtures only when sharing scope across tests is valuable.
- **Demonstrate bugs with failing tests.** When fixing a bug, add a test
  that fails without the fix, then fix it.
- **Test behavior, not implementation.** If a test re-implements the logic
  it's testing, it isn't actually verifying anything. Prefer tests that
  cover important user-story-level behavior over tests that lock down
  subjective API details, since the latter make it harder to evolve
  interfaces. Both have a place, but use API-level tests in moderation.
- **Use xfail for known bugs.** Mark known issues with `pytest.mark.xfail`
  rather than skipping them silently.
- **Exercise meaningful values.** Don't use all-ones for area weights or
  trivial shapes that hide real bugs.
- **Regression tests for checkpoints.** Maintain regression checkpoints from
  specific model releases. Use `strict=True` for state dict loading.

### Code Organization

- Consolidate duplicated code to shared locations (e.g. `fme/core/`).
- Remove unused code, flags, and imports proactively.
- Use `if condition: raise` instead of `assert` in production code (asserts
  can be stripped by `python -O`).
- Use context managers for resource cleanup (timers, distributed contexts).
- Pass composed objects rather than their parts when multiple attributes would
  be used within the function.

### Vendorized Code

When vendorizing external code, commit the unmodified copy first (with
pre-commit checks skipped if needed), then commit your modifications. This
makes review transparent by separating "what we copied" from "what we changed".

## Code Review

Automated CI tests must pass before your pull request will be reviewed. All
pull requests will be reviewed for:
- Correctness and functionality
- Code style, type-hinting and consistency
- Alignment with project goals
- Responsibility isolation and abstraction level consistency
- Appropriate test coverage

Please be responsive to feedback during the review process.

### Review comment conventions

One guide our team finds helpful is [Conventional Comments](https://conventionalcomments.org/).
Reviewers label comments to indicate priority:
- **Issue**: Must be addressed before merge.
- **Suggestion (optional)**: Worth considering but non-blocking.
- **Question**: Seeking clarification; may or may not require changes.
- **nit**: Minor style preference; does not require re-review.

### PR scope

- Keep PRs focused. Split cleanly separable changes into separate PRs.
- Non-blocking suggestions are often deferred to follow-on PRs.
- Config changes typically need multiple rounds of review.

## Developing

You can install a developer environment using the `conda` package manager with
`make create_environment`.

Install pre-commit hooks to ensure linting and type-hinting requirements are met before
committing code: `pre-commit install`.

You can run local tests with `pytest`. As a shortcut for running only faster unit
tests, use `make test_very_fast` from the root of the repository.

## Internal Development

If you are making changes directly to ai2cm/ace, please follow these internal
development guidelines.

When making new branches, use the naming convention:
`<type>/<short-description>`, where `<type>` is one of:
- feature: Any new functionality in the repository including workflows and scripts but not including configurations. Should be the "default" type if it's unclear which to use.
- refactor: No changes to features, just code restructuring or simplification.
- fix: Bug fixes.
- exp: Branch used for running experiments that is likely not intended to merge with mainline code. Functionality changes in these branches should first be PR'd using a feature branch.
- config: Changes to baseline and experimental configurations under config/.
- scripts: Changes isolated to a single script under scripts/ and subject to less rigorous review than changes to core code.
- docs: Documentation changes only.
