
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

## Code Review

Automated CI tests must pass before your pull request will be reviewed. All
pull requests will be reviewed for:
- Correctness and functionality
- Code style, type-hinting and consistency
- Alignment with project goals

Please be responsive to feedback during the review process.

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
