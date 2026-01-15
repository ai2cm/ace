
# Contributing

Thank you for your interest in contributing to this project!This is a research codebase
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

Install pre-commit to ensure linting and type-hinting requirements are met before
committing code: `pre-commit install`.

You can run local tests with `pytest`. As a shortcut for running only faster unit
tests, use `make test_very_fast` from the root of the repository.
