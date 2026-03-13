# AGENTS.md

This file is the single source of agent guidance for the `ai2cm/ace` repository.

## Repository Context

This is a Python machine learning project for atmospheric modeling (ACE - AI2 Climate Emulator).

### Key Conventions

- Code is in the `fme/` directory (ace, core, coupled, diffusion, downscaling modules)
- Tests follow pytest conventions
- Configuration uses YAML files in `configs/`
- The project uses PyTorch for ML components
- The default conda environment for the repo is named `fme`

### Common Commands

- Run all tests: `make test`
- Run fast tests only: `make test_fast`
- Run very fast tests: `make test_very_fast`
- Run tests with coverage: `make test_cov`
- Create development environment: `make create_environment`
- Build Docker image: `make build_docker_image`
- Run pre-commit hooks: `pre-commit run --all-files`

When running tests in a conda environment, use `python -m pytest` (not `pytest`) to ensure the correct interpreter is used.
Pre-commit hooks run ruff, ruff-format, and mypy. If ruff-format modifies files, re-stage and create a new commit (do not amend).

### GitHub MCP Server Setup

To use the PR review rules, configure the GitHub MCP server in Cursor's "Tools & MCP" settings. You will need a read-only personal access token with the following permissions: Pull requests, Issues, Contents, Metadata.

## Code Guidelines for Agents

When writing or reviewing code, read CONTRIBUTING.md for design principles and
testing guidelines. The sections below cover additional details agents are most
likely to miss.

### Naming

- Config classes: append `Config` to the built type (`TrainStepperConfig`).
- Prefer descriptive names (`noise_distribution` not `distribution`).
  Names should reflect the present scope, not the caller's context.
  For example, a function that normalizes any tensor should be
  `normalize(x: Tensor)`, not `normalize_loss(loss: Tensor)`.
- "scatter" implies communication; "localize" when no communication occurs.
- Private functions get a `_` prefix.

### Config design

- Validate in `__post_init__`, not at runtime.
- No redundant options; remove unused fields.
- Backwards compatibility for checkpoint loading is critical; use deprecation
  warnings for config removal.

### Testing

- Prefer fast-running, parsimonious tests.
- Create helpers for repeated test setup (threshold: 3+ instances).
  Prefer explicit helpers over pytest fixtures; use fixtures only when
  sharing scope across tests is valuable.
- When fixing a bug, add a failing test first.
- Tests must test behavior, not re-implement logic. Prefer tests that cover
  user-story-level behavior over tests that lock down subjective API details.
- Use `xfail` for known bugs, not silent skips.
- Use non-trivial values (not all-ones) so tests exercise real behavior.

### Code organization

- Consolidate duplicated code to shared locations (e.g. `fme/core/`).
- Remove unused code, flags, and imports proactively.
- `if/raise` instead of `assert` in production code.
- Context managers for cleanup (timers, distributed contexts).
- Pass composed objects, not their parts, if multiple attributes would be
  used within the function.
- Commit vendorized code unmodified first, then modifications separately.

## Pull Request Review Assistant

Use this same workflow for both initial review and re-review.

### 1) Gather context with GitHub MCP

- `pull_request_read`:
  - `get` for metadata/state
  - `get_diff` for full diff (or current context)
  - `get_review_comments` for review threads
  - `get_comments` for general PR discussion
- `list_commits` and `get_commit` when commit-by-commit analysis is needed

### 2) Scope the review

- **Initial review**: review the full PR diff and all current discussion.
- **Re-review (delta review)**:
  - If user provides a starting SHA, review changes from that point.
  - If not, ask for starting SHA or default to changes since last review comment timestamp.
  - Focus on what changed, whether prior comments were addressed, and whether new issues were introduced.

### 3) Evaluate findings

Use these severity buckets:

- **Critical Issues (Must Fix)**: security vulnerabilities, logic bugs, breaking changes
- **Suggestions (Should Consider)**: performance, error handling, clarity/design improvements
- **Minor/Nitpicks (Optional)**: style, naming, docs polish

For re-reviews, classify prior comments as **Addressed**, **Partially Addressed**, **Unaddressed**, or **Dismissed**. Treat clear author rationale as addressed when appropriate.

### 4) Output format

Write concise markdown with:

1. **PR Summary**: title/number, author, target branch, goal
2. **Changes Overview**: files changed + high-level summary
3. **Code Review Findings**: grouped by severity with file/line references
4. **Discussion Status**: key unresolved comment threads
5. **Testing Assessment**: gaps and edge cases
6. **Recommendation**: Ready to merge / Needs minor changes / Needs revision

For re-reviews, include:

- Commits reviewed (`<start_sha>...<head_sha>`)
- Status of previously raised comments
- Outstanding items before merge

### 5) Practical guidance

- Be specific, constructive, and explicit about blocking vs non-blocking items.
- Prefer delta-focused summaries for re-reviews.
- If the PR was heavily refactored, recommend a fresh full review.
- For large PRs, batch API calls to avoid rate limits.
- Remember that GitHub MCP access is read-only.
