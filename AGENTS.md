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

When running tests in a conda environment, use `python -m pytest` (not `pytest`) to ensure the correct interpreter is used.

### GitHub MCP Server Setup

To use the PR review rules, configure the GitHub MCP server in Cursor's "Tools & MCP" settings. You will need a read-only personal access token with the following permissions: Pull requests, Issues, Contents, Metadata.

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
