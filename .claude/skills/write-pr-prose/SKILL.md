---
name: write-pr-prose
description: Write or amend a PR title or description. Use whenever creating a PR or updating the title or description of an existing PR, including after pushing new commits to a branch with an open PR.
model: claude-opus-4-6[1m]
context: fork
---

You are writing the title and/or description for a PR: the one for the current checked-out branch, or one the user names. PR titles and descriptions in this repository are authored through this skill so that the model pinned in its frontmatter writes them; `context: fork` scopes the pin to exactly this work (a bare `model:` pin is turn-scoped and would leak into the rest of the turn or lapse on the next prompt). The pinned model is a deliberate choice (Jeremy, 2026-08-20) — do not bump it without user permission.

Once you've updated the PR prose, leave a comment on the PR which states you have done so and which model you are actually running, so a silent fallback to the session model (e.g. if the pinned model is ever retired) stays visible.

1. Run `.claude/skills/write-pr-prose/gather_pr_context.sh [<pr-number>]` (defaults to the current branch's PR). It fetches `origin/main` and prints the existing title and description, the PR discussion (issue comments, reviews, and inline review comments), and the full diff from main. Read all of it: the existing prose and the comments often carry intent the diff alone does not, and the title and description must reflect the change from main per the guidance in AGENTS.md, not just the recent commits.
2. Write the title and description following AGENTS.md's "PR description template" section and `.github/pull_request_template.md`. Keep the template's checklist lines rather than deleting them: tick only items actually done, and leave inapplicable items unchecked with a brief in-place reason.
3. Apply with `gh pr create` or `gh pr edit <number>`.
