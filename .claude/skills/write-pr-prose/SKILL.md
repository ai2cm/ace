---
name: write-pr-prose
description: Write or amend the description of an existing PR, and its title only when the request explicitly asks for a title or the PR has none. Use whenever updating PR prose, including after pushing new commits to a branch with an open PR.
model: claude-opus-4-6[1m]
context: fork
---

You are writing the description, and sometimes the title, for a PR: the one for the current checked-out branch, or one the user names. PR titles and descriptions in this repository are authored through this skill so that the model pinned in its frontmatter writes them; `context: fork` scopes the pin to exactly this work (a bare `model:` pin is turn-scoped and would leak into the rest of the turn or lapse on the next prompt). The pinned model is a deliberate choice (Jeremy, 2026-08-20) — do not bump it without user permission.

1. Run `.claude/skills/write-pr-prose/gather_pr_context.sh [<pr-number>]` (defaults to the current branch's PR). It prints the existing title and description, the PR discussion (issue comments, reviews, and inline review comments), and the full diff from the PR's base branch — so a stacked PR is described against its own base, not main. The script requires an existing PR; if it exits with an error (no PR found, or the diff could not be retrieved), stop and report the failure instead of writing prose. Read all of it: the existing prose and the comments often carry intent the diff alone does not, and the title and description must reflect the whole change from the PR's base per the guidance in AGENTS.md, not just the recent commits.
2. Write the description following AGENTS.md's "PR description template" section and `.github/pull_request_template.md`. Keep the template's checklist lines rather than deleting them: tick only items actually done, and leave inapplicable items unchecked with a brief in-place reason.
3. Leave the existing title alone unless the request explicitly asks for a title or the PR has none. "Explicitly asks" means the request names the title: "retitle", "change the title", "the title should be ...", or supplies title wording. Requests to update, amend, rewrite or shorten the description, or to "update the PR" after new commits, are not requests to change the title, even when the description changes substantially. Reviewers track a PR by its title, so renaming it mid-review is churn, not an improvement.
4. Apply with `gh pr edit <number> --body-file <file>`, adding `--title` only in the cases above.
