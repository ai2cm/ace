---
name: write-pr-description
description: Write or amend a PR description. Use whenever creating a PR or updating the description of an existing PR, including after pushing new commits to a branch with an open PR.
model: claude-opus-4-6[1m]
context: fork
---

You are writing the description for a PR: the one for the current checked-out branch, or one the user names. PR descriptions in this repository are authored through this skill so that the model pinned in its frontmatter writes them; `context: fork` scopes the pin to exactly this work (a bare `model:` pin is turn-scoped and would leak into the rest of the turn or lapse on the next prompt). The pinned model is a deliberate choice (Jeremy, 2026-08-20) — do not bump it without his say-so.

Begin your report by stating which model you are actually running, so a silent fallback to the session model (e.g. if the pinned model is ever retired) stays visible.

1. Run `git fetch origin main`, then read the full diff from main (`git diff origin/main...HEAD`, or `gh pr diff <number>`), not just the recent commits, so the description reflects the change from main per the guidance in AGENTS.md.
2. Write the title and description following AGENTS.md's "PR description template" section and `.github/pull_request_template.md`. Keep the template's checklist lines rather than deleting them: tick only items actually done, and leave inapplicable items unchecked with a brief in-place reason.
3. Apply with `gh pr create` or `gh pr edit <number>`.
