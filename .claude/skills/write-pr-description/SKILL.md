---
name: write-pr-description
description: Write or amend a PR description. Use whenever creating a PR or updating the description of an existing PR, including after pushing new commits to a branch with an open PR.
model: claude-opus-4-6[1m]
---

You are writing the description for a PR: the one for the current checked-out branch, or one the user names. PR descriptions in this repository are authored through this skill so that the model pinned in its frontmatter writes them.

1. Read the full diff from main (`git diff origin/main...HEAD`, or `gh pr diff <number>`), not just the recent commits, so the description reflects the change from main per the guidance in AGENTS.md.
2. Write the title and description following AGENTS.md's "PR description template" section and `.github/pull_request_template.md`.
3. Apply with `gh pr create` or `gh pr edit <number>`.
