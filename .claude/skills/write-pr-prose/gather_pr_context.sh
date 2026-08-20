#!/usr/bin/env bash
# Print the context needed to write a PR's title and description: the
# existing title/description, the PR discussion (issue comments, reviews,
# and inline review comments), and the full diff from main.
# Usage: gather_pr_context.sh [<pr-number>]  (defaults to the current branch's PR)
set -euo pipefail

git fetch origin main --quiet || echo "(warning: could not fetch origin/main; diff base may be stale)"

if pr_number=$(gh pr view "$@" --json number --jq .number 2>/dev/null); then
    echo "=== Existing title and description ==="
    gh pr view "$@"
    echo
    echo "=== Discussion (issue comments and reviews) ==="
    gh pr view "$@" --comments
    echo
    echo "=== Inline review comments ==="
    repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
    gh api "repos/${repo}/pulls/${pr_number}/comments" \
        --jq '.[] | "--- \(.user.login) on \(.path) line \(.line // .original_line):\n\(.body)\n"'
    echo "=== Diff from origin/main ==="
    gh pr diff "$@"
else
    echo "(no existing PR found; showing the local diff from origin/main)"
    echo "=== Diff from origin/main ==="
    git diff origin/main...HEAD
fi
