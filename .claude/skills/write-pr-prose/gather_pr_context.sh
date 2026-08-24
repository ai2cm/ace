#!/usr/bin/env bash
# Print the context needed to write a PR's title and description: the
# existing title/description, the PR discussion (issue comments, reviews,
# and inline review comments), and the full diff from the PR's base branch.
# A failure in one section prints a "(warning: ...)" line and the rest still
# runs; treat any such line as a gap in context coverage.
# Usage: gather_pr_context.sh [<pr-number>] [<base-branch>]
#   <pr-number>   defaults to the current branch's PR
#   <base-branch> diff base when no PR exists yet; defaults to the repo's
#                 default branch (an existing PR always uses its own base)
set -uo pipefail

pr_number="${1:-}"
base_override="${2:-}"

if [ -z "$pr_number" ]; then
    pr_number=$(gh pr view --json number --jq .number 2>/dev/null || true)
fi

if [ -n "$pr_number" ]; then
    base=$(gh pr view "$pr_number" --json baseRefName --jq .baseRefName 2>/dev/null || true)
    git fetch origin "${base:-main}" --quiet 2>/dev/null \
        || echo "(warning: could not fetch origin/${base:-main}; diff base may be stale)"
    echo "=== Existing title and description ==="
    gh pr view "$pr_number" || echo "(warning: could not retrieve title/description)"
    echo
    echo "=== Discussion (issue comments and reviews) ==="
    gh pr view "$pr_number" --comments || echo "(warning: could not retrieve discussion)"
    echo
    echo "=== Inline review comments ==="
    repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)
    gh api --paginate "repos/${repo}/pulls/${pr_number}/comments" \
        --jq '.[] | "--- \(.user.login) on \(.path) line \(.line // .original_line):\n\(.body)\n"' \
        || echo "(warning: could not retrieve inline review comments)"
    echo "=== Diff from PR base (${base:-unknown}) ==="
    gh pr diff "$pr_number" || echo "(warning: could not retrieve diff)"
else
    base="$base_override"
    if [ -z "$base" ]; then
        base=$(gh repo view --json defaultBranchRef --jq .defaultBranchRef.name 2>/dev/null || true)
    fi
    base="${base:-main}"
    git fetch origin "$base" --quiet 2>/dev/null \
        || echo "(warning: could not fetch origin/$base; diff base may be stale)"
    echo "(no existing PR found; showing the local diff from origin/$base)"
    echo "=== Diff from origin/$base ==="
    git diff "origin/$base...HEAD" || echo "(warning: could not compute diff)"
fi
