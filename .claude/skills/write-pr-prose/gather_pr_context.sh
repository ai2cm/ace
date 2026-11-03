#!/usr/bin/env bash
# Print the context needed to write a PR's title and description: the
# existing title/description, the PR discussion (issue comments, reviews,
# and inline review comments), and the full diff from the PR's base branch.
# A failure in a discussion section prints a "(warning: ...)" line and the
# rest still runs; a missing PR or a failure to produce the diff is fatal
# (exit 1), since PR prose must not be written without them.
# Usage: gather_pr_context.sh [<pr-number>]
#   <pr-number>  defaults to the current branch's PR
set -uo pipefail

pr_number="${1:-}"

if [ -z "$pr_number" ]; then
    pr_number=$(gh pr view --json number --jq .number 2>/dev/null || true)
fi

if [ -z "$pr_number" ]; then
    echo "ERROR: no existing PR found for this branch; this script only gathers context for existing PRs." >&2
    exit 1
fi

# Everything comes from the GitHub API; no local fetch needed.
repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)
base=$(gh pr view "$pr_number" --json baseRefName --jq .baseRefName 2>/dev/null || true)
[ -z "$base" ] && echo "(warning: could not determine PR base branch)"
echo "=== Existing title and description ==="
gh pr view "$pr_number" || echo "(warning: could not retrieve title/description)"
echo
echo "=== Issue comments ==="
gh api --paginate "repos/${repo}/issues/${pr_number}/comments" \
    --jq '.[] | "--- \(.user.login) (\(.created_at)):\n\(.body)\n"' \
    || echo "(warning: could not retrieve issue comments)"
echo "=== Reviews ==="
gh api --paginate "repos/${repo}/pulls/${pr_number}/reviews" \
    --jq '.[] | select(.body != "") | "--- \(.user.login) (\(.state), \(.submitted_at)):\n\(.body)\n"' \
    || echo "(warning: could not retrieve reviews)"
echo "=== Inline review comments ==="
gh api --paginate "repos/${repo}/pulls/${pr_number}/comments" \
    --jq '.[] | "--- \(.user.login) on \(.path) line \(.line // .original_line):\n\(.body)\n"' \
    || echo "(warning: could not retrieve inline review comments)"
echo "=== Diff from PR base (${base:-unknown}) ==="
if ! gh pr diff "$pr_number"; then
    echo "ERROR: could not retrieve the PR diff; do not write PR prose without it." >&2
    exit 1
fi
