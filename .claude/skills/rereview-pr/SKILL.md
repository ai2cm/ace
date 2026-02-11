# PR Re-review Assistant for ACE Repository

**Skill Name:** rereview-pr
**Description:** Re-review a GitHub pull request to check if feedback was addressed
**Usage:** `/rereview-pr <PR number>` or "Re-review PR #123" or "Check if comments addressed on PR #456"

---

## Instructions

When asked to re-review a pull request (e.g., "Re-review PR #N" or "Check if my comments were addressed on PR #N"):

### Context Detection

First, determine what context is available:

1. **Previous Review in Conversation:** If the current conversation contains a previous PR review from this session, use those findings for comparison
2. **No Previous Context:** Need to fetch all review information from GitHub

---

## Step 1: Determine Starting Point

Ask the user to clarify the review scope:

**Options:**
- **From specific commit:** "from commit `<SHA>`" - Review changes since that commit
- **From last review:** Review all changes since the most recent review comment timestamp
- **Full re-review:** Review entire PR with fresh eyes

**Prompt if unclear:**
> To focus the re-review, please specify:
> - A commit SHA to review from (e.g., "from commit abc1234")
> - Or I can review all changes since the last review comment
> - Or do a full fresh review

---

## Step 2: Fetch PR Data

Use `gh` CLI to gather information:

```bash
# Get current PR state
gh pr view <PR_NUMBER> --json title,number,author,baseRefName,headRefName,body,state,commits,comments,reviews,reviewDecision,updatedAt

# Get list of commits (optionally filtered by date)
gh api repos/ai2cm/ace/pulls/<PR_NUMBER>/commits

# Get specific commit details and diff
gh api repos/ai2cm/ace/commits/<COMMIT_SHA>

# Get all review comments (threads)
gh api repos/ai2cm/ace/pulls/<PR_NUMBER>/comments

# Get general PR discussion comments
gh pr view <PR_NUMBER> --json comments

# Get current diff
gh pr diff <PR_NUMBER>

# Get checks status
gh pr checks <PR_NUMBER>
```

---

## Step 3: Analyze Review Comments

For each review comment thread, determine:

### Resolution Categories

**✅ Addressed**
- Code was changed to resolve the concern
- Author provided explanation and reviewer accepted
- Thread marked as resolved
- **Format:** `[Addressed] Comment about X → Fixed in commit abc1234`

**⚠️ Partially Addressed**
- Some but not all aspects of the feedback were addressed
- Implementation differs from suggestion but addresses core concern
- **Format:** `[Partial] Comment about Y → Fixed A but B still pending`

**❌ Unaddressed**
- No code change or response from author
- Author acknowledged but hasn't implemented yet
- Discussion ongoing without resolution
- **Format:** `[Unaddressed] Comment about Z → No changes yet`

**🗨️ Discussed/Dismissed**
- Author explained why no change is needed
- Design decision documented
- Out of scope for this PR
- **Format:** `[Dismissed] Comment about W → Explained in thread (reason)`

---

## Step 4: Check New Commits

For commits since the starting point:

```bash
# Get diff for each new commit
for commit in <new_commits>; do
  gh api repos/ai2cm/ace/commits/$commit
done
```

**Analyze for:**
- Changes that address review comments
- New functionality added
- Refactoring or cleanup
- New issues introduced

---

## Step 5: Compare Against Previous Assistant Review (if available)

If context from a previous review exists in this conversation:

1. Compare previous findings against new changes
2. Check if issues raised were addressed
3. Identify items not part of formal PR review comments

**Ask user:**
> I previously identified [N] issues in this PR. Some were not part of the formal GitHub review. Would you like me to check if these were addressed?

---

## Output Format

## Re-review Summary

- **PR:** #[number] - [title]
- **Author:** @[username]
- **Review Scope:** [Commits/changes reviewed]
  - **Starting Point:** `<starting_sha>` or [date of last review]
  - **Current HEAD:** `<current_sha>`
  - **New Commits:** [count] commits since last review
- **Last Updated:** [timestamp]
- **Current Status:** [open/approved/changes_requested]

---

## New Changes Overview

**Commits Since Last Review:**

| Commit | Author | Date | Message |
|--------|--------|------|---------|
| `abc1234` | @user | Jan 1 | Fix issue with... |
| `def5678` | @user | Jan 2 | Address review comments |

**Files Modified in New Commits:**
- `fme/core/file1.py` (+25, -10) - [brief description]
- `fme/ace/file2.py` (+5, -2) - [brief description]
- `test_file.py` (+40, -0) - [brief description]

**Summary of Changes:**
[High-level description of what changed since the starting point]

---

## Review Comments Status

### ✅ Addressed Comments ([count])

| Original Comment | File:Line | Resolution | How Addressed |
|------------------|-----------|------------|---------------|
| "Should use vectorized operation here" | `fme/core/metrics.py:123` | ✅ Fixed | Changed to use torch.vmap in `abc1234` |
| "Missing type hints" | `fme/ace/stepper.py:45` | ✅ Fixed | Added type annotations in `def5678` |

### ⚠️ Partially Addressed Comments ([count])

| Original Comment | File:Line | Status | Notes |
|------------------|-----------|--------|-------|
| "Need tests for edge cases A, B, C" | `test_foo.py` | ⚠️ Partial | Tests added for A and B, C still missing |

### ❌ Unaddressed Comments ([count])

| Original Comment | File:Line | Status | Notes |
|------------------|-----------|--------|-------|
| "Performance concern with nested loop" | `fme/coupled/ocean.py:234` | ❌ Not addressed | No changes to this section |
| "Should extract to helper function" | `fme/downscaling/model.py:567` | ❌ Not addressed | Code unchanged |

### 🗨️ Discussed/Dismissed ([count])

| Original Comment | File:Line | Reason | Discussion Summary |
|------------------|-----------|--------|-------------------|
| "Consider caching this computation" | `fme/core/loss.py:89` | Design decision | Author explained computation varies per batch, caching not beneficial |

---

## New Issues in Recent Changes (if any)

If new commits introduce problems not present before:

### ❌ Critical Issues

1. **[Issue Title]**
   - **Introduced in:** commit `abc1234`
   - **File:** `path/to/file.py:123`
   - **Problem:** [Description]
   - **Fix:** [Recommendation]

### ⚠️ Suggestions

[Use same format as initial review]

### 💡 Minor/Nitpicks

[Use same format as initial review]

---

## Previous Assistant Review Findings (if applicable)

If previous review context exists in this conversation:

> **Note:** In my previous review from this conversation, I identified the following items that were NOT part of the GitHub PR review comments:
>
> **Addressed from previous Assistant review:**
> - ✅ [Item that was fixed]
>
> **Still unaddressed from previous Assistant review:**
> - ❌ [Item still pending]
>
> Would you like me to include these in the outstanding items summary?

---

## Test Coverage Status

**Test Changes:**
- New tests added: [count] files, [count] test cases
- Tests modified: [count] files
- Test coverage: [increased/decreased/unchanged]

**Assessment:**
- ✅ Review comments about testing addressed
- ⚠️ Some test gaps remain: [list]
- ❌ Still missing tests for: [list]

---

## ACE-Specific Compliance Check

Quick validation against ACE conventions:

- [ ] Vendorized code not modified
- [ ] Pre-commit hooks passing (ruff format, mypy)
- [ ] Test speed markers correct (`skip_slow` for slow tests)
- [ ] YAML configs valid
- [ ] Docstrings follow Google style
- [ ] No breaking changes to pinned dependencies

---

## Outstanding Items Summary

### Blocking Issues (Must Address Before Merge)

1. **[Critical unaddressed comment]** - `file.py:123`
2. **[New issue introduced]** - `file2.py:456`
3. **[Missing required tests]** - [specific test scenarios]

### Non-Blocking Items (Should Consider)

1. **[Partially addressed feedback]** - [what's missing]
2. **[Suggestion from review]** - [optional improvement]

### Total Outstanding
- ❌ Critical: [count]
- ⚠️ Suggestions: [count]
- 💬 Discussion needed: [count]

---

## Recommendation

**Status:** [Choose one]

- ✅ **Ready to Merge**
  - All review comments addressed
  - No new issues introduced
  - Tests pass, checks green
  - Meets ACE quality standards

- 🔄 **Needs Minor Changes**
  - Most comments addressed
  - [count] non-blocking items remain
  - Can merge after quick fixes

- ❌ **Needs Revision**
  - [count] blocking issues unresolved
  - Critical comments not addressed
  - New problems introduced

**Summary:** [One-sentence recommendation with key next steps]

---

## Re-review Tips

- Focus on **delta** from previous state, not full re-review (unless requested)
- Be concise - highlight what changed, not entire PR history
- Track comment resolution systematically
- Note if PR has been substantially refactored (might need fresh full review)
- Check commit messages for references to issue numbers or review comments
- Validate that fixes actually solve the original concern (not just surface-level changes)
- For large PRs with many new commits, prioritize review comments over comprehensive code review

---

## Special Cases

**If PR has been force-pushed or rebased:**
> ⚠️ This PR was force-pushed, making commit-by-commit comparison difficult. Consider a fresh full review instead of delta review.

**If too many changes:**
> ⚠️ [count] commits since last review is substantial. Would you prefer:
> - Delta review (may take longer)
> - Fresh full review
> - Focus only on review comment resolution

**If all comments resolved:**
> ✅ All review comments have been addressed! Running final checks for any new issues introduced...
