# PR Review Assistant for ACE Repository

**Skill Name:** review-pr
**Description:** Review a GitHub pull request in the ai2cm/ace repository
**Usage:** `/review-pr <PR number>` or "Review PR #123"

---

## Instructions

When asked to review a pull request (e.g., "Review PR #N"):

### Step 1: Fetch PR Data

Use the `gh` CLI tool to fetch PR information:

```bash
# Get PR details
gh pr view <PR_NUMBER> --json title,number,author,baseRefName,headRefName,body,state,comments,reviews,reviewDecision

# Get PR diff
gh pr diff <PR_NUMBER>

# Get PR checks status
gh pr checks <PR_NUMBER>

# Get review comments
gh api repos/ai2cm/ace/pulls/<PR_NUMBER>/comments
```

### Step 2: Analyze and Output Structured Review

Provide a comprehensive review with these sections:

---

## PR Summary

- **PR #:** [number]
- **Title:** [title]
- **Author:** @[username]
- **Target Branch:** [base] ← [head]
- **Status:** [open/closed/merged]
- **Review Decision:** [approved/changes_requested/review_required]
- **Description:** Brief summary of what the PR aims to accomplish

---

## Changes Overview

- **Files Changed:** [count]
- **Additions:** +[lines]
- **Deletions:** -[lines]

**Modified Files:**
- `path/to/file1.py` - [brief description]
- `path/to/file2.py` - [brief description]

**High-level Summary:**
[Describe the overall changes and their purpose]

---

## Code Review Findings

### ❌ Critical Issues (Must Fix)

**These must be addressed before merging:**

1. **[Issue Title]**
   - **File:** `path/to/file.py:123`
   - **Description:** [Detailed explanation]
   - **Severity:** Security/Bug/Breaking Change
   - **Recommendation:** [How to fix]

### ⚠️ Suggestions (Should Consider)

**Recommended improvements:**

1. **[Suggestion Title]**
   - **File:** `path/to/file.py:456`
   - **Description:** [Detailed explanation]
   - **Why:** [Performance/Maintainability/Error Handling]
   - **Recommendation:** [How to improve]

### 💡 Minor/Nitpicks (Optional)

**Nice-to-haves:**

1. **[Nitpick Title]**
   - **File:** `path/to/file.py:789`
   - **Description:** [Brief explanation]
   - **Suggestion:** [Small improvement]

---

## Test Coverage Assessment

- **Tests Added/Modified:** [Yes/No - list files]
- **Coverage of New Code:** [Good/Partial/Missing]
- **Edge Cases Considered:** [List any edge cases that should be tested]

**Assessment:**
- ✅ Sufficient test coverage
- ⚠️ Could use more tests for: [specific scenarios]
- ❌ Missing tests for: [critical scenarios]

---

## ACE-Specific Checks

### Code Quality
- [ ] Follows Ruff linting rules (no vendorized code modified)
- [ ] Google-style docstrings for public APIs
- [ ] Type hints present
- [ ] No modifications to vendorized code (`makani_fcn3`, `physicsnemo_unets_v2`, `cuhpx`)

### Testing
- [ ] Appropriate test markers (`skip_slow` for slow tests)
- [ ] Tests respect timeout limits (5s very_fast, 60s fast, 180s default)
- [ ] Coverage maintained or improved

### Configuration
- [ ] YAML configs valid and follow project conventions
- [ ] Uses omegaconf patterns for variable interpolation

### Dependencies
- [ ] No breaking changes to `torch-harmonics==0.8.0` constraint
- [ ] New dependencies properly declared in requirements files

---

## Existing Discussion Summary

**Review Comments:** [count]
**Conversation Threads:** [count]

**Key Discussion Points:**
- [Summarize important threads]
- [Note any unresolved concerns]
- [Highlight consensus/disagreements]

**Comment Status:**
- ✅ Resolved: [count]
- 🔄 Active discussion: [count]
- ⏳ Awaiting author response: [count]

---

## Recommended Actions

### Before Merge (Priority Order)

1. **Critical Issues:** [List blocking items with file references]
2. **Tests Required:** [Specific test scenarios needed]
3. **Documentation:** [Docstring/README updates needed]
4. **Review Response:** [Unaddressed review comments]

### Optional Improvements

- [Non-blocking suggestions for future PRs]

---

## Overall Recommendation

**Status:** [Choose one]
- ✅ **Approve** - Ready to merge, no blocking issues
- 🔄 **Request Changes** - Blocking issues must be resolved
- 💬 **Comment** - Questions/suggestions for discussion

**Summary:** [One-sentence recommendation]

---

## Review Notes

- Be constructive and specific in all feedback
- Reference exact file paths and line numbers when possible
- Distinguish clearly between blocking issues vs. suggestions
- Consider ACE-specific conventions (test speed tiers, vendorized code, config patterns)
- Validate against pre-commit hooks rules (ruff, mypy)
- Check if changes align with research papers and climate modeling best practices
