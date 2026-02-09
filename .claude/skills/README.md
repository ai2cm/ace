# ACE Claude Code Skills

This directory contains custom skills for Claude Code to assist with ACE repository workflows.

## Available Skills

### 1. **review-pr** - PR Review Assistant

**Purpose:** Conduct comprehensive code review of GitHub pull requests

**Usage:**
```
/review-pr 123
```
or
```
Review PR #123
```

**What it does:**
- Fetches PR data using `gh` CLI
- Analyzes code changes and diff
- Provides structured review with severity levels (Critical/Suggestions/Nitpicks)
- Checks ACE-specific conventions (vendorized code, test markers, docstrings)
- Assesses test coverage
- Summarizes existing discussion
- Provides actionable recommendations

**Prerequisites:**
- `gh` CLI installed and authenticated (`gh auth login`)
- Read access to ai2cm/ace repository

---

### 2. **rereview-pr** - PR Re-review Assistant

**Purpose:** Check if review comments have been addressed after updates

**Usage:**
```
/rereview-pr 123
```
or
```
Re-review PR #123 from commit abc1234
```
or
```
Check if comments addressed on PR #456
```

**What it does:**
- Compares new commits against previous review state
- Categorizes review comments as Addressed/Partially/Unaddressed/Dismissed
- Identifies new issues introduced in recent commits
- Tracks previous Assistant findings from the conversation
- Provides focused delta review
- Recommends if PR is ready to merge

**Prerequisites:**
- `gh` CLI installed and authenticated
- Optionally: commit SHA or timestamp to review from

---

## How Skills Work

1. **Invoke a skill** using `/skill-name` or natural language
2. **Claude executes** the instructions defined in the skill file
3. **Results are presented** in the conversation with structured output

## GitHub CLI Setup

These skills use the GitHub CLI (`gh`) to interact with pull requests.

### Installation

**macOS:**
```bash
brew install gh
```

**Linux:**
```bash
# Debian/Ubuntu
sudo apt install gh

# Fedora/RHEL
sudo dnf install gh
```

**Conda (any platform):**
```bash
conda install -c conda-forge gh
```

### Authentication

```bash
gh auth login
```

Follow the prompts to authenticate with your GitHub account.

### Test Your Setup

```bash
# View a PR
gh pr view 123

# Get PR diff
gh pr diff 123

# List PRs
gh pr list --repo ai2cm/ace
```

## Skill Development

To create a new skill:

1. Create a markdown file in this directory: `.claude/commands/my-skill.md`
2. Add instructions in natural language
3. Document usage and prerequisites
4. Test with Claude Code

**Skill Template:**
```markdown
# My Skill Name

**Skill Name:** my-skill
**Description:** Brief description
**Usage:** `/my-skill <args>` or natural language trigger

---

## Instructions

[Detailed instructions for Claude to follow]

## Output Format

[Expected output structure]
```

## Tips for Effective PR Reviews

**For Initial Reviews:**
- Be thorough but constructive
- Distinguish blocking issues from suggestions
- Check ACE-specific patterns (vendorized code, test speed tiers)
- Verify test coverage for new functionality

**For Re-reviews:**
- Specify starting commit for focused review: "Re-review PR #123 from commit abc1234"
- For substantial changes, request full re-review instead of delta
- Track resolution of your own previous findings

## Related Files

- **`.claude/CLAUDE.md`** - Project context (always loaded)
- **`.claude/settings.json`** - Permissions and hooks configuration
- **`.cursor/rules/`** - Original Cursor AI rules (reference)
