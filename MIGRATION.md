# Repository Migration Guide

## Important Notice

This repository has undergone a **breaking history change** as part of our transition to open development at Ai2 climate modeling. The `main` branch now contains a new, filtered history that removes certain internal files and commits from our previous private repository.  This will streamline new releases and hopefully foster better collaboration with our growing userbase.

**This is a breaking change.** Future updates should maintain normal Git history since this will be our main development repo!

---

## What Changed?

The `main` branch history has been completely rewritten to:
- Remove internal/proprietary files and references\
- Establish a new baseline for open development

**If you have an existing clone of this repository from before this migration, you will encounter errors when pulling.**

---

## Migration Instructions

### Symptoms of Unmigrated Repository

When you try to pull from the repository, you may see errors like:

```bash
$ git pull
fatal: refusing to merge unrelated histories
```

or

```bash
$ git pull
hint: You have divergent branches and need to specify how to reconcile them.
```

This indicates your local repository is still based on the old history.

---

## Option 1: Easy Migration (No Local Work to Preserve)

**Use this if:** You have no local commits or branches you need to keep.

### Steps:

1. **Delete your local repository folder:**
   ```bash
   cd /path/to/parent/directory
   rm -rf ace
   ```

2. **Clone the repository fresh:**
   ```bash
   git clone https://github.com/ai2cm/ace.git
   cd ace
   ```

That's it! You now have the new history.

---

## Option 2: Advanced Migration (Preserving Local Work)

**Use this if:** You have local commits, branches, or work in progress you want to preserve.

### Step 1: Preserve Your Local Work

If you have uncommitted changes, commit them first:
```bash
git add .
git commit -m "WIP: local changes before migration"
```

If you have local commits on the old `main` branch that you want to preserve:
```bash
# Save your current main branch to a new branch name
git checkout main
git branch my-local-work
```

If you have other local branches, they will be preserved automatically.

### Step 2: Update to the New Main Branch

```bash
# Fetch the new history from the remote
git fetch origin

# Switch to main and reset to the new remote version
git checkout main
git reset --hard origin/main
```

**Warning:** This will discard any local commits on your `main` branch. Make sure you saved them to `my-local-work` or another branch in Step 1.

### Step 3: Reconcile Your Local Work (If Needed)

If you preserved local work in `my-local-work` or other branches, you'll need to rebase or cherry-pick those commits onto the new `main` branch.

#### Option A: Rebase Your Branch

```bash
# Switch to your branch with local work
git checkout my-local-work

# Rebase onto the new main
git rebase main
```

**Note:** Since the history has been rewritten, Git may not be able to automatically find the common ancestor. You might need to use:

```bash
git rebase --onto main --root my-local-work
```

This replays all commits from your branch onto the new `main`. You may encounter conflicts that need manual resolution.

#### Option B: Cherry-Pick Specific Commits

If rebasing is too complex or you only need specific commits:

```bash
# Switch to main
git checkout main

# Create a new branch for your work
git checkout -b my-migrated-work

# Cherry-pick individual commits by their SHA
git cherry-pick <commit-sha-1>
git cherry-pick <commit-sha-2>
# etc.
```

To find the commit SHAs you want to preserve:
```bash
git log my-local-work
```

### Step 4: Verify Your Migration

Check that your working directory looks correct:
```bash
git status
git log
```

---

## FAQs

### Why did this migration happen?

We transitioned from a private repository to open development. To protect internal information and provide a clean starting point, we filtered the repository history to remove proprietary content.

### I'm getting merge conflicts during rebase. What should I do?

This is expected due to the history rewrite. You have several options:
1. Resolve conflicts manually as they appear during the rebase
2. Use `git cherry-pick` to selectively apply specific commits (see Option B above)
3. Manually re-apply your changes on top of the new `main` by copying files or code snippets

### What if I already pushed branches to the remote?

If your branches were based on the old `main`:
1. Rebase them onto the new `main` (as described above)
2. Force-push them: `git push --force-with-lease origin your-branch-name`
3. Coordinate with any collaborators who may have pulled your branch

### Who can I contact for help?

If you encounter issues during migration, please:
- Open an issue on the repository
- Check for a pinned migration issue with additional guidance
- Reach out to the repository maintainers

---

## Timeline

- **Migration Date:** 2025-12-17
- **New Main Branch:** Based on `filtered-main-test` branch
- **Old History:** Now under main-archived-2025-12-17

---

## Additional Resources

- [Git Documentation: Rebasing](https://git-scm.com/book/en/v2/Git-Branching-Rebasing)
- [Git Documentation: Cherry-Pick](https://git-scm.com/docs/git-cherry-pick)
- [GitHub Docs: About Git Rebase](https://docs.github.com/en/get-started/using-git/about-git-rebase)
