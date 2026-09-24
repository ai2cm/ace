# beaker-queue Claude Code skill

Source of the personal Claude Code skill that reports a budget's allocated queue on a Beaker
cluster. `SKILL.md` is the skill definition and `scripts/beaker_queue.py` the tool it runs; the
script also works standalone:

```
python3 scripts/beaker_queue.py ai2/jupiter --budget ai2/atec-climate [--include-unallocated]
```

Install locally by copying this directory to `~/.claude/skills/beaker-queue/`, or package it with
the skill-creator skill's `package_skill.py` and upload the `.skill` file under Settings →
Capabilities → Skills on claude.ai to sync it across logins.
