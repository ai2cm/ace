# Gantry examples

This directory contains some examples for submitting batch jobs
to Ai2's [beaker](https://beaker.allen.ai/) platform using the
[gantry](https://github.com/allenai/beaker-gantry) tool.

For documentation on beaker, see [here](https://beaker-docs.apps.allenai.org/).
For help with gantry, see the readme in the
[gantry repository](https://github.com/allenai/beaker-gantry) or call `gantry run --help`.

A few important notes about using these examples:
- you must call `gantry run` from the root of this repository. For this reason, there
are directives provided in the top-level Makefile of this repo. For example,
you can call `make run_ace_evaluator` from the top-level of this repo to submit
the ACE evaluator example.
- gantry will install the current commit of this repo before running your job.
Therefore, you must commit and push changes to remote before submitting.
- you can use `--allow-dirty` to skip committing changes, but this may lead
to unexpected behavior so be careful!

For quick experimentation, feel free to make changes to the run scripts or
yaml configurations and commit these changes to experimental branches.
However, we will only merge changes to the gantry examples when there is a
change we expect all users to want (e.g. updated 'best' ACE configuration).
