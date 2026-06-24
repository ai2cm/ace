---
name: pre-review
description: Prepare a PR for review by a team member. Use when the user asks you to perform a pre-review.
---

A PR (corresponding to the current checked-out branch, or one the user has told you) is ready for review.
That review is going to be performed by another team member, but that member has limited time.
You are tasked with performing a "pre-review" to make this PR as easy to review as possible.

For example, this could mean:
 - Re-structuring the PR, either by splitting it into multiple smaller PRs or by re-ordering commits, to make it easier to review.
 - Catching and fixing any issues you can find before the PR is seen by the reviewer.
 - Making comments on the PR to point out potential issues or areas of concern for the reviewer to focus on.
 - Improving the PR description and title.

Focus specifically on making the reviewer's job easier, not on making the PR perfect.
Issues involving readability might be particularly important, especially if the changes are hard to understand or the reason for them is not immediately clear.

Begin by taking a look at the PR, as well as its description and title, and considering areas for improvement.
Communicate with the user about whether:
 - The PR has significant structural issues that should be fixed with significant refactors of the PR.
 - The PR should be significantly restructured, for example by restructuring the commits to have a more logical flow or by splitting into multiple smaller PRs.
 - The PR structure is acceptable, but there are specific issues that should be fixed before review.
 - The PR is in good shape, perhaps you have some comments to make on the PR or updates to its title/description but the code is ready for the reviewer.

When leaving comments on a PR, always start your comment with "Pre-review agent: ".

## Required audit passes

Before reporting the PR ready, walk the diff and explicitly perform each of these passes. Report findings even if none — silence on a pass means it wasn't done. Address findings inline where small; surface them as comments or to the user where they need a decision.

### 1. Silent-failure pass

For every new branch, default, fallback, optional field, union member, or aggregation introduced by this PR, name what happens when its input is missing, empty, `None`, `0.0`, a default-valued weight, or a key not present in the prior schema. If the answer is "the code takes the default branch quietly," flag it. Specifically check:
 - New dacite-loaded unions: can every previously-valid YAML still parse and produce equivalent behavior? Are deprecation warnings present where a config form is being phased out?
 - New aggregations / weighted sums: behavior at empty input, mismatched keys across calls, zero weight, all-zero weights.
 - "Unsupported" / "not applicable" code paths: do they raise loudly when the user explicitly asked for the unsupported behavior, or only when it was a silent default?

The user's reviewers catch this class of issue more reliably than the author's AI agent does. Treat it as the highest-priority pass.

### 2. Behavior-change pass on existing code paths

For every function, method, or code path that was *modified* (not newly added) by this PR, decide whether the change alters what existing callers observe. AI-generated edits regularly drift the semantics of code they "clean up" — particularly:
 - Reduction changes (sum ↔ mean, mean ↔ weighted mean, mean over channels vs samples).
 - Default values of existing arguments.
 - Which inputs cause a raise vs. a return vs. a log.
 - Fields silently dropped from intermediate containers (`BatchData`-like objects, dicts, dataclasses) that downstream code relied on.
 - Log lines removed, renamed, or moved (downstream dashboards/parsers may depend on the string).
 - Return types narrowed or widened in ways that change downstream branching.

For each behavior change found: if intended, it should be stated in the PR description (and a docstring updated if the public contract shifted); if unintended, revert it. If unclear, seek clarification from the author. Do not let "incidental cleanup" through.

Keep an eye out also for unintended effects of intended behavior changes, especially if the change is in a widely-used utility function or a core code path.

### 3. Differential-test pass

For every test added or modified to cover new behavior, confirm the assertion would actually fail if the new code path were no-oped out. Flag any test whose inputs make masked/unmasked, weighted/unweighted, filtered/unfiltered, or seeded/unseeded outputs numerically identical (e.g., all-ones inputs for a masking test, identical values across channels for a per-channel test, single-element collections for a multi-element behavior). Strengthen inputs in place where the fix is small.

### 4. Sibling / mirror coverage pass

When the PR adds an option, field, or behavior to one builder, config, or registry entry, list the siblings in the same registry or module and decide explicitly for each whether it applies. Common sibling sets in this repo: SFNO variants (SFNO, NoiseConditionedSFNO, etc.), the ace/coupled mirror modules, builder registries, aggregator-config types. If a sibling is intentionally skipped, that decision belongs in the PR description.

### 5. Config back-compat pass (only if the PR touches loaded configs or checkpoint-persisted state)

The only hard back-compat guarantee in this repo is that prior trained checkpoints must still load for **inference**. Resuming an in-progress training job across code or config changes is explicitly not guaranteed (see AGENTS.md). Check:

 - Can every prior trained checkpoint still be loaded for inference? If not, the PR needs explicit user sign-off.
 - Can every prior inference-time YAML still parse? If a field is being removed or restructured, is there a deprecation warning rather than a hard break, unless the user has approved the break?
 - Training-time configs may break across versions, but if a persisted training-state value will produce *silently wrong* output (rather than an error or warning) (and actually wrong, not just stochastically different) when the surrounding config is edited mid-run, that belongs in the silent-failure pass — prefer a warning or a reset.

### 6. Split heuristic

Recommend splitting the PR if more than one of these is true: it touches >20 files; it introduces a new typed config *and* changes behavior *and* adds a back-compat shim; it bundles a refactor with a new feature. Past PRs that hit two of these took 5–8 days to merge and often attracted reviewer-authored companion PRs; splitting the scaffolding from the migration usually cuts that to 1–2 days per piece.

Note: an `fme/ace` change and its `fme/coupled` mirror are *not* a reason to split — they should travel together (or, if separated, the second PR's description should cite the first's review) so reviewers don't re-derive the same questions twice.

### 7. Abstract-or-final / inheritance-depth pass

 - Methods defined on a class with at least one in-repo subclass must be `@abstractmethod` or `@final`. Exception: `__init__` on `nn.Module` subclasses.
 - In-repo inheritance must be at most 1 level deep, ignoring purely-abstract classes (those whose methods are all `@abstractmethod`).

### 8. AI-agent artifact pass

Quick scan for patterns that AI-generated code drops in and human reviewers consistently flag:
 - `Any` return types where a concrete type is available.
 - `isinstance` + `type: ignore` pairs (prefer refactoring the type).
 - Imports inside test function bodies (lift to module top).
 - Abbreviations in identifiers (`cb` → `callback`).
 - Stale log lines or comments left over from a refactor.
 - Unnecessary removal of comments or docstrings unrelated to the change.

### 9. Builder pattern pass

We generally follow a "builder pattern" for configuration in this repo: dataclasses we load using dacite (a `Config`), usually paired with an implementation class that the config's `build()` constructs. The rules to check are:
 - **Field privacy.** Config dataclass fields should be treated as private and only used in methods on that class, even though they are not prefixed with `_`. Reading `config.some_field` from anywhere else (an implementation class, a free function, or another config) is a violation; expose what callers need as a `@property` or method.
 - **Builder constructs sub-objects.** When a builder `Config` class is paired with an implementation class, the builder should construct sub-objects from any sub-configuration fields and pass those to the implementation class, rather than the implementation class building those sub-objects. The implementation class receives built collaborators, never a sub-config to build itself.

Exceptions to the field-privacy rule:
 - **Leaf-config carve.** A config may be passed to the one implementation class its `build()` constructs, and that impl may read the config's fields — but only if the config is a **leaf**, i.e. none of its fields are themselves dataclasses. `list[Config]`, `dict[str, Config]`, and `Config | None` all count as dataclass fields, so they make a config non-leaf. A non-leaf config must not be handed to its impl; its `build()` builds the sub-objects (per the second rule) and may pack the scalars the impl needs into a separate, plain (non-dacite) leaf dataclass passed as an argument.
 - **Parent/sibling reads are not exempt.** A config reading another config's fields — even one it directly contains — is still a violation. Fix it with a public `@property` on the child for a derived value (e.g. `loss.requires_ensemble` rather than `loss.type == "EnsembleLoss"`), or a `validate(...)` method on the child for an invariant (e.g. `metric.validate(required_target="denorm")`). The parent then goes through that sanctioned API.
 - **Tests are exempt** — a test reading a config's fields is not a violation.

For top-level entrypoints the same idea applies as **build-then-execute**: a run-config consumed by a free function (e.g. `run_*_from_config`) that reads the config as a parameter bag is a violation. The config's `build()` should assemble the runtime object(s); a thin entrypoint then performs the process side-effects (IO, logging, distributed setup) and invokes the built object. Pure assembly (which reads the config) belongs in `build()`; the side-effecting shell operates on built objects and explicit values, not the config.

Pre-existing violations of these rules are allowed, but new violations should be flagged, including refactors to existing classes that introduce new violations.

## After audit passes

When audit passes are complete, give a concise summary of the audit pass results.
Start each audit pass with a ✅ if clear, or a ⚠️ or ❌ (depending on severity) if issues are found.

Then, under a separate heading, revisit the overall purpose of the pre-review.
Give the user a clear summary of the state of the PR and what you recommend for next steps.
When giving this summary, do not re-cap on each successful check that has no actions required, but you could give a summary of the overall state.
