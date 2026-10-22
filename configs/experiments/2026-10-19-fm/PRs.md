# PRs merged into exp/alexeyfm2

This branch is `main` plus the PRs below. No code changes are made on this
branch outside `configs/experiments/2026-10-19-fm/` unless they come from a PR
listed here.

| PR | Title | Author |
|----|-------|--------|
| [#1514](https://github.com/ai2cm/ace/pull/1514) | Extend torch.compile to every step type and module builder | yyexela |
| [#1519](https://github.com/ai2cm/ace/pull/1519) | Make irfft, NullTimer.child, CappedGELU, and envelope clipping traceable by torch.compile | yyexela |
| [#1520](https://github.com/ai2cm/ace/pull/1520) | Fix SecondaryDecoder.wrap_module mutating the decoder in place | yyexela |
| [#1526](https://github.com/ai2cm/ace/pull/1526) | Recover wandb metrics lost to preemption from on-disk logs | yyexela |
| [#1536](https://github.com/ai2cm/ace/pull/1536) | Add per-group network normalization | yyexela |
| [#1537](https://github.com/ai2cm/ace/pull/1537) | Add get_pooled_stats.py with per-group stats output | yyexela |
| [#1539](https://github.com/ai2cm/ace/pull/1539) | Add float32_matmul_precision option to SingleModuleStepConfig | yyexela |
| [#1541](https://github.com/ai2cm/ace/pull/1541) | Build unconditional modules as if the dataset carried no labels | yyexela |
| [#1543](https://github.com/ai2cm/ace/pull/1543) | Swin U-Net attention and conditioning performance | yyexela |
| [#1544](https://github.com/ai2cm/ace/pull/1544) | Add patch_size, num_levels and skip_projection options to the Swin U-Net | yyexela |
| [#1545](https://github.com/ai2cm/ace/pull/1545) | Pass the network normalizer to GMR and input masking at call time | yyexela |
| [#1549](https://github.com/ai2cm/ace/pull/1549) | Make Samudra and m2lines CappedGELU traceable by torch.compile (via #1514) | yyexela |
| [#1576](https://github.com/ai2cm/ace/pull/1576) | Add output_dir to inference configs to decouple job dir from data output dir | yyexela |
| [#1588](https://github.com/ai2cm/ace/pull/1588) | Stop dataset properties from aliasing member labels and masks | yyexela |
| [#1592](https://github.com/ai2cm/ace/pull/1592) | Add renamed_variables, derived_variables, and variable_metadata to XarrayDataConfig | yyexela |
| [#1597](https://github.com/ai2cm/ace/pull/1597) | Add label-balanced sampling for multi-source training | yyexela |
