# 2026-09-09 Swin efficiency

Follow-up to the August 2026 4-degree AIMIP foundation-model runs, where the
noise-conditioned Swin transformer (265M parameters) took about 213 GPU-hours
over 150 epochs against 75 GPU-hours for the noise-conditioned SFNO (14M
parameters). Per training epoch the Swin run was 3.3x slower, largely because
its `nn.Linear` layers ran true FP32 matmul while SFNO's convolutions already
used TensorFloat-32, and because the Swin decoder stage ran at `2 * embed_dim`
channels on the full-resolution grid.

`base_configs/` holds the two configs as they were run, minus the
`pre_cooldown_checkpoint_epoch` field that only exists on the experiment
branch they were launched from.
`generate_configs.py` derives the configs in `run_configs/`:

| Config | Changes from base | Params | Fwd GFLOP/sample |
|---|---|---|---|
| `...-nc-sfno-fm-a1-fast` | TF32 matmul; inference sets with weight 0 dropped, remaining every 25 epochs | 14M | 99 |
| `...-nc-swin-v2-fm-a1-fast` | as above, plus `compile: true` and `skip_projection: true` | 238M | 690 |
| `...-nc-swin-v2-fm-a1-fast-compute-matched` | `-fast` with `embed_dim: 192`, `depth_multiplier: 1`, `num_heads: [3, 6, 6, 3]` | 34M | 101 |
| `...-nc-swin-v2-fm-a1-fast-mid` | `-fast` with `embed_dim: 256`, `depth_multiplier: 1`, `mlp_ratio: 8/3` | 47M | 137 |
| `...-nc-swin-v2-fm-a1-fast-param-matched` | `-fast` with `embed_dim: 128`, `depth_multiplier: 1`, `mlp_ratio: 8/3` | 12M | 37 |

FLOP and parameter counts are for a 45x90 grid with 46 input and 51 output
channels, forward pass only, batch size 1.

Expected effect on the Swin run: the FP32 matmul bottleneck (about 65% of the
H100 CUDA-core peak) is removed by TF32, `skip_projection` removes about a
quarter of the FLOPs, and trimming inference removes most of the roughly 29
GPU-hours spent on in-training rollouts. The SFNO `-fast` config exists so the
two architectures can be compared under the same inference schedule and matmul
precision.

The three scaled-down Swin configs bracket SFNO: `compute-matched` has the
same forward FLOPs as SFNO, `param-matched` has a similar parameter count,
and `mid` sits between the two. Layers 2 and 3 hold over 80% of the base
model's parameters, so `depth_multiplier` is the main lever; `embed_dim` is
quadratic; `mlp_ratio: 8/3` is the usual SwiGLU convention.

Regenerate with `python generate_configs.py` from this directory. The test
file checks the committed configs match the generator and parse into
`TrainConfig`.
