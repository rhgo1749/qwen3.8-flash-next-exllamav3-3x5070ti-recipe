# MoE routing histogram probe

This directory contains the **version-sensitive** probe used to discover the hot-expert placement described in the main recipe.

It was validated against ExLlamaV3 1.5.1.

## Why this exists

With 280 GPU-resident experts and 232 CPU-resident experts per layer, the original resident set was badly aligned with real routing frequency.

The probe answers two questions:

1. How much routing currently lands on CPU?
2. If the same number of GPU slots were assigned to the hottest experts, how low could CPU routing go?

On the validation workload the first large collection moved the estimate from roughly:

```text
45.17% CPU hit -> 12.06% ideal CPU hit
```

without changing the number of resident experts.

## Dynamic collection for the main layers

Run a representative workload with dynamic CPU swapping enabled:

```bash
export PYTHONPATH="/path/to/this/directory:${PYTHONPATH:-}"
export EXL3_MOE_HIST_PROBE=1
export EXL3_MOE_HIST_OUT="/tmp/qwen38-routing-stats.json"
export EXL3_MOE_CPU_SWAP=1

# start the model server
```

Do not promote a tiny sample. The original large collection contained more than 68 million routed expert selections.

Watch the printed `cpu_ideal` value and expert ordering until they stop moving materially.

## Passive collection for MTP or a missing static layer

When static placement is already active, use a prefix:

```bash
export PYTHONPATH="/path/to/this/directory:${PYTHONPATH:-}"
export EXL3_MOE_HIST_PROBE=1
export EXL3_MOE_HIST_OUT="/tmp/qwen38-routing-stats.json"
export EXL3_MOE_HIST_PREFIX="mtp."
```

For a missing layer, set the exact layer prefix, for example:

```bash
export EXL3_MOE_HIST_PREFIX="model.language_model.layers.47.mlp"
```

The probe merges sampled keys atomically into the JSON file.

## Production use

Once the histogram is accepted:

```bash
export EXL3_MOE_CPU_SWAP=0
export EXL3_MOE_CPU_SPLIT_STATS="/path/to/qwen38-routing-stats.json"
```

Back up a known-good histogram before replacing it.

## Important

This probe monkey-patches ExLlamaV3 internal classes. It is **not** an upstream stable API.

After an ExLlamaV3 upgrade:

- inspect the changed class/method names;
- run a small canary;
- verify the output histogram;
- do not assume the hook still observes the same routing path.
