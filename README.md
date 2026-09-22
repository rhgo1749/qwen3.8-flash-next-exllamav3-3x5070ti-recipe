# Qwen3.8-Flash-Next on 3× RTX 5070 Ti — ExLlamaV3 recipe

**English** | [한국어](README.ko.md) | [日本語](README.ja.md) | [简体中文](README.zh-CN.md)

A measured serving recipe for **Qwen3.8-Flash-Next EXL3 3.05 bpw** on **three RTX 5070 Ti 16 GB GPUs** using **ExLlamaV3 1.5.1**.

This repository is not a generic "best settings" claim. It documents one specific hardware/model/runtime combination, the optimization path that worked, the settings that failed, and the benchmark method used to separate real gains from workload noise.

## TL;DR

Validated host:

- 3 × NVIDIA GeForce RTX 5070 Ti 16 GB
- physical PCIe link layout: PCIe 5.0 x8 / x8 / x4
- approximate theoretical one-way bandwidth: 31.5 / 31.5 / 15.75 GB/s
- NVIDIA enumeration at measurement time: GPU0 x8, GPU1 x4, GPU2 x8
- PCIe 5.0 signaling rate: 32.0 GT/s per lane; all GPU pairs traverse `PHB`; no NVLink
- Ryzen 9 9950X3D
- 128 GB DDR5 (4 × 32 GB), DDR5-5800 CL40
- FCLK 2000 MHz, VSOC ~1.05 V
- CPU power limit configured around 110 W in BIOS
- NVIDIA driver 615.71.09
- Linux 7.0.0-31-generic
- ExLlamaV3 1.5.1
- Qwen3.8-Flash-Next, EXL3 3.05 bpw
- 262,144 max sequence length
- 524,288-token shared cache budget
- cache mode `8,4`
- MTP enabled, 3 draft tokens

The promoted configuration is:

```text
gpu_split                  = [11.0, 15.0, 15.0]
cpu_moe_split_experts      = 232
cpu_moe_threads            = 24

draft_mode                 = mtp
draft_gpu_split            = [3, 0, 0]
draft_num_tokens           = 3

EXL3_MOE_CPU_SWAP          = 0
EXL3_MOE_CPU_SPLIT_STATS   = /path/to/routing-stats.json

EXL3_MGEMM_N_THRESHOLD     = 2048
EXL3_INT8_GEMV             = 0
```

The biggest gain came from **frequency-guided static hot-expert placement**. The final kernel-selection gain came from a more subtle interaction: for this model on these GPUs, the best controlled C3 result was obtained by **unfusing the 2048-wide Gated DeltaNet bundle while keeping INT8 activation GEMV disabled**.

### Host CPU / RAM tuning used for the measurements

The host was not running JEDEC-default memory or an unlimited CPU profile. The measurements in this repository used:

```text
CPU                 Ryzen 9 9950X3D, 16C/32T
BIOS CPU power cap  ~110 W
RAM                 128 GB total, 4 × 32 GB
DRAM                DDR5-5800 CL40
FCLK                2000 MHz
VSOC                ~1.05 V
main CPU-MoE        24 worker threads
MTP CPU-MoE         16 worker threads
```

The 24/16 worker split is part of the serving recipe; the BIOS/RAM values are host context, not claimed universal optimums. In particular, do not assume the same VSOC or memory clock is stable on another 4-DIMM AM5 system.

## Headline results

The original real-workload three-request aggregate decode looked capped around **72–75 tok/s**.

After expert placement, MTP tuning, CPU thread tuning and the final kernel policy:

- real mixed workload: commonly **~94–101 tok/s aggregate C3**
- good real-workload intervals: **110+ tok/s**
- controlled C1, promoted policy: **84.4 tok/s average TG**
- controlled C2, promoted policy: **104.5 tok/s average aggregate TG**
- controlled C3, promoted policy: **120.4 tok/s average aggregate TG**
- historical real-workload peak observed during optimization: about **119 tok/s**

The controlled result is intentionally reported separately from production sustained throughput. A fixed warm-cache benchmark is much cleaner than a real agent workload with variable prompt lengths, prefix-cache state, draft acceptance and overlapping prefill.

## Why this configuration works

### 1. Static hot-expert placement

With 232 experts on CPU and 280 resident on GPU per split MoE layer, naive/static-tail placement sent far too many routed experts to CPU.

The collected routing histogram showed:

```text
before hot placement:
  CPU hit ~45.17%

ideal hot 280 experts resident:
  CPU hit ~12.06%
  GPU hit ~87.94%
```

After promotion, decode handoff profiling measured about **1.436 CPU expert assignments/token-row**, approximately **14.4% effective CPU hit**, versus roughly **44%** before the static profile.

The MTP layer had a similar problem:

```text
MTP before hot placement:  ~44.3% CPU hit
MTP hot-placement target:   ~7.5% CPU hit
```

A later pass found that layer 47 was missing from the original histogram. Its unoptimized CPU hit was roughly **42.8%**; after collecting that layer separately, its hot-placement estimate dropped to about **7.4%**.

This was the largest optimization in the entire project.

### 2. MTP3 was the practical optimum

We tested MTP1, MTP2, MTP3 and MTP4.

- MTP1 was competitive before the MTP expert placement was fixed, but lost its advantage afterward.
- MTP2-hot did not raise acceptance enough. A real C3 sample reached only about **72.4 tok/s aggregate**.
- **MTP3-hot** became the best production point.
- MTP4 required more VRAM and either failed to load with the display-headroom-preserving split or consumed too much GPU0 headroom. In real traffic it also underperformed MTP3.

The promoted setting is therefore:

```yaml
draft_num_tokens: 3
dynamic_draft: false
```

### 3. CPU MoE thread counts matter

The 9950X3D did not scale monotonically with more worker threads.

Main 48-layer CPU-MoE worker:

- 16 threads: slower
- **24 threads: best tested point**
- 32 threads: much slower

MTP one-layer worker:

- **16 threads: better**
- 24 threads: lower C3 throughput due to host-core contention

So the asymmetry is intentional:

```text
main CPU-MoE worker: 24 threads
MTP CPU-MoE worker:  16 threads
```

### 4. The final kernel policy was not "INT8 is faster"

The relevant text decode widths are:

| Path | Effective width used by the fusion heuristic |
| --- | ---: |
| attention QKV slices | 512 |
| MoE gate/up | 640 |
| Gated DeltaNet qkv+z slice | 2048 |

ExLlamaV3's fusion heuristic uses:

```text
fuse when out_features < EXL3_MGEMM_N_THRESHOLD
```

That makes **2048 vs 2049** a useful boundary test:

- threshold 2048: the 2048-wide GDN bundle is **unfused**
- threshold 2049: the same bundle is **fused**
- 512-wide attention and 640-wide MoE gate/up remain fused in both cases

We then ran a 2×2 controlled C3 test:

| GDN policy | INT8 activation GEMV | 3-run average sum-TG |
| --- | --- | ---: |
| fused, threshold 2049 | on (`2`) | 112.3 tok/s |
| unfused, threshold 2048 | on (`2`) | 109.4 tok/s |
| **unfused, threshold 2048** | **off (`0`)** | **120.4 tok/s** |
| fused, threshold 2049 | off (`0`) | 112.8 tok/s |

So the winning **end-to-end** interaction on this hardware was:

```bash
EXL3_MGEMM_N_THRESHOLD=2048
EXL3_INT8_GEMV=0
```

This is not a claim that the raw non-INT8 GEMV kernel alone is 7–10% faster. MTP draft acceptance also moved between runs, so the measured gain can include both kernel cost and numerical-path effects on speculative acceptance. The result should be interpreted as an end-to-end serving-policy win for this exact model/runtime/hardware combination.

This is an important correction to an earlier natural-workload observation: 1024 initially looked much faster than 2048, but inspection of the model shapes showed that **1024 and 2048 make the same relevant text hot-path fusion decisions**. The apparent difference was workload noise. Controlled A/B testing was required to identify the real boundary.

## Recipe

The full Tabby/ExLlamaV3 model configuration is in:

```text
recipe/tabby_config.yml
```

The promoted environment variables are in:

```text
recipe/env.sh.example
```

The routing histogram collector used for static expert placement is in:

```text
tools/moe_hist_probe/sitecustomize.py
```

It is intentionally version-sensitive. Read `docs/optimization-log.md` before using it.

## Minimal launch policy

Assuming your ExLlamaV3/TabbyAPI installation already works with the model:

```bash
export EXL3_MOE_CPU_SWAP=0
export EXL3_MOE_CPU_SPLIT_STATS=/absolute/path/to/qwen38-routing-stats.json
export EXL3_MGEMM_N_THRESHOLD=2048
export EXL3_INT8_GEMV=0

# start TabbyAPI / your ExLlamaV3 server using recipe/tabby_config.yml
```

Do **not** copy someone else's routing histogram blindly. Expert popularity depends on workload and can drift. Collect your own representative traffic, then freeze the ranking and validate it against your existing placement.

## Controlled concurrency benchmark

`bench/controlled_c3.py` runs a configurable concurrency benchmark:

1. one fixed warm-up request;
2. `CONCURRENCY` identical, warm-cache requests concurrently (default 3);
3. 900 output tokens per request;
4. greedy decoding for repeatability.

On the validated machine the fixed prompt is about 40k tokens.

The benchmark is designed for **relative A/B comparison on the same host**, not for cross-project leaderboard claims.

Examples:

```bash
CONCURRENCY=1 python3 bench/controlled_c3.py
CONCURRENCY=2 python3 bench/controlled_c3.py
CONCURRENCY=3 python3 bench/controlled_c3.py
```

Then use the server-side ExLlamaV3/Tabby logs to record each request's reported token generation rate and sum the concurrent decode rates for that run.

## Things that did not help

These are worth documenting because most of them looked plausible before measurement:

- MTP2-hot: not enough acceptance gain
- MTP4: VRAM/headroom cost and worse real-workload aggregate
- base CPU-MoE 16 threads: slower
- base CPU-MoE 32 threads: much slower
- MTP CPU-MoE 24 threads: worse due to contention
- CPU expert split 228/224: failed to fit at the validated 524k cache budget
- `gpu_split [11,15.5,14.5]`: failed the module/VRAM boundary
- `gpu_split [11,15.25,14.75]`: same problem
- aggressive `MGEMM_N_THRESHOLD=0`: poor/noisy real-workload behavior
- INT8 activation GEMV on the promoted unfused GDN path: slower in controlled C3
- treating one natural traffic sample as proof: several apparent wins disappeared under controlled A/B

See `docs/optimization-log.md` for the sequence and measurements.

## GPU0 headroom is intentional

The first GPU is deliberately limited to 11 GB in the split:

```yaml
gpu_split: [11.0, 15.0, 15.0]
```

This is not a maximum-throughput-only setup. The split intentionally leaves several GiB free on GPU0, which is useful on systems where that GPU is display-attached or also serves normal desktop/OS workloads. A fully headless machine with no desktop workload on GPU0 should re-sweep the split instead of assuming 11/15/15 is globally optimal.

## Reproducibility notes

- The model used here is an **EXL3 3.05 bpw conversion** of Qwen3.8-Flash-Next.
- The model weights are **not** included in this repository.
- The upstream model license applies to model weights separately from this repository.
- A local CPU idle-parking patch was present in the author's runtime image. It targets idle host CPU behavior and is not required to understand or reproduce the throughput tuning described here.
- Static hot-expert rankings are workload-dependent.
- CUDA/driver/runtime updates can change kernel break-even points.
- Re-run the controlled A/B benchmark after changing GPU generation, EXL3 bpw, ExLlamaV3 version, expert residency, cache size, or batch/concurrency shape.

## Repository layout

```text
.
├── README.md
├── README.ko.md
├── README.ja.md
├── README.zh-CN.md
├── RESULTS.md
├── bench/
│   └── controlled_c3.py
├── docs/
│   └── optimization-log.md
├── recipe/
│   ├── env.sh.example
│   └── tabby_config.yml
└── tools/
    └── moe_hist_probe/
        └── sitecustomize.py
```

## License

The code and documentation in this repository are released under the MIT License. Model weights are not included and remain subject to their own license.
