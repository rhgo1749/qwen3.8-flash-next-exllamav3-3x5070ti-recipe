# Optimization log

This is the chronological reasoning behind the promoted recipe.

## 0. Starting point: an apparent ~72–75 tok/s wall

Three RTX 5070 Ti GPUs were serving Qwen3.8-Flash-Next EXL3 3.05 bpw with part of each MoE layer on CPU.

Early concurrent decode repeatedly landed around 72–75 tok/s aggregate, which initially looked close to a hardware ceiling.

GPU telemetry disproved that interpretation later: even during concurrent decode, the GPUs were frequently far below full SM, memory-controller and power utilization. The real ceiling was software/data movement, not raw GPU silicon.

## 1. Find the CPU-MoE leak

The model has 512 routed experts per layer and activates 10 per token.

With 232 experts on CPU and 280 on GPU, the important question is not only "how many experts fit on GPU?" but **which 280 experts are resident**.

A routing histogram from real agent traffic showed that the original physical tail split was poor:

```text
current CPU hit ~45.17%
ideal CPU hit   ~12.06%
```

That meant the machine was paying CPU/GPU handoff cost far more often than necessary even though the number of GPU slots was fixed.

### Decision

Collect expert popularity from representative workload, sort experts hot-to-cold per layer, freeze the permutation, and disable dynamic swapping during production.

This was partly a consequence of the upstream capability available at the time. As of **2026-09-23**, ExLlamaV3 PR [#315](https://github.com/turboderp-org/exllamav3/pull/315) was still open. Its proposed precomputed-profile `seed` mode would start from a workload-trained expert placement and then continue with the existing upstream dynamic swapper, but that path was not available in the release/upstream version used for this experiment. Therefore the experiment compared the existing upstream dynamic policy against a custom frozen histogram placement; it did **not** establish whether static placement beats a properly seeded dynamic policy. If `seed` lands upstream, the fair follow-up is static-vs-seed A/B under the same workload, using decode CPU assignments and end-to-end throughput rather than prefill histogram alone.

This became the dominant optimization.

## 2. MTP had the same problem

The MTP component has its own MoE layer and its routing was also badly aligned with the resident expert set.

Passive sampling was added before the fused CPU split submit path because the earlier hook missed decode traffic.

Measured before/after target:

```text
MTP current CPU hit ~44.3%
MTP hot target       ~7.5%
```

After this correction, MTP3 became much more attractive.

## 3. Revisit the speculative draft length

### MTP1

High acceptance, but the measurement was from the pre-hot-placement era.

### MTP2

We expected MTP2 to become interesting if weighted acceptance climbed into roughly the 65–70% range.

It did not. A real three-request overlap reached about 72.4 tok/s aggregate and MTP2 was rejected.

### MTP3

Best tested real-workload point after hot placement.

### MTP4

Two problems:

1. placement/cache workspace increased enough to violate the desired GPU0 headroom at the original split;
2. real traffic did not outperform MTP3.

MTP3 was retained.

## 4. CPU-MoE worker thread count

The CPU worker was instrumented with ExLlamaV3's handoff/GEMV profiling.

At 24 threads:

```text
gemv_gu ~0.98 ms
gemv_d  ~0.49 ms
```

16 threads was slower. 32 threads was substantially slower.

The one-layer MTP worker behaved differently: keeping it at 16 threads avoided extra contention with the 48-layer main worker.

The final asymmetry is deliberate:

```text
main worker = 24
MTP worker  = 16
```

## 5. Check whether CPU power/thermals were the problem

The 9950X3D sustained roughly 5.25–5.50 GHz in the observed workload and did not show a thermal-limit pattern. Temperatures were also well below a thermal ceiling.

That shifted attention away from CPU power settings and toward:

- actual GEMV latency
- routing frequency
- scheduling gaps
- CPU/GPU overlap

## 6. Try to buy more GPU-resident experts

Histogram math said dropping from CPU232 to CPU224 would reduce CPU hit further.

But real loader tests at the 524k cache budget showed:

```text
CPU232: fits
CPU228: loader VRAM failure
CPU224: loader VRAM failure
```

So CPU232 is effectively the residency boundary for this exact cache/split configuration.

## 7. Stage rebalance did not fit

GPU2 was often the busiest stage, so we tried moving budget from GPU2 to GPU1 while keeping GPU0 fixed:

```text
[11,15.5,14.5]   fail
[11,15.25,14.75] fail
```

The placement boundary is module-granular, not infinitely divisible. These small numeric changes could not create a valid better partition.

## 8. Find the missing layer 47 profile

The first static histogram had layers 0–46. Layer 47 silently remained on the original unpermuted tail.

A dedicated passive collection showed:

```text
layer 47 current CPU hit ~42.8%
layer 47 hot target       ~7.4%
```

Adding it was clearly correct, although the end-to-end improvement was much smaller than the original bulk hot-placement gain.

## 9. Kernel threshold sweep: the dangerous natural-traffic false positive

Natural agent traffic initially suggested:

```text
N threshold 1024 >> 2048 / 4096
```

That looked exciting but did not make architectural sense.

We inspected the actual projection widths and ExLlamaV3's `use_mgemm()` decision:

```text
attention slice = 512
MoE gate/up     = 640
GDN slice       = 2048
```

Thresholds 1024 and 2048 therefore make the same decisions for those relevant widths. The apparent difference had to be workload noise.

This was the point where we stopped trusting natural traffic for the kernel question and built a controlled C3 harness.

## 10. Controlled boundary test

A clean threshold boundary exists at 2048/2049 because the heuristic is strict:

```text
fuse when out_features < threshold
```

So:

```text
threshold 2048 -> GDN width 2048 is unfused
threshold 2049 -> GDN width 2048 is fused
```

We crossed that with activation INT8 on/off.

Three controlled C3 repetitions per cell produced:

```text
2049 + INT8=2: 112.3 tok/s
2048 + INT8=2: 109.4 tok/s
2048 + INT8=0: 120.4 tok/s  <-- winner
2049 + INT8=0: 112.8 tok/s
```

This result is more useful than the earlier natural-workload threshold sweep because:

- same prompt
- same cache state after warmup
- same output length
- same concurrency
- same model/runtime/hardware
- only the two kernel-policy variables change

## 11. Final promoted profile

The original promotion used the headroom-preserving `[11,15,15]` / CPU232 layout. A later 2026-09-24 pass moved the default performance mode to MTP on GPU2, increased base expert residency, moved PLE out of explicit RAM residency, and enabled confidence-calibrated dynamic drafting. The `[11,15,15]` layout remains available as the explicit headroom-preserving mode.

Current promotion:

```text
MTP3 ceiling, dynamic drafting enabled
MTP confidence 0.4
gpu_split [15,15,14]
draft_gpu_split [0,0,3]
CPU experts 208
GPU experts 304
main CPU-MoE threads 24
MTP CPU-MoE threads 16
ngram_ram false
static hot-expert placement including layer 47
MGEMM N threshold 2048
INT8 activation GEMV 0
3-GPU-only live server
```

### 11.1 PLE RAM vs NVMe evidence

The promoted model directory is about 88 GB and the PLE table alone is about 31 GB. With `ngram_ram=false`, the live container used about 33.52 GiB host memory; the 128 GB validation host showed about 41 GiB used and 79 GiB available.

A two-run 40k C3 comparison produced:

```text
PLE explicit RAM: 111.35 tok/s average aggregate
PLE NVMe path:    110.4 tok/s average aggregate
```

The <1% difference was smaller than draft-acceptance variance, while cold-ish 40k prefill was effectively tied (~1.59k vs ~1.61k tok/s). The capacity win therefore outweighed the unproven sub-1% throughput difference, so `ngram_ram=false` was promoted.

### 11.2 Dynamic MTP evidence

Short 142-token-prompt C3 steady-state:

```text
fixed MTP3:                132.6 tok/s aggregate
dynamic MTP3, conf 0.4:    151.9 tok/s aggregate
dynamic MTP3, conf 0.6:    129.7 tok/s aggregate
```

Long 40k runs did not show a stable advantage beyond acceptance noise. The promoted interpretation is therefore workload-sensitive: confidence 0.4 is useful as a mixed-workload default because it can trim unproductive draft positions on short/low-acceptance requests without adding another model forward for the decision.

### 11.3 Capacity contract after promotion

```text
VRAM:       3 x 16 GB validated; clean load 14,456 / 14,746 / 15,240 MiB
RAM:        64 GB recommended minimum; 96 GB recommended; 128 GB directly validated
Host reserve: 8 GiB (`EXL3_HOST_MEM_RESERVE_MB=8192`); 64 GB host not yet directly validated
SSD:        ~88 GB model directory, ~31 GB PLE; 100 GB free minimum, 120 GB+ recommended
Storage:    NVMe strongly recommended; validation host used Crucial T710 NVMe
```

See `docs/resource-requirements.md` for the full caveats.

### 11.4 Forced cooperative-wide geometry rejected

A 2026-09-24 live workload A/B tested `EXL3_MOE_COOP_WIDE=1` while keeping the promoted `layer` profile, MTP3, `EXL3_MGEMM_N_THRESHOLD=2048`, `EXL3_INT8_GEMV=0`, and `EXL3_MOE_PINNED_ARENA=1` otherwise unchanged.

With cooperative-wide geometry forced, repeated real agent decodes landed in the **6.6–26.8 tok/s** range. After removing only the force-wide override and cleanly restarting the server, longer generations recovered repeatedly to **50.1–62.8 tok/s** (with additional short completions reaching 67–84 tok/s). Prefill remained healthy, including ~1.0–1.5k tok/s on the observed partially cached requests.

This was a production-workload A/B rather than an identical synthetic prompt replay, so the exact ratio should not be treated as a universal benchmark. The magnitude and repeated recovery were nevertheless sufficient to reject forced wide geometry for the promoted recipe. **Leave `EXL3_MOE_COOP_WIDE` unset and allow ExLlamaV3's Blackwell geometry heuristic to choose automatically.**

## 12. Interpretation

The original ~72–75 tok/s wall was not a GPU hardware ceiling.

The largest problem was **unnecessary CPU expert traffic**. Once that was reduced, smaller runtime choices became visible and worth measuring.

The final controlled C3 result around 120 tok/s should not be described as universal production sustained throughput. Real mixed agent traffic still varies with prefill, cache state, output length and draft acceptance.

The useful conclusion is narrower:

> On this exact 3×5070 Ti / Qwen3.8-Flash-Next EXL3 3.05 bpw setup, software placement and kernel policy left a large amount of performance on the table. Careful routing and controlled A/B testing recovered much of it without changing model bpw or KV precision.
