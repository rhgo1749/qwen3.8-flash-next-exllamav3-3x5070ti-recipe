# Results

All numbers in this document come from one host unless stated otherwise.

## Validated hardware

- CPU: AMD Ryzen 9 9950X3D, 16C/32T, BIOS power limit around 110 W
- RAM: 128 GB DDR5 (4 × 32 GB), DDR5-5800 CL40
- Fabric / SoC: FCLK 2000 MHz, VSOC ~1.05 V
- GPU: 3 × RTX 5070 Ti 16 GB
- physical PCIe layout: Gen5 x8 / x8 / x4
- approximate theoretical one-way bandwidth: 31.5 / 31.5 / 15.75 GB/s
- NVIDIA enumeration during measurement: GPU0 x8, GPU1 x4, GPU2 x8
- PCIe 5.0 signaling: 32.0 GT/s per lane
- GPU topology: all GPU-to-GPU paths reported as `PHB`; no NVLink
- NVIDIA driver: 615.71.09
- Linux kernel: 7.0.0-31-generic
- ExLlamaV3: 1.5.1
- Model: Qwen3.8-Flash-Next EXL3 3.05 bpw

The physical PCIe layout on the validation machine was x8 / x8 / x4. At PCIe 5.0 rates, that corresponds to approximately 31.5 / 31.5 / 15.75 GB/s of theoretical one-way payload bandwidth. NVIDIA's runtime enumeration did not follow that physical ordering: during measurement GPU0 and GPU2 reported x8 while GPU1 reported x4. All three links reported PCIe Gen5 (32.0 GT/s per lane), and `nvidia-smi topo -m` reported `PHB` between every GPU pair. There was no NVLink. GPU0 headroom was intentionally preserved; this also makes the profile friendlier to systems where GPU0 is display-attached or shares VRAM with normal desktop/OS workloads.

The CPU/RAM tuning above is recorded because CPU-resident MoE work is material to this recipe. The benchmark did **not** use an unrestricted 9950X3D power profile. The serving runtime then used 24 threads for the main CPU-MoE worker and 16 threads for the MTP worker. These host values describe the tested machine rather than a guaranteed optimum for other AM5 memory controllers or DIMM kits.

## Production profile

```text
gpu_split              [11.0, 15.0, 15.0]
CPU experts            232 / 512 per split layer
GPU experts            280 / 512 per split layer
main CPU-MoE threads   24
MTP CPU-MoE threads    16
draft tokens           3
cache mode             8,4
cache size             524,288 tokens
max context            262,144 tokens
MGEMM N threshold      2048
INT8 activation GEMV   disabled
static routing stats   enabled
```

## Progression

The numbers below are not all from identical prompts, so treat this table as an optimization timeline rather than a strict benchmark ladder.

| Stage | Observation |
| --- | --- |
| early 3-GPU real workload | ~72–75 tok/s looked like an aggregate ceiling |
| static base hot-expert placement | CPU routing fell from ~45% toward ~14% effective decode hit |
| MTP hot-expert placement + MTP3 | real C3 moved into roughly 94–101 tok/s sustained territory |
| good real traffic intervals | 105–111+ tok/s |
| historical real traffic peak | ~119 tok/s |
| final controlled C3 kernel A/B winner | **120.4 tok/s average sum-TG** |

## Expert placement

### Base model histogram

Representative routing collection:

- 68,362,240 routed expert selections
- 47 layers in the first collection
- 512 expert counters per layer
- 280 GPU-resident experts / 232 CPU-resident experts

Histogram result:

```text
original placement CPU hit   45.17%
ideal hot placement CPU hit  12.06%
ideal hot placement GPU hit  87.94%
```

Post-promotion decode handoff measurement:

```text
CPU expert assignments/token-row
before: ~4.409
after:   1.436

effective decode CPU hit: ~14.4%
```

### MTP histogram

Before static hot placement:

```text
MTP CPU hit ~44.3%
```

Hot-placement estimate with the same 280/232 split:

```text
MTP CPU hit ~7.5%
```

### Layer 47 repair

Layer 47 was absent from the initial histogram. A dedicated passive collection showed:

```text
unpermuted CPU hit ~42.8%
hot-placement CPU hit ~7.4%
```

The end-to-end gain was small (a few percent at most), but it removed an obvious residual routing leak.

## MTP sweep

### MTP1

Before the MTP expert profile was fixed, MTP1 was competitive:

```text
weighted acceptance ~84%
cached TG ~49.7 tok/s
```

This result should not be compared directly with the final MTP3-hot configuration because the expert placement changed afterward.

### MTP2-hot

A real three-request decode overlap reached only about:

```text
72.4 tok/s aggregate
weighted acceptance around the high-50s/low-60s in the measured batch
```

It did not achieve the ~65–70% acceptance region that would have made it competitive with MTP3 on this host.

### MTP3-hot

Promoted.

Natural real-workload C3 commonly landed around:

```text
94–101 tok/s sustained
```

with good intervals above 110 tok/s.

### MTP4

MTP4 required extra placement budget. It failed to load with the preferred display-headroom-preserving split until GPU0 allocation was raised, which sacrificed the headroom that profile was designed to preserve.

A real sample with the larger split showed roughly:

```text
2+ overlap aggregate ~70 tok/s
```

and was not competitive with MTP3.

## CPU-MoE thread sweep

Main 48-layer worker:

| Threads | Result |
| ---: | --- |
| 16 | about 20–25% slower kernel-level GEMV than 24 |
| **24** | **promoted** |
| 32 | clearly worse; handoff compute median rose to ~2.77 ms/job |

Representative 24-thread profiling:

```text
gemv_gu ~0.98 ms
gemv_d  ~0.49 ms
compute ~1.55–1.70 ms/job
```

MTP one-layer worker:

| Threads | Result |
| ---: | --- |
| **16** | **promoted** |
| 24 | real C3 about 90.3 tok/s in the observed window; worse than the 16-thread profile |

The two worker pools contend for the same host cores, so increasing both does not help.

## VRAM boundary tests

At 524,288 cache tokens and `gpu_split [11,15,15]`:

- CPU expert count 232 loads
- 228 failed the loader budget
- 224 failed the loader budget

Attempted stage rebalance:

- `[11,15.5,14.5]`: failed
- `[11,15.25,14.75]`: failed

The stage boundary is coarse enough that a small budget transfer cannot necessarily move one clean module while preserving transient headroom.

## Kernel selection experiments

### Why 2048 is the interesting boundary

Observed text-path widths:

```text
attention sliced QKV: 512
MoE gate/up:          640
Gated DeltaNet slice: 2048
```

ExLlamaV3 uses a strict `out_features < threshold` condition for this fusion heuristic.

Therefore:

- threshold 2048: 2048-wide GDN is unfused
- threshold 2049: 2048-wide GDN is fused
- 512 and 640 remain fused in both cases

This creates a clean boundary test.

### Controlled C3 protocol

Each run:

- fixed prompt, about 40,102 tokens on the validation tokenizer
- warm prefix first
- three concurrent requests
- 900 output tokens each
- greedy decoding
- sum the three server-reported decode rates
- three repetitions per configuration

### 2×2 result

| Threshold | INT8 activation GEMV | GDN policy | Run sums (tok/s) | Average |
| ---: | ---: | --- | --- | ---: |
| 2049 | 2 | fused | 113.3, 109.6, 114.1 | **112.3** |
| 2048 | 2 | unfused | 108.9, 105.7, 113.6 | **109.4** |
| **2048** | **0** | **unfused** | **123.4, 116.5, 121.3** | **120.4** |
| 2049 | 0 | fused | 110.6, 118.9, 108.8 | **112.8** |

The winner is therefore not "INT8 is faster". The opposite happened here: disabling activation INT8 while keeping the GDN bundle unfused was the best tested **end-to-end** combination.

Do not read the percentage difference as a pure raw-kernel speedup. MTP draft acceptance also varied across the cells (for example, the 2048/INT8=0 runs had somewhat higher aggregate acceptance than some comparison cells), so the observed throughput difference can include both kernel execution cost and numerical-path effects on speculative acceptance.

## Final controlled concurrency scaling

Using the promoted production policy:

```text
EXL3_MGEMM_N_THRESHOLD=2048
EXL3_INT8_GEMV=0
MTP draft tokens=3
static hot-expert placement enabled
```

the same ~40,102-token warm-cache prompt and 900-token output workload was run at C1, C2 and C3.

| Concurrency | Run results | Average aggregate TG | Weighted draft acceptance |
| ---: | --- | ---: | ---: |
| C1 | 78.2, 85.5, 89.5 | **84.4 tok/s** | 68.5% |
| C2 | 103.9, 103.5, 106.1 | **104.5 tok/s** | 58.6% |
| C3 | 123.4, 116.5, 121.3 | **120.4 tok/s** | 64.8% |

For C1, each run result is the single request's TG. For C2 and C3, each run result is the sum of the simultaneously decoding requests' server-reported TG.

These are controlled warm-cache decode measurements, not mixed production sustained throughput. The C1 spread is strongly influenced by MTP draft acceptance, so a single run should not be treated as the host's fixed single-request speed.

### Why natural traffic initially misled us

A natural workload sample made threshold 1024 look much faster than 2048.

That interpretation was wrong.

Inspection of the actual tensor widths showed that 1024 and 2048 make the same relevant hot-path decisions for this model. The difference was caused by prompt/cache/acceptance/concurrency variation in the real agent workload.

This is why the final decision is based on the controlled 2×2 benchmark.

## What remains

At this point the low-cost configuration space was largely exhausted.

The remaining plausible gains require code/runtime work:

- CPU-MoE AVX-512 GEMV optimization
- better CPU/GPU overlap and scheduler behavior
- per-layer expert-slot allocation instead of the same 280/232 count everywhere
- model-shape-specific kernel specialization
- adding a fourth GPU and re-running residency/stage balancing

These are not simple environment-variable sweeps.
