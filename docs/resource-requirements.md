# Resource requirements for the promoted 3-GPU modes

Validated on **2026-09-24** with Qwen3.8-Flash-Next EXL3 3.05 bpw, ExLlamaV3 1.5.1, a 524,288-token shared `8,4` cache, static-hot MoE placement, MTP3, and PLE streamed from NVMe.

The recipe exposes two supported 3-GPU modes:

| Mode | Config | GPU split | CPU experts | MTP placement | Purpose |
| --- | --- | --- | ---: | --- | --- |
| **Performance mode** | `recipe/tabby_config.yml` | `[15,15,14]` | 208 | GPU2 `[0,0,3]` | maximum tested residency/throughput |
| **Headroom-preserving mode** | `recipe/tabby_config.headroom.yml` | `[11,15,15]` | 232 | GPU0 `[3,0,0]` | deliberately keep several GiB free on GPU0 |

Both modes use `ngram_ram: false`, dynamic MTP with confidence `0.4`, the same static routing policy, and the same 524,288-token `8,4` cache. The exact capacity measurements below were taken on **Performance mode** unless stated otherwise.

These numbers are not generic requirements for every Qwen3.8-Flash-Next quantization or runtime.

## GPU / VRAM

**Validated requirement: 3 × 16 GB NVIDIA GPUs.**

Performance-mode split:

```yaml
gpu_split: [15.0, 15.0, 14.0]
cpu_moe_split_experts: 208

draft_model:
  draft_mode: mtp
  draft_gpu_split: [0, 0, 3]
  draft_num_tokens: 3
```

Headroom-preserving mode uses:

```yaml
gpu_split: [11.0, 15.0, 15.0]
cpu_moe_split_experts: 232

draft_model:
  draft_mode: mtp
  draft_gpu_split: [3, 0, 0]
  draft_num_tokens: 3
```

The headroom mode intentionally gives up some GPU expert residency to preserve GPU0 VRAM; do not interpret the performance-mode VRAM measurements below as measurements of the headroom mode.

Observed VRAM after a clean production load:

| GPU | Observed used VRAM | Board memory |
| --- | ---: | ---: |
| RTX 5070 Ti #0 | 14,456 MiB | 16,303 MiB |
| RTX 5070 Ti #1 | 14,746 MiB | 16,303 MiB |
| RTX 5070 Ti #2 | 15,240 MiB | 16,303 MiB |

The third GPU has only about 1 GiB of remaining board memory in this profile. Treat **16 GB per GPU as a real requirement**, not a loose recommendation. A 12 GB card is not expected to fit this exact cache/expert/draft layout without reducing another budget.

The production server is intentionally **3-GPU-only**. A fourth RTX 5060 Ti was tested earlier but adding a fourth pipeline/device stage reduced throughput on this topology, so the promoted server no longer exposes 4-GPU aliases and does not create a CUDA context on the 5060 Ti.

## System RAM

**Validated capacity: 128 GB.**

The important update is that the promoted profile no longer keeps the ~31 GB PLE n-gram table permanently resident in system RAM:

```yaml
ngram_ram: false
```

With PLE streamed from NVMe, the live container reported about **33.52 GiB** memory use and the 128 GB host reported about **41 GiB used / 79 GiB available** after loading the model.

Capacity guidance:

| System RAM | Status |
| --- | --- |
| **128 GB** | validated, comfortable |
| **96 GB** | recommended with ample host/file-cache headroom |
| **64 GB** | **recommended minimum** for the SSD-PLE profile with an 8 GiB host-memory reserve; not yet directly validated on a 64 GB machine |
| **48 GB** | not recommended; transient allocations, pinned buffers, OS memory and page cache leave too little safety margin |

The promoted reserve is now `EXL3_HOST_MEM_RESERVE_MB=8192` (8 GiB). This is an OOM safety margin, not model working memory. The change is justified by moving the ~31 GB PLE table out of explicit RAM residency: the live container used about 33.52 GiB with `ngram_ram: false`, leaving roughly 18 GiB of additional room on a nominal 64 GB host after model use plus the 8 GiB reserve. This **does not constitute direct 64 GB validation**; CPU expert storage, pinned buffers, OS memory, reclaimable file cache and transient load allocations still matter.

## SSD / model storage

The promoted profile expects the PLE table to be **NVMe-backed** rather than locked in RAM.

Measured local storage footprint:

```text
full model directory             ~88 GB
ngram_embedding.safetensors      ~31 GB (32,640,183,408 bytes)
```

Storage guidance:

- **100 GB free**: minimum space for this exact local model directory with little working margin.
- **120 GB+ free**: recommended so conversions, logs, temporary files, and metadata do not compete with the model.
- **NVMe SSD**: strongly recommended for `ngram_ram: false`.
- The validation host used a **Crucial T710 NVMe**. A PCIe 4.0 x4-class or faster NVMe is a sensible target, but the minimum SSD class has not yet been independently benchmarked in this repository.

The disk path still benefits from Linux page cache; `ngram_ram: false` means the full PLE table is not explicitly pinned as a permanent ~31 GB RAM allocation.

## Why PLE moved from RAM to NVMe

A controlled 40k-token C3 comparison was run with the rest of the profile fixed and dynamic MTP confidence 0.6 used to reduce draft-length noise.

Two-run averages:

| PLE backing | Aggregate decode |
| --- | ---: |
| explicit RAM (`ngram_ram: true`) | **111.35 tok/s** |
| NVMe / page-cache path (`ngram_ram: false`) | **110.4 tok/s** |

The difference was **under 1%**, smaller than the run-to-run MTP acceptance variance. Cold-ish 40k prefill was also effectively tied: roughly **1.59k tok/s** for RAM versus **1.61k tok/s** for the NVMe-backed run.

This does **not** prove that every SSD is equivalent to RAM. It shows that on the validated NVMe host, explicitly reserving another ~31 GB of system RAM for PLE did not buy a measurable end-to-end win large enough to justify the capacity cost.

## Dynamic MTP promotion

The mixed-workload default is now:

```yaml
draft_num_tokens: 3
dynamic_draft: true
draft_confidence: 0.4
```

In a short 142-token-prompt C3 steady-state check:

| Draft policy | Aggregate decode |
| --- | ---: |
| fixed MTP3 | **132.6 tok/s** |
| dynamic MTP3, confidence 0.4 | **151.9 tok/s** |
| dynamic MTP3, confidence 0.6 | **129.7 tok/s** |

The 0.4 result was about **14.6% above fixed MTP3** in that short-workload measurement, while the 40k workload stayed within the much larger run-to-run acceptance variance. Therefore `0.4` is promoted as a mixed-workload policy, not as a claim that dynamic drafting always improves every long-context request.

Dynamic drafting does not add another model forward solely to make the decision; it uses the existing draft confidence/acceptance calibration to truncate the MTP window when appropriate.

## Promoted resource profiles

```text
Common
GPU                     3 × RTX 5070 Ti 16 GB (validated)
System RAM              64 GB recommended minimum; 96 GB recommended; 128 GB validated
Local model storage     ~88 GB
PLE table               ~31 GB, NVMe-backed (`ngram_ram: false`)
Host memory reserve     8 GiB (`EXL3_HOST_MEM_RESERVE_MB=8192`)
Free SSD space           100 GB minimum, 120 GB+ recommended
Drafting                 MTP3, dynamic, confidence 0.4
Shared KV cache          524,288 tokens, mode `8,4`
Max sequence length      262,144

Performance mode
VRAM split              [15, 15, 14] GiB budget
CPU experts             208 / 512 per split MoE layer
MTP placement           GPU2 [0,0,3]

Headroom-preserving mode
VRAM split              [11, 15, 15] GiB budget
CPU experts             232 / 512 per split MoE layer
MTP placement           GPU0 [3,0,0]
```
