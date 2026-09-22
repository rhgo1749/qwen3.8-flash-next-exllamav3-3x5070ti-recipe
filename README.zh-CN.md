# Qwen3.8-Flash-Next on 3× RTX 5070 Ti — ExLlamaV3 配置与优化方案

[English](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | **简体中文**

这是一个经过实测的 **Qwen3.8-Flash-Next EXL3 3.05 bpw** 推理优化方案，运行于 **3× RTX 5070 Ti 16 GB**，使用 **ExLlamaV3 1.5.1**。

本仓库并不声称这些参数是通用“最佳设置”。它记录的是一个具体硬件 / 模型 / 运行时组合中真正有效的优化、失败的尝试，以及用于区分真实性能提升与实际工作负载噪声的基准方法。

## TL;DR

验证主机:

- NVIDIA GeForce RTX 5070 Ti 16 GB ×3
- 物理 PCIe 布局: PCIe 5.0 x8 / x8 / x4
- 理论单向带宽约: 31.5 / 31.5 / 15.75 GB/s
- 测试时 NVIDIA enumeration: GPU0 x8, GPU1 x4, GPU2 x8
- PCIe 5.0 signaling rate: 每 lane 32.0 GT/s；所有 GPU 对均经过 `PHB`；无 NVLink
- Ryzen 9 9950X3D
- DDR5 128 GB (4 × 32 GB), DDR5-5800 CL40
- FCLK 2000 MHz, VSOC ~1.05 V
- BIOS CPU power limit ~110 W
- NVIDIA driver 615.71.09
- Linux 7.0.0-31-generic
- ExLlamaV3 1.5.1
- Qwen3.8-Flash-Next, EXL3 3.05 bpw
- 最大 sequence length 262,144
- shared cache budget 524,288 tokens
- cache mode `8,4`
- 启用 MTP，3 个 draft tokens

最终采用的配置:

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

最大的性能提升来自 **基于 routing frequency 的 static hot-expert placement**。最终 kernel 选择还有一个更微妙的交互：在这套模型和 GPU 上，controlled C3 中最快的组合是 **让 2048-wide Gated DeltaNet bundle 保持 unfused，同时关闭 INT8 activation GEMV**。

### 测试时的 CPU / RAM 设置

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

24/16 worker split 是推理方案的一部分；BIOS / RAM 参数只是这台主机的实测条件，并不代表其他 AM5 系统上的通用最优值。

## 主要结果

初始实际工作负载下，3-request aggregate decode 看起来卡在 **72–75 tok/s** 左右。

优化后:

- 实际 mixed workload: 通常 **~94–101 tok/s aggregate C3**
- 较好的实际区间: **110+ tok/s**
- controlled C1: **平均 84.4 tok/s**
- controlled C2: **平均 aggregate 104.5 tok/s**
- controlled C3: **平均 aggregate 120.4 tok/s**
- 实际工作负载中观察到的 historical peak: 约 **119 tok/s**

controlled benchmark 与 production sustained throughput 是刻意分开描述的，因为固定 warm-cache benchmark 更容易控制 prompt 长度、prefix cache、draft acceptance 和 overlapping prefill 等变量。

## 为什么这个配置更快

### 1. Static hot-expert placement

每个 split MoE layer 有 512 个 experts，其中 232 个在 CPU，280 个驻留 GPU。简单的 tail placement 导致过多 routing 落到 CPU。

```text
hot placement 之前:
  CPU hit ~45.17%

把最热的 280 experts 放到 GPU 后:
  CPU hit ~12.06%
  GPU hit ~87.94%
```

promotion 后的 decode handoff profiling 测得 **1.436 CPU expert assignments/token-row**，对应 effective CPU hit 约 **14.4%**。

MTP layer 也存在类似问题:

```text
MTP hot placement 前:  ~44.3% CPU hit
MTP hot-placement 目标: ~7.5% CPU hit
```

后续还发现 layer 47 没有进入最初 histogram。该层未优化时 CPU hit 约 **42.8%**，hot placement 后估计约 **7.4%**。

这是整个项目中收益最大的优化。

### 2. MTP3 是实际最优点

测试了 MTP1/2/3/4:

- MTP1: 修复 MTP expert placement 之前有竞争力
- MTP2-hot: acceptance 提升不足，real C3 约 **72.4 tok/s**
- **MTP3-hot: 最终采用**
- MTP4: VRAM/headroom 成本更高，实际 aggregate 也低于 MTP3

```yaml
draft_num_tokens: 3
dynamic_draft: false
```

### 3. CPU MoE thread 数

Main 48-layer CPU-MoE worker:

- 16 threads: 更慢
- **24 threads: 最佳**
- 32 threads: 明显更慢

MTP one-layer worker:

- **16 threads: 最佳**
- 24 threads: 因 host-core contention 导致 C3 下降

```text
main CPU-MoE worker: 24 threads
MTP CPU-MoE worker:  16 threads
```

### 4. 最终 kernel policy 并不是“INT8 一定更快”

相关 text decode width:

| Path | fusion heuristic 使用的 effective width |
| --- | ---: |
| attention QKV slices | 512 |
| MoE gate/up | 640 |
| Gated DeltaNet qkv+z slice | 2048 |

ExLlamaV3 条件:

```text
fuse when out_features < EXL3_MGEMM_N_THRESHOLD
```

因此 **2048 vs 2049** 是一个很干净的边界测试。

2×2 controlled C3:

| GDN policy | INT8 activation GEMV | 3-run 平均 sum-TG |
| --- | --- | ---: |
| fused, threshold 2049 | on (`2`) | 112.3 tok/s |
| unfused, threshold 2048 | on (`2`) | 109.4 tok/s |
| **unfused, threshold 2048** | **off (`0`)** | **120.4 tok/s** |
| fused, threshold 2049 | off (`0`) | 112.8 tok/s |

最终采用:

```bash
EXL3_MGEMM_N_THRESHOLD=2048
EXL3_INT8_GEMV=0
```

这并不意味着 raw non-INT8 GEMV kernel 本身快了 7–10%。MTP draft acceptance 也会变化，因此 end-to-end throughput 可能同时受到 kernel cost 和 numerical path 对 speculative acceptance 的影响。

还有一个重要修正：在自然工作负载中，threshold 1024 一度看起来比 2048 快很多。但检查实际 tensor width 后发现，**1024 和 2048 在相关 hot path 上做出的 fusion decision 相同**，所以那次差异实际上是 workload noise。

## Recipe

```text
recipe/tabby_config.yml
recipe/env.sh.example
tools/moe_hist_probe/sitecustomize.py
```

routing histogram probe 依赖 ExLlamaV3 内部 API。使用前请阅读 `docs/optimization-log.md`。

## 最小启动策略

```bash
export EXL3_MOE_CPU_SWAP=0
export EXL3_MOE_CPU_SPLIT_STATS=/absolute/path/to/qwen38-routing-stats.json
export EXL3_MGEMM_N_THRESHOLD=2048
export EXL3_INT8_GEMV=0

# 使用 recipe/tabby_config.yml 启动 TabbyAPI / ExLlamaV3 server
```

不要直接复制别人的 routing histogram。expert popularity 与 workload 有关，应当采集自己的代表性流量并做 A/B 验证。

## Controlled benchmark

```bash
CONCURRENCY=1 python3 bench/controlled_c3.py
CONCURRENCY=2 python3 bench/controlled_c3.py
CONCURRENCY=3 python3 bench/controlled_c3.py
```

验证主机上的固定 prompt 约为 40k tokens，每个 request 输出 900 tokens，使用 greedy decoding。

## 没有帮助的尝试

- MTP2-hot: acceptance 提升不足
- MTP4: VRAM/headroom 成本增加，实际 aggregate 更差
- base CPU-MoE 16 threads: 更慢
- base CPU-MoE 32 threads: 明显更慢
- MTP CPU-MoE 24 threads: contention 导致退化
- CPU expert split 228/224: 在 524k cache budget 下加载失败
- `gpu_split [11,15.5,14.5]`: 卡在 module/VRAM boundary
- `gpu_split [11,15.25,14.75]`: 同样失败
- `MGEMM_N_THRESHOLD=0`: 实际 workload 表现差或噪声很大
- promoted unfused GDN path 上的 INT8 activation GEMV: controlled C3 更慢
- 把一次自然 traffic 当作 proof: 多个 apparent win 在 controlled A/B 中消失

## GPU0 headroom 是有意保留的

```yaml
gpu_split: [11.0, 15.0, 15.0]
```

这不是纯最大吞吐量配置。该 split 会在 GPU0 上有意保留数 GiB VRAM，因此更适合 GPU0 同时承担显示输出或普通 desktop / OS workload 的系统。如果 GPU0 完全 headless 使用，不必照搬 11/15/15，可以重新搜索更激进的 split。

## 可复现性说明

- 使用的是 Qwen3.8-Flash-Next 的 **EXL3 3.05 bpw conversion**
- 仓库不包含 model weights
- upstream model license 单独适用
- static hot-expert ranking 与 workload 有关
- CUDA/driver/runtime 更新可能改变 kernel break-even point
- 更换 GPU 世代、EXL3 bpw、ExLlamaV3 版本、expert residency、cache size、batch/concurrency 后，应重新进行 controlled A/B

## License

本仓库代码与文档使用 MIT License。model weights 不包含在本仓库中。
