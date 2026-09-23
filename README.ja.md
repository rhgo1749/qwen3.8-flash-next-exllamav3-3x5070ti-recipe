# Qwen3.8-Flash-Next on 3× RTX 5070 Ti — ExLlamaV3 レシピ

[English](README.md) | [한국어](README.ko.md) | **日本語** | [简体中文](README.zh-CN.md)

**RTX 5070 Ti 16 GB ×3** 上で **ExLlamaV3 1.5.1** を使い、**Qwen3.8-Flash-Next EXL3 3.05 bpw** を実測・最適化したサービングレシピです。

このリポジトリは一般的な「最適設定」を主張するものではありません。特定のハードウェア / モデル / ランタイム構成で有効だった最適化、失敗した設定、実ワークロードのノイズと本当の性能向上を切り分けるために使ったベンチマーク方法を記録しています。

## TL;DR

検証ホスト:

- NVIDIA GeForce RTX 5070 Ti 16 GB ×3
- 物理 PCIe 構成: PCIe 5.0 x8 / x8 / x4
- 理論上の片方向帯域幅: 約 31.5 / 31.5 / 15.75 GB/s
- 測定時の NVIDIA enumeration: GPU0 x8, GPU1 x4, GPU2 x8
- PCIe 5.0 signaling rate: 1 lane あたり 32.0 GT/s; 全 GPU ペアは `PHB` 経由; NVLink なし
- Ryzen 9 9950X3D
- DDR5 128 GB (4 × 32 GB), DDR5-5800 CL40
- FCLK 2000 MHz, VSOC 約 1.05 V
- BIOS CPU power limit 約 110 W
- NVIDIA driver 615.71.09
- Linux 7.0.0-31-generic
- ExLlamaV3 1.5.1
- Qwen3.8-Flash-Next, EXL3 3.05 bpw
- 最大 sequence length 262,144
- shared cache budget 524,288 tokens
- cache mode `8,4`
- MTP 有効、draft token 3

採用設定:

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

最大の改善は **routing frequency に基づく static hot-expert placement** でした。最後の kernel 選択ではさらに微妙な相互作用があり、このモデル / GPU 構成では **2048-wide Gated DeltaNet bundle を unfused のままにし、INT8 activation GEMV を無効化する組み合わせ**が controlled C3 で最速でした。

### 測定時の CPU / RAM 設定

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

24/16 worker split はレシピの一部です。BIOS / RAM 値はこのホストの実測条件であり、他の AM5 システムでの普遍的な最適値ではありません。

## 主な結果

初期の実ワークロードでは 3-request aggregate decode が **72–75 tok/s** 付近で頭打ちに見えました。

最適化後:

- 実 mixed workload: 通常 **~94–101 tok/s aggregate C3**
- 良好な実ワークロード区間: **110+ tok/s**
- controlled C1: **平均 84.4 tok/s**
- controlled C2: **平均 aggregate 104.5 tok/s**
- controlled C3: **平均 aggregate 120.4 tok/s**
- 実ワークロードで観測した historical peak: 約 **119 tok/s**

controlled benchmark と production sustained throughput は意図的に分けて記載しています。

## なぜ速くなったのか

### 1. Static hot-expert placement

各 split MoE layer で 512 experts のうち CPU 232 / GPU 280 とした場合、単純な tail placement では CPU への routing が多すぎました。

```text
hot placement 前:
  CPU hit ~45.17%

最も hot な 280 experts を GPU に置いた場合:
  CPU hit ~12.06%
  GPU hit ~87.94%
```

promotion 後の decode handoff profiling では **1.436 CPU expert assignments/token-row**、effective CPU hit 約 **14.4%** でした。

> **Upstream の制約:** この結果は、固定 static profile が十分に初期化された dynamic policy より本質的に優れていることを示すものではありません。検証時点の **2026-09-23** では、workload-trained placement から開始した後も既存 upstream dynamic swapper を継続利用する `seed` mode を含む ExLlamaV3 PR [#315](https://github.com/turboderp-org/exllamav3/pull/315) はまだ open で、この recipe が使用した release/upstream path では利用できませんでした。そのため既存 upstream dynamic placement と custom static histogram placement は比較できましたが、**static と seeded-dynamic の公正な A/B は実施できていません。** upstream に取り込まれた後に `static` と `seed` を再比較する必要があります。

MTP layer でも:

```text
MTP hot placement 前:  ~44.3% CPU hit
MTP hot-placement target: ~7.5% CPU hit
```

さらに layer 47 が初期 histogram から欠落していることを発見し、約 **42.8% → 7.4%** の改善余地を確認しました。

これが全体で最も大きな最適化でした。

### 2. MTP3 が実用上の最適点

MTP1/2/3/4 を比較しました。

- MTP1: MTP expert placement 修正前は競争力あり
- MTP2-hot: acceptance が足りず real C3 約 **72.4 tok/s**
- **MTP3-hot: 採用**
- MTP4: VRAM/headroom コストが増え、実 aggregate も MTP3 より悪化

```yaml
draft_num_tokens: 3
dynamic_draft: false
```

### 3. CPU MoE thread 数

Main 48-layer CPU-MoE worker:

- 16 threads: 遅い
- **24 threads: 最良**
- 32 threads: さらに遅い

MTP one-layer worker:

- **16 threads: 最良**
- 24 threads: host-core contention で C3 低下

```text
main CPU-MoE worker: 24 threads
MTP CPU-MoE worker:  16 threads
```

### 4. 最終 kernel policy は「INT8 の方が速い」ではなかった

関連する text decode width:

| Path | fusion heuristic が見る effective width |
| --- | ---: |
| attention QKV slices | 512 |
| MoE gate/up | 640 |
| Gated DeltaNet qkv+z slice | 2048 |

ExLlamaV3 の条件:

```text
fuse when out_features < EXL3_MGEMM_N_THRESHOLD
```

したがって **2048 vs 2049** が有効な境界テストになります。

2×2 controlled C3:

| GDN policy | INT8 activation GEMV | 3-run 平均 sum-TG |
| --- | --- | ---: |
| fused, threshold 2049 | on (`2`) | 112.3 tok/s |
| unfused, threshold 2048 | on (`2`) | 109.4 tok/s |
| **unfused, threshold 2048** | **off (`0`)** | **120.4 tok/s** |
| fused, threshold 2049 | off (`0`) | 112.8 tok/s |

採用:

```bash
EXL3_MGEMM_N_THRESHOLD=2048
EXL3_INT8_GEMV=0
```

これは raw non-INT8 GEMV kernel 単体が 7–10% 速いという意味ではありません。MTP draft acceptance も変動しているため、end-to-end throughput には kernel cost と speculative acceptance の両方が影響します。

また、自然ワークロードでは一度 threshold 1024 が 2048 より大幅に速く見えましたが、tensor width を確認すると **1024 と 2048 は関連 hot path で同じ fusion decision** をします。これは workload noise でした。

## Recipe

```text
recipe/tabby_config.yml
recipe/env.sh.example
tools/moe_hist_probe/sitecustomize.py
```

routing histogram probe は ExLlamaV3 内部 API に依存します。利用前に `docs/optimization-log.md` を参照してください。

## 最小 launch policy

```bash
export EXL3_MOE_CPU_SWAP=0
export EXL3_MOE_CPU_SPLIT_STATS=/absolute/path/to/qwen38-routing-stats.json
export EXL3_MGEMM_N_THRESHOLD=2048
export EXL3_INT8_GEMV=0

# recipe/tabby_config.yml で TabbyAPI / ExLlamaV3 server を起動
```

他人の routing histogram をそのままコピーしないでください。expert popularity は workload dependent です。

## Controlled benchmark

```bash
CONCURRENCY=1 python3 bench/controlled_c3.py
CONCURRENCY=2 python3 bench/controlled_c3.py
CONCURRENCY=3 python3 bench/controlled_c3.py
```

検証ホストでは固定 prompt は約 40k tokens、各 request は 900 output tokens、greedy decoding です。

## 効果がなかったもの

- MTP2-hot: acceptance 改善不足
- MTP4: VRAM/headroom コスト増 + 実 aggregate 悪化
- base CPU-MoE 16 threads: 遅い
- base CPU-MoE 32 threads: さらに遅い
- MTP CPU-MoE 24 threads: contention で悪化
- CPU expert split 228/224: 524k cache budget で load 失敗
- `gpu_split [11,15.5,14.5]`: module/VRAM boundary で失敗
- `gpu_split [11,15.25,14.75]`: 同様
- `MGEMM_N_THRESHOLD=0`: 実ワークロードで不安定 / 不利
- promoted unfused GDN path で INT8 activation GEMV: controlled C3 で低速
- 自然 traffic 1 回だけを proof とすること: controlled A/B で複数の apparent win が消えた

## GPU0 headroom は意図的

```yaml
gpu_split: [11.0, 15.0, 15.0]
```

最大 throughput 専用構成ではありません。GPU0 に数 GiB の VRAM headroom を意図的に残しているため、その GPU をディスプレイ接続や通常の desktop / OS workload にも使うシステムに向いています。GPU0 を完全に headless で使う場合は、11/15/15 をそのまま採用せず split を再調整する余地があります。

## 再現性に関する注意

- モデルは Qwen3.8-Flash-Next の **EXL3 3.05 bpw conversion**
- model weights は含まれません
- upstream model license は別途適用
- static hot-expert ranking は workload dependent
- CUDA/driver/runtime 更新で kernel break-even point は変わる可能性あり
- GPU 世代、EXL3 bpw、ExLlamaV3 version、expert residency、cache size、batch/concurrency を変更した場合は controlled A/B を再実行してください

## License

このリポジトリのコードとドキュメントは MIT License です。model weights は含みません。
