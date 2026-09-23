# Qwen3.8-Flash-Next on 3× RTX 5070 Ti — ExLlamaV3 레시피

[English](README.md) | **한국어** | [日本語](README.ja.md) | [简体中文](README.zh-CN.md)

**RTX 5070 Ti 16 GB ×3**에서 **ExLlamaV3 1.5.1**로 **Qwen3.8-Flash-Next EXL3 3.05 bpw**를 서빙하며 측정한 최적화 레시피입니다.

이 저장소는 일반적인 "최적 설정"을 주장하지 않습니다. 특정 하드웨어/모델/런타임 조합에서 실제로 효과가 있었던 최적화 과정, 실패한 설정, 그리고 실전 워크로드의 노이즈와 진짜 성능 향상을 구분하기 위해 사용한 벤치마크 방법을 기록합니다.

## TL;DR

검증한 호스트:

- NVIDIA GeForce RTX 5070 Ti 16 GB ×3
- 물리 PCIe 링크 구성: PCIe 5.0 x8 / x8 / x4
- 이론상 단방향 대역폭 약 31.5 / 31.5 / 15.75 GB/s
- 측정 당시 NVIDIA enumeration: GPU0 x8, GPU1 x4, GPU2 x8
- PCIe 5.0 신호 속도: lane당 32.0 GT/s; 모든 GPU 쌍은 `PHB` 경유; NVLink 없음
- Ryzen 9 9950X3D
- DDR5 128 GB (4 × 32 GB), DDR5-5800 CL40
- FCLK 2000 MHz, VSOC 약 1.05 V
- BIOS CPU power limit 약 110 W
- NVIDIA driver 615.71.09
- Linux 7.0.0-31-generic
- ExLlamaV3 1.5.1
- Qwen3.8-Flash-Next, EXL3 3.05 bpw
- 최대 sequence length 262,144
- shared cache budget 524,288 tokens
- cache mode `8,4`
- MTP 활성화, draft token 3개

레시피에는 3-GPU 모드를 두 가지로 유지합니다. 기본은 **성능 우선 모드**, 대안은 GPU0에 수 GiB의 여유 VRAM을 남기는 **헤드룸 확보 모드**입니다.

성능 우선 모드 (`recipe/tabby_config.yml`):

```text
gpu_split                  = [15.0, 15.0, 14.0]
cpu_moe_split_experts      = 208
cpu_moe_threads            = 24
ngram_ram                  = false

draft_mode                 = mtp
draft_gpu_split            = [0, 0, 3]
draft_num_tokens           = 3
dynamic_draft              = true
draft_confidence           = 0.4
```

헤드룸 확보 모드 (`recipe/tabby_config.headroom.yml`):

```text
gpu_split                  = [11.0, 15.0, 15.0]
cpu_moe_split_experts      = 232
cpu_moe_threads            = 24
ngram_ram                  = false

draft_mode                 = mtp
draft_gpu_split            = [3, 0, 0]
draft_num_tokens           = 3
dynamic_draft              = true
draft_confidence           = 0.4
```

두 모드 모두 아래 runtime policy를 공유합니다.

```text
EXL3_MOE_CPU_SWAP          = 0
EXL3_MOE_CPU_SPLIT_STATS   = /path/to/routing-stats.json
EXL3_MGEMM_N_THRESHOLD     = 2048
EXL3_INT8_GEMV             = 0
```

가장 큰 성능 향상은 **routing frequency 기반 static hot-expert placement**에서 나왔습니다. 마지막 커널 선택에서는 더 미묘한 상호작용이 있었습니다. 이 모델과 GPU 조합에서는 **2048-wide Gated DeltaNet bundle은 unfused 상태로 두고, INT8 activation GEMV는 끄는 조합**이 controlled C3에서 가장 빨랐습니다.

### 측정 당시 CPU / RAM 설정

이 호스트는 JEDEC 기본 메모리나 무제한 CPU 전력 프로파일로 돌린 것이 아닙니다.

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

24/16 worker split은 서빙 레시피의 일부입니다. BIOS/RAM 값은 해당 호스트의 실측 조건이지 모든 시스템의 보편적 최적값이라는 뜻은 아닙니다.

### 최종 promotion의 VRAM / RAM / SSD 요구사항

2026-09-24 실측으로 용량 요구가 크게 바뀌었습니다.

- **VRAM:** 검증된 요구사항은 **16 GB GPU ×3**입니다. clean load 직후 RTX 5070 Ti 3장의 사용량은 약 **14,456 / 14,746 / 15,240 MiB**였습니다.
- **System RAM:** **128 GB는 실측 검증됨**. PLE를 명시적 RAM 상주에서 제외한 뒤 live container는 약 **33.52 GiB**, host 전체 used는 약 **41 GiB**였습니다. 따라서 **96 GB+를 실용 권장치**로 보고, **64 GB는 미검증이며 CPU208 + 24 GiB host reserve를 고려하면 빡빡한 용량**으로 봅니다.
- **SSD:** local model directory가 약 **88 GB**, 그중 PLE n-gram table이 약 **31 GB**입니다. **빈 공간 100 GB 최소, 120 GB+ 권장**, NVMe 사용을 권장합니다. 검증 호스트는 Crucial T710 NVMe를 사용했습니다. PCIe 4.0 x4급 이상을 합리적 목표로 보지만, 최소 SSD 등급 자체를 별도로 A/B한 것은 아닙니다.

세부 실측과 주의사항은 [`docs/resource-requirements.md`](docs/resource-requirements.md)에 정리했습니다.

## 주요 결과

초기 실전 3-request aggregate decode는 약 **72–75 tok/s** 부근에서 막히는 것처럼 보였습니다.

최적화 후:

- 실제 mixed workload: 보통 **~94–101 tok/s aggregate C3**
- 좋은 실전 구간: **110+ tok/s**
- controlled C1: **평균 84.4 tok/s**
- controlled C2: **평균 aggregate 104.5 tok/s**
- controlled C3: **평균 aggregate 120.4 tok/s**
- 실전 워크로드에서 관측한 historical peak: 약 **119 tok/s**

controlled 결과와 production sustained throughput은 의도적으로 구분합니다. 고정 warm-cache benchmark가 실제 agent workload보다 prompt 길이, prefix cache 상태, draft acceptance, overlapping prefill 등의 변수를 훨씬 잘 통제하기 때문입니다.

## 왜 이 설정이 빨랐나

### 1. Static hot-expert placement

각 split MoE layer에서 512개 expert 중 232개를 CPU, 280개를 GPU에 두었을 때 단순한 tail placement는 CPU로 너무 많은 routing을 보냈습니다.

```text
hot placement 이전:
  CPU hit ~45.17%

GPU에 hottest 280 experts 배치 시:
  CPU hit ~12.06%
  GPU hit ~87.94%
```

실제 promotion 이후 decode handoff profiling에서는 **CPU expert assignment 1.436/token-row**, 즉 effective CPU hit 약 **14.4%**를 기록했습니다.

> **Upstream 한계:** 이 결과는 고정된 static profile이 잘 초기화된 dynamic policy보다 본질적으로 우월하다는 뜻이 아닙니다. 검증 시점인 **2026-09-23**에는 workload-trained placement에서 시작한 뒤 기존 upstream dynamic swapper를 계속 사용하는 `seed` 모드를 포함한 ExLlamaV3 PR [#315](https://github.com/turboderp-org/exllamav3/pull/315)이 아직 open 상태였고, 이 레시피가 사용한 release/upstream 경로에서는 사용할 수 없었습니다. 따라서 본 프로젝트는 기존 upstream dynamic placement와 custom static histogram placement는 비교했지만, **static vs seeded-dynamic의 공정한 A/B는 수행하지 못했습니다.** 해당 기능이 upstream에 들어오면 `static`과 `seed`를 다시 비교해야 합니다.

MTP layer도 비슷했습니다.

```text
MTP hot placement 이전:  ~44.3% CPU hit
MTP hot-placement 목표:   ~7.5% CPU hit
```

나중에 layer 47이 초기 histogram에서 빠져 있다는 것도 발견했습니다. 해당 layer의 unoptimized CPU hit는 약 **42.8%**, hot placement 추정치는 약 **7.4%**였습니다.

이 최적화가 전체 프로젝트에서 가장 큰 효과를 냈습니다.

### 2. MTP3가 실용적 최적점

MTP1/2/3/4를 모두 테스트했습니다.

- MTP1: MTP expert placement 수정 전에는 경쟁력 있었음
- MTP2-hot: acceptance가 충분히 올라가지 않아 real C3 약 **72.4 tok/s**
- **MTP3-hot: 최종 채택**
- MTP4: VRAM/headroom 비용이 커지고 실전 aggregate도 MTP3보다 낮았음

MTP3는 최종 **ceiling**으로 유지하되, mixed workload 기본값은 confidence 기반 dynamic truncation으로 바뀌었습니다.

```yaml
draft_num_tokens: 3
dynamic_draft: true
draft_confidence: 0.4
```

142-token prompt의 짧은 C3 steady-state 실측에서는 fixed MTP3가 **132.6 tok/s aggregate**, dynamic 0.4가 **151.9 tok/s**, dynamic 0.6이 **129.7 tok/s**였습니다. 40k long-context에서는 run-to-run acceptance variance 안에 머물렀으므로, 이 값은 mixed-workload 기본 정책이지 dynamic이 모든 긴 요청에서 항상 빠르다는 뜻은 아닙니다.

### 3. CPU MoE thread 수

9950X3D는 worker thread를 늘린다고 선형으로 빨라지지 않았습니다.

Main 48-layer CPU-MoE worker:

- 16 threads: 느림
- **24 threads: 최적**
- 32 threads: 훨씬 느림

MTP one-layer worker:

- **16 threads: 더 빠름**
- 24 threads: host-core contention으로 C3 저하

따라서 최종적으로:

```text
main CPU-MoE worker: 24 threads
MTP CPU-MoE worker:  16 threads
```

### 4. 최종 kernel policy는 "INT8이 더 빠르다"가 아니었음

관련 text decode width:

| Path | fusion heuristic이 보는 effective width |
| --- | ---: |
| attention QKV slices | 512 |
| MoE gate/up | 640 |
| Gated DeltaNet qkv+z slice | 2048 |

ExLlamaV3의 조건:

```text
fuse when out_features < EXL3_MGEMM_N_THRESHOLD
```

따라서 **2048 vs 2049**는 의미 있는 경계입니다.

- threshold 2048: 2048-wide GDN은 **unfused**
- threshold 2049: 같은 GDN은 **fused**
- 512 attention과 640 MoE gate/up은 둘 다 fused 유지

2×2 controlled C3 결과:

| GDN policy | INT8 activation GEMV | 3-run 평균 sum-TG |
| --- | --- | ---: |
| fused, threshold 2049 | on (`2`) | 112.3 tok/s |
| unfused, threshold 2048 | on (`2`) | 109.4 tok/s |
| **unfused, threshold 2048** | **off (`0`)** | **120.4 tok/s** |
| fused, threshold 2049 | off (`0`) | 112.8 tok/s |

최종 채택:

```bash
EXL3_MGEMM_N_THRESHOLD=2048
EXL3_INT8_GEMV=0
```

이 숫자를 raw GEMV kernel 자체가 7–10% 빨라졌다는 뜻으로 해석하면 안 됩니다. MTP draft acceptance도 run마다 움직였으므로, 측정된 차이에는 kernel cost와 numerical path가 speculative acceptance에 주는 영향이 함께 들어갈 수 있습니다.

또 중요한 교정이 하나 있었습니다. 자연 workload에서는 처음에 threshold 1024가 2048보다 훨씬 빨라 보였지만, 실제 tensor width를 확인하니 **1024와 2048은 관련 hot path에서 같은 fusion 결정을 내립니다.** 즉 그 차이는 workload noise였고, 최종 결론은 controlled A/B에서만 채택했습니다.

## Recipe

전체 Tabby/ExLlamaV3 model configuration:

```text
recipe/tabby_config.yml
```

최종 environment variable:

```text
recipe/env.sh.example
```

static expert placement용 routing histogram collector:

```text
tools/moe_hist_probe/sitecustomize.py
```

이 probe는 ExLlamaV3 내부 API에 의존합니다. 사용 전에 `docs/optimization-log.md`를 읽어주세요.

## 최소 실행 정책

```bash
export EXL3_MOE_CPU_SWAP=0
export EXL3_MOE_CPU_SPLIT_STATS=/absolute/path/to/qwen38-routing-stats.json
export EXL3_MGEMM_N_THRESHOLD=2048
export EXL3_INT8_GEMV=0

# recipe/tabby_config.yml을 사용해 TabbyAPI / ExLlamaV3 서버 시작
```

다른 사람의 routing histogram을 그대로 복사하지 마세요. expert popularity는 workload에 따라 달라질 수 있습니다. 본인 workload를 충분히 수집한 뒤 ranking을 고정하고 기존 placement와 A/B 검증하는 것을 권장합니다.

## Controlled benchmark

`bench/controlled_c3.py`는 concurrency를 바꿀 수 있습니다.

1. 고정 warm-up request 1회
2. 동일 warm-cache request를 `CONCURRENCY`개 동시 실행
3. request당 output 900 tokens
4. 반복성을 위해 greedy decoding 사용

검증 머신에서 고정 prompt는 약 40k tokens입니다.

```bash
CONCURRENCY=1 python3 bench/controlled_c3.py
CONCURRENCY=2 python3 bench/controlled_c3.py
CONCURRENCY=3 python3 bench/controlled_c3.py
```

server-side ExLlamaV3/Tabby log에서 각 request의 TG를 확인하고, 해당 run에서 동시에 decode한 request들의 TG를 합산합니다.

## 효과 없었던 것들

- MTP2-hot: acceptance 향상이 부족
- MTP4: VRAM/headroom 비용 증가 + 실전 aggregate 저하
- base CPU-MoE 16 threads: 느림
- base CPU-MoE 32 threads: 훨씬 느림
- MTP CPU-MoE 24 threads: contention으로 악화
- 과거 `[11,15,15]` + GPU0 MTP 배치에서는 CPU expert split 228/224도 load 실패했지만, 이후 MTP를 GPU2로 옮기고 `[15,15,14]`로 재배치해 **CPU208**까지 fit
- `gpu_split [11,15.5,14.5]`: 과거 module/VRAM boundary에서 실패
- `gpu_split [11,15.25,14.75]`: 동일
- `MGEMM_N_THRESHOLD=0`: 실전 workload에서 좋지 않거나 noise가 큼
- promoted unfused GDN path에서 INT8 activation GEMV: controlled C3에서 더 느림
- 자연 traffic 한 번의 결과를 proof로 간주: controlled A/B에서 여러 apparent win이 사라짐

자세한 순서와 측정은 `docs/optimization-log.md` 참고.

## 현재 3-GPU residency 경계

최종 promotion split은 다음과 같습니다.

```yaml
gpu_split: [15.0, 15.0, 14.0]
cpu_moe_split_experts: 208
draft_gpu_split: [0, 0, 3]
```

이 값은 **성능 우선 모드**입니다. 검증 호스트는 desktop을 iGPU로 출력하므로 RTX 5070 Ti 3장을 compute 전용으로 사용할 수 있습니다. clean load 기준 대략 14.1 / 14.4 / 14.9 GiB를 사용했고 세 번째 GPU가 가장 빡빡한 VRAM 경계였습니다.

GPU0에 의도적으로 여유 VRAM이 필요한 시스템은 별도의 **헤드룸 확보 모드**인 `recipe/tabby_config.headroom.yml`을 사용하면 됩니다. 이 모드는 `[11,15,15]`, CPU232, MTP GPU0 배치를 유지하면서도 새로 채택한 SSD PLE 및 dynamic MTP 정책은 그대로 사용합니다.

네 번째 RTX 5060 Ti는 최종 서버 구성에 포함하지 않습니다. 테스트한 PHB/no-P2P topology에서는 4번째 stage를 추가했을 때 expert residency가 늘어도 throughput이 오히려 내려갔습니다. 따라서 live promotion은 RTX 5070 Ti 3장만 노출합니다.

## 재현성 주의사항

- 사용 모델은 Qwen3.8-Flash-Next의 **EXL3 3.05 bpw conversion**
- model weight는 이 저장소에 포함하지 않음
- upstream model license는 별도 적용
- 작성자 runtime image에는 CPU idle-parking용 로컬 patch가 있었지만 throughput tuning의 핵심은 아님
- static hot-expert ranking은 workload dependent
- CUDA/driver/runtime 업데이트로 kernel break-even point가 달라질 수 있음
- GPU 세대, EXL3 bpw, ExLlamaV3 버전, expert residency, cache size, batch/concurrency를 바꾸면 controlled A/B를 다시 수행할 것

## 저장소 구조

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
│   ├── tabby_config.yml                 # 성능 우선 모드
│   └── tabby_config.headroom.yml        # 헤드룸 확보 모드
└── tools/
    └── moe_hist_probe/
        └── sitecustomize.py
```

## License

이 저장소의 코드와 문서는 MIT License로 배포합니다. 모델 weight는 포함하지 않으며 해당 모델의 별도 license를 따릅니다.
