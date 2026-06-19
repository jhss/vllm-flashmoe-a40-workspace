네. 레포 기준으로 보면 **핵심 수정 위치는 `deepep_ht.py`가 아니라 `experts/triton_moe.py`**입니다.

이미 레포에 `VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS` 실험이 있어서 DeepEP의 local expert ID를 global ID로 되돌리는 단계는 어느 정도 제거해 보셨습니다. 그런데 이후에도 `TritonExperts.apply()`가 범용 `_prepare_expert_assignment()`를 호출하기 때문에, 결국 generic `moe_align_block_size`가 `topk_ids`를 다시 스캔합니다. 실험 문서에도 local-ID 경로만으로는 개선되지 않았고, 다음 단계로 DeepEP HT 전용 assignment kernel이 필요하다고 정리돼 있습니다. ([GitHub][1])

## 현재 레포의 실행 흐름

현재 코드를 단순화하면 다음과 같습니다.

```text
prepare_finalize/deepep_ht.py
    DeepEP dispatch/receive
    ├─ expert_x
    ├─ expert_topk_ids
    ├─ expert_topk_weights
    └─ expert_num_tokens_per_expert_list
              ↓
modular_kernel.py
    ExpertTokensMetadata 생성
              ↓
experts/triton_moe.py
    TritonExperts.apply()
              ↓
fused_moe.py
    _prepare_expert_assignment()
              ↓
moe_align_block_size.py
    sorted_token_ids
    expert_ids
    num_tokens_post_padded 생성
              ↓
A100 또는 일반 Triton MoE GEMM
```

DeepEP receiver는 이미 local expert ID와 expert별 token count를 얻지만, `TritonExperts.apply()`에서 다시 범용 assignment 경로를 호출합니다. 현재 receiver는 local-ID 실험이 켜졌을 때 `-1`을 local sentinel로 바꾸고, 그렇지 않으면 local ID를 global expert ID로 offset합니다. 또 `expert_num_tokens_per_expert_list`를 `ExpertTokensMetadata`로 변환합니다. ([GitHub][2])

---

# 수정해야 할 파일

## 1. 가장 중요한 파일

```text
vllm/vllm/model_executor/layers/fused_moe/experts/triton_moe.py
```

여기가 **실제 direct assignment 경로를 선택해야 하는 위치**입니다.

현재 `TritonExperts.apply()` 내부에서 대략 이런 호출을 합니다.

```python
sorted_token_ids, expert_ids, num_tokens_post_padded = (
    _prepare_expert_assignment(
        topk_ids,
        config,
        global_num_experts,
        expert_map,
        ...
    )
)
```

그리고 W1이나 W2용 tuning config의 `BLOCK_SIZE_M`이 달라지면 assignment를 다시 만들기도 합니다. 즉 같은 forward에서 base/W1/W2 schedule 생성이 중복될 가능성도 있습니다. ([GitHub][3])

이 호출을 다음과 같은 helper로 감싸는 것이 좋습니다.

```python
def prepare_assignment_for_config(
    topk_ids,
    config,
    expert_tokens_meta,
    global_num_experts,
    expert_map,
    ...
):
    if (
        envs.VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT
        and expert_tokens_meta.assignment_layout
            == ExpertAssignmentLayout.DEEPEP_HT_LOCAL
    ):
        return deepep_ht_prepare_expert_assignment(
            topk_ids=topk_ids,
            expert_num_tokens=expert_tokens_meta.expert_num_tokens,
            block_size_m=config["BLOCK_SIZE_M"],
            num_local_experts=expert_tokens_meta.expert_num_tokens.numel(),
            ...
        )

    return _prepare_expert_assignment(
        topk_ids,
        config,
        global_num_experts,
        expert_map,
        ...
    )
```

그리고 현재 `_prepare_expert_assignment()`를 직접 부르는 모든 위치를 이 helper로 바꿉니다.

### `BLOCK_SIZE_M`별 cache도 같이 넣는 것이 좋습니다

assignment 결과는 GEMM의 `BLOCK_SIZE_N`, `BLOCK_SIZE_K`와는 관계없고 `BLOCK_SIZE_M`에 의해 달라집니다.

따라서 다음처럼 cache할 수 있습니다.

```python
assignment_cache = {}

def get_assignment(config):
    block_m = config["BLOCK_SIZE_M"]

    if block_m not in assignment_cache:
        assignment_cache[block_m] = prepare_assignment_for_config(
            topk_ids=topk_ids,
            config=config,
            expert_tokens_meta=expert_tokens_meta,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )

    return assignment_cache[block_m]
```

그러면 W1과 W2 config가 달라도 `BLOCK_SIZE_M`이 같다면 동일한 schedule을 재사용할 수 있습니다.

---

## 2. 새로운 direct assignment kernel 파일

새 파일을 하나 추가하는 것이 가장 깔끔합니다.

```text
vllm/vllm/model_executor/layers/fused_moe/deepep_ht_expert_assignment.py
```

또는:

```text
vllm/vllm/model_executor/layers/fused_moe/experts/deepep_ht_assignment.py
```

첫 번째 위치를 추천합니다. DeepEP prepare 단계와 expert backend의 경계에서 사용하는 공용 코드이기 때문입니다.

함수 계약은 다음 정도가 좋습니다.

```python
def deepep_ht_prepare_expert_assignment(
    topk_ids: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    block_size_m: int,
    num_local_experts: int,
    ignore_invalid_experts: bool,
) -> tuple[
    torch.Tensor,  # sorted_token_ids
    torch.Tensor,  # expert_ids
    torch.Tensor,  # num_tokens_post_padded
]:
    ...
```

입력과 출력은 다음과 같습니다.

```text
입력
  topk_ids:
    [M_recv, top_k]
    local expert ID 또는 invalid sentinel

  expert_num_tokens:
    [num_local_experts]
    DeepEP가 알려준 local expert별 token 수

  block_size_m:
    현재 W1 또는 W2 GEMM config의 BLOCK_SIZE_M

출력
  sorted_token_ids
  expert_ids
  num_tokens_post_padded
```

기존 `moe_align_block_size()`와 **동일한 출력 ABI**를 만들면, 뒤의 A100 MoE kernel이나 일반 Triton GEMM 코드는 바꾸지 않아도 됩니다. 현재 A100 전용 kernel도 이미 이 세 metadata를 입력으로 받습니다. ([GitHub][4])

---

# direct kernel이 실제로 해야 할 일

## 1단계: expert별 시작 위치 계산

DeepEP count가 다음과 같다고 합시다.

```text
expert_num_tokens = [5, 0, 13, 7]
BLOCK_SIZE_M = 8
```

padding 후 크기는:

```text
expert 0: 5  → 8
expert 1: 0  → 0
expert 2: 13 → 16
expert 3: 7  → 8
```

prefix sum을 계산하면:

```text
expert_start = [0, 8, 8, 24]
num_tokens_post_padded = 32
```

동시에 block별 `expert_ids`를 만들 수 있습니다.

```text
expert_ids = [
    0,       # sorted_token_ids[0:8]
    2, 2,    # sorted_token_ids[8:24]
    3,       # sorted_token_ids[24:32]
]
```

## 2단계: token-expert pair를 각 expert 영역에 쓰기

`top_k=8`이면 flattened pair ID는 다음과 같습니다.

```python
pair_id = token_id * 8 + topk_slot
```

GPU kernel은 `topk_ids`를 스캔하면서:

```python
expert = topk_ids[token_id, topk_slot]

if expert is valid:
    position = atomic_add(write_cursor[expert], 1)
    sorted_token_ids[position] = pair_id
```

를 수행합니다.

순서가 안정적일 필요는 없습니다. 같은 expert에 속한 pair들이 그 expert 영역 안에만 모이면 됩니다.

## 3단계: padding slot 채우기

padding 위치에는 기존 GEMM이 invalid로 판정할 수 있는 sentinel을 넣습니다.

```python
padding_pair_id = topk_ids.numel()
```

기존 fused MoE GEMM은 `sorted_token_ids`에서 pair ID를 읽고:

```text
input token row = pair_id // top_k
router weight   = flattened_topk_weights[pair_id]
```

형태로 접근합니다. 따라서 schedule kernel이 `topk_weights`를 별도로 정렬하거나 복사할 필요가 없습니다. ([GitHub][5])

여기서 앞의 설명을 한 가지 정정하면:

> direct assignment kernel이 실제로 필요한 것은 `topk_ids`와 expert count입니다. `topk_weights`는 schedule 생성 입력으로 필요하지 않고, GEMM이 `sorted_token_ids`의 원래 pair ID를 이용해 기존 weight tensor에서 직접 읽습니다.

---

# invalid expert 처리가 중요합니다

DeepEP receive의 `topk_ids`에는 현재 rank가 처리하지 않는 slot이 존재합니다.

레포의 local-ID 실험은 이를 다음과 같이 바꿉니다.

```text
유효 local expert: 0 ... E_local - 1
invalid slot:      E_local
```

그리고 identity map의 마지막 값을 `-1`로 두어 invalid expert로 처리합니다. ([GitHub][2])

direct scheduler에서는 두 모드를 지원하는 것이 안전합니다.

### 일반 모드

invalid pair도 pseudo-expert 영역으로 묶고, 해당 block의 `expert_ids`를 `-1`로 기록합니다.

```python
invalid_count = topk_ids.numel() - expert_num_tokens.sum()
```

이 방식은 기존 GEMM의 “invalid expert block은 output을 0으로 기록”하는 의미를 보존하기 쉽습니다.

### invalid-skip 최적화 모드

레포에 이미 있는 `VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS` 경로와 함께 사용할 때는 invalid pair를 schedule에서 완전히 제외할 수 있습니다.

다만 invalid pair를 제외하면 W1/W2 intermediate의 해당 slot이 쓰이지 않으므로, masked activation과 masked reduce 경로가 반드시 함께 활성화되어야 합니다. 따라서 첫 구현에서는 **invalid pseudo-expert block을 포함하는 안전한 버전부터** 만드는 것을 추천합니다.

---

## 3. DeepEP receiver 수정

파일:

```text
vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py
```

이 파일에서 final GEMM schedule을 만들면 안 됩니다.

이유는 DeepEP receiver 시점에는 어떤 `BLOCK_SIZE_M`을 사용할지 확정되지 않았기 때문입니다. 실제로 현재 레포에서는 W1과 W2 config가 다르면 각각 다른 `BLOCK_SIZE_M`으로 assignment를 다시 만들 수 있습니다. ([GitHub][1])

여기서 해야 할 일은 두 가지뿐입니다.

```text
1. local expert ID를 유지한다.
2. 이 metadata가 DeepEP HT local layout이라는 사실을 표시한다.
```

예:

```python
use_direct_assignment = envs.VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT
use_local_ids = (
    use_direct_assignment
    or envs.VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS
)

if use_local_ids:
    # 기존 local-ID 실험 경로 재사용:
    # - local ID 유지
    # - -1은 local sentinel E_local로 변경
    expert_topk_ids = remap_deepep_ht_topk_ids(
        expert_topk_ids,
        rank_expert_offset=0,
        invalid_expert_id=num_local_experts,
    )

expert_tokens_meta = ExpertTokensMetadata.make_from_list(
    expert_num_tokens_per_expert_list,
    device=expert_x.device,
    assignment_layout=ExpertAssignmentLayout.DEEPEP_HT_LOCAL,
)
```

즉 기존 `VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS` 구현을 버리는 것이 아니라, **direct scheduler의 전처리 단계로 재사용**하면 됩니다.

---

## 4. metadata에 layout 정보 추가

파일:

```text
vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py
```

현재 `ExpertTokensMetadata`는 주로 다음 정보를 갖습니다.

```python
expert_num_tokens
expert_num_tokens_cpu
```

그리고 `make_from_list()`는 Python list로부터 CPU tensor를 만든 뒤 device tensor로 복사합니다. ([GitHub][6])

여기에 다음과 같은 layout 표시를 추가합니다.

```python
from enum import Enum, auto


class ExpertAssignmentLayout(Enum):
    GENERIC = auto()
    DEEPEP_HT_LOCAL = auto()


@dataclass
class ExpertTokensMetadata:
    expert_num_tokens: torch.Tensor
    expert_num_tokens_cpu: torch.Tensor | None
    assignment_layout: ExpertAssignmentLayout = (
        ExpertAssignmentLayout.GENERIC
    )
```

단순한 prototype이면:

```python
is_deepep_ht_local: bool = False
```

로 해도 됩니다. 다만 upstream PR까지 고려하면 enum이 더 명확합니다.

`sorted_token_ids` 자체를 이 metadata에 저장하는 것은 추천하지 않습니다. schedule은 `BLOCK_SIZE_M`에 의존하므로 receiver 단계가 아니라 `TritonExperts.apply()`에서 생성해야 하기 때문입니다.

---

## 5. 환경변수 추가

파일:

```text
vllm/vllm/envs.py
```

다음 flag를 추가합니다.

```text
VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT=1
```

의미는 다음과 같이 잡는 것이 좋습니다.

```text
DeepEP HT local IDs 사용
+
count-aware direct assignment 사용
+
지원하지 않는 조건에서는 generic path로 fallback
```

사용자가 기존 local-ID flag까지 두 개 켜지 않아도 되도록:

```python
use_local_ids = (
    envs.VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS
    or envs.VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT
)
```

로 처리하는 편이 좋습니다.

---

# `moe_align_block_size.py`는 처음에는 수정하지 않는 편이 좋습니다

파일:

```text
vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py
```

현재 generic implementation은 최대 크기의 output buffer를 만들고 `ops.moe_align_block_size()`를 호출하며, 필요하면 `expert_map`도 적용합니다. ([GitHub][7])

이 함수는 다음 경우를 모두 담당하는 범용 fallback입니다.

```text
non-DeepEP
global expert IDs
다른 all-to-all backend
다른 fused experts backend
일반 EP/non-EP
```

따라서 이것을 DeepEP 전용으로 바꾸기보다는:

```text
TritonExperts.apply()
    ├─ DeepEP direct path
    └─ 기존 generic moe_align_block_size fallback
```

구조로 두는 편이 안전합니다.

---

# `a100_moe_kernels.py`도 첫 단계에서는 수정할 필요가 없습니다

파일:

```text
vllm/vllm/model_executor/layers/fused_moe/a100_moe_kernels.py
```

현재 A100 kernel은 이미 다음 metadata를 받습니다.

```text
sorted_token_ids
expert_ids
num_tokens_post_padded
```

direct scheduler가 같은 contract를 만들면 A100 W1/W2 kernel은 그대로 사용할 수 있습니다. ([GitHub][4])

즉 이번 프로젝트의 경계는:

```text
DeepEP receive metadata
        ↓
새 direct assignment kernel
        ↓
기존 A100/Triton MoE GEMM
```

입니다.

`FlashMoE/` 디렉터리 역시 이 첫 구현에서는 건드릴 필요가 없습니다.

---

# 실제 수정 파일 요약

| 파일                                         | 변경 내용                                                  |
| ------------------------------------------ | ------------------------------------------------------ |
| `vllm/envs.py`                             | `VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT` 추가                  |
| `fused_moe/modular_kernel.py`              | metadata에 `DEEPEP_HT_LOCAL` layout 표시 추가               |
| `prepare_finalize/deepep_ht.py`            | local ID 유지, metadata layout 표시                        |
| `fused_moe/deepep_ht_expert_assignment.py` | 새로운 count-aware GPU assignment kernel                  |
| `experts/triton_moe.py`                    | direct/generic 경로 선택, `BLOCK_SIZE_M`별 assignment cache |
| `tests/kernels/moe/...`                    | schedule 및 GEMM correctness 테스트                        |
| `benchmarks/kernels/...`                   | generic/direct assignment 및 full forward 측정            |

---

# 테스트는 이렇게 해야 합니다

현재 테스트 디렉터리에는 이미 MoE alignment와 DeepEP 테스트가 있으므로 다음 파일을 추가하는 편이 좋습니다. ([GitHub][8])

```text
vllm/tests/kernels/moe/test_deepep_ht_expert_assignment.py
```

atomic scatter를 사용하면 `sorted_token_ids` 내부 순서가 generic 결과와 달라질 수 있습니다. 따라서 tensor를 그대로 `torch.equal()`로 비교하면 안 됩니다.

검증해야 할 invariant는 다음입니다.

```text
1. 모든 valid token-expert pair가 정확히 한 번 등장한다.
2. invalid pair가 잘못된 local expert 영역에 들어가지 않는다.
3. 각 block의 pair들이 expert_ids[block]과 일치한다.
4. num_tokens_post_padded가 expert별 padding 합과 일치한다.
5. 기존 GEMM 경로와 최종 output이 allclose다.
```

테스트 shape는 최소한 다음을 포함해야 합니다.

```text
M_recv: 1, 16, 128, 512, 1024
top_k: 8
local experts: 64
BLOCK_SIZE_M: 16, 32, 64, 128

routing:
  balanced
  한 expert에 집중
  empty expert 다수
  invalid slot 다수
```

DeepEP의 expert count는 receive된 `topk_idx`의 expert별 개수와 대응하며, 현재 vLLM 경로가 `expert_alignment=1`을 사용하므로 padding 전 count로 사용할 수 있습니다. ([GitHub][9])

---

# 성능 측정 시 기대치를 조심해야 합니다

현재 레포의 profiling 결과에서는 generic `moe_align_block_size` 자체가 한 실험에서 forward당 약 `9 μs` 수준이었습니다. 반면 DeepEP receiver metadata 처리와 remap 쪽은 수십 μs 수준으로 측정된 구간도 있습니다. ([GitHub][1])

따라서 단순히:

```text
generic moe_align_block_size
→ 새로운 Triton kernel
```

만 바꾸면 오히려 kernel launch가 늘어서 느려질 수도 있습니다.

포트폴리오 결과를 만들려면 다음을 묶어야 합니다.

```text
local → global → local 변환 제거
assignment count 단계 재사용
BLOCK_SIZE_M별 schedule cache
W1/W2 중복 assignment 제거
향후 CPU count metadata 경로 제거
```

그리고 현재 `ExpertTokensMetadata.make_from_list()`가 Python/CPU list에서 device tensor를 만드는 구조이므로, 제가 위에서 설명한 1차 구현만으로는 **CPU metadata 경로가 완전히 제거되지 않습니다**. ([GitHub][6])

진짜 GPU-only 경로는 2차 단계에서 다음 중 하나가 필요합니다.

```text
A. topk_ids histogram/count를 direct scheduler 안에서 GPU로 재계산
B. DeepEP를 수정해 expert count의 CUDA tensor를 직접 노출
```

DeepEP의 현재 공개 경로가 per-expert count를 host-side list 형태로 제공해 다른 프로젝트들이 이를 device로 복사하는 문제는 별도로도 논의된 바 있습니다. ([GitHub][10])

---

# 제가 추천하는 구현 순서

```text
Commit 1
  metadata layout 추가
  TritonExperts의 assignment helper/cache 추가
  direct flag off 상태에서 기존 동작 동일하게 유지

Commit 2
  count-aware direct assignment kernel 구현
  기존 local-ID sentinel 경로 재사용
  schedule correctness 테스트 추가

Commit 3
  W1/W2의 같은 BLOCK_SIZE_M schedule 재사용
  assignment microbenchmark와 full MoE forward 측정

Commit 4
  GPU-native count 생성
  CPU list → tensor 복사 제거
  가능하면 remap kernel까지 제거
```

첫 MVP 범위는 다음으로 제한하는 것이 가장 현실적입니다.

```text
DeepEP HT
TritonExperts
BF16
top-k=8
EP=2
local experts=64
A100
quantization/LoRA 없음
```

핵심 변경점은 결국 이것입니다.

> **`deepep_ht.py`에서는 local routing 정보만 보존하고, `triton_moe.py`가 실제 GEMM config의 `BLOCK_SIZE_M`을 확인한 뒤 새 DeepEP 전용 assignment kernel을 호출하도록 변경해야 합니다.** Final schedule을 receiver에서 미리 만드는 방식은 현재 레포의 W1/W2 tuning 구조와 맞지 않습니다.

[1]: https://github.com/jhss/vllm-flashmoe-a40-workspace/blob/main/vllm/benchmarks/results/a100_sxm_moe_ep_code_changes.md "vllm-flashmoe-a40-workspace/vllm/benchmarks/results/a100_sxm_moe_ep_code_changes.md at main · jhss/vllm-flashmoe-a40-workspace · GitHub"
[2]: https://raw.githubusercontent.com/jhss/vllm-flashmoe-a40-workspace/main/vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py "raw.githubusercontent.com"
[3]: https://github.com/jhss/vllm-flashmoe-a40-workspace/blob/main/vllm/vllm/model_executor/layers/fused_moe/experts/triton_moe.py "vllm-flashmoe-a40-workspace/vllm/vllm/model_executor/layers/fused_moe/experts/triton_moe.py at main · jhss/vllm-flashmoe-a40-workspace · GitHub"
[4]: https://github.com/jhss/vllm-flashmoe-a40-workspace/blob/main/vllm/vllm/model_executor/layers/fused_moe/a100_moe_kernels.py "vllm-flashmoe-a40-workspace/vllm/vllm/model_executor/layers/fused_moe/a100_moe_kernels.py at main · jhss/vllm-flashmoe-a40-workspace · GitHub"
[5]: https://github.com/jhss/vllm-flashmoe-a40-workspace/blob/main/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py "vllm-flashmoe-a40-workspace/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py at main · jhss/vllm-flashmoe-a40-workspace · GitHub"
[6]: https://github.com/jhss/vllm-flashmoe-a40-workspace/blob/main/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py "vllm-flashmoe-a40-workspace/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py at main · jhss/vllm-flashmoe-a40-workspace · GitHub"
[7]: https://raw.githubusercontent.com/jhss/vllm-flashmoe-a40-workspace/main/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py "raw.githubusercontent.com"
[8]: https://github.com/jhss/vllm-flashmoe-a40-workspace/tree/main/vllm/tests/kernels/moe "vllm-flashmoe-a40-workspace/vllm/tests/kernels/moe at main · jhss/vllm-flashmoe-a40-workspace · GitHub"
[9]: https://github.com/deepseek-ai/DeepEP/blob/main/tests/elastic/test_ep.py?utm_source=chatgpt.com "DeepEP/tests/elastic/test_ep.py at main"
[10]: https://github.com/deepseek-ai/DeepEP/issues/505?utm_source=chatgpt.com "[Question] Question on using num_worst_tokens · Issue #505"
