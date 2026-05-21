# AssertionError: num_computed_tokens 33023 vs max(...) = 33024

## 에러 메시지

```
[core.py:1105] AssertionError: Preempted request chatcmpl-81fd233407202f9b-86dcc5ae
has num_computed_tokens 33023
but max(lmcache_cached_tokens, vllm_cached_tokens) = 33024
```

---

## assertion의 의도

`vllm_v1_adapter.py:1571-1586` (CachedRequestData 재스케줄 경로):

```python
if preempted:
    assert request.num_computed_tokens == max(
        lmcache_cached_tokens, load_spec.vllm_cached_tokens
    )
```

Preempted 요청이 재스케줄될 때, vLLM이 직전 스케줄링 단계에서 할당한 블록 수
(`request.num_computed_tokens`)가 LoadSpec에 기록된 캐시 히트 토큰 수와 일치해야
한다는 불변 조건을 검사한다.

정상 경우: `lmcache_cached_tokens = 33024`, vLLM이 33024 토큰을 할당
→ `33024 == max(33024, 0)` ✓

---

## 근본 원인: full-prompt-hit 시 -1 조정

`vllm_v1_adapter.py:1295-1337` `get_num_new_matched_tokens`:

```python
num_external_hit_tokens = ...  # 33024 (LMCache 전체 히트)

if num_external_hit_tokens == request.num_tokens:  # 33024 == 33024 → full-prompt-hit
    need_to_allocate -= 1  # 33024 → 33023

self.load_specs[req_id] = LoadSpec(
    vllm_cached_tokens=num_computed_tokens,        # 0 (vLLM 자체 캐시 없음)
    lmcache_cached_tokens=num_external_hit_tokens, # 33024 ← 조정 전 원본값
    can_load=False,
)
```

- `need_to_allocate = 33023`으로 감소 → vLLM 스케줄러가 33023개 블록 할당
- → `request.num_computed_tokens = 33023`
- 그러나 `LoadSpec.lmcache_cached_tokens = 33024` (원본값 그대로)

assertion: `33023 == max(33024, 0)` → **FAILS**

### -1 조정이 필요한 이유

LMCache가 모든 토큰을 반환하면 마지막 토큰을 다시 계산(recompute)해야 KV 연산이
가능하다. `need_to_allocate -= 1`은 "마지막 토큰 1개는 GPU에서 다시 계산"을
의미하며 `LoadSpec`에는 반영되지 않아 off-by-one이 발생한다.

---

## 간헐적 → 지속적 전환의 원인

| 상태 | LMCache SSD 내용 | full-prompt-hit 발생 | assertion 결과 |
|------|-----------------|---------------------|----------------|
| SPO flush 전 | 없음 또는 부분 | 미발생 (partial hit) | 통과 |
| SPO flush 후 | 전체 시퀀스 (33024토큰) | 매번 발생 | **항상 실패** |

SPO 모드에서 최초로 전체 시퀀스(33024토큰)가 SSD에 flush된 이후:

1. 해당 요청(또는 동일 prefix를 가진 신규 요청)이 재스케줄될 때 LMCache lookup이
   전체 히트를 반환한다.
2. `get_num_new_matched_tokens`에서 full-prompt-hit 분기로 진입 → `need_to_allocate -= 1`.
3. assertion이 매 preemption마다 실패한다.

"처음에는 잘 되다가 어느 순간부터 계속 실패"하는 패턴이 이 메커니즘과 정확히 일치한다.

---

## 코드 경로 요약

```
SPO: _flush_evicted → engine.store(33024 tokens) → SSD 기록

이후 preemption + reschedule:
  get_num_new_matched_tokens (line 1295)
    └─ num_external_hit_tokens = 33024  (full match)
    └─ need_to_allocate -= 1  → 33023  (line 1301)
    └─ LoadSpec(lmcache_cached_tokens=33024)  (line 1333-1337)

  vLLM scheduler: request.num_computed_tokens = 33023  (sched/scheduler.py:804)

  vllm_v1_adapter.py:1579:
    assert 33023 == max(33024, 0)  → AssertionError
```

---

## 버그 위치

`vllm_v1_adapter.py`의 assertion이 full-prompt-hit 시의 -1 조정을 고려하지 않는다.

수정 방향 (참고용):
- `LoadSpec.lmcache_cached_tokens`에 조정된 값(`need_to_allocate`)을 저장하거나,
- assertion에서 full-prompt-hit 케이스를 별도 처리한다.

```python
# 수정 예시 (vllm_v1_adapter.py)
full_hit = (lmcache_cached_tokens == request.num_tokens)
effective_lmcache = lmcache_cached_tokens - (1 if full_hit else 0)
assert request.num_computed_tokens == max(effective_lmcache, load_spec.vllm_cached_tokens)
```

---

## 관련 소스 위치

| 파일 | 줄 | 내용 |
|------|----|------|
| `lmcache/integration/vllm/vllm_v1_adapter.py` | 1301-1302 | full-prompt-hit `-1` 조정 |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | 1333-1337 | `LoadSpec` 생성 (원본값 사용) |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | 1571-1586 | preemption assertion |
| `vllm/v1/core/sched/scheduler.py` | 804 | `request.num_computed_tokens` 설정 |
