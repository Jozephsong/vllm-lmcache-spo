# LMCacheSPOConnector — Technical Analysis (Q&A)

Source-code evidence only. All line numbers refer to the state of the codebase
at the time of writing.

---

## Q1. 동일 system prompt 재요청 시 SSD 중복 저장 여부

**상황**: n=4 system prompt, 같은 prompt를 가진 두 번째 요청이 들어올 때 이전에
LMCache에서 hit → load → 다시 eviction → store. 이미 SSD에 써진 동일 청크를
또 쓰는가?

### 결론: 비-layerwise 경로에서는 중복 체크 없이 re-store 됨

`cache_engine.py:474`의 non-layerwise `store()` 경로:

```python
# cache_engine.py:458-481
for start, end, key in self.token_database.process_tokens(...):
    memory_obj = self.storage_manager.allocate(   # ← 무조건 할당
        kv_shapes, kv_dtypes, ...
    )
    ...
```

`storage_manager.allocate()` → `batched_from_gpu()` → `batched_put()` 순서로
진행되며, 어디에도 "이 key가 이미 존재하는가"를 사전 확인하는 코드가 없다.

반면 layerwise 경로(`store_layer()`)에는 명시적 중복 체크가 있다:

```python
# cache_engine.py:651-654
if self.storage_manager.contains(
    keys_multi_layer[0], self.retrieve_locations
):
    continue          # ← 이미 있으면 건너뜀
```

따라서:

- **`use_layerwise=False` (기본값)**: 동일 청크를 동일 키로 다시 `store()` 하면
  **중복 저장됨**. 이전 내용을 덮어쓰는 형태.
- **`use_layerwise=True`**: `contains()` 체크로 건너뜀. 단, SPO 모드는
  `use_layerwise=True`와 호환되지 않아 자동으로 write-through로 폴백됨
  (`lmcache_spo_connector.py:95-100`).

**요약**: SPO + 비-layerwise 조합에서는 same-key 재저장 시 dedup 없이 SSD에 다시
기록된다. 이는 Q3에서 설명하는 과잉 저장의 원인 중 하나이다.

---

## Q2. Block 단위 eviction인데 request 전체를 저장하는 것이 맞는가?

**상황**: eviction은 블록 단위로 발생하는데, `_flush_evicted`는 해당 블록을 포함한
request 전체(slot_mapping 전체)를 store한다. 필요 이상으로 저장하는 것 아닌가?

### 결론: 의도된 설계이며, 일부 over-storing이 있음

`_flush_evicted` 내부 로직:

```python
# lmcache_spo_connector.py:234-244
for req_id, spec in list(self._pending_store.items()):
    spec_blocks = {
        int(b)
        for b in torch.unique(spec.slot_mapping // block_size).tolist()
    }
    if spec_blocks & evicted_block_ids:   # ← 하나라도 겹치면
        triggered[req_id] = spec          # ← request 전체를 store 대상으로 지정
```

evicted 블록과 하나라도 겹치는 request는 **전체 sequence**를 store한다. 예를 들어
1024 토큰 시퀀스에서 마지막 16-token 블록 하나만 evict되어도 1024 토큰 전체가
저장된다.

**그러나 이것이 올바른 이유**:

LMCache의 캐시 단위는 chunk(`_lmcache_chunk_size`)이며, 키는 prefix-hash 기반이다.
이미 SSD에 있는 청크는 재로딩 시 hit로 처리되므로, 전체 sequence를 저장해도
lookup 시 이미 존재하는 prefix 청크는 skip된다(단, Q1에서 설명한 dedup 부재 문제는
별도임).

store를 블록 단위로 분할하면 오히려 prefix-hash 체인이 깨져서 불완전한 KV가
저장될 위험이 있다. 따라서 "블록 하나라도 evict되면 전체 저장"은 정합성을 위한
의도된 설계이다.

**부작용**: 아직 HBM에 남아있는 블록의 KV도 함께 복사됨 → CPU-SSD 대역폭 소모.
이 비용은 해당 블록이 이후에 재사용되어 eviction 시 두 번 저장되는 것을 방지한다
(이미 SSD에 있으면 다음 eviction 시 lookup hit → store 불필요).

---

## Q3. SPO가 write-through보다 SSD 기록량이 2배 이상 많은 이유

**상황**: n=4 이상에서 eviction이 발생하고 flush가 되는데, 동일 조건 대비 SPO
모드에서 총 기록량이 write-through보다 훨씬 많다.

### 원인 분석

세 가지 요인이 복합적으로 작용한다.

#### 원인 1: skip_leading_tokens 누적 불가 (Q4에서 상세 설명)

Write-through는 `wait_for_save()`가 호출될 때마다
`save_spec.skip_leading_tokens = len(token_ids)`로 갱신된다
(`vllm_v1_adapter.py:1186`). 즉, 이미 저장한 토큰은 다음 step에서 건너뛴다.

SPO는 eviction이 발생하기 전까지 `engine.store()`를 호출하지 않으므로
`save_spec.skip_leading_tokens`가 갱신될 기회가 없다. 그러나
`tracker.num_saved_tokens`는 metadata 생성 시점(`from_request_tracker()`
`vllm_v1_adapter.py:361`)에 이미 증가하므로 `save_spec.skip_leading_tokens`도
매 step 증가한다.

`_capture_pending`에서 `min()` 로직으로 이를 보정하지만, 첫 capture 시
`existing`이 None이면 `skip = save_spec.skip_leading_tokens`가 그대로 쓰인다.
이 값이 0이 아닌 경우, 즉 lookup hit prefix가 있는 경우에는 올바르게 skip이 설정된다.

**핵심**: lookup hit가 없는(skip=0) request는 매 eviction마다 전체 토큰을 저장.
Write-through는 이미 저장된 토큰을 skip하므로 증분 저장.

#### 원인 2: 중복 request가 각각 독립적으로 저장됨

동일 system prompt를 가진 다른 user의 요청들이 각각 `_pending_store`에
별도 entry로 존재한다. 이 중 여러 entry의 블록이 evict되면 같은 토큰을 가진
여러 request가 각각 `engine.store()`를 호출한다.

Write-through는 첫 번째 저장 이후 다른 request의 lookup이 hit로 처리되어
`save_spec.can_save = False` 또는 skip이 max로 설정되므로 중복 저장이 줄어든다
(`vllm_v1_adapter.py:1115-1118`).

SPO는 flush 시점이 eviction 타이밍에 종속되어 여러 request가 동시에 flush 대상이
될 때 동일 청크를 반복 저장한다.

#### 원인 3: Q1에서 설명한 dedup 부재

Non-layerwise 경로에서는 이미 SSD에 존재하는 청크를 `contains()` 체크 없이 덮어씀.
Write-through는 `skip_leading_tokens` 증가로 사실상 dedup 효과를 얻는 반면,
SPO는 그렇지 않다.

**정량적 예시** (chunk_size=256, block_size=16):

- 시스템 프롬프트 512 토큰, 64명 사용자, n=4 system prompt
- Write-through: 최초 1회 512 토큰 저장 후 이후 요청은 hit → skip=512 → 0 토큰 저장
- SPO: eviction이 발생하는 매 request마다 512 토큰 저장 → user 수 비례 증가

---

## Q4. skip_leading_tokens의 누적 업데이트 문제

**질문**: 같은 request가 여러 step에 걸쳐 들어올 때 skip_leading_tokens는 누적
업데이트되어야 하지 않나? min() 로직이 오히려 저장량을 늘리는 게 아닌가?

### 결론: min() 로직은 multi-chunk prefill에서 필요하나, skip 누적 문제는 별도로 존재

#### `tracker.num_saved_tokens`의 동작

```python
# vllm_v1_adapter.py:327
skip_leading_tokens = tracker.num_saved_tokens

# vllm_v1_adapter.py:360-361
if not skip_save:
    tracker.num_saved_tokens = num_tokens_to_save   # ← 매 step 갱신
```

`tracker.num_saved_tokens`는 `from_request_tracker()`에서 매 step
`num_tokens_to_save`로 설정된다. Write-through에서는 실제로 `engine.store()`를
호출했기 때문에 이 값이 "이미 저장된 토큰 수"를 정확히 반영한다.

SPO에서는 `engine.store()`를 호출하지 않음에도 불구하고 `tracker.num_saved_tokens`가
`num_tokens_to_save`로 설정된다. 이는 SPO 모드에서 "미래에 eviction 시 저장할
예정인 토큰 수"를 추적하는 것으로 볼 수 있으나, 실제 저장이 발생하지 않아
다음 step에서 `skip_leading_tokens = tracker.num_saved_tokens`가 커진 값이 된다.

#### min() 로직의 역할과 한계

```python
# lmcache_spo_connector.py:190-193
existing = self._pending_store.get(req.req_id)
skip = save_spec.skip_leading_tokens
if existing is not None:
    skip = min(existing.skip_leading_tokens, skip)
```

**역할**: multi-chunk prefill 시 첫 청크 capture 이후 두 번째 청크 capture에서
`save_spec.skip_leading_tokens`는 첫 청크 크기만큼 증가한다. min()을 적용하지 않으면
첫 청크가 skip되어 전체 prefill의 일부만 저장된다. min()은 이를 방지하여 lookup hit
prefix(skip=N 이상)는 건너뛰되, 새로 계산된 청크들은 빠짐없이 저장하게 한다.

**한계**: eviction이 발생한 뒤 `_cleanup_req()`로 entry가 제거된다. 이후 동일
req_id로 **새 request**가 들어오면 `existing`이 None이 되고 새 `save_spec.skip_leading_tokens`
(lookup hit prefix 크기)이 그대로 skip으로 사용된다. 이는 의도된 동작이다.

**Write-through와의 차이**: write-through는 `wait_for_save()`에서 실제 저장 후
`save_spec.skip_leading_tokens = len(token_ids)` (`vllm_v1_adapter.py:1186`)로
설정하여 다음 step에서는 모든 토큰이 skip된다. SPO는 이 갱신이 없으므로 eviction 시에
전체 (혹은 lookup hit prefix 이후 전체)를 저장하게 된다. 이것이 Q3의 저장량 증가의
핵심 원인이다.

**결론**: min() 자체가 저장량을 늘리는 것은 아니다. min()은 multi-chunk prefill
정합성을 위해 필요하다. 저장량이 많은 원인은 SPO에서 `engine.store()` 미호출로 인해
`save_spec.skip_leading_tokens`가 올바르게 "이미 저장됨" 상태로 갱신되지 않는 구조
때문이다.

---

## Q5. Write-through가 SPO보다 TTFT/prefill time이 느린 이유

**상황**: eviction이 발생하지 않는 범위에서 테스트. SSD 저장은 write-through에서만
발생. write-through의 TTFT/prefill이 SPO보다 느림.

### 결론: write-through의 engine.store()는 SSD 백엔드에서 동기(blocking) 호출

#### GPU connector의 synchronize

```python
# gpu_connectors.py:365-369
if not memory_obj.tensor.is_cuda:
    # Force a synchronize if the target buffer is NOT CUDA device
    # NOTE: for better performance, we may not want to sync for every
    # memory object
    self.store_stream.synchronize()
```

SSD 오프로딩 백엔드에서 target buffer는 CPU tensor이므로 `not memory_obj.tensor.is_cuda`
가 True. 따라서 `store_stream.synchronize()`가 청크마다 호출된다.

`gpu_connector.batched_from_gpu()` → `from_gpu()` (per chunk) → 위 synchronize 호출
순서이며, `cache_engine.store()` (`vllm_v1_adapter.py:1171-1180`)의 호출은
이 과정이 완료될 때까지 블록킹된다.

#### wait_for_save 호출 위치

```python
# kv_connector_model_runner_mixin.py:117 (finally block)
self.wait_for_save()
```

`wait_for_save()`는 forward pass **이후** finally 블록에서 호출된다. 그러나
이것이 다음 forward의 TTFT에 영향을 준다:

1. Step N의 finally에서 `wait_for_save()` → `engine.store()` 호출 → blocking
2. Step N의 blocking이 끝나야 scheduler로 리턴 → Step N+1 스케줄링 시작
3. TTFT는 Step N+1의 첫 토큰 생성 시간이므로, Step N의 저장 지연이 전파됨

또한 write-through는 prefill step마다 `engine.store()`를 호출하므로 **prefill
자체가 저장 I/O에 의해 연장**된다.

#### SPO의 경우

```python
# lmcache_spo_connector.py:139-142
def wait_for_save(self) -> None:
    if not self._is_spo:
        super().wait_for_save()
        return
    self._capture_pending()   # ← dict write만, GPU/SSD I/O 없음
```

SPO의 `wait_for_save()`는 `_capture_pending()`만 호출한다. 이는 메모리 상의 dict
연산으로 GPU/SSD I/O가 없어 거의 즉시 리턴된다.

`handle_block_evictions()`의 `_flush_evicted()`는 forward pass **이전**에 호출되나
(`gpu_model_runner.py:3577-3581`), eviction이 발생하지 않는 실험 범위에서는
호출되지 않는다.

**결론**: write-through는 매 prefill step의 `wait_for_save()`에서 SSD 저장 I/O가
동기적으로 blocking되어 다음 요청의 TTFT 및 prefill time이 증가한다. SPO는 동일 조건
(eviction 없음)에서 `wait_for_save()`가 lightweight하므로 이 지연이 없다. 실험
결과의 TTFT/prefill 차이는 `store_stream.synchronize()`로 인한 blocking이 직접적인
원인이다.

---

## 참조 파일 및 라인 번호

| 파일 | 라인 | 내용 |
|------|------|------|
| `LMCache/lmcache/v1/cache_engine.py` | 363–565 | `store()` non-layerwise 경로 (dedup 없음) |
| `LMCache/lmcache/v1/cache_engine.py` | 644–654 | `store_layer()` `contains()` 체크 (layerwise only) |
| `LMCache/lmcache/v1/gpu_connector/gpu_connectors.py` | 365–369 | `store_stream.synchronize()` for non-CUDA target |
| `LMCache/lmcache/integration/vllm/vllm_v1_adapter.py` | 327 | `skip_leading_tokens = tracker.num_saved_tokens` |
| `LMCache/lmcache/integration/vllm/vllm_v1_adapter.py` | 360–362 | `tracker.num_saved_tokens = num_tokens_to_save` |
| `LMCache/lmcache/integration/vllm/vllm_v1_adapter.py` | 1084–1186 | `wait_for_save()` write-through 경로 |
| `LMCache/lmcache/integration/vllm/vllm_v1_adapter.py` | 1184–1186 | `save_spec.skip_leading_tokens = len(token_ids)` |
| `vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py` | 117 | `wait_for_save()` finally 블록 호출 위치 |
| `vllm/vllm/v1/worker/gpu_model_runner.py` | 3577–3581 | `handle_block_evictions()` forward 이전 호출 |
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_spo_connector.py` | 157–201 | `_capture_pending()` min() skip 로직 |
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_spo_connector.py` | 204–291 | `_flush_evicted()` 전체 sequence store |
