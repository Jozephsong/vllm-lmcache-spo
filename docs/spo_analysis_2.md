# LMCacheSPOConnector — 추가 기술 분석 (Q&A #2)

소스 코드 기반 답변. 참조 라인은 작성 시점 기준.

---

## Q1. SSD에 이미 저장된 데이터에 대해 두 번째 요청이 오면 re-store되는가?

**Q1 보정**: 이전 분석(`spo_analysis.md` Q1)에서 "non-layerwise는 dedup 없이
re-store된다"고 했는데, lookup hit 시의 skip 메커니즘으로 인해 실제로는
상황에 따라 달라진다.

### 핵심 메커니즘: `num_saved_tokens = lmcache_cached_tokens`

새 요청이 들어올 때 `RequestTracker`가 생성된다:

```python
# vllm_v1_adapter.py:194, 1457-1460
lmcache_cached_tokens = 0
if load_spec is not None:
    lmcache_cached_tokens = load_spec.lmcache_cached_tokens  # ← lookup hit 토큰 수

RequestTracker(..., num_saved_tokens=lmcache_cached_tokens, ...)
```

이후 `from_request_tracker()`에서:

```python
# vllm_v1_adapter.py:327
skip_leading_tokens = tracker.num_saved_tokens  # = lmcache_cached_tokens (e.g., 512)
```

### Case A: 순차 요청 (두 번째 요청이 첫 번째 완료 후 진입)

시스템 프롬프트 512 토큰이 SSD에 저장된 상황:
- 두 번째 요청 → lookup hit → `lmcache_cached_tokens = 512`
- `tracker.num_saved_tokens = 512` → `skip_leading_tokens = 512`

`_capture_pending()`에서:

```python
# lmcache_spo_connector.py:179-181
token_ids = req.token_ids         # e.g., len=512 (system prompt only)
if save_spec.skip_leading_tokens >= len(token_ids):
    continue                       # 512 >= 512 → capture 자체 생략 → re-store 없음 ✓
```

신규 토큰이 붙어 `len(token_ids) > 512`이 되면:
- `skip = 512`, 새 토큰만 저장 → 512 토큰 prefix는 re-store 안 됨 ✓

**순차 요청에서는 lookup hit + skip 메커니즘에 의해 이미 캐시된 청크는
다시 저장되지 않는다.**

### Case B: 동시 요청 (같은 시스템 프롬프트, 여러 요청 동시 인-플라이트)

- 요청 A, B 모두 lookup miss 상태(SSD에 아직 없거나, A가 evict되기 전에 B 진입)
- 두 요청 모두 `num_saved_tokens = 0`, `skip = 0`으로 `_pending_store`에 캡처됨
- eviction 시 두 요청 모두 flush → **동일 청크를 두 번 `engine.store()`** → 중복 저장 발생

### `local_disk_backend`의 in-progress dedup

```python
# local_disk_backend.py:309-311
if self.exists_in_put_tasks(key):   # 이미 진행 중인 put task면 skip
    return None
```

이는 동시 in-progress put 중복은 막지만, **이미 완료된(기존 SSD 저장) 항목에는
적용되지 않는다**. 즉, 두 요청이 서로 다른 eviction step에서 flush되면 같은 키로
두 번 저장된다.

### 결론

| 시나리오 | 중복 저장 여부 |
|----------|---------------|
| 순차 요청 (lookup hit 후 새 요청) | ✓ 없음 (skip 메커니즘으로 방지) |
| 동시 인-플라이트 요청 (같은 내용, lookup miss 상태) | ✗ 발생 |
| 동시 in-progress put (같은 청크 key) | ✓ 없음 (`exists_in_put_tasks` 방지) |

---

## Q2. `engine.store()` 에서 GPU→CPU 동기, CPU→SSD 비동기인가?

### GPU→CPU: 동기 (synchronous)

```python
# gpu_connectors.py:337-369
with torch.cuda.stream(self.store_stream):
    lmc_ops.multi_layer_kv_transfer(
        memory_obj.tensor, ..., TransferDirection.D2H, ...
    )                                              # ← GPU→CPU D2H 전송 (CUDA stream)

if not memory_obj.tensor.is_cuda:
    # Force a synchronize if the target buffer is NOT CUDA device
    self.store_stream.synchronize()               # ← CPU/SSD target → 동기 블록킹
```

`memory_obj.tensor`은 `storage_manager.allocate()`가 반환한 CPU tensor이므로
`not memory_obj.tensor.is_cuda` 조건은 항상 True → 청크마다 `synchronize()` 호출.

### CPU→SSD: 비동기 (asynchronous)

```python
# local_disk_backend.py:346-355
asyncio.run_coroutine_threadsafe(
    self.disk_worker.submit_task(
        "put",
        self.async_save_bytes_to_disk,   # ← 실제 디스크 쓰기
        key=key,
        memory_obj=memory_obj,
        ...
    ),
    self.loop,                           # ← 별도 이벤트 루프/스레드 풀
)
```

`AsyncPQThreadPoolExecutor` (max_workers=4)가 별도 스레드에서 실제 디스크 I/O를
처리한다. `batched_submit_put_task()`의 docstring도 "store KV caches to disk
**asynchronously**"라고 명시 (`local_disk_backend.py:366`).

### `local_cpu=False`여도 CPU 버퍼를 거친다

`local_cpu` 설정은 `LocalCPUBackend`(DRAM hot cache) 활성화 여부를 제어한다.
이와 무관하게 `from_gpu()` 내 `memory_obj.tensor`은 CPU 상의 임시 버퍼이며,
GPU→CPU D2H copy 후 이 버퍼를 `batched_submit_put_task()`에 넘겨 SSD에 쓴다.
따라서 **`local_cpu=False`로 설정해도 GPU→CPU 동기 전송은 반드시 발생한다**.

### SSD write 양이 증가할수록 TTFT가 늘어나는가?

`engine.store()`에서 청크 수만큼 `store_stream.synchronize()` 가 호출된다
(`cache_engine.py:532-533`의 `batched_from_gpu`는 청크마다 `from_gpu` 호출).

```
TTFT 지연 ≈ N_청크 × (GPU→CPU D2H 전송 시간 + synchronize 오버헤드)
```

저장하는 청크 수(= 저장 토큰 수 / chunk_size)에 비례하여 `wait_for_save()`
또는 `handle_block_evictions()`에서 블록킹 시간이 늘어난다.

**write 양이 많을수록 TTFT는 계속 늘어난다**. CPU→SSD 디스크 쓰기 자체는 비동기로
TTFT에 직접 영향이 없지만, GPU→CPU 동기화 시간은 store 토큰 수에 정비례한다.

---

## Q3. SPO에서 중복 저장을 막으려면 어떻게 해야 하나? (예시 코드)

중복 저장의 두 경로:
1. **동시 in-flight 요청** — 여러 요청이 같은 prefix로 동시에 `_pending_store`에 있다가 동시에 flush
2. **반복 eviction** — 같은 내용이 다른 타이밍에 두 번 flush (순차 요청에서는 lookup hit으로 방지되나, edge case 존재)

### 방법 1: `_flush_evicted` 전에 LMCache lookup으로 이미 캐시된 토큰 수 확인

```python
# 예시 코드 (수정 불가 - 개념 설명용)
def _flush_evicted(self, evicted_block_ids: set[int]) -> None:
    impl = self._lmcache_engine
    engine = impl.lmcache_engine

    ...  # triggered dict 구성 (기존과 동일)

    for req_id, spec in triggered.items():
        token_ids = spec.token_ids
        slot_mapping = spec.slot_mapping.to(impl.device)
        skip = spec.skip_leading_tokens // chunk_size * chunk_size

        # [NEW] flush 직전에 이미 캐시된 토큰 수를 확인
        # engine.retrieve()는 캐시 hit 토큰 mask를 반환하므로
        # 이를 이용해 이미 저장된 prefix를 건너뜀
        already_cached = impl._get_lmcache_cached_tokens(
            req_id, token_ids  # ← 내부적으로 token_database.process_tokens와
                               #   storage_manager.contains()를 사용
        )
        adjusted_skip = max(skip, already_cached // chunk_size * chunk_size)

        if adjusted_skip >= len(token_ids):
            self._cleanup_req(req_id)
            continue

        store_mask = torch.ones(len(token_ids), dtype=torch.bool)
        store_mask[:adjusted_skip] = False

        engine.store(token_ids, mask=store_mask, ...)
        self._cleanup_req(req_id)
```

현재 `LMCacheConnectorV1Impl`에는 이런 메서드가 없으므로 실제로는
`engine.token_database.process_tokens()` + `engine.storage_manager.contains()`를
직접 호출해야 하며, 이는 내부 API에 해당한다.

### 방법 2: req_id + token count 기반 중복 추적 (SPO 레벨 dedup)

같은 req_id가 다른 step에서 이미 flush된 경우를 막는다.
단, 동일 내용의 **다른 req_id** 요청 간 중복은 막지 못한다.

```python
# 예시 코드 (수정 불가 - 개념 설명용)

# __init__ 에 추가:
self._flushed_tokens: dict[str, int] = {}
# req_id → 마지막으로 flush한 토큰 수

# _flush_evicted 내:
for req_id, spec in triggered.items():
    token_ids = spec.token_ids
    skip = spec.skip_leading_tokens // chunk_size * chunk_size

    # 이미 이 req_id에 대해 이 범위를 flush했으면 skip
    prev_flushed = self._flushed_tokens.get(req_id, 0)
    adjusted_skip = max(skip, prev_flushed // chunk_size * chunk_size)
    if adjusted_skip >= len(token_ids):
        self._cleanup_req(req_id)
        continue

    store_mask = torch.ones(len(token_ids), dtype=torch.bool)
    store_mask[:adjusted_skip] = False

    engine.store(token_ids, mask=store_mask, ...)
    self._flushed_tokens[req_id] = len(token_ids)  # 기록
    self._cleanup_req(req_id)

# _cleanup_req 에서 함께 정리:
def _cleanup_req(self, req_id: str) -> None:
    self._pending_store.pop(req_id, None)
    self._flushed_tokens.pop(req_id, None)
```

### 방법 3: `engine.store()` 내부에 contains() 체크 추가 (LMCache 레벨)

근본적 해결책. `cache_engine.py` non-layerwise `store()` 경로에
layerwise와 동일한 `contains()` 사전 체크를 추가:

```python
# 예시 코드 (수정 불가 - 개념 설명용)
# cache_engine.py store() 의 process_tokens 루프 내:
for start, end, key in self.token_database.process_tokens(...):
    # [NEW] layerwise와 동일한 contains 체크
    if self.storage_manager.contains(key, self.retrieve_locations):
        continue  # ← 이미 존재하면 skip

    memory_obj = self.storage_manager.allocate(...)
    ...
```

이 방법이 SPO 뿐 아니라 write-through에서의 중복도 방지하는
가장 포괄적인 수정이다.

---

## Q4. SPO 모드에서 `save_spec`은 실제로 어떻게 사용되는가?

### `save_spec`이 읽히는 곳: `_capture_pending()`

```python
# lmcache_spo_connector.py:175-181
save_spec = req.save_spec
if save_spec is None or not save_spec.can_save:
    continue                                    # ① can_save 게이트

token_ids = req.token_ids
if save_spec.skip_leading_tokens >= len(token_ids):
    continue                                    # ② 저장할 토큰 없으면 skip

existing = self._pending_store.get(req.req_id)
skip = save_spec.skip_leading_tokens            # ③ initial skip 값으로 사용
if existing is not None:
    skip = min(existing.skip_leading_tokens, skip)
```

SPO에서 `save_spec`은 두 가지 용도:

1. **`can_save` 게이트**: `skip_save=True`(decode phase, priority 등)인 요청을
   capture에서 제외. lookup hit로 인해 `skip_leading_tokens >= len(token_ids)`인
   경우 capture 자체를 skip하여 SSD 재저장 방지.

2. **초기 skip 값**: `save_spec.skip_leading_tokens`는 `tracker.num_saved_tokens`
   로부터 온다. lookup hit가 있으면 이미 캐시된 prefix 토큰 수가 초기 skip으로
   설정되어, 해당 prefix는 flush 시에도 저장하지 않는다.

### `save_spec`이 업데이트되지 않는 이유

Write-through는 `engine.store()` 호출 직후 `save_spec`을 갱신한다:

```python
# vllm_v1_adapter.py:1184-1186 (write-through wait_for_save 내)
if get_pp_group().is_last_rank:
    save_spec.skip_leading_tokens = len(token_ids)  # ← 실제 저장 후 갱신
```

SPO는 `engine.store()`를 `_flush_evicted()`(eviction 시점)에서 호출하며,
이 시점에는 `save_spec` 객체에 대한 참조가 없다. `_PendingSpec`에 캡처된
`skip_leading_tokens`를 사용하므로 `save_spec.skip_leading_tokens`는 갱신되지 않는다.

### `tracker.num_saved_tokens` 진행과의 관계

`save_spec.skip_leading_tokens = tracker.num_saved_tokens`이고,
`tracker.num_saved_tokens`은 `from_request_tracker()` 내에서 매 step 진행된다
(`vllm_v1_adapter.py:361`). SPO에서 실제 저장이 없어도 이 값은 증가하므로,
`save_spec.skip_leading_tokens`는 "optimistic skip"(저장 완료를 가정한 추정치)이 된다.

`_capture_pending()`의 `min()` 로직은 이 optimistic 진행을 보정한다:

```python
# lmcache_spo_connector.py:190-193
existing = self._pending_store.get(req.req_id)
skip = save_spec.skip_leading_tokens   # optimistic (매 step 증가)
if existing is not None:
    skip = min(existing.skip_leading_tokens, skip)  # ← 첫 캡처의 실제 skip 보존
```

### 요약

| 항목 | Write-through | SPO |
|------|--------------|-----|
| `save_spec` 읽기 | `wait_for_save()` 내 | `_capture_pending()` 내 |
| `save_spec.skip_leading_tokens` 쓰기 | `engine.store()` 후 갱신 (`=len(token_ids)`) | **갱신 없음** |
| skip 추적 대체재 | `save_spec.skip_leading_tokens` | `_PendingSpec.skip_leading_tokens` (min 보정) |
| `can_save` 역할 | store 여부 결정 | capture 여부 결정 |

SPO에서 `save_spec`은 "이 요청을 캡처할 필요가 있는지, 어디서부터 저장해야 하는지"를
판단하는 입력 소스이지만, 실제 SSD 저장 완료 후 갱신되지 않아
"이미 SSD에 저장됨" 상태를 반영하지 못한다. 이 역할은 `_pending_store` entry 삭제
(`_cleanup_req`)가 간접적으로 담당한다.

---

## 참조 파일 및 라인 번호

| 파일 | 라인 | 내용 |
|------|------|------|
| `LMCache/lmcache/integration/vllm/vllm_v1_adapter.py` | 194 | `num_saved_tokens = lmcache_cached_tokens` |
| `LMCache/lmcache/integration/vllm/vllm_v1_adapter.py` | 327 | `skip_leading_tokens = tracker.num_saved_tokens` |
| `LMCache/lmcache/integration/vllm/vllm_v1_adapter.py` | 360–362 | `tracker.num_saved_tokens` 갱신 및 `SaveSpec` 생성 |
| `LMCache/lmcache/integration/vllm/vllm_v1_adapter.py` | 1184–1186 | write-through의 `save_spec.skip_leading_tokens` 갱신 |
| `LMCache/lmcache/v1/gpu_connector/gpu_connectors.py` | 337–369 | D2H transfer 및 `store_stream.synchronize()` |
| `LMCache/lmcache/v1/storage_backend/abstract_backend.py` | 71–94 | `batched_submit_put_task` 비동기 설명 |
| `LMCache/lmcache/v1/storage_backend/local_disk_backend.py` | 291–355 | `submit_put_task`: in-progress dedup 및 async 디스크 write |
| `LMCache/lmcache/v1/storage_backend/local_disk_backend.py` | 358–378 | `batched_submit_put_task`: "store to disk asynchronously" |
| `LMCache/lmcache/v1/storage_backend/storage_manager.py` | 378–427 | `batched_put`: "Non-blocking function" |
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_spo_connector.py` | 157–201 | `_capture_pending`: `save_spec` 사용 위치 |
