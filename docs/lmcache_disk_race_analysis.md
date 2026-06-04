# LMCache LocalDiskBackend — 고QPS에서 SSD 중복 쓰기 분석

## 증상

QPS를 매우 높게 설정(예: 1,000,000)한 multi-round QA 벤치마크에서
LMCache metric의 local disk usage가 예상보다 훨씬 높게 측정된다.

---

## 코드 흐름

**lookup 시점** (forward pass 이전, `vllm_v1_adapter.py`):
```
storage_manager.contains(key)
  → local_disk_backend.contains()  [disk_lock]
    → key not in self.dict  →  cache miss  →  skip_leading_tokens = 0
```

**wait_for_save 시점** (forward pass 이후):
```
engine.store()
  → storage_manager.batched_put()
    → local_disk_backend.submit_put_task(key)
      → exists_in_put_tasks(key)   ← put_tasks만 확인
      → (self.dict는 여기서 확인하지 않음)
```

---

## 근본 원인: lookup–store 사이의 타이밍 gap

`submit_put_task` (`local_disk_backend.py:309`)는 **`exists_in_put_tasks`만 확인하고
`self.dict`(이미 완료된 write)는 확인하지 않는다.**

```python
# skip repeated save
if self.exists_in_put_tasks(key):   # put_tasks만 체크
    return None
# self.dict 확인 없음 → 완료된 write에 대한 재시도 가능
```

중복 write가 발생하는 시나리오:

```
[Step N]   Request A: lookup → self.dict에 key 없음 → skip=0
           Request A: submit_put_task(key) → put_tasks에 추가, async write 시작

           ... forward pass 여러 번 진행 ...
           ... thread pool에서 write 완료:
               self.dict에 key 추가 (line 519)
               put_tasks에서 key 제거 (line 521) ...

[Step N+K] Request B (동일 prefix):
           lookup → self.dict에 key 없음 (write 진행 중이었음) → skip=0

           ... 이 사이에 Request A의 write 완료 →
               put_tasks에서 key 제거 ...

           submit_put_task(key)
           → exists_in_put_tasks(key) = False  ← 방금 제거됨
           → self.dict 확인 없음
           → 중복 write 발생!
```

핵심은 **lookup 시점**과 **submit_put_task 시점** 사이에 async write가 완료되는 경우다.
이 window에서:
- lookup: key가 `self.dict`에 없어 miss → `skip_leading_tokens = 0`
- async write 완료: key가 `put_tasks`에서 제거됨
- submit_put_task: `exists_in_put_tasks` = False, `self.dict` 미확인 → 중복 write

---

## QPS가 높을수록 빈번한 이유

thread pool `max_workers=4` (`local_disk_backend.py:46`)로 고정되어 있다.
QPS가 극단적으로 높으면 write 요청이 폭주하여 queue가 쌓이고,
하나의 write가 완료되기까지 수십~수백 개의 forward pass step이 지나간다.

그 사이 동일 prefix를 가진 새 요청들이 lookup → miss → submit_put_task 경로로
진입하는 빈도가 급증하여, lookup–store 사이의 타이밍 gap에 write 완료가
겹치는 확률이 높아진다.

---

## 관련 코드 위치

| 파일 | 줄 | 내용 |
|------|----|------|
| `lmcache/v1/storage_backend/local_disk_backend.py` | 46 | `max_workers=4` (thread pool) |
| `lmcache/v1/storage_backend/local_disk_backend.py` | 180–188 | `contains()` — `self.dict`만 확인, `put_tasks` 미확인 |
| `lmcache/v1/storage_backend/local_disk_backend.py` | 309–311 | `submit_put_task` — `exists_in_put_tasks`만 확인, `self.dict` 미확인 |
| `lmcache/v1/storage_backend/local_disk_backend.py` | 519 | async write 완료 후 `self.dict`에 key 추가 |
| `lmcache/v1/storage_backend/local_disk_backend.py` | 521 | `remove_put_task` — `put_tasks`에서 key 제거 |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | 1110–1186 | `wait_for_save` — lookup 결과로 skip 계산 후 `engine.store()` 호출 |

---

## 수정 내용

두 곳을 수정하여 중복 write를 차단한다.

### 1. `contains()` — in-flight write도 hit로 처리

```python
# 변경 전
def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
    with self.disk_lock:
        if key not in self.dict:
            return False
        ...

# 변경 후
def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
    if self.disk_worker.exists_in_put_tasks(key):  # ← 추가
        return True
    with self.disk_lock:
        if key not in self.dict:
            return False
        ...
```

lookup 시점에 in-flight write도 hit로 반환하므로 `skip_leading_tokens`가 정확히 설정되어
`engine.store()` 단계에서 해당 chunk가 마스킹된다. `submit_put_task`까지 도달하지 않는다.

### 2. `submit_put_task()` — 완료된 write도 차단

```python
# 변경 전
if self.exists_in_put_tasks(key):
    logger.debug(f"Put task for {key} is already in progress.")
    return None

# 변경 후
if self.exists_in_put_tasks(key):
    logger.debug(f"Put task for {key} is already in progress.")
    return None
if self.contains(key):                              # ← 추가
    logger.debug(f"Key {key} already exists on disk, skipping.")
    return None
```

lookup과 submit_put_task 사이에 write가 완료된 경우를 이 check가 잡는다.

### 두 수정의 역할

| 수정 | 방어 시점 | 방어 대상 |
|------|-----------|-----------|
| `contains()`에 `put_tasks` 체크 추가 | lookup 시점 | write 진행 중인 경우 + 불필요한 KV copy 방지 |
| `submit_put_task()`에 `contains()` 추가 | write 직전 | lookup 이후 write가 완료된 경우 |

---

## 추가 수정: 남아있는 두 문제

### 문제 1: `batched_async_contains()` — async lookup 경로에도 같은 문제

`contains()`는 수정됐지만 async lookup 경로(`async_lookup_and_prefetch`)에서 사용되는
`batched_async_contains()`는 여전히 `self.dict`만 확인했다.

```python
# 변경 전
async def batched_async_contains(self, ...):
    with self.disk_lock:
        for key in keys:
            if key not in self.dict:   # put_tasks 미확인
                return num_hit_counts
            ...

# 변경 후
async def batched_async_contains(self, ...):
    with self.disk_lock:
        for key in keys:
            if self.disk_worker.exists_in_put_tasks(key):  # ← 추가
                num_hit_counts += 1
                continue
            if key not in self.dict:
                return num_hit_counts
            ...
```

in-flight write를 hit로 처리하여 불필요한 GPU→CPU KV copy와 `submit_put_task` 호출을 차단한다.

---

### 문제 2: `interval_stored_tokens` 과다 계산

`cache_engine.store()` (observability.py) 에서 `on_store_request`가 dedup 체크보다
**앞서** `interval_stored_tokens`를 증가시켜 실제보다 많은 토큰이 저장된 것으로 집계됐다.

```python
# 변경 전: on_store_request에서 즉시 카운트 (dedup 이전)
def on_store_request(self, num_tokens):
    self.interval_stored_tokens += num_tokens   # dedup 전에 카운트
    ...

# 변경 후: on_store_finished에서 실제 처리된 토큰 수로 카운트
def on_store_request(self, num_tokens):
    # interval_stored_tokens 증가 제거
    ...

def on_store_finished(self, store_stats, num_stored_tokens=-1):
    if num_stored_tokens >= 0:
        store_stats.num_tokens = num_stored_tokens
    self.interval_stored_tokens += store_stats.num_tokens  # ← 추가
```

이로써:
- memory 할당 실패로 `on_store_finished`가 호출되지 않는 경우 → 카운트 안 됨 (정확)
- freeze mode early return → 카운트 안 됨 (정확)
- `contains()` 수정으로 lookup 단계에서 skip된 chunk → `process_tokens`에서 제외 → `tot_token_num`에 포함 안 됨 → 카운트 안 됨 (정확)

### 전체 수정 요약

| 수정 | 파일 | 효과 |
|------|------|------|
| `contains()`에 `put_tasks` 체크 추가 | `local_disk_backend.py` | sync lookup: in-flight write hit 처리 |
| `submit_put_task()`에 `contains()` 추가 | `local_disk_backend.py` | 완료된 write 재시도 차단 |
| `batched_async_contains()`에 `put_tasks` 체크 추가 | `local_disk_backend.py` | async lookup: in-flight write hit 처리 |
| `interval_stored_tokens` 카운트 시점 이동 | `observability.py` | stored token metric 정확도 개선 |
