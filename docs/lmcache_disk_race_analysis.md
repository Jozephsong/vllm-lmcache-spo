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

## 수정 방향

`submit_put_task`에서 `self.dict`도 함께 확인한다:

```python
# 현재
if self.exists_in_put_tasks(key):
    return None

# 수정안: 완료된 write도 차단
if self.exists_in_put_tasks(key) or self.contains(key):
    return None
```

두 check가 별도 lock 획득이므로 완전한 원자성은 아니지만,
실용적으로 대부분의 중복 write를 방지할 수 있다.
완전히 막으려면 check와 insert를 단일 `put_lock` 구간으로 합쳐야 한다:

```python
def check_and_insert_put_task(self, key) -> bool:
    """Returns False if already stored or in-flight (skip); True if newly inserted."""
    with self.put_lock:
        if key in self.put_tasks:
            return False
    with self.disk_lock:
        if key in self.dict:
            return False
    with self.put_lock:
        # double-check after acquiring lock
        if key in self.put_tasks:
            return False
        self.put_tasks.append(key)
        return True
```
