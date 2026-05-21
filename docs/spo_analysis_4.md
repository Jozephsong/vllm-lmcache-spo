# LMCacheSPOConnector — 역방향 인덱스 적용 (#4)

`spo_analysis_3.md` Q1의 개선 방법 2(역방향 인덱스)를 적용한 코드 수정 내용.

---

## 변경 동기

기존 `_flush_evicted`의 triggered 탐색:

```python
for req_id, spec in list(self._pending_store.items()):
    spec_blocks = {
        int(b)
        for b in torch.unique(spec.slot_mapping // block_size).tolist()
    }
    if spec_blocks & evicted_block_ids:
        triggered[req_id] = spec
```

- **복잡도**: O(N × T log T) — 활성 요청 수 × 토큰 수 × 정렬
- 매 eviction step마다 모든 요청의 slot_mapping에 대해 torch.unique 호출
- N=200 요청, T=1024 토큰이면 step당 수천 건의 텐서 연산

---

## 변경 내용

### 1. `_PendingSpec`: `block_ids` 필드 추가

```python
block_ids: frozenset[int]  # pre-computed for inverted-index lookup
```

캡처 시 1회 계산한 block_ids를 스펙에 저장.
`_cleanup_req`에서 역인덱스를 정리할 때 이 값을 참조한다.

### 2. `__init__`: `_block_to_reqs` 추가

```python
self._block_to_reqs: dict[int, set[str]] = {}
```

block_id → req_id 집합의 역방향 인덱스.
eviction 시 영향 받는 req_id를 O(|evicted_blocks|)로 조회한다.

### 3. `_capture_pending`: 인덱스 유지 로직 추가

```python
block_size = self._vllm_config.cache_config.block_size  # 루프 밖 1회

# 기존 엔트리 덮어쓸 때 이전 block_ids를 인덱스에서 제거
if existing is not None:
    skip = min(existing.skip_leading_tokens, skip)
    for block_id in existing.block_ids:
        s = self._block_to_reqs.get(block_id)
        if s is not None:
            s.discard(req.req_id)
            if not s:
                del self._block_to_reqs[block_id]

# 새 block_ids 계산 및 인덱스 등록
block_ids = frozenset(
    int(b) for b in torch.unique(req.slot_mapping // block_size).tolist()
)
self._pending_store[req.req_id] = _PendingSpec(..., block_ids=block_ids)
for block_id in block_ids:
    self._block_to_reqs.setdefault(block_id, set()).add(req.req_id)
```

- decode 단계에서 블록이 추가 할당되면 slot_mapping이 커진다. 기존 block_ids를
  먼저 제거하고 새 block_ids를 등록하여 인덱스를 최신 상태로 유지한다.
- `block_size`를 루프 밖으로 이동 — 기존 `_flush_evicted`에 있던 변수와 동일값,
  중복 없이 재배치.

### 4. `_flush_evicted`: O(N) 스캔 → O(|evicted|) 조회로 교체

삭제:
```python
block_size = self._vllm_config.cache_config.block_size  # ← 제거

for req_id, spec in list(self._pending_store.items()):
    spec_blocks = { int(b) for b in torch.unique(...).tolist() }
    if spec_blocks & evicted_block_ids:
        triggered[req_id] = spec
```

추가:
```python
triggered: dict[str, _PendingSpec] = {}
for block_id in evicted_block_ids:
    for req_id in self._block_to_reqs.get(block_id, ()):
        if req_id in self._pending_store:
            triggered[req_id] = self._pending_store[req_id]
```

- 복잡도: O(|evicted_blocks| × average_reqs_per_block)
- `evicted_block_ids`가 수십 개라면 반복 횟수는 수십~수백 회에 불과
- `torch.unique` 호출 완전 제거 — GPU 동기화 없음

### 5. `_cleanup_req`: 역인덱스 정리 추가

```python
def _cleanup_req(self, req_id: str) -> None:
    spec = self._pending_store.pop(req_id, None)
    if spec is not None:
        for block_id in spec.block_ids:
            s = self._block_to_reqs.get(block_id)
            if s is not None:
                s.discard(req_id)
                if not s:
                    del self._block_to_reqs[block_id]
```

빈 set은 즉시 삭제하여 메모리 누수 방지.
`_flush_evicted`의 모든 cleanup 경로(early exit, dedup skip, finally)가 이 함수를
거치므로 인덱스 정합성이 보장된다.

---

## 복잡도 비교

| 구간 | 변경 전 | 변경 후 |
|------|---------|---------|
| `_capture_pending` (per req) | O(T log T) 블록 계산 없음 | O(T log T) 1회 + O(B) 인덱스 |
| `_flush_evicted` triggered 탐색 | O(N × T log T) | O(\|evicted\| × B_avg) |
| `_cleanup_req` | O(1) | O(B) 인덱스 정리 |

`T` = 토큰 수, `B` = 요청당 블록 수, `N` = `_pending_store` 크기,
`|evicted|` = 이번 step에서 evict된 블록 수.

일반적으로 `|evicted|` ≪ N이므로 flush 탐색 비용이 크게 줄어든다.
block_ids 계산 비용이 flush에서 capture로 이동했으나 capture는 요청당 1회이고
이미 수행해야 하는 연산이므로 순 이득이다.

---

## 변경 파일

- `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_spo_connector.py`
