# LMCacheSPOConnector — 효율화 분석 및 중복 저장 수정 (#3)

---

## Q1. `_flush_evicted`에서 triggered 탐색 비용 분석

### 현재 구현

```python
for req_id, spec in list(self._pending_store.items()):
    spec_blocks = {
        int(b)
        for b in torch.unique(spec.slot_mapping // block_size).tolist()
    }
    if spec_blocks & evicted_block_ids:
        triggered[req_id] = spec
```

요청 수 N, 요청당 평균 토큰 T, 블록당 토큰 수 B라 하면:
- `spec.slot_mapping // block_size`: O(T) 텐서 연산 (CPU)
- `torch.unique(...)`: O(T log T) 정렬
- `.tolist()` + set comprehension: O(T/B) (블록 수)
- 집합 교차: O(|evicted|)

**전체 복잡도**: O(N × T log T) — N이 수백, T가 수천이면 매 eviction step마다 상당한 비용.

### 개선 방법 1: `_PendingSpec`에 block_ids 사전 계산

```python
# _PendingSpec에 block_ids 필드 추가
@dataclass
class _PendingSpec:
    token_ids: list[int]
    slot_mapping: torch.Tensor
    is_last_prefill: bool
    skip_leading_tokens: int
    request_configs: Optional[dict]
    block_ids: frozenset[int]          # ← 캡처 시 1회만 계산

# _capture_pending에서:
block_ids = frozenset(
    int(b)
    for b in torch.unique(spec.slot_mapping // block_size).tolist()
)
self._pending_store[req.req_id] = _PendingSpec(..., block_ids=block_ids)

# _flush_evicted에서:
if spec.block_ids & evicted_block_ids:
    triggered[req_id] = spec
```

- 캡처 시 O(T log T) 1회, 이후 flush마다 O(B) 집합 교차만
- N번의 flush가 발생하면 총 절약: O(N × T log T) → O(T log T + N × B)

### 개선 방법 2: 역방향 인덱스 (block_id → req_ids)

```python
# __init__에 추가:
self._block_to_reqs: dict[int, set[str]] = defaultdict(set)

# _capture_pending에서:
for block_id in block_ids:
    self._block_to_reqs[block_id].add(req_id)

# _cleanup_req에서 정리:
for block_id in spec.block_ids:
    self._block_to_reqs[block_id].discard(req_id)

# _flush_evicted에서 (전체 스캔 제거):
triggered_req_ids: set[str] = set()
for block_id in evicted_block_ids:
    triggered_req_ids.update(self._block_to_reqs.get(block_id, set()))
```

- Lookup: O(|evicted_block_ids|) — eviction당 evict되는 블록 수에만 비례
- N이 매우 커도 탐색 비용은 evicted block 수에만 의존 → 가장 확장성 있는 방법
- 단, cleanup 시 역인덱스 유지 비용이 추가됨

### 비교

| 방법 | Flush 탐색 복잡도 | 추가 메모리 | 구현 복잡도 |
|------|-----------------|------------|------------|
| 현재 | O(N × T log T) | 없음 | 낮음 |
| 방법 1 | O(N × B) | O(N × B) | 낮음 |
| 방법 2 | O(\|evicted\|) | O(N × B) | 중간 |

**결론**: 일반적인 사용 패턴(N < 수백, T < 수천)에서는 방법 1이 충분하며 구현 단순성이 높다.
N이 매우 크거나(수천 동시 요청) eviction 빈도가 높은 경우에는 방법 2가 적합하다.

---

## 코드 수정: 중복 prefix 저장 방지

### 문제 케이스

| 케이스 | 설명 | 기존 동작 |
|--------|------|----------|
| 동일 스텝 동시 flush | 같은 system prompt를 가진 req A, B가 같은 eviction step에서 flush | 둘 다 SSD에 저장 (중복) |
| 다른 스텝 순차 flush | req A 저장 완료 후 req B(A와 같은 내용, 먼저 시작)가 이후 eviction에서 flush | 다시 저장 (중복) |
| lookup hit 후 신규 요청 | 이미 SSD에 있는 prefix를 가진 새 요청 → `save_spec.can_save=False` | 캡처 자체 skip ✓ |

### 수정 내용

**1. `__init__`: `_stored_chunk_sigs` 추가**

```python
self._stored_chunk_sigs: set[int] = set()
```

성공적으로 flush된 요청의 첫 번째 저장 청크의 hash값을 저장한다.
`int` 하나 = 8 bytes, 고유 system prompt 수에 비례 → 메모리 부담 없음.

**2. `_flush_evicted`: 정렬 후 early-exit 추가**

```python
if skip >= len(token_ids):
    self._cleanup_req(req_id)
    continue
```

`is_last_prefill=False` 정렬 이후 `aligned_len <= skip`이 될 수 있는 edge case 처리.
기존 코드는 이 경우 `engine.store()`를 all-False mask로 호출했음(낭비).

**3. `_flush_evicted`: chunk_sig 검사**

```python
chunk_sig = hash(tuple(token_ids[skip:skip + chunk_size]))
if chunk_sig in self._stored_chunk_sigs:
    logger.debug("SPO dedup: req=%s first-chunk already stored — skip", req_id)
    self._cleanup_req(req_id)
    continue
```

`token_ids[skip:skip+chunk_size]`는 첫 번째 저장할 청크.
두 요청이 같은 prefix를 가지면 이 슬라이스가 동일 → 같은 hash → 두 번째 요청은 skip.

**4. `_flush_evicted`: 성공 후 sig 등록**

```python
try:
    engine.store(...)
    self._stored_chunk_sigs.add(chunk_sig)  # ← store 성공 시에만 등록
except Exception:
    logger.warning(...)
finally:
    self._cleanup_req(req_id)
```

`engine.store()` 예외 발생 시 sig를 등록하지 않아 다음 eviction에서 재시도 가능.

### 커버되는 케이스

**동시 요청 같은 eviction step**:
- req A: `chunk_sig` 계산 → 미등록 → store → sig 등록
- req B: 같은 `chunk_sig` → 이미 등록 → skip ✓

**다른 eviction step (sequential cross-step)**:
- Step N: req A flush → sig 등록
- Step N+M: req B flush → 같은 `chunk_sig` → 이미 등록 → skip ✓

**Store 실패**:
- try에서 exception → sig 미등록 → 다음 eviction에서 재시도 가능 ✓

**Skip이 다른 경우 (다른 prefix hit 크기)**:
- `chunk_sig = hash(tuple(token_ids[skip:skip+chunk_size]))` — skip이 다르면
  첫 청크 내용도 다르므로 다른 hash → 독립적으로 저장 ✓

**process 재시작**:
- `_stored_chunk_sigs`는 in-memory; 재시작 시 초기화
- 그러나 재시작 후 SSD에 이미 있는 content에 대한 새 요청은 lookup hit →
  `save_spec.can_save=False` → `_capture_pending`에서 skip → 저장 안 됨 ✓

### 주의사항

**hash 충돌**: Python의 `hash(tuple[int,...])` (SipHash-based, 64-bit)은 충돌 확률이
극히 낮다. 그러나 충돌 발생 시 서로 다른 content가 같은 sig → 한쪽이 저장되지 않음
(false dedup). 이를 절대 방지하려면 `set[tuple[int,...]]`를 사용해야 하나
메모리가 청크 크기에 비례해 증가하는 trade-off가 있다.

**메모리 증가**: `_stored_chunk_sigs`는 정리되지 않고 누적된다. 고유한 system prompt
수에 비례하므로 일반 환경에서 문제 없으나, 수백만 종류의 고유 prefix를 처리하는
환경에서는 LRU 기반 bounded set으로 대체 검토 필요.

---

## 전체 코드 리뷰

### `_PendingSpec` (변경 없음)

이상 없음. `slot_mapping`이 CPU tensor임은 확인됨.

### `__init__`

- `_stored_chunk_sigs`는 `_pending_store` 직후 초기화. write-through 모드에서도
  초기화되나, `_flush_evicted`는 SPO에서만 호출되므로 접근되지 않음. 무해.

### `handle_block_evictions` (변경 없음)

- `_is_spo` 가드 → write-through 모드 무영향 ✓
- `finished_req_ids` loop 없음 (premature-cleanup 버그 수정 완료) ✓

### `wait_for_save` (변경 없음)

- SPO: `_capture_pending()` → no GPU/SSD I/O ✓
- write-through: parent delegate ✓

### `_capture_pending` (변경 없음)

- `lookup_unpin` 호출 ✓
- `can_save` / `skip >= len(token_ids)` 이중 게이트 ✓
- min() skip 보정 ✓

### `_flush_evicted` (수정됨)

| 항목 | 확인 |
|------|------|
| `engine is None` guard | ✓ |
| `kv_caches` 등록 확인 | ✓ |
| block scan → triggered dict | ✓ |
| `skip >= len` pre-alignment check | ✓ |
| alignment (is_last_prefill=False) | ✓ |
| **NEW** post-alignment `skip >= len` check | ✓ edge case 처리 |
| **NEW** `chunk_sig` 계산 (정렬 후) | ✓ 실제 저장될 청크 기준 |
| **NEW** dedup check | ✓ |
| `store_mask` 생성 | ✓ |
| `engine.store()` | ✓ |
| **NEW** sig 등록 (try 성공 후) | ✓ exception 시 미등록 |
| `_cleanup_req` in finally | ✓ 예외 여부 무관 정리 |

### `_cleanup_req` (변경 없음)

`_stored_chunk_sigs`를 정리하지 않음 — 의도적.
sig는 "이 content가 이미 SSD에 있음"의 영속 마커이므로 req 정리 시 삭제하면 안 됨.

---

## 변경 파일

- `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_spo_connector.py`
  - `__init__`: `_stored_chunk_sigs: set[int] = set()` 추가
  - `_flush_evicted`: post-alignment skip check, chunk_sig dedup 로직 추가
