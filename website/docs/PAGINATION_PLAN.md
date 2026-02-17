# Pagination Plan (Composite-Key Cursor + Page Size Controls)

## Goal
Add robust pagination for search results with stable ordering and user-friendly controls at both top and bottom of the results list.

## UX Requirements
1. Page size selector with options: **10, 25, 50, 100**.
2. Pagination controls at **top and bottom** of results (no forced scroll to page-turn).
3. URL-addressable state so pagination can be refreshed/shared.

---

## URL Contract
Use query params on `/search/`:
- `q`: search query
- `ps`: page size (`10|25|50|100`, default `25`)
- `cursor`: opaque cursor token for next/prev navigation
- `dir`: `next|prev` (used with cursor; optional if stack-based)

Examples:
- `/search/?q=test-time&ps=25`
- `/search/?q=test-time&ps=50&cursor=<token>&dir=next`

---

## Sorting Contract (deterministic)
For FTS search, always sort by a stable composite key:
1. `score ASC` (BM25)
2. `digest_id ASC` (unique tie-breaker)

This ensures deterministic pagination even when many rows have equal score.

---

## Composite-Key Cursor Format
Cursor payload contains:
- query hash (`qh`)
- page size (`ps`)
- last row score (`s`)
- last row digest_id (`id`)

Suggested serialized payload (base64url JSON):
```json
{"qh":"...","ps":25,"s":-12.34,"id":"2404.18923_..."}
```

Validation on decode:
- `qh` matches current query
- `ps` matches current page size
- required fields exist and parse correctly

---

## SQL Strategy

### First page
```sql
SELECT ...
FROM digests_fts ...
WHERE digests_fts MATCH :q
ORDER BY score ASC, digest_id ASC
LIMIT :limit_plus_one;
```

### Next page (seek by composite key)
```sql
SELECT ...
FROM digests_fts ...
WHERE digests_fts MATCH :q
  AND (
    score > :last_score
    OR (score = :last_score AND digest_id > :last_digest_id)
  )
ORDER BY score ASC, digest_id ASC
LIMIT :limit_plus_one;
```

`limit_plus_one` determines `has_more`.

---

## Prev Page Behavior
v1 approach:
- Maintain client-side cursor history stack per query/page-size.
- "Prev" pops from stack; no reverse SQL needed in v1.

This keeps implementation simple while preserving good UX.

---

## UI Components
Add pagination controls in two places:
1. **Top controls** (between search box/status and result list)
2. **Bottom controls** (below result list)

Each control group includes:
- Prev button
- Next button
- Current page indicator
- Page size selector (10/25/50/100)

Behavior:
- Changing page size resets pagination to first page.
- Buttons disabled when no previous/next page available.

---

## State Model
Track in memory:
- `query`
- `page_size`
- `current_page`
- `next_cursor`
- `cursor_stack` (for Prev)
- `has_more`

Persist in URL:
- `q`, `ps`, and current `cursor` (if not first page)

---

## Error Handling
- Invalid cursor: ignore cursor and restart at first page with warning.
- Query/page size mismatch with cursor: reset to first page.
- FTS unavailable in range backend: fail fast (current policy), avoid broad fallback scans.

---

## Acceptance Criteria
1. User can page through results with stable ordering (no duplicates/skips).
2. Page size selector supports 10/25/50/100 at both top and bottom controls.
3. Next/Prev available at both top and bottom controls.
4. URL reflects current query + page size + cursor state.
5. Refresh retains current page context.
6. Changing page size resets to page 1.
