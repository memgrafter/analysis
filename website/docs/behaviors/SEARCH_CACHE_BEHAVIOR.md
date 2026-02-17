# Search Cache Behavior

This document describes the current client-side search caching behavior in `assets/search.js`.

## Scope

Caching is implemented in browser `sessionStorage` (tab/session scoped), keyed by:

- DB file (`manifest.db_file`)
- query text
- sort
- page size
- cursor token
- selected years
- selected scopes

This means cache entries are specific to a full search state, not just query text.

---

## Cache-first rules

### 1) Initial page load with querystring is cache-first
If the user lands on `/search/?q=...` (with optional `sort`, `ps`, `cursor`, `y`, `f`), the app checks cache first.

- If hit: cached results render immediately and status indicates cache.
- If miss: a live search runs.

### 2) Query/filter/sort/page-size/cursor changes are cache-first
For non-manual flows (including switching queries or changing filters), the app attempts cache first for that exact state.

Example:

- search `transformer`
- search `test`
- search `transformer` again

The second `transformer` attempt reuses cache first (if present).

### 3) Cached result writes
After each successful live search, the result page is written to cache for that exact state.

---

## Manual Search button behavior

Pressing **Search** behaves as follows:

- If the submitted state is the **same state currently displayed**, Search forces a **fresh live query** (cache bust).
- If the submitted state is **different** from what is currently displayed, Search uses cache-first for that new state.

This gives users a deterministic way to refresh stale cached results without losing fast cache reuse for normal navigation/state changes.

---

## Notes

- Cache storage is best-effort; quota/errors are ignored and search still works.
- Cache lifetime is session-scoped (cleared when session storage is cleared or tab/session ends).
- Cache keys include DB hash/name, so DB rebuilds naturally invalidate old entries.
