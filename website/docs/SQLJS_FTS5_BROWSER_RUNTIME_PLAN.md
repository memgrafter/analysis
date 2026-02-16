# Plan: True FTS5 in Browser Search Runtime (sql.js)

## Status
Planned. No implementation in this document.

## Why this plan
Current search vertical slice works, but may fall back from FTS to `LIKE` if the loaded sql.js WASM runtime lacks FTS5 support. We want guaranteed in-browser FTS search semantics.

## Scope
- Ensure browser-side SQLite runtime supports FTS5.
- Remove ambiguity from CDN-provided sql.js binaries.
- Keep static hosting model.

---

## Clarification: What is failing today
- Build-time DB generation uses local Python `sqlite3` and successfully creates FTS5 tables.
- Query-time browser runtime is a separate SQLite engine compiled to WASM (`sql.js`).
- If that WASM binary is not compiled with `SQLITE_ENABLE_FTS5`, browser queries fail with `no such module: fts5`.
- This is not a user browser installation issue; it is a runtime binary/compile-flags issue.

---

## Deliverables
1. Pinned, self-hosted sql.js runtime assets in repo (or pinned artifact bucket).
2. Build recipe for sql.js WASM with required SQLite compile flags.
3. Verification script that proves FTS5 support before app usage.
4. Search page configured to load only verified runtime assets.
5. Integration test gate that fails when FTS5 is unavailable.

---

## Required Runtime Features
Minimum required SQLite compile option:
- `SQLITE_ENABLE_FTS5`

Optional but recommended:
- `SQLITE_ENABLE_JSON1`

---

## Implementation Phases

### Phase 1 — Runtime Pinning
- Stop relying on unverified CDN runtime for core behavior.
- Add versioned runtime paths, e.g.:
  - `/assets/sqljs/sql-wasm.js`
  - `/assets/sqljs/sql-wasm.wasm`

### Phase 2 — Build sql.js with FTS5
- Build sql.js WASM from source with explicit SQLite flags.
- Record exact source version + build command in docs.
- Commit checksums for produced artifacts.

### Phase 3 — Runtime Verification
- Add a script that executes runtime checks:
  1. `CREATE VIRTUAL TABLE t USING fts5(x)` succeeds.
  2. Insert/query with `MATCH` succeeds.
- Fail pipeline if verification fails.

### Phase 4 — App Wiring
- Update search client to load local pinned runtime assets.
- Keep temporary fallback mode only during transition.

### Phase 5 — Enforce FTS Path
- Remove fallback mode after repeated green runs.
- Require FTS mode in integration tests.

---

## Test/Validation Criteria
- Browser search queries execute via FTS table (`digests_fts`) without fallback.
- Query examples using prefix/boolean syntax behave as expected.
- Integration runner fails if FTS5 unavailable.
- Same runtime behavior locally and on S3 deploy.

---

## Risks
- WASM build reproducibility drift across machines.
- Accidental runtime swap (CDN/local mismatch).
- Increased asset management burden (versioning/checksums).

## Mitigations
- Pin source version and hash artifacts.
- Always load runtime from controlled local/static path.
- Add runtime self-check before enabling search UI.

---

## Exit Criteria
- FTS5 confirmed in-browser from pinned runtime.
- Search UI uses FTS-only mode.
- Integration suite enforces FTS availability as a hard gate.
