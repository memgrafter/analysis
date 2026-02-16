# Plan: Static Data Plane for MCP-Compatible Search via SQLite Range Requests

## Status
Exploratory plan, sequenced after `SQLJS_FTS5_BROWSER_RUNTIME_PLAN.md`.

## Intent
Support MCP-driven search while preserving static hosting for data (`manifest + sqlite` on S3) and using HTTP Range Requests for efficient access.

## Important Constraint
A **pure static website alone cannot host an MCP server**. MCP requires an executable server process (stdio/http transport).

So this plan targets:
- **Static data plane**: fully static S3-hosted SQLite artifacts.
- **Thin MCP control plane**: minimal runtime process that reads static artifacts and serves MCP tools.

---

## Architecture (Recommended)

### 1) Static Data Plane (S3)
- `search/manifest.json`
- `search/search-<hash>.sqlite`
- Optional shard/manifests

Requirements:
- CORS configured for MCP runtime host(s)
- HTTP Range enabled (S3 supports this)
- Cache-control: immutable for DB, short/no-cache for manifest

### 2) MCP Thin Server (Runtime)
A lightweight MCP server exposes tools such as:
- `search_digests(query, limit, mode)`
- `get_digest_by_id(digest_id)`
- `resolve_arxiv_id(arxiv_id)`

Internals:
- Reads `manifest.json`
- Opens remote SQLite via HTTP range-capable VFS/runtime
- Executes SQL queries and returns JSON rows

### 3) Clients
- LLM agents use MCP tools.
- Web UI can continue using direct static browser path.

---

## Why this split works
- Keeps search corpus and index deployment static and cheap.
- Avoids building/operating full search backend index service.
- MCP layer remains small and mostly stateless.

---

## Query Path Options for MCP Server

### Option A — Remote SQLite over HTTP Range (Preferred)
- MCP runtime uses SQLite-compatible HTTP VFS.
- No full DB download per query.
- Closest to static-first architecture.

### Option B — Manifest + Precomputed JSON (Not for general search)
- Works only for predefined lookups/slices.
- Insufficient for arbitrary ranked search queries.

### Option C — Full DB download to local cache at MCP startup
- Simpler implementation, higher cold-start/network cost.
- Acceptable only for small DBs or controlled environments.

---

## Proposed MCP Tool Contract (v1)

### `search_digests`
Inputs:
- `q: string`
- `limit: number (default 20, max 100)`
- `offset: number (default 0)`
- `mode: "fts" | "arxiv_exact"`

Output:
- `{ digest_id, arxiv_id, title, score?, snippet? }[]`

### `get_digest_by_id`
Inputs:
- `digest_id: string`

Output:
- metadata row from SQLite + canonical raw URL `/view/<digest_id>.md`

### `resolve_arxiv_id`
Inputs:
- `arxiv_id: string`

Output:
- canonical digest id using `arxiv_latest`

---

## Security / Abuse Controls
- Enforce max `limit` and query timeout.
- Parameterized SQL only.
- Optional allowlist for query syntax.
- Rate limiting at MCP process boundary if exposed remotely.

---

## Integration/Validation Plan
1. Verify static artifact access:
   - manifest fetch
   - range reads on sqlite object
2. MCP smoke tests:
   - `search_digests("transformers")`
   - `resolve_arxiv_id("2404.18923")`
3. Correctness checks:
   - tool outputs match browser UI top results for sample queries
4. Failure behavior:
   - missing manifest/db
   - range request errors
   - malformed query

---

## Sequencing
1. Complete and stabilize true in-browser FTS runtime plan.
2. Reuse same SQLite schema/artifacts for MCP thin server.
3. Implement MCP tool surface.
4. Add MCP integration checks.

---

## Non-Goals (v1)
- Replacing static web viewer/search UI.
- Building a full dynamic search cluster/API.
- Supporting arbitrary write operations to SQLite.

---

## Exit Criteria
- Static artifacts remain single source of truth.
- MCP server can answer ranked search and lookup tools against static SQLite data.
- Range-based access validated in integration checks.
