# S3 Static Site Search: Three Architecture Proposals

Design proposals for a fully client-side search interface over ~91,000 structured
markdown analysis documents, served entirely from S3 static hosting.

## Corpus Profile

| Metric | Value |
|---|---|
| Total documents | ~91,285 (52,100 in 2025 + 39,185 in 2024) |
| Average document size | ~11 KB markdown |
| Raw corpus size | ~980 MB |
| Indexed metadata (SQLite) | 45 MB (52K records currently indexed) |
| Document format | YAML frontmatter + standardized markdown sections |
| Frontmatter fields | `title`, `arxiv_id`, `tags`, `core_contribution` |
| Searchable sections | Executive Summary, Method Summary, Key Results, Mechanisms, etc. |

## Common Constraints

- **No server-side compute**: all search runs in the browser via JavaScript.
- **S3 static hosting**: assets served via CloudFront or S3 website endpoint.
- **Cold-start latency**: user should see a usable search box within 2 seconds.
- **Target search latency**: < 200 ms per keystroke for real-time results.
- **Progressive enhancement**: basic metadata search first, full-text later.

---

## Plan A: Prebuilt Lunr.js Sharded Index

### Overview

Build a full-text search index at deploy time using [Lunr.js](https://lunrjs.com/)
(or the maintained fork [Elasticlunr.js](http://elasticlunr.com/)), serialize it to
JSON shards, and load shards on demand from S3. The client assembles the index in
memory and executes queries locally.

### Preprocessing Pipeline

```
┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐
│ Markdown     │───▶│ Extract + Index  │───▶│ Serialized Shards │
│ Corpus (1GB) │    │ (Node.js build)  │    │ on S3 (~60-80 MB) │
└──────────────┘    └──────────────────┘    └───────────────────┘
```

1. **Extract phase** (Python or Node.js script):
   - Parse each markdown file's YAML frontmatter → `title`, `arxiv_id`,
     `core_contribution`, `tags`.
   - Extract first 300 words of body text as a `snippet` field.
   - Emit one JSON record per document with `id`, `title`, `core_contribution`,
     `snippet`, `arxiv_id`, `tags`.

2. **Index phase** (Node.js):
   - Feed all records into Lunr.js, building a single inverted index.
   - Serialize the index to JSON.
   - Split serialized index into N shards (~2 MB each) by term-range
     (a–d, e–h, …) or by document partition.

3. **Metadata catalog** (tiny JSON, ~5 MB compressed):
   - Flat array of `{id, title, arxiv_id, core_contribution_preview}` for
     instant display before full index loads.

### Index Size Estimates

| Component | Raw Size | Gzipped |
|---|---|---|
| Lunr inverted index (91K docs, 3 fields) | ~120 MB | ~30–40 MB |
| Metadata catalog | ~18 MB | ~4–5 MB |
| Per-shard average (26 shards) | ~4.6 MB | ~1.2 MB |

### Client Architecture

```
index.html
├── app.js (search UI, ~15 KB)
├── catalog.json.gz (metadata, ~5 MB) ← loaded immediately
├── shards/
│   ├── shard-00.json.gz (~1.2 MB)
│   ├── shard-01.json.gz
│   └── ...shard-25.json.gz
└── docs/  ← full markdown files served on demand
```

**Load sequence:**
1. Page loads → fetch `catalog.json.gz` → display search box.
2. User types → load relevant shards on demand based on query terms.
   - Shard manifest maps term prefixes → shard files.
   - Alternatively: load all shards in background via `requestIdleCallback`.
3. Assemble partial Lunr index from loaded shards → execute query → rank results.
4. Display result cards with title, arxiv_id, core_contribution preview.
5. Click result → fetch full markdown file → render with
   [marked.js](https://marked.js.org/).

### Search UX

- **Metadata-first**: instant search over catalog (title + core_contribution)
  while full index loads.
- **Debounced input**: 150 ms debounce on keystrokes, search fires on idle.
- **Faceted filters**: dropdown for year (2023/2024/2025), tag checkboxes.
- **Result preview**: highlighted snippets from `core_contribution`.

### Real-Time Search Extension

**Option 1 — Service Worker pre-cache:**
Populate a Service Worker cache with all shards on first visit. Subsequent searches
are instant from cache — no network round-trips.

**Option 2 — WebSocket live index updates:**
Add a lightweight WebSocket relay (Lambda + API Gateway) that pushes new document
metadata as papers are indexed. Client appends to in-memory Lunr index without
page reload.

**Option 3 — Streaming index via chunked transfer:**
Serve the full index as a streaming JSON response. Parse incrementally using a
streaming JSON parser (e.g., `oboe.js`). Search becomes available progressively as
tokens arrive.

### Trade-offs

| Strength | Weakness |
|---|---|
| Mature library, well-documented | Lunr index can be large in memory (~200 MB decoded) |
| Sharding controls cold-start cost | Shard assembly adds complexity |
| Good relevance ranking (BM25-like) | No fuzzy/typo tolerance without plugin |
| Works fully offline once cached | Build step required on every corpus update |

---

## Plan B: SQLite-over-HTTP with sql.js + HTTP Range Requests

### Overview

Preprocess the corpus into a SQLite database with FTS5 full-text search, host the
`.sqlite` file on S3, and use [sql.js](https://sql.js.org/) (SQLite compiled to
WebAssembly) in the browser. Leverage **HTTP Range Requests** to load only the
database pages needed for each query, avoiding downloading the entire file.

This approach is proven at scale by projects like
[Datasette Lite](https://lite.datasette.io/) and
[sql.js-httpvfs](https://github.com/niccokunzmann/niccokunzmann.github.io).

### Preprocessing Pipeline

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐
│ Markdown     │───▶│ Build FTS5 SQLite │───▶│ papers.sqlite    │
│ Corpus (1GB) │    │ (Python script)   │    │ on S3 (~100 MB)  │
└──────────────┘    └───────────────────┘    └──────────────────┘
```

1. **Extract + load** (Python, extending existing `index_frontmatter.py`):
   - Parse frontmatter: `title`, `arxiv_id`, `tags`, `core_contribution`.
   - Extract first 500 words of body as `abstract_text`.
   - Insert into `papers` table and `papers_fts` FTS5 virtual table.

2. **FTS5 configuration**:
   ```sql
   CREATE VIRTUAL TABLE papers_fts USING fts5(
       title, core_contribution, abstract_text, tags,
       content='papers', content_rowid='id',
       tokenize='porter unicode61'
   );
   ```

3. **Optimize for range requests**:
   - Run `PRAGMA page_size=4096;` and `VACUUM;` to align pages.
   - Generate a page-index sidecar file mapping FTS segment pages to byte
     offsets (used by `sqlite-httpvfs` or `wa-sqlite` HTTP backend).

### Index Size Estimates

| Component | Raw Size | Notes |
|---|---|---|
| SQLite DB with FTS5 (91K docs) | ~100–140 MB | FTS5 adds ~40% overhead |
| sql.js WASM binary | ~1.2 MB | Gzips to ~400 KB |
| Initial page fetch (query) | ~200–500 KB | Only pages touched by query |

### Client Architecture

```
index.html
├── app.js (search UI + sql.js glue, ~20 KB)
├── sql-wasm.wasm (~400 KB gzipped)
├── papers.sqlite (hosted on S3, ~120 MB, accessed via Range)
└── docs/  ← full markdown files served on demand
```

**Using [sql.js-httpvfs](https://github.com/niccokunzmann/niccokunzmann.github.io)
or [wa-sqlite](https://niccokunzmann.github.io/) with an HTTP VFS backend:**

1. Page loads → load sql.js WASM → open HTTP VFS pointing at `papers.sqlite` URL.
2. User types → SQL query runs:
   ```sql
   SELECT title, arxiv_id, snippet(papers_fts, 1, '<b>', '</b>', '…', 40)
   FROM papers_fts WHERE papers_fts MATCH ? ORDER BY rank LIMIT 20;
   ```
3. sql.js issues HTTP Range Requests for only the SQLite pages needed.
4. S3/CloudFront serves the byte ranges; browser caches them.
5. Subsequent queries reuse cached pages — response < 50 ms.

### Search UX

- **Instant cold-start**: WASM loads in ~300 ms, first query fetches ~500 KB.
- **SQL-powered**: full FTS5 syntax — phrase queries, `AND`/`OR`/`NOT`, prefix
  matching, column filters.
- **Real-time typing**: each keystroke appends `*` for prefix matching:
  ```sql
  SELECT ... FROM papers_fts WHERE papers_fts MATCH 'transform*';
  ```
- **Facets via SQL**: `GROUP BY` on year, tag counts, etc.
- **Pagination**: `LIMIT/OFFSET` with total count.

### Real-Time Search Extension

**Option 1 — Delta DB overlay:**
Maintain a small `delta.sqlite` (~1 MB) containing only papers added since last
full build. Client loads this fully into memory. Queries run against both the main
Range-accessed DB and the in-memory delta, results merged client-side.

**Option 2 — Lambda@Edge query proxy:**
For truly real-time search, deploy a Lambda@Edge function that runs `better-sqlite3`
against the same `papers.sqlite` on EFS or S3. Client sends query → Lambda
executes FTS5 → returns JSON. Graceful fallback to client-side if Lambda is cold
or unavailable.

**Option 3 — Streaming inserts via EventSource:**
New papers pushed as Server-Sent Events from a lightweight API. Client inserts into
an in-memory sql.js instance that shadows the main DB. Searches hit both.

### Trade-offs

| Strength | Weakness |
|---|---|
| Minimal preprocessing (extend existing scripts) | sqlite-httpvfs adds complexity |
| FTS5 is fast, well-tested, supports ranked search | Large DB file on S3 (~120 MB) |
| Range requests = low bandwidth per query | Requires S3 Range Request support + CORS |
| SQL flexibility (facets, aggregations, joins) | WASM cold-start ~300 ms |
| Leverages existing `index_frontmatter.py` | Browser memory ~50 MB for sql.js |

---

## Plan C: Precomputed Trigram Index with Tiered Payload Delivery

### Overview

Build a custom trigram-based inverted index at deploy time. Serve it as a tiered
set of static JSON files: a small **metadata tier** (~5 MB) for instant results,
a **trigram posting-list tier** (~20 MB) for fuzzy full-text search, and optional
**document chunk** files for deep content search. All search logic runs in a
Web Worker to keep the UI thread responsive.

This approach provides the best cold-start latency and built-in typo tolerance,
at the cost of a custom index format.

### Preprocessing Pipeline

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Markdown     │───▶│ Trigram Indexer   │───▶│ Tiered JSON on S3   │
│ Corpus (1GB) │    │ (Python build)   │    │ (~25 MB compressed) │
└──────────────┘    └──────────────────┘    └─────────────────────┘
```

1. **Tier 0 — Metadata catalog** (built first, served first):
   - `catalog.json`: array of `{id, title, arxiv_id, year, contribution_preview}`.
   - ~18 MB raw, ~4 MB gzipped. Enables instant title/ID search.

2. **Tier 1 — Trigram posting lists** (built second):
   - For each document, extract trigrams from `title + core_contribution`.
   - Build inverted index: `trigram → [doc_id, doc_id, ...]`.
   - Split into 256 shard files by first-byte hash of trigram.
   - Each shard ~80 KB gzipped.

3. **Tier 2 — Document content chunks** (optional, built last):
   - Group documents by arxiv_id prefix (e.g., `2024/`, `2025/`).
   - Each chunk: 500 documents' body text (first 500 words each), ~2 MB gzipped.
   - Loaded on demand for deep content search.

### Index Size Estimates

| Tier | Purpose | Raw Size | Gzipped | Load Trigger |
|---|---|---|---|---|
| 0 — Catalog | Instant title search | ~18 MB | ~4 MB | Page load |
| 1 — Trigram shards | Fuzzy full-text | ~50 MB total | ~12 MB total | First query |
| 2 — Content chunks | Deep body search | ~200 MB total | ~50 MB total | On demand |

### Client Architecture

```
index.html
├── app.js (UI controller, ~10 KB)
├── search-worker.js (Web Worker, ~8 KB)
├── catalog.json.gz (~4 MB) ← Tier 0
├── trigrams/
│   ├── 00.json.gz (~80 KB) ← Tier 1, loaded per query
│   ├── 01.json.gz
│   └── ...ff.json.gz
├── chunks/
│   ├── 2024-000.json.gz (~2 MB) ← Tier 2, on demand
│   └── ...
└── docs/  ← full markdown files
```

**Load sequence:**
1. Page loads → fetch `catalog.json.gz` → index in Web Worker.
2. User types 1 character → search catalog (title substring match, < 10 ms).
3. User types 3+ characters → extract trigrams from query → fetch relevant
   trigram shard files → intersect posting lists → rank results.
4. Worker posts ranked results back to main thread → render.
5. User clicks "Deep Search" → load relevant Tier 2 chunks → re-rank with
   body text matches.

### Trigram Search Algorithm

```javascript
// In search-worker.js
function search(query) {
  const trigrams = extractTrigrams(query.toLowerCase());
  // e.g., "transformer" (11 chars) → 9 trigrams:
  // ["tra","ran","ans","nsf","sfo","for","orm","rme","mer"]

  // Fetch only the shard files containing these trigrams
  const shardIds = new Set(trigrams.map(t => hashByte(t)));
  const postings = await loadShards([...shardIds]);

  // Intersect posting lists with scoring
  const scores = new Map();
  for (const tri of trigrams) {
    for (const docId of postings[tri] || []) {
      scores.set(docId, (scores.get(docId) || 0) + 1);
    }
  }

  // Rank: docs matching more trigrams score higher
  // Normalize by total trigrams for relevance
  return [...scores.entries()]
    .map(([id, hits]) => ({ id, score: hits / trigrams.length }))
    .filter(r => r.score > 0.5)  // require >50% trigram overlap
    .sort((a, b) => b.score - a.score)
    .slice(0, 20);
}
```

### Search UX

- **Progressive refinement**:
  - 1–2 chars → catalog title filter (instant).
  - 3+ chars → trigram search (< 100 ms after shards cached).
  - Optional deep search button for body text.
- **Built-in fuzzy matching**: trigrams naturally handle typos — "transformr"
  shares 7/9 trigrams with "transformer" (78% overlap, above threshold).
- **Web Worker**: all index operations off main thread, UI stays at 60 fps.
- **Highlighted results**: trigram positions mapped back to display text.

### Real-Time Search Extension

**Option 1 — Incremental shard updates:**
New documents generate a small `delta-catalog.json` and `delta-trigrams/` overlay.
Client fetches delta files (< 100 KB) and merges into in-memory index. Rebuild
full shards on a schedule (daily/weekly).

**Option 2 — Broadcast Channel synchronization:**
If the user has multiple tabs open, one tab fetches updates and broadcasts to
others via `BroadcastChannel` API. Combined with periodic polling of a
`manifest.json` (containing build timestamp), this gives near-real-time updates
with zero server cost.

**Option 3 — CloudFront Functions query router:**
Deploy a CloudFront Function (~1 ms execution) that intercepts search requests.
For queries matching a known hot-path (trending terms), return a precomputed result
file. For new queries, fall through to client-side trigram search. This hybrid
approach handles burst traffic without Lambda costs.

### Trade-offs

| Strength | Weakness |
|---|---|
| Best cold-start (catalog only = 4 MB) | Custom index format, no community library |
| Built-in typo/fuzzy tolerance | Trigram index less precise than BM25 ranking |
| Web Worker keeps UI responsive | Higher shard count (~256 files) on S3 |
| Granular bandwidth (load only needed shards) | More complex build pipeline |
| No WASM dependency | Deep search (Tier 2) adds significant payload |

---

## Comparison Matrix

| Criterion | Plan A: Lunr Shards | Plan B: SQLite-over-HTTP | Plan C: Trigram Tiers |
|---|---|---|---|
| **Cold-start payload** | ~5 MB (catalog) | ~400 KB (WASM) + ~500 KB (first query pages) | ~4 MB (catalog) |
| **Search latency (cached)** | < 100 ms | < 50 ms | < 100 ms |
| **Fuzzy/typo tolerance** | Plugin required | No (exact + prefix only) | Built-in |
| **Ranking quality** | Good (BM25-like) | Excellent (FTS5 rank) | Moderate (trigram overlap) |
| **Build complexity** | Medium | Low (extend existing scripts) | High |
| **S3 storage** | ~40 MB | ~120 MB | ~70 MB |
| **Browser memory** | ~200 MB | ~50 MB | ~30 MB (Tier 0+1) |
| **Offline support** | Excellent (SW cache) | Moderate (need all pages) | Good (cache shards) |
| **Real-time extension** | Moderate effort | Low effort (delta DB) | Low effort (delta files) |
| **Existing code reuse** | None (new pipeline) | High (`index_frontmatter.py`) | None (new pipeline) |

## Recommendation

**Plan B (SQLite-over-HTTP)** is recommended as the primary implementation for this
corpus because:

1. **Leverages existing infrastructure**: the `index_frontmatter.py` script and
   `research_index.sqlite` database are already built and maintained.
2. **Lowest build complexity**: extend the existing Python script to add FTS5.
3. **Best search quality**: FTS5 with Porter stemming provides the strongest
   relevance ranking of all three options.
4. **Lowest browser memory**: only ~50 MB vs 200 MB for Lunr.
5. **Range requests minimize bandwidth**: only database pages touched by the query
   are fetched, making it efficient even for the full 120 MB database.
6. **Simplest real-time extension**: a small delta database overlay is trivial to
   implement and requires no server-side components.

Plans A and C are strong alternatives if specific constraints apply — Plan A if
offline-first is critical, Plan C if typo tolerance is a must-have feature.

## Implementation Sketch (Plan B, Recommended)

### Step 1: Extend the build script

```python
# In scripts/index_frontmatter.py, add after table creation:
cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
        title, core_contribution, abstract_text,
        content='papers', content_rowid='id',
        tokenize='porter unicode61'
    );
""")
```

### Step 2: S3 deployment

```bash
# Build optimized DB
python scripts/build_search_db.py --output search.sqlite
sqlite3 search.sqlite "PRAGMA page_size=4096; VACUUM;"

# Upload to S3 with proper headers
aws s3 cp search.sqlite s3://bucket/search.sqlite \
  --content-type application/octet-stream \
  --cache-control "public, max-age=86400"

# Upload static site
aws s3 sync site/ s3://bucket/ --exclude "*.sqlite"
```

### Step 3: Minimal client

```html
<!DOCTYPE html>
<html>
<head><title>ML Research Search</title></head>
<body>
  <input type="search" id="q" placeholder="Search 91,000 papers…">
  <div id="results"></div>
  <script type="module">
    import { createDbWorker } from "sql.js-httpvfs";
    const worker = await createDbWorker(
      [{ from: "jsonconfig", configUrl: "/config.json" }],
      "/sql.js-httpvfs/sqlite.worker.js",
      "/sql.js-httpvfs/sql-wasm.wasm"
    );
    document.getElementById("q").addEventListener("input", async (e) => {
      const q = e.target.value.trim();
      if (q.length < 2) return;
      const results = await worker.db.query(
        `SELECT title, arxiv_id,
                snippet(papers_fts, 1, '<b>', '</b>', '…', 40) as snip
         FROM papers_fts WHERE papers_fts MATCH ?
         ORDER BY rank LIMIT 20`,
        [q + "*"]
      );
      // render results...
    });
  </script>
</body>
</html>
```
