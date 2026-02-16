# Search DB Build Plan (SQLite)

## Goal
Generate a reproducible SQLite search database from curated markdown digests, with easy full regeneration as curation changes.

## Scope (v1)
- Full rebuild on each run (no incremental updates).
- Source directories read in place (no asset moves).
- Output one SQLite DB + one manifest JSON.
- Integration-only testing via one shell entrypoint (no other test suites).

---

## Source Inputs
Read from:
- `../ml_research_analysis_2023/`
- `../ml_research_analysis_2024/`
- `../ml_research_analysis_2025/`

Rules:
- Include all `*.md` recursively.
- `digest_id = <filename without .md>`
- Flatten all years into one namespace.
- Build fails on duplicate `digest_id` collisions.

---

## SQLite Schema (v1)

### `digests`
Primary metadata + preview content used by UI and joins.

Columns:
- `id INTEGER PRIMARY KEY`
- `digest_id TEXT UNIQUE NOT NULL`
- `arxiv_id TEXT`
- `title TEXT`
- `core_contribution TEXT`
- `tags TEXT` (joined/normalized string)
- `body_preview TEXT` (first N words, default 500)
- `source_path TEXT NOT NULL`
- `year INTEGER`
- `timestamp_suffix TEXT` (parsed from filename if present)

Indexes:
- `UNIQUE(digest_id)`
- `INDEX(arxiv_id)`

### `digests_fts`
FTS5 virtual table for ranked search.

Suggested config:
```sql
CREATE VIRTUAL TABLE digests_fts USING fts5(
  title,
  core_contribution,
  tags,
  body_preview,
  content='digests',
  content_rowid='id',
  tokenize='porter unicode61'
);
```

### `arxiv_latest`
Canonical digest mapping for future `/view/?id=<arxiv-id>` resolution.

Columns:
- `arxiv_id TEXT PRIMARY KEY`
- `digest_id TEXT NOT NULL`

Selection rule:
- If multiple digests share arXiv ID, choose latest by parsed timestamp suffix.

---

## Build Script Contract
Implemented script:
- `scripts/build_search_db.py`

Typical CLI (local serving-friendly output path):
```bash
python3 scripts/build_search_db.py \
  --source-dir ../ml_research_analysis_2023 \
  --source-dir ../ml_research_analysis_2024 \
  --source-dir ../ml_research_analysis_2025 \
  --output-dir search \
  --preview-words 500
```

Optional flags:
- `--db-name search.sqlite` (default logical name)
- `--max-files <N>` (optional bounded run)
- `--fail-on-parse-error` / `--no-fail-on-parse-error`
- `--max-parse-errors <N>`

---

## Build Steps
1. Discover markdown files from all source dirs.
2. Parse frontmatter + body.
3. Insert rows into `digests`.
4. Populate/rebuild `digests_fts` from `digests`.
5. Build `arxiv_latest` mapping.
6. Optimize DB and finalize file.
7. Emit manifest JSON with metadata.

---

## SQLite Pragmas / Finalization
During ingest (speed):
- `PRAGMA journal_mode=MEMORY;`
- `PRAGMA synchronous=OFF;`
- `PRAGMA temp_store=MEMORY;`

Finalize (serving-friendly):
- `INSERT INTO digests_fts(digests_fts) VALUES('optimize');`
- `PRAGMA page_size=4096;`
- `VACUUM;`

---

## Output Artifacts
Write to configured `--output-dir` (typically `search/` for local serving):
- `search-<build_hash>.sqlite`
- `manifest.json`
- `search-manifest.json` (compat copy)

Manifest fields (v1):
- `build_hash`
- `built_at`
- `db_file`
- `digest_count`
- `arxiv_count`
- `source_dirs`

---

## S3 Publish Pattern
- Upload immutable DB file:
  - `search/search-<build_hash>.sqlite`
  - cache-control: long max-age, immutable
- Upload mutable manifest:
  - `search/manifest.json`
  - cache-control: short/no-cache

Client behavior:
1. Load `search/manifest.json`.
2. Open referenced DB file.

---

## Validation / Build Failure Conditions
Fail build when:
- duplicate `digest_id` detected
- schema creation fails
- parse errors exceed allowed threshold (v1 default: any parse error fails)

Log summary each run:
- scanned files
- indexed rows
- unique arXiv IDs
- DB file size
- build duration

---

## Local Serving for End-to-End Testing
Use the existing local server to test viewer + raw markdown + search assets without moving source digests.

Run:
```bash
./scripts/run.sh
```

Expected local URLs:
- Home: `http://localhost:8000/`
- Viewer: `http://localhost:8000/view/?id=<digest-id>`
- Viewer alias: `http://localhost:8000/view?id=<digest-id>`
- Raw markdown: `http://localhost:8000/view/<digest-id>.md`

Search URLs (after DB build):
- `http://localhost:8000/search/` (search UI vertical slice)
- `http://localhost:8000/search/?q=<query>`
- `http://localhost:8000/search/manifest.json`
- `http://localhost:8000/search/search-<build_hash>.sqlite`

Range request check (required for sqlite-over-HTTP behavior):
```bash
curl -i -H "Range: bytes=0-1023" \
  http://localhost:8000/search/search-<build_hash>.sqlite
```
Expected: `206 Partial Content`.

---

## Integration Test Runner (Only Test Suite)
Use a single shell entrypoint for all integration checks:
- `scripts/integration_test.sh`

Single command contract:
```bash
./scripts/integration_test.sh
```

Runner responsibilities (v1):
1. Build SQLite + manifest via `scripts/build_search_db.py`.
2. Start local server (`scripts/run.sh`) on localhost test port.
3. Validate HTTP routes and outputs with shell checks (`curl`, `jq`, `sqlite3` as needed):
   - `/` returns 200
   - `/view/` returns 200
   - `/view?id=<sample>` resolves
   - `/view/<sample>.md` returns 200
   - `/search/` returns 200
   - `/search/manifest.json` returns 200 and references DB file
   - `/search/<db_file>` returns 200
   - Range request on DB returns `206 Partial Content`
4. Exit non-zero on any failed check.

Testing policy:
- No separate unit tests.
- No separate Python/JS test runners.
- Integration runner is the only required/official test path.

---

## Regeneration Workflow
When curation changes:
1. Re-run build script.
2. Upload new hashed DB + refreshed manifest.
3. Client picks up new DB via manifest on next load.

No data migration required in v1.
