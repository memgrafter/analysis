# Search DB Build Plan (SQLite)

## Goal
Generate a reproducible SQLite search database from curated markdown digests, with easy full regeneration as curation changes.

## Scope (v1)
- Full rebuild on each run (no incremental updates).
- Source directories read in place (no asset moves).
- Output one SQLite DB + one manifest JSON.

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
Planned script:
- `scripts/build_search_db.py`

Planned CLI:
```bash
python3 scripts/build_search_db.py \
  --source-dir ../ml_research_analysis_2023 \
  --source-dir ../ml_research_analysis_2024 \
  --source-dir ../ml_research_analysis_2025 \
  --output-dir build \
  --preview-words 500
```

Optional flags:
- `--db-name search.sqlite` (default logical name)
- `--fail-on-parse-error` (default true)
- `--max-parse-errors <N>` (if we later allow partial)

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
Write to `build/`:
- `search-<build_hash>.sqlite`
- `search-manifest.json`

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

## Regeneration Workflow
When curation changes:
1. Re-run build script.
2. Upload new hashed DB + refreshed manifest.
3. Client picks up new DB via manifest on next load.

No data migration required in v1.
