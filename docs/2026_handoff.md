# 2026 Split — Handoff Doc

## What's done

### Commit `d4c3536eb` — Create 2026 bucket
- Created `ml_research_analysis_2026/` with 4,740 `.md` files (3,888 Jan `2601.*`, 852 Feb `2602.*`)
- Moved from `ml_research_analysis_2025/` which now has 47,359 `.md` files
- Pruned 4,742 orphaned `26xx` rows from `analysis_outputs/research_index.sqlite` (now 116,503 rows)
- The 2026 files are **not yet indexed** in `research_index.sqlite`

### Commit `1a0ddfded` — Update website build pipeline for 2026
- `website/scripts/build_db.sh` — added `SRC_2026`
- `website/scripts/build_search_db.py` — added 2026 to `DEFAULT_SOURCE_DIRS`
- `website/scripts/build_cloud_terms.py` — added 2026 to `YEAR_OPTIONS`, `YEAR_DIR_MAP`, canonical seed year, union term preference
- `website/scripts/integration_test.sh` — passes `SRC_2026` to build and local server
- Created `website/data/word_clouds/2026/.gitkeep` (empty, needs seed terms)

### Current state
```
ml_research_analysis_2023/   29,961 files
ml_research_analysis_2024/   39,185 files
ml_research_analysis_2025/   47,359 .md files + 1 bad/ subdir (120 early test files)
ml_research_analysis_2026/    4,740 .md files (not indexed, not in website search DB)
```
- `research_index.sqlite`: 116,503 rows (2023+2024+2025 only)
- Website search DB (`website/search/`): stale, last built Feb 17 with old 2025 folder (includes 2026 papers). Needs rebuild.
- No changes pushed to origin yet.

---

## Remaining work

### 1. Index 2026 in `research_index.sqlite`
```bash
python scripts/index_frontmatter.py ml_research_analysis_2026 --full
```
This appends ~4,740 rows. No collisions expected (filenames are globally unique). Verify:
```bash
sqlite3 analysis_outputs/research_index.sqlite "SELECT COUNT(*) FROM papers;"
# Should be ~121,243
```

### 2. Digest remaining 2026 papers
The pipeline lives at `memgrafter/research_crawler_flatagents` (external repo).
1. Get the full list of `2601.*` and `2602.*` ML papers from arXiv
2. Cross-reference against existing filenames in `ml_research_analysis_2026/`
3. Run the FlatAgents pipeline on missing papers
4. Copy outputs into `ml_research_analysis_2026/`
5. Re-run the indexer: `python scripts/index_frontmatter.py ml_research_analysis_2026 --full`

### 3. Populate word cloud seed terms for 2026
Create `.txt` files in `website/data/word_clouds/2026/` matching the structure of `website/data/word_clouds/2025/`:
```bash
ls website/data/word_clouds/2025/
# Use those as templates — one term per line, # for comments
```
Without seed terms, the 2026 cloud tab will be empty but won't break the build.

### 4. Rebuild website search DB
```bash
cd website && ./scripts/build_db.sh
```
This scans all four year folders, builds a new FTS5 SQLite, and regenerates `cloud-terms.json`. Takes ~10 min on the full corpus. The old `search-1a2ab3b72666.sqlite` (917 MB) gets replaced.

Quick smoke test:
```bash
MAX_FILES=1500 ./scripts/integration_test.sh
```

### 5. Update `AGENTS.md`
Update the Layout section, file counts, and `papers` table row count. Add `ml_research_analysis_2026/` to the layout block.

### 6. Push
```bash
git push origin main
```

---

## Gotchas

| Issue | Detail |
|---|---|
| **`--prune` is now source_dir-scoped** | Each row tracks which folder it was indexed from. Pruning `ml_research_analysis_2025` only deletes rows tagged `source_dir=ml_research_analysis_2025` that are missing from disk. Other years are untouched. All 116,503 existing rows have been backfilled. |
| **`bad/` subdir in 2025** | Contains 120 early pipeline test files (Feb 6–8 runs). Not indexed, not harmful. Clean up whenever. |
| **2024 folder has 2026-timestamped files** | These are reruns of 2024 papers generated during 2026 pipeline runs. They stay in `ml_research_analysis_2024/`. The arxiv ID prefix (`24xx`) determines the bucket, not the generation timestamp. |
| **Older papers in 2025 bucket** | 703 files with `24xx`/`23xx`/`22xx` prefixes — reruns of older papers from 2025 pipeline runs. Same logic: they stay. |
| **Website `local_server.py` source dirs** | The integration test passes `--source-dir` for 2026 to the local server. Check if `local_server.py` needs the same update for standalone `./scripts/run.sh` usage. |
