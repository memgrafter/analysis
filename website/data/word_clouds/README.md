# Local Word Cloud Seed Terms

This directory vendors a local copy of the yearly word-cloud seed terms used for `/cloud` and cloud-term scoring.

## Source provenance

Copied from:

- `~/code/research_crawler/research_paper_analysis_v2/queries/word_clouds_2023_organic_semantic_cleaned` → `data/word_clouds/2023/`
- `~/code/research_crawler/research_paper_analysis_v2/queries/word_clouds_2024` → `data/word_clouds/2024/`
- `~/code/research_crawler/research_paper_analysis_v2/queries/word_clouds` (canonical/2025) → `data/word_clouds/2025/`

Only the canonical `.txt` files present in the 2025 set are vendored for all years, so each year has the same themed file set.

## Why this exists

The website build should not depend on another repository being present locally.

By keeping the seed terms here, `scripts/build_cloud_terms.py` can always build `search/cloud-terms.json` and SQLite cloud cache tables (`cloud_term`, `cloud_term_postings`) from in-repo data.
