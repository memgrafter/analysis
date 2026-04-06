# AGENTS.md — ML Research Analysis Corpus

_Last arXiv sync: 2026-04-04 (scan date)._

## What this repo is

Output-only corpus of LLM-generated markdown analyses of ML arXiv papers (2023–2026). Pipeline code lives at [memgrafter/research_crawler_flatagents](https://github.com/memgrafter/research_crawler_flatagents); do not look for runner code here.

## Layout

```
ml_research_analysis_2023/   30,070 files (30,070 unique IDs)
ml_research_analysis_2024/   39,693 files (38,529 unique IDs, 1,164 reruns)
ml_research_analysis_2025/   46,753 files (46,232 unique IDs, 521 reruns)
ml_research_analysis_2026/   18,950 files (18,870 unique IDs, 80 reruns)
analysis_outputs/            research_index.sqlite + digests
scripts/                     index_frontmatter.py, search_topic.py
spot_analyses/               grouped deep-dives by topic
website/                     static browse UI
```

## File format

Named `{arxiv_id}_{slug}_{timestamp}.md`. YAML frontmatter has `arxiv_id`, `core_contribution`, `tags`. Body sections: Quick Facts, Executive Summary, Method Summary, Key Results, Mechanism Analysis, Reproduction Notes, Limitations & Confidence. **Tags are unreliable** — use `core_contribution` and body text for search.

## SQLite index

`analysis_outputs/research_index.sqlite` — two tables:

- **`papers`** (116,503 rows; 2023–2025 buckets indexed): `id`, `filename`, `title`, `arxiv_id`, `tags` (JSON array), `core_contribution`, `indexed_at`, `file_mtime_ns`, `file_size`, `source_dir`. Indexed on `arxiv_id`.
- **`spot_analysis_paper_groups`**: `group_name`, `arxiv_id`, `title`, `source_url`, `filepath`. 1,824 rows across 8 groups (e.g. `test_time_compute_scaling`, `reasoning_distillation`, `multi_agent_debate`).

## Searching

```bash
# ripgrep full-text across years
rg -l "speculative decoding" ml_research_analysis_202*/

# script search (handles noisy tags)
python scripts/search_topic.py --topic "mixture of experts" --alias moe

# reindex after adding files
python scripts/index_frontmatter.py ml_research_analysis_2025
```

## Pipeline summary

Three-phase FlatAgents: (1) Prep — PDF download, text extraction, FTS matching; (2) Expensive — parallel mechanism analysis, reproduction notes, open questions; (3) Wrap — limits/confidence, tagging, assembly, quality judge + repair. 2025 used pony-alpha (glm-5) for phase 2; 2023–2024 used Trinity Large throughout.

## Known gaps

~190 total failures across all years: PDF 404s (~106), context overflow >256k (~60), provider errors (~9), PDF parse errors (~15). These are permanent; no pending retries.
