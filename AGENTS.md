# AGENTS.md — ML Research Analysis Corpus

## What this repo is

Output-only corpus of LLM-generated markdown analyses of ML arXiv papers (2023–2026). Pipeline code lives at [memgrafter/research_crawler_flatagents](https://github.com/memgrafter/research_crawler_flatagents); do not look for runner code here.

## Layout

```
ml_research_analysis_2023/   29,961 files (29,961 unique IDs)
ml_research_analysis_2024/   39,185 files (38,027 unique IDs, 1,158 reruns)
ml_research_analysis_2025/   52,099 files (51,517 unique IDs, 582 reruns)
ml_research_analysis_2026/   _ files (backfill in progress)
analysis_outputs/            research_index.sqlite + digests
scripts/                     index_frontmatter.py, search_topic.py
spot_analyses/               grouped deep-dives by topic
website/                     static browse UI
```

## File format

Named `{arxiv_id}_{slug}_{timestamp}.md`. YAML frontmatter has `arxiv_id`, `core_contribution`, `tags`. Body sections: Quick Facts, Executive Summary, Method Summary, Key Results, Mechanism Analysis, Reproduction Notes, Limitations & Confidence. **Tags are unreliable** — use `core_contribution` and body text for search.

## SQLite index

`analysis_outputs/research_index.sqlite` — two tables:

- **`papers`** (52,099 rows, 2025 bucket indexed): `id`, `filename`, `title`, `arxiv_id`, `tags` (JSON array), `core_contribution`, `indexed_at`, `file_mtime_ns`, `file_size`. Indexed on `arxiv_id`.
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
