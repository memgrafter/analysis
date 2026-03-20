# ML Research Analysis Corpus

A structured library of **121,245** machine-learning paper analyses covering arXiv publications from 2023–2025. Each paper is distilled into a standardised markdown report so you can survey findings, compare mechanisms, and spot trends without reading thousands of PDFs.

| Year | Analyses | Unique papers |
|------|----------|---------------|
| 2023 | 29,961   | 29,961        |
| 2024 | 39,185   | 38,027        |
| 2025 | 52,099   | 51,517        |

---

## What an analysis looks like

Every file follows the same template. Here's a trimmed example (`2511.21730_a-benchmark-for-procedural-memory-retrieval_…md`):

```yaml
# frontmatter
arxiv_id: '2511.21730'
core_contribution: >
  Introduces the first benchmark for evaluating procedural
  memory retrieval in language agents, isolating retrieval
  from execution …
tags: [procedural, retrieval, memory, …]   # ⚠ see caveat below
```

```markdown
## Quick Facts          — arXiv link, authors, headline numbers
## Executive Summary    — what the paper does in one paragraph
## Method Summary       — experimental setup, models, data
## Key Results          — quantitative findings
## Mechanism Analysis   — *why* the approach works (multiple sub-sections)
## Reproduction Notes   — hyperparameters, compute, data details
## Limitations & Confidence
```

> **Tag caveat:** Auto-generated `tags` are noisy (every record shares a long generic tail). Prefer searching `core_contribution`, titles, and body text. Tag regeneration is planned.

---

## How to use this corpus

### Browse a topic

Pick a year folder and search with ripgrep:

```bash
# Find all papers mentioning mixture-of-experts
rg -l "mixture of experts|MoE" ml_research_analysis_2025/

# Full-text search across every year
rg -n "speculative decoding" ml_research_analysis_202*/
```

### Structured search via script

The search script works around noisy tags by matching across title, `core_contribution`, and filename:

```bash
python scripts/search_topic.py --topic "mixture of experts" --alias moe
python scripts/search_topic.py --topic "reinforcement learning" --alias rl --limit 25 --json
```

### Query the SQLite index

`analysis_outputs/research_index.sqlite` indexes the 2025 bucket (52,099 rows) with columns: `title`, `arxiv_id`, `core_contribution`, `tags`, `filename`, `file_size`.

```bash
# papers whose core contribution mentions "distillation"
sqlite3 analysis_outputs/research_index.sqlite \
  "SELECT arxiv_id, title FROM papers WHERE core_contribution LIKE '%distillation%' LIMIT 10"

# look up a specific paper
sqlite3 analysis_outputs/research_index.sqlite \
  "SELECT filename FROM papers WHERE arxiv_id = '1706.03762'"
```

### Explore curated topic groups

The `spot_analyses/` directory and the `spot_analysis_paper_groups` table contain deep-dive clusters across eight research themes:

| Group | Theme |
|-------|-------|
| `test_time_compute_scaling` | Scaling compute at inference |
| `reasoning_distillation` | Distilling reasoning capabilities |
| `multi_agent_debate` | Multi-agent argumentation |
| `process_reward_models` | Step-level reward modelling |
| `agentic_workflow_pipeline_design` | LLM agent architectures |
| `adaptive_compute_allocation` | Dynamic compute budgets |
| `test_time_adaptation` | Adapting models at test time |
| `continual_online_tta` | Continual / online TTA |

### Browse the website

The `website/` directory contains a static site with full-text search. See `website/README.md` for build and deploy instructions.

---

## Repository layout

```
ml_research_analysis_2023/   Per-paper markdown analyses
ml_research_analysis_2024/
ml_research_analysis_2025/
analysis_outputs/            SQLite index, digests, assessment outputs
scripts/                     index_frontmatter.py, search_topic.py
spot_analyses/               Curated topic deep-dives (8 groups, 1,824 papers)
website/                     Static browse/search UI
docs/                        Internal reference documents
archive/                     Superseded v1 analyses
```

---

## How analyses are generated

A three-phase [FlatAgents pipeline](https://github.com/memgrafter/research_crawler_flatagents) produces each report:

1. **Prep** — download arXiv PDF, extract text, match against ML terminology corpus
2. **Expensive** — parallel LLM calls for mechanism analysis, reproduction notes, and open questions
3. **Wrap** — limitations/confidence, tagging, report assembly, quality judge + auto-repair

The 2025 batch used GLM-5 (pony-alpha) for the expensive phase; 2023–2024 used Trinity Large throughout. Pipeline code, configs, and execution databases live in the pipeline repo — this repository is output only.

---

## Known limitations

- **~190 permanent failures** across all years: PDF 404s (~106), context overflow >256k (~60), provider errors (~9), PDF parse errors (~15). No pending retries.
- **Tags are unreliable** — the tail of every tag list contains generic terms. Use `core_contribution` and full-text search instead.
- **Duplicate filenames** exist where papers were rerun (1,158 in 2024, 582 in 2025). The SQLite index and filenames are deduplicated by `(arxiv_id, timestamp)`.

---

## Reindexing

After adding or removing analysis files, rebuild the SQLite index:

```bash
python scripts/index_frontmatter.py ml_research_analysis_2025
python scripts/index_frontmatter.py ml_research_analysis_2025 --prune  # also remove deleted files
```
