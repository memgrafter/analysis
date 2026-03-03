# Process Reward Models — Overview

## Corpus Coverage

- **111 papers** in 2025/2026 corpus slice (`ml_research_analysis_2025/`)
- Focus: step-level reward/process-supervision methods for reasoning and agent trajectories

## Phase-0 Definition (final)

Group built from `papers.title` + `papers.core_contribution` keyword union (2025/2026 only):

- `process reward model` / `process reward`
- `process supervision`
- `step-level reward` / `step-wise reward`

This intentionally prioritizes precision over raw mention count from full-file grep, yielding a tractable PRM-focused set.

## Phase-0 Quality Gate

- `DB_COUNT`: **111**
- `papers.md` lines: **111**
- `dup_group_ids`: **0**
- `broken_links`: **0**
- `duplicate_arxiv_rows_in_papers`: **0**

Scope smoke test: acceptable precision; dominant PRM/process-supervision mechanisms with a small boundary tail (general process-RL and reward-shaping papers).

## Extraction Status

- Option C: [option_c_extraction.md](option_c_extraction.md)
- Option A: [option_a_refinement.md](option_a_refinement.md)
- Final merged analysis: [../process_reward_models.md](../process_reward_models.md)

## Paper List

See [papers.md](papers.md).
