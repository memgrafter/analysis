# Adaptive Compute Allocation — Overview

## Corpus Coverage

- **51 papers** in 2025 corpus (spin-off from `test_time_compute_scaling`)
- Focus: runtime allocation of inference-time compute (how much reasoning to spend, when to stop, where to route budget)

## Phase-0 Definition (final)

Starting from `test_time_compute_scaling`, selected papers whose title/core_contribution matched allocation cues including:

- adaptive/dynamic compute allocation
- overthinking/underthinking + stopping cues
- token/reasoning budget controls
- difficulty-aware/query-adaptive allocation
- compute-optimal / budget-constraint tradeoff cues

## Phase-0 Quality Gate

- `DB_COUNT`: **51**
- `papers.md` lines: **51**
- `dup_group_ids`: **0**
- `broken_links`: **0**
- `duplicate_arxiv_rows_in_papers`: **0**

Scope smoke-test quality: acceptable precision (dominantly in-scope; small context/evaluation tail retained).

## Extraction Status

- Option C: [option_c_extraction.md](option_c_extraction.md)
- Option A: [option_a_refinement.md](option_a_refinement.md)
- Final merged analysis: [../adaptive_compute_allocation.md](../adaptive_compute_allocation.md)

## Paper List

See [papers.md](papers.md).
