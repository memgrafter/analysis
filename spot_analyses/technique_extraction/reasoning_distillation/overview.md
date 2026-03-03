# Reasoning Distillation — Overview

## Corpus Coverage

- **86 papers** in 2025/2026 corpus slice (`ml_research_analysis_2025/`)
- Focus: distilling reasoning behaviors/traces from stronger teachers into smaller or cheaper students

## Phase-0 Definition (final)

Group built from `papers.title` + `papers.core_contribution` (2025/2026 only), requiring:

1. Distillation cue: `distill*`
2. Reasoning cue: `reasoning` OR `chain-of-thought` OR `rationale`
3. Compression/transfer cue: `teacher` OR `student` OR `small language model` OR `compact` OR `compression`

This narrows generic KD papers to a reasoning-transfer slice aligned with the queue scope.

## Phase-0 Quality Gate

- `DB_COUNT`: **86**
- `papers.md` lines: **86**
- `dup_group_ids`: **0**
- `broken_links`: **0**
- `duplicate_arxiv_rows_in_papers`: **0**

Scope smoke test: good precision with a small boundary tail (domain-heavy papers where reasoning distillation is a component rather than the central mechanism).

## Extraction Status

- Option C: [option_c_extraction.md](option_c_extraction.md)
- Option A: [option_a_refinement.md](option_a_refinement.md)
- Final merged analysis: [../reasoning_distillation.md](../reasoning_distillation.md)

## Paper List

See [papers.md](papers.md).
