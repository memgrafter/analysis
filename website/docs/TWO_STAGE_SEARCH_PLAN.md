# Two-Stage Search Plan (Compact Index + Subset Full-Text Refilter)

## Goal
Keep search fast and cheap while preserving the ability to do deeper full-text relevance when needed.

## Core idea
1. **Stage 1 (global retrieval):** query a compact search index across all documents.
2. **Stage 2 (local refinement):** run deeper full-text filtering only on the Stage 1 subset.

This avoids indexing/scanning full raw text globally.

---

## Stage 1: Compact global index (required)

### Indexed fields
- `title`
- `core_contribution`
- `tags`
- `arxiv_id`
- short body signal (`body_preview`, e.g. first 500 words)

### Output
- Top-K candidate `digest_id`s (e.g. K=100)
- Basic rank score (BM25)

### Constraints
- Keep DB relatively small (target ~100–300 MB)
- Keep query transfer low and predictable

---

## Stage 2: Subset full-text refinement (optional per query)

### Input
- Candidate list from Stage 1 (`digest_id` set)

### Data source options
- Option A: fetch raw markdown for candidates (`/view/<digest-id>.md`)
- Option B: fetch prebuilt deep-text chunks for candidates

### Refinement behavior
- Re-score candidates using full text signals
- Return refined top-N results

### Guardrails
- Hard cap candidate set size for refinement (e.g. max 100)
- Timeout/byte budget for refinement step
- If refinement budget exceeded, fall back to Stage 1 ranking

---

## Build artifacts

### Required
- `search/manifest.json`
- `search/search-<hash>.sqlite` (compact index)

### Optional (for faster Stage 2)
- `search/deep_text/<digest-id>.json` or year-sharded chunk files

---

## Query flow
1. User enters query.
2. Stage 1 returns top-K candidates quickly.
3. If query is broad/high-value, run Stage 2 refinement on K.
4. Show final ranked results.

---

## Initial defaults
- `preview_words`: 500 for Stage 1
- Stage 1 `K`: 100
- Final shown results: 20
- Stage 2 byte budget: configurable (start conservative)

---

## Why this plan
- Prevents giant global full-text index costs
- Preserves high-quality retrieval via targeted deep pass
- Fits static hosting constraints and incremental evolution

---

## Acceptance criteria
- Stage 1 remains fast and stable on full corpus
- Stage 2 improves relevance on ambiguous/broad queries
- Transfer/cost stays bounded by explicit budgets
- No requirement for server-side dynamic search backend
