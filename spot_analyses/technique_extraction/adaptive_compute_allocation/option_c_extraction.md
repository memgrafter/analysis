# Adaptive Compute Allocation — Option C (Grep-Seeded)

**Date:** 2026-02-15  
**Group:** `adaptive_compute_allocation` (51 papers)  
**Coverage:** 51/51 matched (100%)

---

## Group Definition Used (Phase 0)

Spin-off from `test_time_compute_scaling` with allocation-focused cues in title/core_contribution:

- adaptive/dynamic compute and allocation terms
- overthinking/underthinking and early-stop cues
- reasoning/token budget control cues
- compute-optimal and difficulty-aware strategy cues

This yielded 51 papers in the focused adaptive allocation slice.

---

## Grep-Seeded Categories

### 1) Overthinking & Reasoning-Length Trimming (19 papers)
Patterns: `overthinking`, `underthinking`, `trim`, `shorter`, `brevity`, `reasoning completion point`, `redundant`.

Largest mechanism cluster: detect redundant reasoning and shorten trajectories without harming accuracy.

### 2) Reward / Uncertainty / Confidence-Guided Allocation (15 papers)
Patterns: `confidence`, `certainty`, `uncertainty`, `reward model`, `process reward`, `reward feedback`, `self-verified`.

Use intermediate confidence/reward signals to allocate or terminate compute.

### 3) Budget Control Interfaces (13 papers)
Patterns: `thinking budget`, `reasoning budget`, `token budget`, `budget constraints`, `budget-aware`, `control fields`.

Interfaces that expose runtime budget control directly to the model or prompting layer.

### 4) Difficulty-Adaptive Allocation (10 papers)
Patterns: `difficulty-aware`, `query-adaptive`, `task complexity`, `difficulty classifier`, `adaptive compute allocation`.

Per-query or per-step compute depth chosen by estimated problem difficulty.

### 5) Compute-Optimal Tradeoff Modeling (9 papers)
Patterns: `compute-optimal`, `pareto`, `trade-off`, `cost-per-token`, `effective token cost`, `fixed budget`.

Modeling-oriented papers that characterize cost/accuracy frontiers and guide allocation policy.

### 6) Stopping / Halting Policies (7 papers)
Patterns: `early stopping`, `optimal stopping`, `dynamic terminate`, `entropy after`, `halting`.

Formal or heuristic stopping rules that end reasoning once marginal gains drop.

### 7) Search & Planning Budget Reallocation (7 papers)
Patterns: `dual-phase`, `planning and execution`, `rollout length`, `hierarchical trajectory search`, `deep search agents`.

Allocate budget across phases/branches/subproblems, not just overall token count.

### 8) Domain-Specific Allocation Settings (7 papers)
Patterns: `medical`, `robot control`, `table reasoning`, `image generation`, `diffusion language`, `program termination`.

Applied settings where adaptive compute is customized to domain constraints.

### 9) Bandit / Scheduler Controllers (5 papers)
Patterns: `bandit`, `UCB`, `predictive scheduling`, `utility optimization`, `cumulative regret`.

Explicit controller algorithms for online compute scheduling.

### 10) Architectural / Routing Mechanisms (5 papers)
Patterns: `thinking/non-thinking modes`, `draft model`, `deferral`, `collaboration graph`, `distributed placement/routing`.

Architectural pathways (mode switching, draft-verify routing, multi-model graph allocation).

---

## Overlap Notes

- Strong overlap among **Overthinking**, **Stopping/Halting**, and **Difficulty-Adaptive** buckets.
- **Reward/Uncertainty guidance** cross-cuts nearly all allocation mechanisms.
- **Tradeoff modeling** papers are often evaluative/analytical rather than introducing a standalone runtime controller.
- **Domain-specific** bucket is context, not a mechanism family.

---

## Coverage Summary

- Total papers: **51**
- Matched by grep categories: **51 (100%)**
- Likely context-heavy (survey/report/evaluation) rather than mechanism-first: small minority (~5–8 papers)
