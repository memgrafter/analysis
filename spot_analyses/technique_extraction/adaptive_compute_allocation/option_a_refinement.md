# Adaptive Compute Allocation — Option A (Semantic Refinement)

**Date:** 2026-02-15  
**Input:** 51 core_contribution summaries (`/tmp/adaptive_compute_allocation_core_contributions.txt`)  
**Goal:** Reconcile grep buckets into mechanism-level families for runtime compute budgeting.

---

## Refined Taxonomy (Semantic)

### 1) Budgeted Halting & Overthinking Suppression (~20)
Core family: methods that decide when to stop reasoning and/or compress traces once they become redundant.

**Sub-families**
- Early-stop detectors (entropy/uncertainty/reflective-step signals)
- Optimal stopping formulations (cost vs reward)
- Post-hoc trimming / concise-answer selection

**Representative papers:** 2502.10954, 2509.26522, 2510.01394, 2510.10103, 2509.06174

---

### 2) Difficulty-Aware Compute Routing (~14)
Allocate more compute to harder queries and less to easy ones.

**Sub-families**
- Difficulty estimators/classifiers
- Query-adaptive candidate/beam/rollout allocation
- Dynamic mode switching (fast vs deliberate reasoning)

**Representative papers:** 2506.12721, 2509.09864, 2509.05226, 2512.00466, 2602.01237

---

### 3) Controller-Driven Allocation Policies (~10)
Explicit runtime controllers that optimize budget usage online.

**Sub-families**
- Bandit/UCB threshold tuning
- Utility optimization over latency + token cost
- Scheduling policies for sequential scaling

**Representative papers:** 2503.07572, 2506.12721, 2510.10103, 2602.01120, 2602.01237

---

### 4) Reward/Uncertainty-Guided Budgeting (~12)
Use confidence or reward estimates as decision signals for continuing, stopping, or reallocating compute.

**Sub-families**
- PRM/reward-aware allocation
- Confidence/certainty probing during reasoning
- Self-evaluation-assisted allocation

**Representative papers:** 2502.06703, 2505.17454, 2507.14958, 2509.07820, 2509.25420

---

### 5) Structural Allocation (Search, Phases, Multi-Agent, Routing) (~10)
Budget assignment across internal structures: phases, branches, experts, or collaborating agents.

**Sub-families**
- Planning vs execution dual-phase budgets
- Search-trajectory branching/re-ranking budgets
- Multi-agent/module-level budget planning
- Draft-verify deferral and model-graph routing

**Representative papers:** 2509.25420, 2511.00086, 2512.11213, 2601.14224, 2602.01842

---

### 6) Compute-Accuracy Frontier Modeling (~8)
Analytical/empirical work that characterizes frontier behavior and informs budget policy.

**Sub-families**
- Compute-optimal scaling analyses
- Pareto frontier evaluation
- System-level latency/cost accounting

**Representative papers:** 2505.18065, 2509.19645, 2510.02228, 2512.24776

---

### 7) Deployment-Specific Adaptive Allocation (~8)
Domain-constrained instantiations (medical, robotics, table reasoning, distributed serving) where allocation policy is task/environment aware.

**Representative papers:** 2506.13102, 2511.20906, 2512.21884, 2511.11233

---

## Semantic Reconciliation Notes

### What Option C got right
- Correctly surfaced overthinking mitigation as the dominant cluster.
- Correctly identified uncertainty/reward signals and budget interfaces as central building blocks.
- Captured nontrivial architectural directions (multi-agent planning, routing, draft-verify).

### What Option C over-split
- `bandit_scheduler`, `stopping_halting`, and part of `difficulty_adaptive` are often one controller loop in practice.
- `domain_specific` should be treated as overlay/context, not a peer mechanism category.

### What Option C under-expressed
- Structural allocation across search/planning phases is a distinct design axis beyond token-level early stopping.
- Frontier-modeling papers play a policy-selection role even when they do not introduce a new runtime algorithm.

### Edge/context papers
- A small tail are report/survey/evaluation heavy (still useful for policy guidance and benchmarking).
