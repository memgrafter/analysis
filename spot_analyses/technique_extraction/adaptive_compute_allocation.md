# Adaptive Compute Allocation — Final Analysis

**Date:** 2026-02-15  
**Group:** `adaptive_compute_allocation` (51 papers, 2025 corpus)  
**Grep Coverage:** 51/51 (100%)  
**Sources:** [Option C (grep-seeded)](adaptive_compute_allocation/option_c_extraction.md) · [Option A (semantic refinement)](adaptive_compute_allocation/option_a_refinement.md)

---

## Method Summary

- **Option C:** high-recall grep pass over core_contribution summaries to enumerate allocation-related categories.
- **Option A:** semantic consolidation into mechanism-level families (stopping, routing, controller policy, structural allocation).
- **Merge rule:** preserve Option C recall, but use Option A boundaries when categories overlap heavily.

---

## Final Taxonomy (Merged)

### 1) Budgeted Halting & Overthinking Suppression (~20)
Runtime policies that cut off redundant reasoning once marginal utility drops.

- Early-stop detectors from entropy/certainty/trace signals
- Reasoning-length trimming and concise-path preference
- Optimal stopping formulations balancing quality vs cost

**Key papers:** 2502.10954, 2509.26522, 2510.01394, 2510.10103, 2509.06174  
**Agreement:** Strong Option C + Option A agreement (dominant cluster).

### 2) Difficulty-Aware Compute Routing (~14)
Per-query/per-step compute budgeting driven by estimated difficulty.

- Difficulty classifiers and query-adaptive allocation
- Adaptive depth/candidate/beam/rollout assignment
- Fast-vs-slow reasoning mode selection

**Key papers:** 2506.12721, 2509.09864, 2509.05226, 2512.00466, 2602.01237  
**Agreement:** High confidence, appears across both passes.

### 3) Controller-Driven Allocation Policies (~10)
Explicit control algorithms for online compute scheduling.

- Bandit/UCB threshold controllers
- Utility optimization over token and latency cost
- Sequential scheduling policies for test-time scaling loops

**Key papers:** 2503.07572, 2506.12721, 2510.10103, 2602.01120, 2602.01237  
**Agreement:** Option A merges parts of C's bandit/scheduler + stopping buckets.

### 4) Reward / Uncertainty-Guided Budgeting (~12)
Use confidence/reward signals to continue, stop, or reallocate compute.

- PRM/reward-aware compute decisions
- Certainty probes during generation
- Self-evaluation and confidence-aware self-training

**Key papers:** 2502.06703, 2505.17454, 2507.14958, 2509.07820, 2509.25420  
**Agreement:** Stable cross-cutting family in both extractions.

### 5) Structural Allocation Across Search/Agents/Routing (~10)
Compute is allocated over structure (phases, branches, modules, agents), not just raw token length.

- Planning vs execution dual-phase budgeting
- Branch/rerank budget policies in deep search
- Multi-agent/module planning under fixed budgets
- Draft-verify deferral and model-graph routing

**Key papers:** 2509.25420, 2511.00086, 2512.11213, 2601.14224, 2602.01842  
**Agreement:** Option A elevated this from split Option C categories.

### 6) Compute-Accuracy Frontier Modeling (~8)
Papers that characterize compute-quality frontiers and inform policy design.

- Compute-optimal scaling analyses
- Pareto-style cost/accuracy evaluation
- System-level cost-per-token and latency accounting

**Key papers:** 2505.18065, 2509.19645, 2510.02228, 2512.24776  
**Agreement:** Treated as decision-support family (not always a runtime algorithm contribution).

### 7) Deployment-Specific Allocation Policies (~8)
Domain-constrained adaptive allocation in medical, robotics, tables, and serving systems.

**Key papers:** 2506.13102, 2511.20906, 2511.11233, 2512.21884  
**Agreement:** Kept as overlay category in final merge.

---

## Extraction Reconciliation

### Where Option C and Option A agree
- Overthinking mitigation is the most active mechanism thread.
- Difficulty-aware adaptive budgeting is central.
- Confidence/reward signals are primary control inputs for allocation.

### What Option A added
- A clear split between **token-level halting** and **structure-level allocation**.
- Unified controller view (bandit/scheduler/stopping as one policy loop family).
- Frontier-modeling papers reframed as policy guidance rather than mechanism duplicates.

### Merge / drop decisions
- Merged: `bandit_scheduler` + parts of `stopping_halting` + parts of `difficulty_adaptive` -> **Controller-Driven Allocation Policies**.
- Merged: `search_planning_alloc` + `arch_routing_mechanisms` -> **Structural Allocation**.
- Demoted: domain buckets treated as overlays, not primary mechanism axes.

---

## Coverage Reconciliation

| Metric | Count |
|---|---:|
| Total papers in group | 51 |
| Grep-matched (Option C) | 51 (100%) |
| Semantically classifiable into final taxonomy | 44 |
| Context/survey/report/evaluation-heavy | 5 |
| Clearly tangential | 2 |
| Empty/malformed entries | 0 |

---

## Application / Domain Summary

| Domain | ~Papers | Note |
|---|---:|---|
| General LLM reasoning | ~30 | Overthinking suppression, stopping, query-adaptive budgets |
| Multi-agent / tool-use systems | ~6 | Budget-aware planning/routing under fixed resource limits |
| Robotics / control | ~4 | Difficulty-aware policy compute scaling during control |
| Medical reasoning VLM/LLM | ~3 | Task-aware token budget tuning in high-stakes setting |
| Distributed inference systems | ~3 | Placement/routing/resource allocation for serving latency |

---

## Paper List

See [papers.md](adaptive_compute_allocation/papers.md) (51 papers).

## Extraction Artifacts

- [overview.md](adaptive_compute_allocation/overview.md)
- [option_c_extraction.md](adaptive_compute_allocation/option_c_extraction.md)
- [option_a_refinement.md](adaptive_compute_allocation/option_a_refinement.md)
