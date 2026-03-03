# Reasoning Distillation — Final Analysis

**Date:** 2026-02-16  
**Group:** `reasoning_distillation` (86 papers, 2025/2026 corpus slice)  
**Grep Coverage:** 86/86 (100%)  
**Sources:** [Option C (grep-seeded)](reasoning_distillation/option_c_extraction.md) · [Option A (semantic refinement)](reasoning_distillation/option_a_refinement.md)

---

## Method Summary

- **Option C:** high-recall grep pass over distillation-focused core_contribution summaries to enumerate recurring clusters.
- **Option A:** semantic consolidation into mechanism-level families; domain/modality and evaluation buckets separated as overlays.
- **Merge rule:** preserve Option C recall; use Option A boundaries where categories are heavily overlapping (especially teacher-student, trace, and data-selection clusters).

---

## Final Taxonomy: 7 Core Technique Categories + 2 Overlay Categories

Categories **1–7** are mechanism-level reasoning-distillation families. Categories **8–9** are overlays (deployment context and analysis infrastructure).

### 1) Trace-Structured Reasoning Transfer (~32)
Distillation targets intermediate reasoning structure (CoT/rationale/trajectory), not only final answer outputs.

- CoT/rationale trace imitation objectives
- Structured decomposition targets (step/function/plan level)
- Informative trajectory alignment metrics

**Key papers:** 2507.01921, 2510.03988, 2512.17053, 2601.14249, 2601.09088  
**Agreement:** Strong Option C + Option A agreement.

### 2) Teacher-Trace Selection & Data Curation (~22)
Performance gains driven by *which* teacher traces are selected, filtered, and mixed.

- Difficulty/quality-aware rationale selection
- Data-centric augmentation/mixing/filtering
- Provenance-aware trace filtering

**Key papers:** 2505.18759, 2509.23574, 2601.10109, 2512.20908, 2508.09883  
**Agreement:** Stable in both passes; Option A tightened boundaries.

### 3) Compression & Budget-Aware Distillation (~20)
Mechanisms that compress or shorten reasoning while preserving student quality.

- Prefix/sequence truncation and token pruning
- Dual-granularity CoT compression
- Budget-aware reasoning length control

**Key papers:** 2505.19716, 2511.01470, 2512.21002, 2601.10064, 2601.20467, 2510.02312  
**Agreement:** Strong; overlaps with category 1 by design.

### 4) RL / Preference-Optimized Distillation (~16)
Reasoning distillation combined with RL and preference objectives.

- KD + RL hybrid post-training
- Preference-optimized distillation (DPO/ORPO-like)
- On-policy reward-guided refinement

**Key papers:** 2505.24850, 2506.02208, 2507.00054, 2509.25100, 2512.07461, 2602.02244  
**Agreement:** Option A merged RL and preference fragments into one family.

### 5) Multi-Teacher Alignment & Conflict Handling (~8)
Methods for reconciling multiple, drifting, or inconsistent teacher sources.

- Drifting-teacher concept alignment
- Knowledge purification under teacher conflicts
- Cross-architecture teacher-student transfer control

**Key papers:** 2510.04142, 2601.21288, 2602.01064, 2509.25100, 2512.18174  
**Agreement:** Small but distinct family.

### 6) Self-Distillation & Autonomous Bootstrapping (~12)
Students improve reasoning via self-generated traces and low-label feedback loops.

- On-policy self-distillation
- Reflection-driven self-improvement
- Data-free or low-supervision bootstrapping

**Key papers:** 2512.04072, 2601.18734, 2508.12387, 2602.01222, 2602.02366  
**Agreement:** Option A elevated this from scattered Option C mentions.

### 7) Agentic / Social Reasoning Distillation (~9)
Distilling multi-agent or socially structured reasoning into compact models.

- Debate/interaction-graph distillation
- Integrating multi-agent reasoning into single students
- Socratic interaction distillation

**Key papers:** 2511.05528, 2507.21166, 2510.14406, 2509.24726  
**Agreement:** Consistent niche category with growing activity.

### 8) Overlay: Domain & Modality-Specific Deployments (~24)
Applications in code, SQL, retrieval, document QA, speech/audio, and embodied VLA settings.

**Key papers:** 2511.22521, 2510.17598, 2510.18817, 2508.16998, 2509.14930, 2601.09708

### 9) Overlay: Evaluation / Critical Analyses (~8)
Benchmarks and analyses probing when/why reasoning distillation works or fails.

**Key papers:** 2509.22193, 2505.13792, 2504.02521, 2504.14772, 2510.00579

---

## Extraction Reconciliation

### Where Option C and Option A agree
- Teacher-student transfer with trace-level supervision is the backbone of this area.
- Data curation and efficiency controls are as important as objective design.
- RL/preference integration is a major acceleration path for student capability.

### What Option A added
- Cleaner separation between **core mechanism families** and **domain/evaluation overlays**.
- Identification of **self-distillation/bootstrapping** as a standalone strategy.
- Consolidation of multi-teacher papers into a coherent conflict-resolution family.

### Merge / drop decisions
- Merged: Option C `teacher_student_transfer` + parts of `trace_objectives` into a mechanism hierarchy where trace-supervision is primary and teacher-student framing is cross-cutting.
- Merged: Option C `multi_teacher` + cross-architecture conflict papers into one explicit alignment family.
- Demoted: domain-specific and benchmark-heavy buckets to overlays rather than core mechanism axes.

---

## Coverage Reconciliation

| Metric | Count |
|---|---:|
| Total papers in group | 86 |
| Grep-matched (Option C) | 86 (100%) |
| Semantically classifiable into core technique categories | ~76 |
| Overlay/context-heavy (domain + eval dominant) | ~7 |
| Clearly tangential to reasoning-distillation mechanisms | ~3 |
| Empty/malformed entries | 0 |

---

## Application / Domain Summary (overlay)

| Domain | ~Papers | Note |
|---|---:|---|
| Math/general reasoning SLMs | ~30 | Core trace distillation, data selection, and compression methods |
| Multimodal (vision/audio/VLA) | ~16 | Cross-modal reasoning transfer and latent alignment |
| Code / software engineering | ~10 | Structural reasoning transfer for code generation/repair |
| Retrieval / ranking / search | ~9 | Distilled reasoning for reranking and relevance optimization |
| Enterprise/vertical tasks (industry, e-commerce, healthcare docs) | ~8 | Domain-constrained student deployment under cost limits |

---

## Paper List

See [papers.md](reasoning_distillation/papers.md) (86 papers).

## Extraction Artifacts

- [overview.md](reasoning_distillation/overview.md)
- [option_c_extraction.md](reasoning_distillation/option_c_extraction.md)
- [option_a_refinement.md](reasoning_distillation/option_a_refinement.md)
