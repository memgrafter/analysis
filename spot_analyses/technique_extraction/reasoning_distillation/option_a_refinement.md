# Reasoning Distillation — Option A (Semantic Refinement)

**Date:** 2026-02-16  
**Input:** 86 core_contribution summaries (`/tmp/reasoning_distillation_core_contributions.txt`)  
**Goal:** Reconcile grep buckets into mechanism-level reasoning-distillation families.

---

## Refined Taxonomy (Semantic)

### 1) Trace-Structured Distillation Objectives (~32)
Core family: transferring *how* a model reasons (trace structure), not just final answers.

**Sub-families**
- CoT/rationale trace imitation
- Structured decomposition (step, function, query-plan level)
- Trajectory-level informative-alignment objectives

**Representative papers:** 2507.01921, 2510.03988, 2512.17053, 2601.14249, 2601.09088

---

### 2) Teacher-Trace Selection & Data Curation (~22)
Core family: selecting which teacher traces to distill for maximum student gain.

**Sub-families**
- Difficulty- and quality-aware rationale selection
- Data-centric augmentation/mixing/filtering
- Provenance-aware and alignment-aware trace filtering

**Representative papers:** 2505.18759, 2509.23574, 2601.10109, 2512.20908, 2508.09883

---

### 3) Compression & Budget-Aware Reasoning Transfer (~20)
Core family: preserve reasoning performance while shrinking trace length/compute.

**Sub-families**
- Prefix/sequence truncation and token pruning
- Dual-granularity CoT compression
- Budget-aware reasoning distillation

**Representative papers:** 2505.19716, 2511.01470, 2512.21002, 2601.10064, 2601.20467, 2510.02312

---

### 4) RL / Preference-Optimized Distillation (~16)
Core family: distillation objectives augmented with policy optimization signals.

**Sub-families**
- KD + RL hybrid post-training
- Preference-optimized distillation (e.g., ORPO/DPO-style)
- On-policy or reward-guided self-improvement loops

**Representative papers:** 2505.24850, 2506.02208, 2507.00054, 2509.25100, 2512.07461, 2602.02244

---

### 5) Multi-Teacher Alignment & Conflict Resolution (~8)
Core family: combining multiple teachers without amplifying contradictory signals.

**Sub-families**
- Drifting-teacher concept alignment
- Teacher disagreement/purification mechanisms
- Cross-architecture teacher-student transfer controls

**Representative papers:** 2510.04142, 2601.21288, 2602.01064, 2509.25100, 2512.18174

---

### 6) Self-Distillation & Autonomous Bootstrapping (~12)
Core family: students improve reasoning using self-generated or minimally supervised traces.

**Sub-families**
- On-policy self-distillation
- Reflection-driven bootstrapping
- Data-free or low-label cognitive behavior distillation

**Representative papers:** 2512.04072, 2601.18734, 2508.12387, 2602.01222, 2602.02366

---

### 7) Agentic / Social Reasoning Distillation (~9)
Core family: distilling collaborative or role-specialized reasoning into compact models.

**Sub-families**
- Multi-agent/debate-graph distillation
- Integrating teacher-agent orchestration into single student models
- Socratic interaction distillation

**Representative papers:** 2511.05528, 2507.21166, 2510.14406, 2509.24726

---

### Overlay A) Domain & Modality-Specific Deployments (~24)
Multimodal and domain-heavy instantiations (code, SQL, retrieval, document QA, speech, driving, embodied VLA).

**Representative papers:** 2511.22521, 2510.17598, 2510.18817, 2508.16998, 2509.14930, 2601.09708

### Overlay B) Evaluation / Critical Analyses (~8)
Benchmarks, probing studies, and analyses of distillation failure modes or limits.

**Representative papers:** 2509.22193, 2505.13792, 2504.02521, 2504.14772, 2510.00579

---

## Semantic Reconciliation Notes

### What Option C got right
- Captured teacher-student transfer as the dominant frame.
- Correctly surfaced trace-level objectives, data curation, compression, and RL as recurring mechanisms.
- Detected multimodal/domain expansion as a major trend.

### What Option C over-split
- Domain and multimodal groups were more deployment overlays than separate mechanism axes.
- Multi-teacher and cross-architecture papers are best treated as a single alignment/conflict family.

### What Option C under-expressed
- Self-distillation/bootstrapping appears as a distinct design path, not just a subset of RL.
- Evaluation/probing papers provide important failure-analysis signals and should be tracked explicitly.
