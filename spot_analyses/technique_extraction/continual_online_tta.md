# Continual / Online Test-Time Adaptation — Final Analysis

**Date:** 2026-02-14  
**Group:** `continual_online_tta` (60 papers, 2025 corpus)  
**Grep Coverage:** 58/60 (96.7%)  
**Sources:** [Option C (grep-seeded)](continual_online_tta/option_c_extraction.md) · [Option A (semantic refinement)](continual_online_tta/option_a_refinement.md)

---

## Method Summary

- **Option C (grep-seeded):** Built high-recall buckets from core_contribution summaries using continual/online stream patterns (drift, forgetting, CTTA terms, source-free constraints, prompt/routing, TTT loops).
- **Option A (semantic refinement):** Merged overlapping buckets into mechanism-level families centered on long-horizon stability under non-stationary deployment.
- **Reconciliation rule used:** Keep Option C for recall and counts; use Option A for category boundaries and mechanism semantics.

---

## Final Taxonomy (Merged)

### I. Control & Stability for Non-Stationary Streams

#### 1) Drift-Aware Continual Control Loops (~18)
Mechanisms that detect distribution regime changes and alter adaptation behavior (update, pause, or reset).

- Domain-diversity / shift scoring in dynamic streams
- Drift-triggered reset policies (entropy/KL/threshold-based)
- Order-aware temporal adaptation in stream inference

**Key papers:** 2408.08056, 2510.03839, 2601.15544, 2601.21012, 2506.05736  
**Agreement:** Strong Option C + Option A agreement.

#### 2) Forgetting Mitigation & Retention Anchors (~14)
Mechanisms that preserve prior knowledge while adapting online.

- Slow/fast teacher decomposition for plasticity vs retention
- EMA anchors / conservative weight trajectories
- Ranked entropy and collapse-prevention losses

**Key papers:** 2511.18468, 2510.05530, 2505.16441, 2509.02982, 2502.20677  
**Agreement:** Option A merged several Option C buckets (teacher/memory + entropy-stability).

#### 3) Probabilistic & Uncertainty-Guided CTTA (~8)
Adaptation policies driven by uncertainty estimates and sequential Bayesian updates.

- Variational continual adaptation
- Conformal reliability for pseudo-labeling
- Recursive/sequential Bayesian filtering in latent/function space

**Key papers:** 2402.08182, 2502.02998, 2602.00458, 2601.21012, 2512.02882  
**Agreement:** Option A surfaced this as a coherent mechanism family.

---

### II. Online Update Interfaces

#### 4) Continual Test-Time Training & Self-Improvement Loops (~12)
Persistent test-time learning with self-supervision or task-generated curricula.

- Streaming TTT objectives and online update rules
- Self-generated curricula for on-task improvement
- Mixed on/off-policy test-time optimization loops

**Key papers:** 2512.23675, 2601.22628, 2508.08641, 2510.04786, 2504.01489  
**Agreement:** High confidence; present in both extractions.

#### 5) Parameter-Efficient Online Adaptation (Prompt/Routing) (~7)
Updating prompts/routing/logits rather than full model weights.

- Online prompt tuning with collapse control
- Fisher-preconditioned prompt updates
- MoE router rewiring and prompt transfer across models

**Key papers:** 2501.16404, 2510.03839, 2510.14853, 2512.01420, 2601.23253  
**Agreement:** Option C split across prompt/routing/training-free; Option A merged as one interface family.

#### 6) Source-Free / Privacy-Constrained Continual Adaptation (~7)
Methods designed for settings where source data cannot be revisited.

- Source-free CTTA with online safeguards
- Federated continual adaptation under privacy constraints
- Data-free adaptation in deployment-only conditions

**Key papers:** 2505.13643, 2501.07585, 2509.02982, 2506.05736, 2511.18468  
**Agreement:** Stable category across both passes.

#### 7) Resource-Bounded / On-Device Online Adaptation (~8)
Adaptation constrained by latency, memory, or hardware.

- Low-memory BN-centric CTTA
- Neuromorphic online threshold modulation
- Gradient-free / training-free online updates

**Key papers:** 2502.20677, 2505.05375, 2504.15323, 2509.25495, 2602.00458  
**Agreement:** Option A reframed this from multiple smaller efficiency buckets.

---

### III. Deployment-Driven Clusters

#### 8) Longitudinal Domain Clusters (EEG/Time-Series/Malware) (~10)
Application domains where non-stationarity is first-order and continual adaptation is operationally required.

- EEG/sleep/drowsiness subject/session shifts
- Time-series forecasting + anomaly drift
- Malware drift and stream-class evolution

**Key papers:** 2511.22030, 2509.26301, 2510.14814, 2602.01635, 2505.18734  
**Agreement:** Treated as domain overlay (not a pure mechanism) in final merge.

---

## Extraction Reconciliation

### Where Option C and Option A agree

- Drift handling is the central axis of continual/online TTA.
- Long-horizon forgetting prevention is a distinct, mature design problem.
- TTT-style continuous update loops are a major mechanism family.
- Source-free constraints are frequent and practical.

### What Option A added

- Unified **stability** view (teacher/memory + entropy/anchors) rather than fragmented buckets.
- Clear **probabilistic/sequential filtering** family as a standalone mechanism cluster.
- Separation of **mechanism categories** vs **domain overlays**.

### Merges / drops from Option C

- Merged: `teacher_memory` + parts of `entropy_norm` -> **Forgetting Mitigation & Retention Anchors**.
- Merged: `prompt_routing` + parts of `trainingfree` -> **Parameter-Efficient Online Adaptation**.
- Demoted to context (not mechanism): `surveys_benchmarks` bucket.

---

## Coverage Reconciliation

| Metric | Count |
|---|---:|
| Total papers in group | 60 |
| Grep-matched (Option C) | 58 (96.7%) |
| Semantically classifiable into final taxonomy | 52 |
| Background/survey/context papers | 6 |
| Clearly tangential | 2 |
| Empty/malformed entries | 0 |

---

## Application / Domain Summary

| Domain | ~Papers | Note |
|---|---:|---|
| VLM / Vision-language adaptation | ~10 | Prompt/routing-heavy online adaptation under visual domain shift |
| LLM reasoning/agent adaptation | ~18 | Continual self-improvement, online policy updates, test-time curricula |
| EEG / biosignal monitoring | ~6 | Strong subject/session non-stationarity; online calibration avoidance |
| Time-series forecasting/anomaly | ~8 | Concept drift and temporal regime shifts dominate |
| Edge / federated deployment | ~9 | Memory/privacy/resource constraints shape mechanism choices |

---

## Paper List

See [papers.md](continual_online_tta/papers.md) (60 papers).

## Extraction Artifacts

- [overview.md](continual_online_tta/overview.md)
- [option_c_extraction.md](continual_online_tta/option_c_extraction.md)
- [option_a_refinement.md](continual_online_tta/option_a_refinement.md)
