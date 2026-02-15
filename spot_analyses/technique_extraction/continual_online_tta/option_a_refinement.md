# Continual / Online TTA — Option A (Semantic Refinement)

**Date:** 2026-02-14  
**Input:** 60 core_contribution summaries (`/tmp/continual_online_tta_core_contributions.txt`)  
**Goal:** Merge overlapping grep buckets into mechanism-level categories specific to non-stationary, long-horizon test-time adaptation.

---

## Refined Taxonomy (Semantic)

### 1) Drift-Aware Continual Control Loops (~18)
Online adaptation systems that explicitly monitor shift and alter update behavior.

**Sub-families**
- Drift scoring / regime detection (domain-diversity, martingale triggers)
- Adaptive reset policies (hard/soft reset, entropy/KL gating)
- Stream-order-aware inference (temporal priors)

**Representative papers:** 2408.08056, 2510.03839, 2601.15544, 2601.21012, 2506.05736

---

### 2) Stability Under Long-Horizon Adaptation (~14)
Mechanisms to prevent collapse and catastrophic forgetting during continual updates.

**Sub-families**
- Dual-teacher / slow-fast teacher retention
- EMA anchors and conservative update constraints
- Rank-structured entropy objectives

**Representative papers:** 2511.18468, 2510.05530, 2505.16441, 2509.02982, 2502.20677

---

### 3) Continual Test-Time Training & Self-Improvement (~12)
Test-time optimization loops that keep learning during deployment.

**Sub-families**
- Self-supervised TTT in streams
- Curriculum-driven online improvement
- Online policy updates for black-box objectives

**Representative papers:** 2512.23675, 2601.22628, 2508.08641, 2510.04786, 2504.01489

---

### 4) Probabilistic / Uncertainty-Guided CTTA (~8)
Bayesian state-tracking and uncertainty-aware adaptation policies.

**Sub-families**
- Variational continual adaptation
- Conformal uncertainty for pseudo-label reliability
- Sequential Bayesian filtering in latent/function space

**Representative papers:** 2402.08182, 2502.02998, 2602.00458, 2601.21012, 2512.02882

---

### 5) Source-Free and Privacy-Constrained Continual Adaptation (~7)
Adaptation without source data, often in decentralized or regulated deployment settings.

**Sub-families**
- Source-free CTTA in healthcare/EEG settings
- Federated continual TTA
- Data-free routing adaptation

**Representative papers:** 2505.13643, 2501.07585, 2509.02982, 2506.05736, 2510.14853

---

### 6) Parameter-Efficient Online Adaptation Interfaces (~7)
Lightweight adaptation interfaces that avoid full fine-tuning.

**Sub-families**
- Prompt-only online adaptation
- Routing-only MoE adaptation
- Training-free prompt transfer / covariance-based adaptation

**Representative papers:** 2501.16404, 2510.03839, 2510.14853, 2512.01420, 2601.23253

---

### 7) Resource-Bounded / On-Device Adaptation (~8)
Methods optimized for latency, memory, and hardware constraints.

**Sub-families**
- BN-focused low-memory CTTA
- Neuromorphic-compatible online updates
- Gradient-free adaptation procedures

**Representative papers:** 2502.20677, 2505.05375, 2504.15323, 2509.25495, 2602.00458

---

### 8) Application-Driven Non-Stationary Streams (~10)
Domain clusters where continual adaptation is operationally critical.

**Sub-families**
- EEG/sleep/drowsiness adaptation
- Time-series forecasting and anomaly monitoring
- Malware/cyber drift environments

**Representative papers:** 2511.22030, 2509.26301, 2510.14814, 2602.01635, 2505.18734

---

## Semantic Reconciliation Notes

### What Option C got right
- Correctly surfaced the core axes: **drift handling**, **continual setup**, **TTT loops**, **teacher/memory stabilization**, and **probabilistic filtering**.
- High recall (58/60 matched) means little mechanism signal was missed.

### What Option C over-split
- Separate buckets for entropy/normalization vs. forgetting mitigation are better treated jointly in long-horizon CTTA.
- Prompt/routing and training-free categories overlap heavily with resource-constrained deployment and should be interpreted as adaptation interfaces, not independent problem settings.

### What should be treated as context, not mechanisms
- Survey/tutorial/benchmark/dissertation entries should remain in coverage accounting but excluded from final mechanism taxonomy.

### Edge cases
- A small number of papers are adjacent but not core CTTA mechanisms (e.g., broad continual-learning or industrial transfer-learning under drift).
