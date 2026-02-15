# Continual / Online TTA Technique Extraction — Option C (Grep-Seeded)

**Date:** 2026-02-14  
**Group:** `continual_online_tta` (60 papers)  
**Coverage:** 58/60 matched (96.7%)  
**Unmatched:** 2 papers, both tangential to core CTTA mechanisms (general embodied reasoning proceduralization; industrial ANN transfer-learning under drift).

---

## Group Definition Used (Phase 0)

Spin-off from the existing `test_time_adaptation` group, filtered to papers whose **title/core_contribution** contain continual/online stream cues:

- `continual`, `online`, `stream`, `drift`, `forget`, `non-stationary`, `incremental`, `sequential`

This yielded 60 papers with acceptable scope precision for continual/online TTA.

---

## Grep-Seeded Categories

### 1) Drift / Stream Shift Handling (22 papers)
Patterns: `concept drift`, `drift detection`, `dynamic domain shift`, `non-stationary`, `data stream`, `distribution shift`.

Methods explicitly modeling evolving deployment streams and change-points.

### 2) Continual/Online CTTA Core Setting (12 papers)
Patterns: `continual test-time adaptation`, `CTTA`, `CoTTA`, `online test-time adaptation`, `FedCTTA`.

Papers framed directly as continual or online test-time adaptation.

### 3) TTT / Online Self-Improvement Loops (13 papers)
Patterns: `test-time training`, `TTT`, `online learning rule`, `self-supervised`, `curriculum`, `on-policy`, `off-policy`.

Online parameter updates via self-supervision, curricula, and streaming feedback.

### 4) Forgetting Mitigation via Teacher/Memory Anchors (10 papers)
Patterns: `teacher-student`, `dual-teacher`, `slow-teacher`, `fast-teacher`, `memory`, `buffer`, `prototype`, `cache`.

Stability mechanisms for retaining prior knowledge while adapting to new domains.

### 5) Bayesian / Uncertainty / Sequential Filtering (9 papers)
Patterns: `Bayesian`, `variational`, `conformal`, `martingale`, `recursive`, `transition matrix`, `latent state`, `uncertainty`.

Probabilistic state-tracking and uncertainty-aware control for continuous adaptation.

### 6) Entropy + Normalization Update Family (9 papers)
Patterns: `entropy minimization`, `Tent`, `batch norm`, `normalization`, `entropy gate`, `ranked entropy`, `open-set`.

Tent-style and BN/statistics-based updates, often with safety gates in online settings.

### 7) Source-Free / Data-Free / Federated Constraints (7 papers)
Patterns: `source-free`, `data-free`, `without access to source`, `federated`, `privacy-preserving`.

Adaptation under deployment/privacy constraints where source data cannot be used.

### 8) Prompt / Routing / Expert Adaptation (6 papers)
Patterns: `prompt`, `adapter`, `PromptBridge`, `routing`, `mixture-of-experts`.

Continual adaptation through prompt updates or dynamic expert routing.

### 9) RL / Policy Adaptation at Test Time (6 papers)
Patterns: `reinforcement learning`, `policy optimization`, `agent`, `rollout`, `GRPO`.

Online policy updates and reasoning adaptation loops in dynamic environments.

### 10) Training-Free / Gradient-Free Adaptation (5 papers)
Patterns: `training-free`, `gradient-free`, `without back-propagation`, `without gradient updates`.

Low-overhead online adaptation designed for resource-constrained deployment.

### 11) Domain Cluster: EEG/Time-Series/Biosignal Streams (9 papers)
Patterns: `EEG`, `sleep staging`, `drowsiness`, `EMG`, `time series`, `forecasting`, `anomaly`, `recommendation`.

Strong applied cluster where non-stationarity is explicit and continual adaptation is central.

### 12) Surveys / Tutorial / Benchmark Context (6 papers)
Patterns: `survey`, `tutorial`, `benchmark`, `chapter`, `dissertation`.

Context-setting papers in-group but not always proposing a new CTTA mechanism.

### 13) Continual-Learning-Adjacent (3 papers)
Patterns: `continual learning`, `online concept learning`.

Adjacent continual-learning papers touching TTA concerns but not always pure CTTA mechanisms.

---

## Notes on Overlap

- Heavy overlap among **Drift Handling**, **CTTA Core**, and **TTT/Online Loops**.
- **Entropy/Normalization** overlaps with **Forgetting Mitigation** and **Source-Free** deployment papers.
- **Domain cluster** and **Survey context** are not mechanism categories; kept for coverage accounting.

---

## Coverage Summary

- **Total papers:** 60  
- **Matched by grep categories:** 58 (96.7%)  
- **Unmatched:** 2 (both tangential to core continual-online TTA mechanism focus)
