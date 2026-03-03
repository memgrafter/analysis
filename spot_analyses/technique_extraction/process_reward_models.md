# Process Reward Models — Final Analysis

**Date:** 2026-02-16  
**Group:** `process_reward_models` (111 papers, 2025/2026 corpus slice)  
**Grep Coverage:** 111/111 (100%)  
**Sources:** [Option C (grep-seeded)](process_reward_models/option_c_extraction.md) · [Option A (semantic refinement)](process_reward_models/option_a_refinement.md)

---

## Method Summary

- **Option C:** high-recall grep pass over PRM/process-supervision summaries to enumerate recurring clusters.
- **Option A:** semantic merge into mechanism-level families, with explicit separation of overlays (benchmarks/theory and PRM-free critiques).
- **Merge rule:** keep Option C recall; adopt Option A boundaries when overlap is structurally high.

---

## Final Taxonomy: 7 Core Technique Categories + 2 Overlay Categories

Categories **1–7** are mechanism-level technique families. Categories **8–9** are overlays (context/evaluation/contrast class).

### 1) PRM Architecture & Scoring Design (~30)
Mechanisms that improve step-level evaluation quality through model/formulation design.

- Generative PRMs with critique/repair outputs
- Bidirectional/context-aware scoring
- Hierarchical and trajectory-aware reward formulations

**Key papers:** 2504.00891, 2504.16828, 2508.01682, 2509.24460, 2511.19314, 2602.00760  
**Agreement:** Strong Option C + Option A agreement.

### 2) Data-Centric PRM Supervision Pipelines (~20)
Mechanisms for constructing process labels efficiently and with lower annotation burden.

- Ensemble auto-label + reverse verification
- Active-learning step selection
- Weak/self-supervised pseudo-labeling
- Search-generated synthetic process labels

**Key papers:** 2502.11520, 2504.10559, 2506.03570, 2508.01773, 2509.24351, 2512.03244  
**Agreement:** High confidence; Option A unified scattered data-engineering buckets.

### 3) PRM-Guided Test-Time Search & Compute Control (~24)
PRMs used as external verifiers/selectors to guide inference trajectories.

- Verifier-guided branching/beam/MCTS
- Best-of-N trajectory rescoring
- Early rejection/selective expansion
- Budget-aware test-time compute policies

**Key papers:** 2502.00271, 2503.21961, 2505.13672, 2510.16449, 2510.19767, 2602.01070  
**Agreement:** Strong, with some overlap into RL-optimization papers.

### 4) Process-Supervised RL Optimization (~23)
Training-time methods that use dense intermediate rewards for better credit assignment.

- Thought/step-level reward shaping in RL
- Hybrid process+outcome reward schemes
- PRM-free alternatives deriving process-like supervision intrinsically

**Key papers:** 2502.10325, 2507.23317, 2601.07182, 2601.10201, 2601.20649, 2507.01551  
**Agreement:** Option A merged PRM-based and PRM-free training loops into one family.

### 5) Reliability, Calibration, and Anti-Hacking for PRMs (~18)
Methods that make process rewards trustworthy under noise, confounders, and adversarial behaviors.

- Length-bias/confound mitigation
- Label-noise correction and reflection-aware relabeling
- Reward-hacking detection and causal adjustment
- Outcome-process consistency alignment

**Key papers:** 2505.14391, 2507.15698, 2508.04216, 2601.07349, 2601.12748, 2502.14361  
**Agreement:** Option A elevated this from a secondary bucket to a core axis.

### 6) Agentic and Tool-Use Process Rewarding (~20)
Step-level reward models for long-horizon action trajectories in agents.

- Function-calling and tool invocation evaluators
- Web/GUI/computer-use trajectory reward models
- Multi-agent per-action reward decomposition

**Key papers:** 2502.02584, 2510.14703, 2510.24803, 2511.08325, 2601.21872, 2509.23738  
**Agreement:** Strong; partially overlaps with category 3 (inference control).

### 7) Domain-Specialized & Multimodal PRMs (~20)
PRMs adapted to domain-specific constraints and non-text modalities.

- Vision-language and medical verification PRMs
- Table/tool-grounded reasoning PRMs
- Finance, audio, biomedical, and specialized settings

**Key papers:** 2503.10291, 2506.09532, 2508.15202, 2510.06217, 2510.23217, 2508.03733  
**Agreement:** Clear in both passes; Option A treated this as mechanism+deployment bridge.

### 8) Overlay: Benchmarking, Evaluation, and Theory (~19)
Benchmarks and analytical work that quantify PRM capabilities and limits.

**Key papers:** 2501.01290, 2503.20271, 2505.18065, 2505.23474, 2601.12294, 2510.00492

### 9) Overlay: PRM-Free / Critical Counterpoints (~8)
Work that minimizes dependence on explicit PRMs or challenges PRM assumptions.

**Key papers:** 2505.11227, 2507.01551, 2507.14202, 2509.25604, 2601.20649

---

## Extraction Reconciliation

### Where Option C and Option A agree
- PRM usage sits on three major operational fronts: **inference-time verification**, **training-time RL supervision**, and **agent/tool trajectory evaluation**.
- Data pipeline quality and robustness are decisive for downstream PRM utility.

### What Option A added
- A cleaner mechanism boundary between **architecture**, **supervision pipeline**, and **robustness/calibration** families.
- Explicit treatment of **PRM-free/critical** papers as a coherent contrast class.
- Better separation of core mechanism categories from benchmark/context overlays.

### Merge / drop decisions
- Merged: Option C `tts_search` + parts of `agents_tools` into a shared control stack, while preserving agent/tool family as its own core category.
- Merged: Option C data and weak-supervision fragments into one **Data-Centric Supervision Pipeline** family.
- Demoted: benchmark/theory and PRM-free critiques to overlays (important but not primary mechanism axes).

---

## Coverage Reconciliation

| Metric | Count |
|---|---:|
| Total papers in group | 111 |
| Grep-matched (Option C) | 111 (100%) |
| Semantically classifiable into core mechanism categories | ~98 |
| Overlay/context-heavy (bench/theory/critical) | ~10 |
| Clearly tangential | ~3 |
| Empty/malformed entries | 0 |

---

## Application / Domain Summary (overlay)

| Domain | ~Papers | Note |
|---|---:|---|
| Math/reasoning LLMs | ~42 | PRM architecture, data pipelines, verifier-guided search |
| Agent/tool-use systems | ~24 | Function-calling, web/GUI/computer-use trajectory rewards |
| Multimodal/vision/medical | ~20 | Visual and radiology-oriented process verification |
| Code generation | ~10 | Function-level and intermediate-step verification for coding tasks |
| Retrieval/RAG-heavy tasks | ~9 | Process supervision over retrieval decisions and decomposition |

---

## Paper List

See [papers.md](process_reward_models/papers.md) (111 papers).

## Extraction Artifacts

- [overview.md](process_reward_models/overview.md)
- [option_c_extraction.md](process_reward_models/option_c_extraction.md)
- [option_a_refinement.md](process_reward_models/option_a_refinement.md)
