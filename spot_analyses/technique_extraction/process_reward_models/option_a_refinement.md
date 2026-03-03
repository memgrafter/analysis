# Process Reward Models — Option A (Semantic Refinement)

**Date:** 2026-02-16  
**Input:** 111 core_contribution summaries (`/tmp/process_reward_models_core_contributions.txt`)  
**Goal:** Reconcile grep buckets into mechanism-level PRM families.

---

## Refined Taxonomy (Semantic)

### 1) PRM Architecture Design for Step-Level Evaluation (~30)
Core family: model-side innovations that improve process-level scoring quality and interpretability.

**Sub-families**
- Generative PRMs with rationale/repair outputs
- Bidirectional/context-aware scoring
- Hierarchical and trajectory-aware reward formulations
- Structured-action PRMs (tool calls, agent steps)

**Representative papers:** 2504.00891, 2504.16828, 2508.01682, 2509.24460, 2511.19314, 2602.00760

---

### 2) Data-Centric PRM Training Pipelines (~20)
Core family: methods that construct process labels efficiently and with reduced annotation burden.

**Sub-families**
- Ensemble/auto-label + reverse verification pipelines
- Active-learning selection of informative steps
- Weak/self-supervised pseudo-labeling
- Search-generated supervision (AMCS, synthetic verification traces)

**Representative papers:** 2502.11520, 2504.10559, 2506.03570, 2508.01773, 2509.24351, 2512.03244

---

### 3) PRM-Guided Inference-Time Search and Control (~24)
Core family: using PRMs as verifiers/selectors for test-time trajectory exploration.

**Sub-families**
- Verifier-guided branching/beam/MCTS
- Best-of-N and trajectory rescoring
- Early rejection and selective expansion
- Budget-aware test-time compute control

**Representative papers:** 2502.00271, 2503.21961, 2505.13672, 2510.16449, 2510.19767, 2602.01070

---

### 4) Process-Supervised RL for Reasoning/Agents (~23)
Core family: training-time integration of dense intermediate rewards to improve credit assignment.

**Sub-families**
- Thought/step-level reward shaping for policy optimization
- RL formulations that combine process and outcome rewards
- PRM-free process-supervision alternatives with intrinsic/derived rewards

**Representative papers:** 2502.10325, 2507.23317, 2601.07182, 2601.10201, 2601.20649, 2507.01551

---

### 5) Reliability, Calibration, and Anti-Hacking for PRMs (~18)
Core family: improving robustness of process rewards under noise, bias, and adversarial pressure.

**Sub-families**
- Length-bias and confound mitigation
- Label-noise correction and reflection-aware relabeling
- Reward-hacking detection/causal adjustment
- Outcome-process consistency alignment

**Representative papers:** 2505.14391, 2507.15698, 2508.04216, 2601.07349, 2601.12748, 2502.14361

---

### 6) Agentic/Tool-Use Process Rewarding (~20)
Core family: process rewards over action trajectories in tool-using or multi-agent systems.

**Sub-families**
- Function-calling step evaluators
- Web/GUI/computer-use trajectory PRMs
- Multi-agent per-action reward decomposition

**Representative papers:** 2502.02584, 2510.14703, 2510.24803, 2511.08325, 2601.21872, 2509.23738

---

### 7) Domain-Specialized & Multimodal PRMs (~20)
Core family: adapting PRMs/process supervision to non-text-heavy or high-stakes domains.

**Sub-families**
- Vision-language and medical verification PRMs
- Tabular/tool-grounded reasoning PRMs
- Finance, audio, biomedical, and domain-specific systems

**Representative papers:** 2503.10291, 2506.09532, 2508.15202, 2510.06217, 2510.23217, 2508.03733

---

### Overlay A) Benchmarks & Theory (~19)
Infrastructure and analysis papers that evaluate PRMs or characterize limits/frontiers.

**Representative papers:** 2501.01290, 2503.20271, 2505.18065, 2505.23474, 2601.12294, 2510.00492

### Overlay B) PRM-Free / Critical Counterpoints (~8)
Papers that reduce dependence on explicit PRMs or challenge PRM assumptions.

**Representative papers:** 2505.11227, 2507.01551, 2507.14202, 2509.25604, 2601.20649

---

## Semantic Reconciliation Notes

### What Option C got right
- Correctly surfaced the three dominant operational zones: **inference-time verification**, **process-supervised RL**, and **agent/tool trajectory scoring**.
- Correctly identified robustness/noise and benchmark clusters as substantial side streams.

### What Option C over-split
- Domain-specific papers and architecture variants were partially fragmented; many are one mechanism family with different deployment contexts.
- RL + PRM-free alternatives were better treated as one optimization family with internal subtypes.

### What Option C under-expressed
- Reliability/calibration is now a first-class design axis (not just an error bucket).
- PRM-free and critical papers form a coherent contrast class that meaningfully reframes PRM usage decisions.
