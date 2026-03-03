# Process Reward Models — Option C (Grep-Seeded)

**Date:** 2026-02-16  
**Group:** `process_reward_models` (111 papers)  
**Coverage:** 111/111 matched (100%)

---

## Group Definition Used (Phase 0)

2025/2026-only papers where title/core contribution contains PRM/process-supervision cues:

- `process reward model` / `process reward`
- `process supervision`
- `step-level reward` / `step-wise reward`

---

## Grep-Seeded Categories

### 1) Multimodal & Domain-Specialized Process Rewards (47 papers)
Patterns: `multimodal|vision|vlm|radiology|financial|table|audio|biomedical|medical|knowledge graph|theorem proving`.

PRM/process-supervision adapted to domain constraints (medical VLMs, tabular reasoning, finance, tool-heavy vertical tasks).

### 2) Agent/Tool-Centric Process Rewarding (42 papers)
Patterns: `agent|tool|function calling|gui|web|computer use|multi-turn|tool-use|action candidates`.

Step-level reward modeling for action-sequence quality in tool-use and long-horizon agents.

### 3) PRM-Guided Test-Time Search/Decoding (34 papers)
Patterns: `test-time|inference-time|verifier-guided|best-of-n|mcts|beam search|branching|decoding|trajectory|early rejection`.

PRMs used as external verifiers/selectors for search, branching, and trajectory pruning.

### 4) Process-Supervised RL Optimization (34 papers)
Patterns: `reinforcement learning|grpo|policy optimization|credit assignment|process reward learning|prm-free|step-dpo`.

Training-time use of step-level rewards to densify feedback and improve policy updates.

### 5) PRM Architecture & Scoring Variants (25 papers)
Patterns: `genprm|thinkprm|biprm|contextprm|groundedprm|toolprm|agentprm|webarbiter|funprm|apr`.

Model-design innovations for process-level scoring quality and controllability.

### 6) Benchmarking, Evaluation, and Theory (19 papers)
Patterns: `benchmark|bench|toolprmbench|socratic-prmbench|toolcomp|vilbench|pac-bayes|unified evaluation`.

Evaluation infrastructure and analytical work characterizing PRM behavior/performance.

### 7) Robustness, Bias, and Noise Correction (19 papers)
Patterns: `length bias|reward hacking|noise|false positives|outcome-process inconsistency|scaling flaws|hallucinat|generalization`.

Methods targeting PRM reliability under label noise, confounders, and distribution shift.

### 8) PRM Data Construction & Annotation Pipelines (16 papers)
Patterns: `aurora|actprm|freeprm|spark|active learning|weakly supervised|pseudo labels|annotation|synthetic verification data`.

Pipelines for generating process labels at lower human-label cost.

---

## Overlap Notes

- Strong overlap among **test-time search**, **agent/tool PRMs**, and **RL optimization**.
- **Domain-specialized** category is broad and often co-occurs with architecture or data-pipeline contributions.
- **Benchmark/theory** papers are often contextual overlays rather than standalone mechanism proposals.

---

## Coverage Summary

- Total papers: **111**
- Matched by grep categories: **111 (100%)**
- Dominant core: PRM mechanisms + process-supervision RL loops
- Boundary tail: PRM-free alternatives and broad process-reward applications
