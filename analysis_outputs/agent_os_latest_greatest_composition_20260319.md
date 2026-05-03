# Latest-and-Greatest Composition Guide for an "Agent OS" (API/Test-Time Focus)

**Date:** 2026-03-19  
**Method:** database-first scan of `analysis_outputs/research_index.sqlite`, then targeted curation from recent papers.

---

## 1) Corpus slice used

Scoped to 8 high-signal groups in `spot_analysis_paper_groups`:

- `test_time_compute_scaling` (987)
- `test_time_adaptation` (284)
- `multi_agent_debate` (153)
- `process_reward_models` (111)
- `agentic_workflow_pipeline_design` (92)
- `reasoning_distillation` (86)
- `continual_online_tta` (60)
- `adaptive_compute_allocation` (51)

**Total group rows:** 1,824  
**Unique arXiv IDs:** 1,633

Recent concentration inside this scoped set:
- **2026:** 189
- **2025:** 1,431

Recent-group signal (row counts by group):
- `test_time_compute_scaling`: 2602=19, 2601=81, 2512=72
- `test_time_adaptation`: 2602=7, 2601=34, 2512=27
- `process_reward_models`: 2602=2, 2601=12, 2512=5
- `multi_agent_debate`: 2602=2, 2601=12, 2512=17
- `agentic_workflow_pipeline_design`: 2602=1, 2601=12, 2512=9

---

## 2) What is converging for Agent OS

A practical Agent OS stack is converging around these planes:

1. **Workflow control plane** (graph topology + orchestration)
2. **Runtime scheduler** (state-aware scheduling + compute allocation)
3. **Tool runtime** (tool routing + speculative tool calls)
4. **Verification plane** (PRM/verifier/judge)
5. **Deliberation plane** (debate/consensus only when needed)
6. **Memory/context plane** (overflow handling + episodic memory)
7. **Safety/governance plane** (red-team + attack resilience)
8. **Evaluation/observability plane** (continuous eval and diagnostics)

---

## 3) High-confidence paper set (best current base)

These are the strongest building blocks for an API-only Agent OS.

### A. Workflow control + orchestration
- **2601.22037** — *Optimizing Agentic Workflows using Meta-tools*  
  Compile repeated tool-call subsequences into deterministic meta-tools.
- **2601.22305** — *BayesFlow*  
  Training-free workflow generation via Bayesian posterior sampling over workflow code.
- **2601.07477** — *JudgeFlow*  
  Block-level failure attribution to improve workflows with targeted edits.
- **2512.15751** — *GLOW*  
  Predict workflow performance without full expensive execution.
- **2512.14142** — *Astraea*  
  State-aware scheduler tuned for agentic workloads mixing model calls + external APIs.

### B. Compute governor (test-time only)
- **2602.01237** — *Predictive Scheduling for Efficient Inference-Time Reasoning*  
  Predict per-query budget instead of fixed token spend.
- **2602.01120** — *MarkovScale*  
  Sequential scaling model for deciding when additional reasoning is worth it.
- **2510.01394** — *Optimal Stopping vs Best-of-N*  
  Principled stop/continue framing for inference-time optimization.
- **2502.10954** — *Learning to Stop Overthinking at Test Time*  
  Core signal for per-instance compute control.
- **2602.01842** — *Prism*  
  Hierarchical search + self-verification (strong test-time scaling pattern).

### C. Tool runtime
- **2512.15834** — *Optimizing Agentic Language Model Inference via Speculative Tool Calls*  
  Core method for latency hiding in API-heavy pipelines.
- **2511.17006** — *Budget-Aware Tool-Use Enables Effective Agent Scaling*  
  Tool usage must be budget-coupled, not always-on.
- **2512.07132** — *DART*  
  Use disagreement to trigger/route tool recruitment.

### D. Verification plane (PRM + judge)
- **2602.01070** — *What If We Allocate Test-Time Compute Adaptively?*  
  Adaptive compute guided by process signals.
- **2602.00760** — *APR*  
  Penalize redundant reasoning tails; useful for anti-bloat control.
- **2601.22249** — *FunPRM*  
  Function-as-step PRM for tool-heavy/code tasks.
- **2601.21872** — *WebArbiter*  
  Principle-guided process rewarding for web agents.
- **2601.22230** — *DAJ*  
  Data-reweighted judge to improve judging reliability.

### E. Deliberation plane
- **2602.00454** — *Cross-Modal Memory Compression for Efficient Multi-Agent Debate*  
  Addresses debate token explosion.
- **2511.11306** — *iMAD*  
  Efficient debate inference framing.
- **2505.21503** — *Silence is Not Consensus*  
  Anti-sycophancy pattern (critical for robust deliberation).

### F. Memory + context plane
- **2512.16970** — *PAACE*  
  Plan-aware context engineering for long-horizon coherence.
- **2511.22729** — *Solving Context Window Overflow in AI Agents*  
  Practical overflow mitigation.
- **2511.17775** — *Episodic Memory in Agentic Frameworks*  
  Next-task suggestion and long-horizon continuity.

### G. Safety + governance + eval
- **2601.13518** — *AgenticRed*  
  Automated red-team system design loop.
- **2601.14662** — *Graph extraction attacks on GraphRAG*  
  Important threat model for retrieval-centric systems.
- **2601.07504** — *FROAV*  
  RAG observation + agent verification infrastructure.
- **2601.22025** — *Evaluation-Driven Iteration*  
  Treat evaluation as a continuous cycle, not one-off.

---

## 4) Frontier 2026 additions (promising, still volatile)

These appear highly relevant, but are newer / less triangulated in spot analyses.

- **2602.01848** — *ROMA* (recursive meta-agent framework)
- **2602.01797** — *ORCH* (deterministic multi-agent orchestrator)
- **2602.01664** — *FlowSteer* (interactive workflow orchestration)
- **2602.01204** — *ASTER* (tool-integrated extended reasoning, interaction-collapse focus)
- **2602.02369** — *Live-Evo* (online agentic memory evolution)
- **2602.02366** — *ReasonCACHE* (reasoning improvement without weight updates)
- **2602.02196** — *TIDE* (trajectory-level diagnostics for test-time improvement)
- **2602.02164** — *Co-RedTeam* (orchestrated security discovery)
- **2602.02219 / 2602.02287** — LLM-judge bias and cross-lingual stability

Use these in a sandbox lane before production policies.

---

## 5) Critical counter-signals (avoid naive designs)

- **2602.01011** — *Multi-Agent Teams Hold Experts Back*  
  More agents do not guarantee synergy.
- **2601.12307** — *Strong Single Agent Baseline*  
  Homogeneous multi-agent workflows can sometimes be simulated cheaper by one strong agent.

**Design implication:** default to single-agent path, escalate to multi-agent only on uncertainty/verification triggers.

---

## 6) Agent OS composition (API-only blueprint)

### Stage 1 (minimum viable Agent OS)
1. **Workflow compiler/orchestrator**: 2601.22037, 2601.22305
2. **State-aware scheduler + compute governor**: 2512.14142, 2602.01237, 2510.01394
3. **Speculative tool runtime**: 2512.15834
4. **Verifier/judge gate**: 2602.01070, 2601.22230
5. **Continuous evaluation loop**: 2601.22025, 2601.07504

### Stage 2 (robustness upgrades)
6. **Debate escalation mode**: 2602.00454, 2511.11306, 2505.21503
7. **Context/memory plane**: 2512.16970, 2511.22729, 2511.17775
8. **Safety/attack governance**: 2601.13518, 2601.14662

### Stage 3 (frontier lane)
9. Add ROMA/ORCH/FlowSteer/ASTER/ReasonCACHE as opt-in policy experiments.

---

## 7) Practical policy defaults from this scan

1. **Escalation policy**: single-agent -> verifier check -> debate only if unresolved.
2. **Budget policy**: per-query predictive scheduling + optimal stopping.
3. **Tool policy**: speculative calls only for read-only/high-hit tools.
4. **Judge policy**: multi-judge + calibration checks (position/framing bias tests).
5. **Security policy**: mandatory red-team runs + GraphRAG extraction defense tests.
6. **Observability policy**: trajectory-level diagnostics, not just final-answer accuracy.

---

## 8) If picking only 10 papers right now

1. 2601.22037 — Meta-tools workflow optimization  
2. 2601.22305 — BayesFlow workflow generation  
3. 2512.14142 — State-aware scheduling  
4. 2512.15834 — Speculative tool calls  
5. 2602.01237 — Predictive scheduling  
6. 2510.01394 — Optimal stopping vs BoN  
7. 2602.01070 — Adaptive compute + process rewards  
8. 2601.22230 — Data-reweighted judge  
9. 2512.16970 — Plan-aware context engineering  
10. 2601.13518 — AgenticRed (automated red-teaming)

This 10-paper set is the strongest compact foundation for composing an API-first Agent OS from current corpus trends.

---

## 9) Session context note (verbatim request)

> "Put this message inside as well. That's a funny habit. i need all the context here."
>
> timestamp 2026-03-19T22:14:50
