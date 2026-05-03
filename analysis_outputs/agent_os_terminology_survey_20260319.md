# Agent OS Terminology Survey (Database-First)

**Date:** 2026-03-19  
**Scope:** terminology scan for "Agent OS"-adjacent research language in this corpus.  
**Primary source:** `analysis_outputs/research_index.sqlite`.

---

## 1) Method (careful scan, bounded reads)

1. Used `spot_analysis_paper_groups` to restrict scope to 8 relevant groups:
   - `agentic_workflow_pipeline_design` (92)
   - `multi_agent_debate` (153)
   - `process_reward_models` (111)
   - `adaptive_compute_allocation` (51)
   - `continual_online_tta` (60)
   - `test_time_adaptation` (284)
   - `test_time_compute_scaling` (987)
   - `reasoning_distillation` (86)
2. De-duplicated by `arxiv_id` -> **1,633 unique papers** (from 1,824 group rows).
3. Ran phrase/regex scans over `title + core_contribution` in SQLite-derived records.
4. Cross-checked terminology against spot-analysis summaries for mechanism-level meaning.

### Time distribution in this scoped set
- **2026:** 189
- **2025:** 1,431
- **2024:** 12
- **Older:** 1

---

## 2) Quick takeaway

"Agent OS" is not standardized as a literal phrase. The field is converging through **adjacent terms**:

- **control/orchestration vocabulary** (workflow, scheduler, meta-agent)
- **runtime efficiency vocabulary** (test-time compute, budget-aware, speculative tools)
- **quality-control vocabulary** (PRM, verifier, judge)
- **deliberation vocabulary** (debate, consensus/disagreement)
- **state/memory vocabulary** (context engineering, episodic memory)
- **governance vocabulary** (red-teaming, GraphRAG attacks, evaluation-driven iteration)

---

## 3) Exact-phrase signal table (lower-bound counts)

Counts below are exact/near-exact phrase matches in the 1,633-paper scoped set.

| Term | Paper hits |
|---|---:|
| process reward model (PRM) | 86 |
| verifier | 97 |
| judge | 36 |
| multi-agent debate | 45 |
| consensus | 20 |
| disagreement | 4 |
| test-time compute | 45 |
| overthinking | 16 |
| early-stopping | 4 |
| agentic workflow | 22 |
| workflow orchestration | 14 |
| meta-agent | 5 |
| state-aware scheduling | 1 |
| speculative tool calls | 1 |
| context engineering | 2 |
| context window overflow | 1 |
| episodic memory | 2 |
| plan-aware | 1 |
| evaluation-driven iteration | 1 |
| red-teaming | 5 |
| GraphRAG extraction attack | 3 |
| budget-aware tool-use | 1 |

**Interpretation:** low counts for some terms do **not** imply low importance; they often indicate naming fragmentation.

---

## 4) Cluster-level terminology (regex families; broader signal)

| Cluster | Approx. papers matched |
|---|---:|
| verification/prm | 210 |
| evaluation/observability | 185 |
| debate/deliberation | 150 |
| online adaptation | 145 |
| safety/governance | 102 |
| compute governance | 71 |
| workflow orchestration | 69 |
| tool runtime | 20 |
| memory/context | 20 |

These broader counts capture synonym drift and variant phrasings.

---

## 5) Recommended Agent OS lexicon (canonical terms + aliases)

Use these canonical labels to reduce naming drift.

### A. Control Plane
- **Canonical:** `Agent Orchestration Control Plane`
- **Aliases seen:** orchestration policy, coordinator, meta-agent routing, workflow optimizer

### B. Workflow Topology
- **Canonical:** `Workflow Graph Topology`
- **Aliases:** planner-executor decomposition, hierarchical roles, constrained topology

### C. Runtime Scheduler
- **Canonical:** `State-Aware Runtime Scheduler`
- **Aliases:** scheduling engine, predictive scheduling, execution policy loop

### D. Tool Runtime
- **Canonical:** `Tool Execution Runtime`
- **Aliases:** tool-use orchestration, function-calling coordination, MCP control
- **Specialized term:** `Speculative Tool Calls`

### E. Compute Governor
- **Canonical:** `Adaptive Test-Time Compute Governor`
- **Aliases:** budget-aware inference, halting/early-stop policy, overthinking suppression

### F. Verification Plane
- **Canonical:** `Verifier & Judge Plane`
- **Aliases:** PRM-guided verification, block judge, LLM-as-judge, self-verification

### G. Deliberation Plane
- **Canonical:** `Multi-Agent Deliberation Layer`
- **Aliases:** debate protocols, consensus fusion, disagreement-triggered escalation

### H. Memory/Context Plane
- **Canonical:** `Context & Episodic Memory Plane`
- **Aliases:** context engineering, overflow mitigation, memory compression, plan-aware context

### I. Governance/Safety Plane
- **Canonical:** `Safety & Attack-Surface Governance`
- **Aliases:** agentic red-teaming, guardrails, extraction-attack resilience

### J. Evaluation/Observability Plane
- **Canonical:** `Evaluation & Observability Plane`
- **Aliases:** evaluation-driven iteration, workflow performance prediction, RAG observation

---

## 6) High-value paper anchors by terminology cluster

### Control plane / workflow / scheduling
- **2601.22037** — *Optimizing Agentic Workflows using Meta-tools*
- **2601.22305** — *BayesFlow: A Probability Inference Framework for Meta-Agent Assisted Workflow Generation*
- **2511.20693** — *A^2Flow: Automating Agentic Workflow Generation via Self-Adaptive Abstraction Operators*
- **2512.14142** — *Astraea: A State-Aware Scheduling Engine for LLM-Powered Agents*
- **2601.07477** — *JudgeFlow: Agentic Workflow Optimization via Block Judge*

### Tool runtime / speculative execution
- **2512.15834** — *Optimizing Agentic Language Model Inference via Speculative Tool Calls*
- **2511.17006** — *Budget-Aware Tool-Use Enables Effective Agent Scaling*
- **2512.07132** — *DART: Leveraging Multi-Agent Disagreement for Tool Recruitment in Multimodal Reasoning*
- **2512.20996** — *TrafficSimAgent ... with MCP Control*

### Verification plane (PRM/verifier/judge)
- **2511.08325** — *AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress*
- **2602.00760** — *APR: ... Anchor-based Process Rewards*
- **2602.01070** — *What If We Allocate Test-Time Compute Adaptively?*
- **2601.22230** — *DAJ: Data-Reweighted LLM Judge for Test-Time Scaling in Code Generation*
- **2502.00271** — *Scaling Flaws of Verifier-Guided Search in Mathematical Reasoning*

### Deliberation / consensus control
- **2511.11306** — *iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference*
- **2505.21503** — *Silence is Not Consensus ...*
- **2507.19090** — *Debating Truth ...*
- **2602.00454** — *Cross-Modal Memory Compression for Efficient Multi-Agent Debate*
- **2601.22297** — *Prepare Reasoning Language Models for Multi-Agent Debate ...*

### Compute governance
- **2602.01237** — *Predictive Scheduling for Efficient Inference-Time Reasoning in LLMs*
- **2502.10954** — *Learning to Stop Overthinking at Test Time*
- **2510.10103** — *Stop When Enough: Adaptive Early-Stopping for CoT Reasoning*
- **2510.01394** — *Optimal Stopping vs Best-of-N for Inference Time Optimization*
- **2602.01842** — *Prism: ... Hierarchical Search and Self-Verification ...*

### Memory/context/governance/eval
- **2511.22729** — *Solving Context Window Overflow in AI Agents*
- **2512.16970** — *PAACE: A Plan-Aware Automated Agent Context Engineering Framework*
- **2511.17775** — *Episodic Memory in Agentic Frameworks: Suggesting Next Tasks*
- **2601.13518** — *AgenticRed: Optimizing Agentic Systems for Automated Red-teaming*
- **2601.14662** — *Query-Efficient Agentic Graph Extraction Attacks on GraphRAG Systems*
- **2601.07504** — *FROAV: Framework for RAG Observation and Agent Verification*
- **2601.22025** — *When "Better" Prompts Hurt: Evaluation-Driven Iteration for LLM Applications*

---

## 7) Terminology coherence notes

1. **"Agent OS" is currently a synthesis label**, not a dominant paper keyword.
2. The most stable terminology today is around **verification (PRM/verifier/judge)** and **deliberation (debate/consensus)**.
3. The strongest emerging systems terms are **state-aware scheduling**, **speculative tool calls**, **plan-aware context engineering**, and **evaluation-driven iteration**.
4. A practical naming strategy is to define an Agent OS as a stack of planes:
   - control
   - runtime
   - verification
   - deliberation
   - memory/context
   - safety
   - observability

---

## 8) Suggested standardized glossary (short form)

If a single concise glossary is needed for future docs, use:

- Agent Orchestration Control Plane
- Workflow Graph Topology
- State-Aware Runtime Scheduler
- Tool Execution Runtime
- Adaptive Test-Time Compute Governor
- Verifier & Judge Plane
- Multi-Agent Deliberation Layer
- Context & Episodic Memory Plane
- Safety & Attack-Surface Governance
- Evaluation & Observability Plane

This set captures the strongest recurring mechanisms in the corpus while remaining implementation-neutral.
