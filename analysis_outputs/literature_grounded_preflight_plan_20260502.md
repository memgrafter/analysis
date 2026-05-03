# Literature-Grounded Plan for a Coding-Context Preflight Judge

**Timestamp:** 2026-05-02T20:07:46

## Goal
Build a **preflight judge** that decides whether a coding task + supplied context should be:

- **submitted as-is** to the coding agent,
- **expanded** with more retrieval, or
- **restructured** into a better context before submission.

This plan is **grounded in the literature** rather than based on ad hoc lexical matching. The guiding idea is:

> Context quality should be estimated from **task sufficiency**, **dependency completeness**, and **stability under growth**, not from token count alone.

The strongest research anchors are:

- **ContextBench** for human-verified gold contexts and minimal-sufficient-code evaluation
- **InlineCoder** and **CodeCompass** for dependency-aware context shape
- **LOCA-bench**, **Active Context Compression**, and **OPENDEV** for context growth / rot / compaction
- **PoC** for predict-and-select style choice among context budgets or variants
- **Agent Skill Framework** for progressive disclosure and staged context reveal

---

## Literature grounding by design choice

### 1) Sufficiency is about evidence coverage, not token length
**Paper anchor:** ContextBench

- Digest: [ContextBench: A Benchmark for Context Retrieval in Coding Agents](../ml_research_analysis_2026/2602.05892_contextbench-a-benchmark-for-context-retrieval-in-coding-agents_20260401_164932.md) — **arXiv:2602.05892**

**Implication for the judge:**
- Estimate whether the provided context covers the **minimal sufficient evidence** for the task.
- Prefer **file/block/line evidence coverage** and **gold-context-style overlap** when available.
- Do not use token count as the primary proxy.

---

### 2) Structure matters: hidden dependencies can invalidate an otherwise plausible prefix
**Paper anchors:** InlineCoder, CodeCompass

- Digest: [In Line with Context: Repository-Level Code Generation via Context Inlining](../ml_research_analysis_2026/2601.00376_in-line-with-context-repository-level-code-generation-via-context-inlining_20260208_123538.md) — **arXiv:2601.00376**
- Digest: [CodeCompass: Navigating the Navigation Paradox in Agentic Code Intelligence](../ml_research_analysis_2026/2602.20048_codecompass-navigating-the-navigation-paradox-in-agentic-code-intelligence_20260402_232714.md) — **arXiv:2602.20048**

**Implication for the judge:**
- Score whether the context contains the relevant **call graph neighborhood**, **imports**, **callers/callees**, and other structural neighbors.
- A prefix that is lexically relevant but structurally incomplete should be marked for **restructure**, not mere caching.

---

### 3) Longer context can hurt; compaction is not optional
**Paper anchors:** LOCA-bench, Active Context Compression, OPENDEV

- Digest: [LOCA-bench: Benchmarking Language Agents Under Controllable and Extreme Context Growth](../ml_research_analysis_2026/2602.07962_loca-bench-benchmarking-language-agents-under-controllable-and-extreme-context-growth_20260401_150556.md) — **arXiv:2602.07962**
- Digest: [Active Context Compression: Autonomous Memory Management in LLM Agents](../ml_research_analysis_2026/2601.07190_active-context-compression-autonomous-memory-management-in-llm-agents_20260210_161827.md) — **arXiv:2601.07190**
- Digest: [Building Effective AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned](../ml_research_analysis_2026/2603.05344_building-effective-ai-coding-agents-for-the-terminal-scaffolding-harness-context-engineering-and-lessons-learned_20260403_112109.md) — **arXiv:2603.05344**

**Implication for the judge:**
- Estimate **growth brittleness**: will adding more context likely improve or degrade behavior?
- Prefer explicit compaction / restructuring when the current context shows signs of rot, repetition, or exploration suppression.

---

### 4) Candidate contexts should be compared by predicted utility, not guessed
**Paper anchor:** PoC

- Digest: [PoC: Performance-oriented Context Compression for Large Language Models via Performance Prediction](../ml_research_analysis_2026/2603.19733_poc-performance-oriented-context-compression-via-performance-prediction_20260403_144018.md) — **arXiv:2603.19733**

**Implication for the judge:**
- When two candidate contexts are possible, estimate which one is more likely to meet a success floor.
- Choose the cheaper context that still clears the floor.

This means the preflight judge should behave like a **predict-and-select** module, not a static heuristic.

---

### 5) Context reveal should be staged, not all-or-nothing
**Paper anchor:** Agent Skill Framework

- Digest: [Agent Skill Framework: Perspectives on the Potential of Small Language Models in Industrial Environments](../ml_research_analysis_2026/2602.16653_agent-skill-framework-perspectives-on-the-potential-of-small-language-models-in-industrial-environments_20260402_151011.md) — **arXiv:2602.16653**

**Implication for the judge:**
- The judge should support **progressive disclosure**: reveal more context only when the current state is insufficient.
- This makes the judge a gate in a staged workflow, not a one-shot verdict on a monolithic prompt.

---

## Proposed system architecture

### Stage 1: Non-agentic preflight scoring
This stage is deterministic or rule-based and runs **before** any coding agent is invoked.

It computes four scores:

1. **Sufficiency score**
   - Does the context include the task-relevant evidence?
   - Approximate with coverage of files, blocks, lines, symbols, tests, errors, or diff hunks.

2. **Dependency completeness score**
   - Does the context include the structural neighborhood the task depends on?
   - Approximate with call graph / import graph / symbol graph neighborhood coverage.

3. **Growth brittleness score**
   - Is the context already bloated or noisy enough that more accumulation will likely hurt?
   - Approximate with prompt length, repetition, low code density, low test density, error density, and prior context-growth signals.

4. **Predicted restructure gain**
   - Would a better-shaped context likely improve success enough to justify rebuild cost?
   - Approximate with a lightweight predictor or a ranked comparison among candidate context layouts.

### Stage 2: Agentic fallback for ambiguity
If the non-agentic stage is uncertain, escalate to an **LLM judge** that can reason about:

- task type (repair / implement / refactor / verify)
- missing dependencies
- whether context should be expanded or reorganized
- whether the task is likely to thrash if submitted now

This is where **agentic** checks enter: not as the first line of defense, but as a fallback when the structural signals are ambiguous.

---

## Non-agentic checks vs. agentic checks

### Non-agentic checks
These are computed outside the model and should be the primary gate.

- evidence coverage
- dependency graph coverage
- context growth / rot signals
- prediction of utility under alternative context layouts
- reuse value of a cached prefix

### Agentic checks
These rely on model judgment and are used sparingly.

- task-phase detection
- “what is missing?” analysis
- “should I retrieve more?” decision
- progressive disclosure recommendations

### Design principle
The judge should **trust non-agentic checks first** and use agentic checks only when the signal is unclear.

---

## Practical v1 rubric

### Input
- task text
- context text or files
- optional repo metadata if available

### Outputs
- decision: `submit`, `expand_context`, or `restructure`
- confidence score
- sufficiency score
- dependency completeness score
- growth brittleness score
- predicted restructure gain
- explanation of missing evidence or missing dependencies

### Decision logic
A simple first pass:

- **submit** if sufficiency is high, dependency completeness is high, and growth brittleness is low.
- **expand_context** if sufficiency is borderline but the current structure is plausible.
- **restructure** if coverage is low, dependencies are missing, or context rot is likely.

This is not a universal threshold from the literature; it is a research-grounded implementation of the literature’s logic.

---

## Implementation plan

### Phase A: Build non-agentic scoring first
Implement scores from the task/context only:

- lexical and structural evidence coverage
- path / symbol / test / error / diff signals
- growth brittleness from length and repetition
- simple task typing

**Do not start with a purely lexical matcher.**
Lexical matching is only a weak proxy for gold-context sufficiency.

### Phase B: Add repo-aware structure
If repository metadata exists, extend the judge with:

- AST-derived symbols
- call graph neighbors
- import dependencies
- file ownership / module grouping
- test association

This brings the system closer to the structural ideas in InlineCoder and CodeCompass.

### Phase C: Add candidate re-layout comparison
Given two or more candidate context layouts, estimate which one is more likely to succeed.
This follows the PoC pattern of **predict and select**.

### Phase D: Add an LLM fallback
Only if the non-agentic score is uncertain.
The LLM should explain:

- why the current prefix is insufficient
- what dependencies are missing
- whether more retrieval or a restructured context would help more

### Phase E: Calibrate on known tasks
Tune thresholds using tasks with known outcomes, especially tasks where gold-context annotations or strong retrieval targets exist.

---

## Evaluation plan grounded in literature

### Primary evaluation
Use **ContextBench-style** tasks where gold-context references exist.

Measure:
- whether the judge predicts submit vs restructure correctly
- correlation with file/block/line coverage
- false submit rate on incomplete contexts
- false restructure rate on already sufficient contexts

### Secondary evaluation
Use tasks with known structural dependencies:
- repository-level repair
- class-level generation
- multi-file editing

Measure:
- how often dependency-aware scoring improves over lexical coverage alone

### Growth / rot evaluation
Use long-context settings inspired by LOCA-bench:
- vary context length while holding task semantics fixed
- test whether the judge flags contexts that later degrade agent performance

### Ablation evaluation
Remove each signal in turn:
- no dependency graph
- no growth brittleness signal
- no gold-context proxy
- no candidate comparison

This identifies which signal actually carries the most value.

---

## What not to do
To stay grounded in the literature, avoid these failure modes:

- **Do not** use raw token length as the main decision criterion.
- **Do not** rely on lexical overlap alone.
- **Do not** treat a cached prefix as automatically sufficient.
- **Do not** use the LLM as the first and only gate.
- **Do not** collapse all context-quality questions into a single confidence number without interpretable sub-scores.

---

## Success criteria
The judge is useful if it can reliably answer:

1. Is this context already close to the task’s gold-sufficient evidence?
2. Are important dependencies missing?
3. Is this context likely to degrade if we keep growing it?
4. Would a restructured prefix likely improve success enough to justify rebuild cost?

If yes, then the judge is doing something the literature supports.

---

## Summary
This plan is grounded in the literature because it follows the main research signals:

- **ContextBench** → gold-context sufficiency
- **InlineCoder / CodeCompass** → structural dependencies matter
- **LOCA-bench / OPENDEV / Active Context Compression** → context growth can hurt
- **PoC** → compare candidate context variants by predicted performance
- **Agent Skill Framework** → staged / progressive context disclosure

The resulting system should be a **preflight gate**, not a magical context oracle:
- first compute non-agentic structural signals,
- then use agentic judgment only for ambiguity,
- and calibrate against known outcomes.

That is the literature-grounded way to decide whether to submit a coding task, expand its context, or restructure it first.

---

## References
1. [ContextBench: A Benchmark for Context Retrieval in Coding Agents](../ml_research_analysis_2026/2602.05892_contextbench-a-benchmark-for-context-retrieval-in-coding-agents_20260401_164932.md) — **arXiv:2602.05892**
2. [In Line with Context: Repository-Level Code Generation via Context Inlining](../ml_research_analysis_2026/2601.00376_in-line-with-context-repository-level-code-generation-via-context-inlining_20260208_123538.md) — **arXiv:2601.00376**
3. [CodeCompass: Navigating the Navigation Paradox in Agentic Code Intelligence](../ml_research_analysis_2026/2602.20048_codecompass-navigating-the-navigation-paradox-in-agentic-code-intelligence_20260402_232714.md) — **arXiv:2602.20048**
4. [LOCA-bench: Benchmarking Language Agents Under Controllable and Extreme Context Growth](../ml_research_analysis_2026/2602.07962_loca-bench-benchmarking-language-agents-under-controllable-and-extreme-context-growth_20260401_150556.md) — **arXiv:2602.07962**
5. [Active Context Compression: Autonomous Memory Management in LLM Agents](../ml_research_analysis_2026/2601.07190_active-context-compression-autonomous-memory-management-in-llm-agents_20260210_161827.md) — **arXiv:2601.07190**
6. [Building Effective AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned](../ml_research_analysis_2026/2603.05344_building-effective-ai-coding-agents-for-the-terminal-scaffolding-harness-context-engineering-and-lessons-learned_20260403_112109.md) — **arXiv:2603.05344**
7. [PoC: Performance-oriented Context Compression for Large Language Models via Performance Prediction](../ml_research_analysis_2026/2603.19733_poc-performance-oriented-context-compression-for-large-language-models-via-performance-prediction_20260403_144018.md) — **arXiv:2603.19733**
8. [Agent Skill Framework: Perspectives on the Potential of Small Language Models in Industrial Environments](../ml_research_analysis_2026/2602.16653_agent-skill-framework-perspectives-on-the-potential-of-small-language-models-in-industrial-environments_20260402_151011.md) — **arXiv:2602.16653**
