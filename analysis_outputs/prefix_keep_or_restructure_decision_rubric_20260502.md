# Prefix Keep vs. Restructure: A Standalone Decision Rubric for Coding Contexts

**Timestamp:** 2026-05-02T19:27:15

## Preface: Applicability status
This rubric is **partly applicable now** as a preflight framework, but it is **not yet fully operationalized**. The sufficiency, dependency, and growth checks are actionable today; however, the exact thresholds, scoring cutoffs, and a universally validated predictor for "restructure vs. keep" remain partly theoretical.

## Purpose
This document turns the literature into an operational rubric for deciding whether a prompt/prefix should be:

1. **kept and cached as-is**,
2. **compressed**, or
3. **restructured into a better context**.

The core idea is that a prefix is not valuable just because it is reusable or long; it is valuable if it is **task-sufficient**, **dependency-complete**, and **stable under growth**.

The strongest prior art for this framing is **ContextBench**, which defines **human-verified gold contexts** as the minimal sufficient code regions needed for a task and measures how well agents retrieve them.

- Digest: [ContextBench: A Benchmark for Context Retrieval in Coding Agents](../ml_research_analysis_2026/2602.05892_contextbench-a-benchmark-for-context-retrieval-in-coding-agents_20260401_164932.md) — **arXiv:2602.05892**

---

## One-line takeaway

> **Keep/cache a prefix only when it is already close to the task’s gold-sufficient context and behaves stably; otherwise restructure it around the missing dependencies or the likely failure mode.**

---

## What the literature says about “gold context”

### Direct quantification
**ContextBench** is the clearest paper that directly quantifies a gold context.

It evaluates agent context retrieval against human-verified gold contexts using:
- file recall / precision / F1
- block recall / precision / F1
- line recall / precision / F1
- efficiency (AUC-Cov)
- redundancy
- evidence drop

This makes it the best basis for a keep-vs-restructure policy.

### Partial quantification of sufficiency
Other papers quantify *sufficiency proxies* rather than gold context itself:

- **SRI** shows that code infilling often works with a relatively small editable window, but degrades when the local context becomes too broad.
  - Digest: [From Completion to Editing: Unlocking Context-Aware Code Infilling via Search-and-Replace Instruction Tuning](../ml_research_analysis_2026/2601.13384_from-completion-to-editing-unlocking-context-aware-code-infilling-via-search-and-replace-instruction-tuning_20260208_203654.md) — **arXiv:2601.13384**
- **InlineCoder** shows that repository-level generation often benefits from explicitly inlining the call-graph neighborhood.
  - Digest: [In Line with Context: Repository-Level Code Generation via Context Inlining](../ml_research_analysis_2026/2601.00376_in-line-with-context-repository-level-code-generation-via-context-inlining_20260208_123538.md) — **arXiv:2601.00376**
- **CodeCompass** argues that the real bottleneck is often hidden dependencies, not token count.
  - Digest: [CodeCompass: Navigating the Navigation Paradox in Agentic Code Intelligence](../ml_research_analysis_2026/2602.20048_codecompass-navigating-the-navigation-paradox-in-agentic-code-intelligence_20260402_232714.md) — **arXiv:2602.20048**

### Context growth and rot
- **LOCA-bench** shows that longer contexts can actively worsen performance, and that programmatic tool calling can reduce trajectory length and improve accuracy.
  - Digest: [LOCA-bench: Benchmarking Language Agents Under Controllable and Extreme Context Growth](../ml_research_analysis_2026/2602.07962_loca-bench-benchmarking-language-agents-under-controllable-and-extreme-context-growth_20260401_150556.md) — **arXiv:2602.07962**
- **Active Context Compression** and **OPENDEV** show that context management must include explicit compaction and pruning.
  - Digest: [Active Context Compression: Autonomous Memory Management in LLM Agents](../ml_research_analysis_2026/2601.07190_active-context-compression-autonomous-memory-management-in-llm-agents_20260210_161827.md) — **arXiv:2601.07190**
  - Digest: [Building Effective AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned](../ml_research_analysis_2026/2603.05344_building-effective-ai-coding-agents-for-the-terminal-scaffolding-harness-context-engineering-and-lessons-learned_20260403_112109.md) — **arXiv:2603.05344**

### Predictive selection
- **PoC** is the strongest general template for making a resource decision from a predictor: estimate performance across candidate budgets and choose the cheapest option that clears a floor.
  - Digest: [PoC: Performance-oriented Context Compression for Large Language Models via Performance Prediction](../ml_research_analysis_2026/2603.19733_poc-performance-oriented-context-compression-for-large-language-models-via-performance-prediction_20260403_144018.md) — **arXiv:2603.19733**

This suggests the same structure for prefixes: estimate task success under the current prefix and under a restructured prefix, then choose the cheaper option that meets the target.

---

## Agentic vs. non-agentic checks

The literature points to a useful distinction:

### Agentic checks
These are checks where the **agent participates in deciding or discovering** what context to keep.

Examples:
- task-phase detection / “when to compact”
- agent-generated summaries
- agent-issued tool calls to fetch missing dependencies
- progressive disclosure of additional context only when needed
- reflection on whether a subtask is complete

Relevant papers:
- **Active Context Compression** — agent-controlled `start_focus` / `complete_focus`
- **OPENDEV** — adaptive context compaction within a coding-agent loop
- **Agent Skill Framework** — progressive disclosure / reveal actions (context revealed only when needed)
  - Digest: [Agent Skill Framework: Perspectives on the Potential of Small Language Models in Industrial Environments](../ml_research_analysis_2026/2602.16653_agent-skill-framework-perspectives-on-the-potential-of-small-language-models-in-industrial-environments_20260402_151011.md) — **arXiv:2602.16653**
- **CodeCompass** — the agent must choose to invoke structural navigation tools

### Non-agentic checks
These are checks performed by an **external controller, predictor, or benchmark harness**.

Examples:
- retrieval recall against a gold context
- dependency-graph coverage computed outside the model
- predicted performance under a candidate compression ratio
- context-length stress testing
- evidence-drop scoring from logs

Relevant papers:
- **ContextBench** — external gold-context metrics
- **PoC** — predictor chooses compression ratio
- **LOCA-bench** — external context-growth benchmark

### Practical interpretation
A robust policy usually combines both:

1. **Non-agentic verification** decides whether the current prefix is good enough.
2. **Agentic behavior** decides how to refine or rebuild it.

That is why the best systems are hybrids: they do not trust the agent blindly, but they also do not freeze the prefix without allowing task-aware restructuring.

---

## The 3-gate preflight rubric

### Gate A: Sufficiency
**Question:** Does the prefix contain the minimal evidence needed for the task?

Measure:
- file recall
- block recall
- line recall
- gold-context overlap
- presence of key tests, stack traces, call sites, or diffs

**Interpretation**
- High coverage → keep/cache is plausible.
- Low coverage → restructure.

**Literature anchor:** ContextBench.

---

### Gate B: Dependency completeness
**Question:** Does the prefix include the structural dependencies the task requires?

Measure:
- call-graph neighborhood coverage
- import / symbol graph coverage
- caller/callee presence
- cross-file invariant presence
- hidden-dependency risk

**Interpretation**
- Dependencies present → keep/cache is plausible.
- Dependencies missing or likely hidden → restructure around the dependency graph.

**Literature anchors:** InlineCoder, CodeCompass.

---

### Gate C: Stability under growth
**Question:** Does adding more context degrade reasoning, exploration, or tool use?

Measure:
- trajectory length plateau
- reduced exploration breadth
- reduced tool usage
- higher error rate under longer prompt variants
- evidence drop after consolidation
- context rot / lost-in-the-middle symptoms

**Interpretation**
- Stable behavior → keep/cache is plausible.
- Growth-induced degradation → compress/restructure.

**Literature anchors:** LOCA-bench, Active Context Compression, OPENDEV.

---

## Decision rule

### Keep / cache the prefix when all are true:
1. The prefix is sufficiently close to gold context.
2. Structural dependencies are mostly present.
3. Growth does not make behavior worse.
4. The prefix is likely to be reused.
5. The predicted task success is above the required floor.

### Restructure the prefix when any are true:
1. Coverage is incomplete.
2. Hidden dependencies are likely.
3. Longer context is already causing rot or reduced exploration.
4. A better context layout is likely to improve success enough to justify the rebuild cost.

### Compress the prefix when:
- coverage is adequate but the context is bloated,
- the task is stable but the history is noisy,
- or the current structure is good but too expensive to keep verbatim.

---

## Suggested scoring bundle
A practical rubric should compute a few normalized scores:

### 1) Sufficiency score `S`
How close is the prefix to the gold / gold-like context?
- based on file/block/line coverage and task-relevant artifacts

### 2) Dependency score `D`
How complete is the structural neighborhood?
- based on call graph, import graph, symbol graph, invariant coverage

### 3) Growth brittleness score `B`
How likely is the prefix to degrade the model as more context is appended?
- based on trajectory plateau, tool suppression, or accuracy drop at longer lengths

### 4) Restructure gain score `G`
How much better is a candidate structured context expected to be?
- based on a predictor or diagnostic rerun

### 5) Reuse value `R`
How many future tasks or turns will benefit from caching this prefix?
- based on expected reuse frequency and similarity of future tasks

---

## Example policy
A simple policy could be:

- **Keep/cache** if `S >= τS`, `D >= τD`, `B <= τB`, and `G < τG`.
- **Restructure** if `S < τS` or `D < τD` or `B > τB`.
- **Compress** if `S` and `D` are acceptable but the prefix is expensive and `R` is high.

This is not a universal threshold from the literature; it is the correct operational shape of the decision.

---

## How to implement the preflight

### Non-agentic verification step
Run these checks outside the model:
- compare the prefix against known gold context
- measure dependency coverage
- inspect prompt growth effects across variants
- if possible, predict performance under current vs. restructured context

### Agentic refinement step
If verification fails, let the agent:
- fetch missing dependencies,
- rewrite the context into a better structure,
- compress redundant history,
- preserve artifacts like tests, diffs, errors, and assumptions.

### Final decision
Pick the option with the highest predicted utility:

`expected task success - rebuild cost - future context cost`

That is the most faithful synthesis of the literature.

---

## Why this matters for coding agents
For code, the prefix is rarely just text. It is a mixture of:
- task statement
- repo state
- failed attempts
- tests
- stack traces
- symbol references
- dependency cues
- prior decisions

So the question is not “should I cache this prompt?”
It is:

> **Is this the right representation of the task state, or should it be rebuilt into a more sufficient one?**

That is the real preflight question.

---

## References
1. [ContextBench: A Benchmark for Context Retrieval in Coding Agents](../ml_research_analysis_2026/2602.05892_contextbench-a-benchmark-for-context-retrieval-in-coding-agents_20260401_164932.md) — **arXiv:2602.05892**
2. [CodeCompass: Navigating the Navigation Paradox in Agentic Code Intelligence](../ml_research_analysis_2026/2602.20048_codecompass-navigating-the-navigation-paradox-in-agentic-code-intelligence_20260402_232714.md) — **arXiv:2602.20048**
3. [In Line with Context: Repository-Level Code Generation via Context Inlining](../ml_research_analysis_2026/2601.00376_in-line-with-context-repository-level-code-generation-via-context-inlining_20260208_123538.md) — **arXiv:2601.00376**
4. [From Completion to Editing: Unlocking Context-Aware Code Infilling via Search-and-Replace Instruction Tuning](../ml_research_analysis_2026/2601.13384_from-completion-to-editing-unlocking-context-aware-code-infilling-via-search-and-replace-instruction-tuning_20260208_203654.md) — **arXiv:2601.13384**
5. [Scaling Test-Driven Code Generation from Functions to Classes: An Empirical Study](../ml_research_analysis_2026/2602.03557_scaling-test-driven-code-generation-from-functions-to-classes-an-empirical-study_20260208_123615.md) — **arXiv:2602.03557**
6. [LOCA-bench: Benchmarking Language Agents Under Controllable and Extreme Context Growth](../ml_research_analysis_2026/2602.07962_loca-bench-benchmarking-language-agents-under-controllable-and-extreme-context-growth_20260401_150556.md) — **arXiv:2602.07962**
7. [Active Context Compression: Autonomous Memory Management in LLM Agents](../ml_research_analysis_2026/2601.07190_active-context-compression-autonomous-memory-management-in-llm-agents_20260210_161827.md) — **arXiv:2601.07190**
8. [Building Effective AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned](../ml_research_analysis_2026/2603.05344_building-effective-ai-coding-agents-for-the-terminal-scaffolding-harness-context-engineering-and-lessons-learned_20260403_112109.md) — **arXiv:2603.05344**
9. [Agent Skill Framework: Perspectives on the Potential of Small Language Models in Industrial Environments](../ml_research_analysis_2026/2602.16653_agent-skill-framework-perspectives-on-the-potential-of-small-language-models-in-industrial-environments_20260402_151011.md) — **arXiv:2602.16653**
10. [PoC: Performance-oriented Context Compression for Large Language Models via Performance Prediction](../ml_research_analysis_2026/2603.19733_poc-performance-oriented-context-compression-for-large-language-models-via-performance-prediction_20260403_144018.md) — **arXiv:2603.19733**
