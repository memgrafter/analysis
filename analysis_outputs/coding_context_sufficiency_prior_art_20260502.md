# Minimum Context Needed to Write Correct Code: Prior Art and What It Actually Measures

**Timestamp:** 2026-05-02T18:53:30

## Bottom line
There is **no single universal minimum context length** in the literature for “writing correct code.” Instead, the research studies a more precise question:

> **What is the minimal sufficient context — files, functions, lines, tests, dependency edges, or examples — needed for a specific coding task?**

In other words, the literature treats “minimum context” as a **task-specific sufficiency set**, not a universal token count.

## What the literature directly covers

### 1) ContextBench: minimal sufficient code regions for issue resolution
This is the clearest direct prior art. ContextBench builds **human-verified gold contexts** — the minimal code regions that must be inspected/edited to solve a task — and evaluates agent retrieval against them using recall, precision, F1, evidence drop, and efficiency.

- Digest: [ContextBench: A Benchmark for Context Retrieval in Coding Agents](../ml_research_analysis_2026/2602.05892_contextbench-a-benchmark-for-context-retrieval-in-coding-agents_20260401_164932.md)
- arXiv:2602.05892

**Why it matters:** it operationalizes “minimum context” as **gold context coverage**, not token budget.

### 2) InlineCoder: enough context is often the call graph neighborhood
InlineCoder turns repository-level generation into a function-level task by inlining caller/callee context. That is an explicit statement that the model often needs the **function plus its structural neighborhood**, not the whole repo.

- Digest: [In Line with Context: Repository-Level Code Generation via Context Inlining](../ml_research_analysis_2026/2601.00376_in-line-with-context-repository-level-code-generation-via-context-inlining_20260208_123538.md)
- arXiv:2601.00376

**Why it matters:** the minimum sufficient context may be a **call graph slice**, not arbitrary surrounding text.

### 3) CodeCompass: the key issue can be hidden dependencies, not token count
CodeCompass argues that many coding failures are due to **navigation failure over architecturally hidden dependencies** rather than pure context limits.

- Digest: [CodeCompass: Navigating the Navigation Paradox in Agentic Code Intelligence](../ml_research_analysis_2026/2602.20048_codecompass-navigating-the-navigation-paradox-in-agentic-code-intelligence_20260402_232714.md)
- arXiv:2602.20048

**Why it matters:** the minimum context is not just “more tokens”; it is the **right structural dependencies**.

### 4) SRI: one-shot completion becomes easier when reformulated as explicit editing
Search-and-Replace Infilling reframes code completion as context-aware micro-editing. This suggests that “minimum context” often depends on the **format of the task**: a completion task may need less context when it is reformulated as a localized edit with explicit anchors.

- Digest: [From Completion to Editing: Unlocking Context-Aware Code Infilling via Search-and-Replace Instruction Tuning](../ml_research_analysis_2026/2601.13384_from-completion-to-editing-unlocking-context-aware-code-infilling-via-search-and-replace-instruction-tuning_20260208_203654.md)
- arXiv:2601.13384

**Why it matters:** local edit framing can shrink the context requirement.

### 5) Class-level TDD: the minimum context expands when the artifact becomes multi-step
When code generation moves from functions to classes, the paper finds that dependency-aware scheduling plus method-level tests improve correctness substantially.

- Digest: [Scaling Test-Driven Code Generation from Functions to Classes: An Empirical Study](../ml_research_analysis_2026/2602.03557_scaling-test-driven-code-generation-from-functions-to-classes-an-empirical-study_20260208_123615.md)
- arXiv:2602.03557

**Why it matters:** as task granularity increases, the needed context becomes the **dependency schedule + test scaffold + partially built artifact**.

## So what is the “minimum context” relationship?

The literature suggests three increasingly strong notions:

1. **Local sufficiency**: enough to write the next token/function/method.
2. **Structural sufficiency**: enough to satisfy dependencies and invariants.
3. **Process sufficiency**: enough to support iterative repair and verification.

For coding tasks, the needed context usually grows from (1) to (3) as you move from:
- code completion →
- edit/infilling →
- repository repair →
- class/module generation →
- long-horizon agentic debugging.

## Practical synthesis
If you want a more precise answer than “how much context?”, use this decomposition:

- **Scope**: function / class / file / repository
- **Dependency type**: lexical / structural / behavioral / test-based
- **Task phase**: synthesis / diagnosis / repair / verification
- **Artifact type**: code / tests / diffs / call graphs / execution traces

Then the question becomes:

> What is the smallest artifact set that preserves the task-relevant dependencies and verification signals?

That is the actual research object in most of the literature.

## What is *not* well established
- A universal token threshold for correct code generation
- A model-independent “minimum context size” across tasks
- A single benchmark that measures minimal sufficiency for all code tasks

## Best current reading of the field
The field is converging on this view:

- **Correct code is usually a sufficiency problem, not a raw context-length problem.**
- The right unit is often **files/lines/functions/tests/dependencies**, not tokens.
- For harder tasks, **incremental context plus structured verification** tends to beat a single large prompt.

## References
1. [ContextBench: A Benchmark for Context Retrieval in Coding Agents](../ml_research_analysis_2026/2602.05892_contextbench-a-benchmark-for-context-retrieval-in-coding-agents_20260401_164932.md) — arXiv:2602.05892
2. [In Line with Context: Repository-Level Code Generation via Context Inlining](../ml_research_analysis_2026/2601.00376_in-line-with-context-repository-level-code-generation-via-context-inlining_20260208_123538.md) — arXiv:2601.00376
3. [CodeCompass: Navigating the Navigation Paradox in Agentic Code Intelligence](../ml_research_analysis_2026/2602.20048_codecompass-navigating-the-navigation-paradox-in-agentic-code-intelligence_20260402_232714.md) — arXiv:2602.20048
4. [From Completion to Editing: Unlocking Context-Aware Code Infilling via Search-and-Replace Instruction Tuning](../ml_research_analysis_2026/2601.13384_from-completion-to-editing-unlocking-context-aware-code-infilling-via-search-and-replace-instruction-tuning_20260208_203654.md) — arXiv:2601.13384
5. [Scaling Test-Driven Code Generation from Functions to Classes: An Empirical Study](../ml_research_analysis_2026/2602.03557_scaling-test-driven-code-generation-from-functions-to-classes-an-empirical-study_20260208_123615.md) — arXiv:2602.03557
