# One-Shot vs. Incremental Coding: What the Literature Says

**Timestamp:** 2026-05-02T18:53:30

## Bottom line
The literature does **not** support a simple “one-shot is better” or “incremental is better” claim.

Instead, it suggests:

- **One-shot coding** is strong for local, well-specified tasks.
- **Incremental coding with intermediate context** is stronger for multi-step, stateful, or dependency-heavy tasks.
- **Hybrid systems** are usually best in practice: one-shot within each local step, incremental across the full workflow.

## 1) One-shot coding: best for local synthesis
One-shot coding means the model receives the relevant context once and emits code directly, often as a completion or edit.

### Evidence patterns
- **SRI** improves code infilling by turning it into explicit editing rather than unconstrained completion. This makes the local step more one-shot-friendly. 
  - Digest: [From Completion to Editing: Unlocking Context-Aware Code Infilling via Search-and-Replace Instruction Tuning](../ml_research_analysis_2026/2601.13384_from-completion-to-editing-unlocking-context-aware-code-infilling-via-search-and-replace-instruction-tuning_20260208_203654.md) — arXiv:2601.13384
- **InlineCoder** inlines the call graph neighborhood so the model can solve a repository-level task with a more local prompt. 
  - Digest: [In Line with Context: Repository-Level Code Generation via Context Inlining](../ml_research_analysis_2026/2601.00376_in-line-with-context-repository-level-code-generation-via-context-inlining_20260208_123538.md) — arXiv:2601.00376

### Interpretation
One-shot works when the task context can be made **explicit enough** that the model does not need to search or revise much.

### Failure mode
If hidden dependencies or cross-file invariants exist, one-shot generation tends to miss them.

---

## 2) Incremental coding: best for stateful or dependency-heavy tasks
Incremental coding means the model builds an artifact through multiple stages, using intermediate outputs, tests, and updated context.

### Evidence patterns
- **ClassEval-TDD** scales generation from functions to classes by using dependency-aware scheduling, method-level public tests, and bounded repair. It improves class correctness by **12–26 absolute points** and reaches up to **71% fully correct classes**.
  - Digest: [Scaling Test-Driven Code Generation from Functions to Classes: An Empirical Study](../ml_research_analysis_2026/2602.03557_scaling-test-driven-code-generation-from-functions-to-classes-an-empirical-study_20260208_123615.md) — arXiv:2602.03557
- **ContextBench** shows that good retrieval quality matters, but also that context can be dropped during consolidation; process-aware retrieval matters more than raw prompt size.
  - Digest: [ContextBench: A Benchmark for Context Retrieval in Coding Agents](../ml_research_analysis_2026/2602.05892_contextbench-a-benchmark-for-context-retrieval-in-coding-agents_20260401_164932.md) — arXiv:2602.05892
- **CodeCompass** shows that agents fail when they cannot navigate hidden dependencies, even if the raw codebase is available.
  - Digest: [CodeCompass: Navigating the Navigation Paradox in Agentic Code Intelligence](../ml_research_analysis_2026/2602.20048_codecompass-navigating-the-navigation-paradox-in-agentic-code-intelligence_20260402_232714.md) — arXiv:2602.20048

### Interpretation
Incremental coding helps when the problem has:
- prerequisites,
- shared state,
- verification loops,
- or cross-module dependencies.

### Failure mode
If the intermediate context is not compacted carefully, the system accumulates noise and loses key evidence.

---

## 3) The context layer is the hidden variable
The biggest relationship in the literature is not just about generation style; it is about **how context evolves over time**.

### Relevant results
- **Focus / Active Context Compression**: agent-controlled checkpoints and summarization reduce context bloat while keeping task-relevant knowledge. 
  - Digest: [Active Context Compression: Autonomous Memory Management in LLM Agents](../ml_research_analysis_2026/2601.07190_active-context-compression-autonomous-memory-management-in-llm-agents_20260210_161827.md) — arXiv:2601.07190
- **OPENDEV**: adaptive compaction increases session length and reduces peak context usage.
  - Digest: [Building Effective AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned](../ml_research_analysis_2026/2603.05344_building-effective-ai-coding-agents-for-the-terminal-scaffolding-harness-context-engineering-and-lessons-learned_20260403_112109.md) — arXiv:2603.05344
- **Agent Skill Framework**: progressive disclosure helps only when the model can route to the right skill; revealing too much upfront can hurt small models.
  - Digest: [Agent Skill Framework: Perspectives on the Potential of Small Language Models in Industrial Environments](../ml_research_analysis_2026/2602.16653_agent-skill-framework-perspectives-on-the-potential-of-small-language-models-in-industrial-environments_20260402_151011.md) — arXiv:2602.16653

### Interpretation
Incremental workflows are beneficial only if the system can:
1. decide when to reveal more context,
2. preserve the right intermediate artifacts,
3. compress old context without deleting essential dependencies.

---

## 4) A useful rule of thumb from the literature

### Choose one-shot when:
- the task is local,
- dependencies are explicit,
- the artifact is small,
- the model already has enough relevant context.

### Choose incremental when:
- the task is multi-step,
- the code spans multiple files or methods,
- tests/verification matter,
- the agent must revise after seeing failures.

### Choose hybrid when:
- you want one-shot local edits inside a larger incremental workflow.

That hybrid pattern is the most common winning shape across the papers.

---

## 5) The main literature tension
There is a real tension between:

- **more context** to avoid missing dependencies, and
- **less context** to avoid bloat and distraction.

The literature resolves this not by choosing one side, but by using:
- retrieval,
- structural navigation,
- staged disclosure,
- and compression.

That is why the same broad research thread keeps recurring across code, memory, and agent workflows.

---

## 6) Practical synthesis
For coding systems, the most defensible design is:

1. **One-shot** for local generation or edit proposals.
2. **Incremental** for dependency resolution and repair.
3. **Compression** between stages.
4. **Structured artifacts** instead of free-text only.

This aligns with the strongest results in the literature and avoids the worst failure modes of both extremes.

## References
1. [From Completion to Editing: Unlocking Context-Aware Code Infilling via Search-and-Replace Instruction Tuning](../ml_research_analysis_2026/2601.13384_from-completion-to-editing-unlocking-context-aware-code-infilling-via-search-and-replace-instruction-tuning_20260208_203654.md) — arXiv:2601.13384
2. [In Line with Context: Repository-Level Code Generation via Context Inlining](../ml_research_analysis_2026/2601.00376_in-line-with-context-repository-level-code-generation-via-context-inlining_20260208_123538.md) — arXiv:2601.00376
3. [Scaling Test-Driven Code Generation from Functions to Classes: An Empirical Study](../ml_research_analysis_2026/2602.03557_scaling-test-driven-code-generation-from-functions-to-classes-an-empirical-study_20260208_123615.md) — arXiv:2602.03557
4. [ContextBench: A Benchmark for Context Retrieval in Coding Agents](../ml_research_analysis_2026/2602.05892_contextbench-a-benchmark-for-context-retrieval-in-coding-agents_20260401_164932.md) — arXiv:2602.05892
5. [CodeCompass: Navigating the Navigation Paradox in Agentic Code Intelligence](../ml_research_analysis_2026/2602.20048_codecompass-navigating-the-navigation-paradox-in-agentic-code-intelligence_20260402_232714.md) — arXiv:2602.20048
6. [Active Context Compression: Autonomous Memory Management in LLM Agents](../ml_research_analysis_2026/2601.07190_active-context-compression-autonomous-memory-management-in-llm-agents_20260210_161827.md) — arXiv:2601.07190
7. [Building Effective AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned](../ml_research_analysis_2026/2603.05344_building-effective-ai-coding-agents-for-the-terminal-scaffolding-harness-context-engineering-and-lessons-learned_20260403_112109.md) — arXiv:2603.05344
8. [Agent Skill Framework: Perspectives on the Potential of Small Language Models in Industrial Environments](../ml_research_analysis_2026/2602.16653_agent-skill-framework-perspectives-on-the-potential-of-small-language-models-in-industrial-environments_20260402_151011.md) — arXiv:2602.16653
