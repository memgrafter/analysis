# Agentic vs. Non-Agentic Compaction for Coding Agents

**Timestamp:** 2026-05-02T15:43:32

## Bottom line
For coding agents, the best pattern is usually **agentic control + non-agentic structured compression**.

- **Agentic compaction**: the agent decides *when* to compact, based on task phase boundaries and what it considers salient.
- **Non-agentic compaction**: an external rule, predictor, or compressor decides *how much* to compact and/or what to keep.

The survey on autonomous agent memory frames memory as a **write–manage–read loop** and classifies methods along temporal scope, representation, and control policy. Its control-policy axis is the key distinction here: heuristic, prompted, or learned management. See the digest for exact framing: [Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers](../ml_research_analysis_2026/2603.07670_memory-for-autonomous-llm-agents-mechanisms-evaluation-and-emerging-frontiers_20260402_083855.md) (arXiv:2603.07670).

## What the evidence suggests

### 1) Agentic compaction works best when task boundaries are meaningful
The clearest agentic example I found is **Focus / Active Context Compression**: the agent uses `start_focus` and `complete_focus` to decide when to summarize and prune context. It reported a 22.7% token reduction while maintaining the same accuracy on a small SWE-bench Lite sample. Digest: [Active Context Compression: Autonomous Memory Management in LLM Agents](../ml_research_analysis_2026/2601.07190_active-context-compression-autonomous-memory-management-in-llm-agents_20260210_161827.md) (arXiv:2601.07190).

**Takeaway:** agentic compaction is strong when the model can recognize natural phase transitions like “debugging is done; preserve the lesson, discard the trial history.”

### 2) Non-agentic compaction is better when you want predictability and hard budgets
Several methods are fundamentally non-agentic:

- **Fast KV Compaction via Attention Matching** preserves attention outputs and mass under reference queries, achieving up to 50× KV cache compaction. Digest: [Fast KV Compaction via Attention Matching](../ml_research_analysis_2026/2602.16284_fast-kv-compaction-via-attention-matching_20260402_233842.md) (arXiv:2602.16284).
- **COMI** uses coarse-to-fine compression with marginal information gain to allocate budget adaptively. Digest: [COMI: Coarse-to-fine Context Compression via Marginal Information Gain](../ml_research_analysis_2026/2602.01719_comi-coarse-to-fine-context-compression-via-marginal-information-gain_20260210_114755.md) (arXiv:2602.01719).
- **Read As Human (RAM)** splits context into segments and decides between close reading and skimming. Digest: [Read As Human: Compressing Context via Parallelizable Close Reading and Skimming](../ml_research_analysis_2026/2602.01840_read-as-human-compressing-context-via-parallelizable-close-reading-and-skimming_20260210_054736.md) (arXiv:2602.01840).
- **PoC** chooses a compression ratio based on a performance predictor and a target performance floor. Digest: [PoC: Performance-oriented Context Compression for Large Language Models via Performance Prediction](../ml_research_analysis_2026/2603.19733_poc-performance-oriented-context-compression-for-large-language-models-via-performance-prediction_20260403_144018.md) (arXiv:2603.19733).

**Takeaway:** non-agentic compaction is preferable when you care about reproducibility, latency, or a strict token budget more than model-side judgment.

### 3) Coding agents benefit from structured preservation, not just free-text summaries
The coding-agent literature points to a hybrid design:

- **OPENDEV** uses adaptive context compaction to reduce peak context consumption and extend session length. Digest: [Building Effective AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned](../ml_research_analysis_2026/2603.05344_building-effective-ai-coding-agents-for-the-terminal-scaffolding-harness-context-engineering-and-lessons-learned_20260403_112109.md) (arXiv:2603.05344).
- **ContextEvolve** decomposes context into code abstraction, trajectory guidance, and exemplar selection using multiple specialized agents. Digest: [ContextEvolve: Multi-Agent Context Compression for Systems Code Optimization](../ml_research_analysis_2026/2602.02597_contextevolve-multi-agent-context-compression-for-systems-code-optimization_20260403_170529.md) (arXiv:2602.02597).
- **Developing Adaptive Context Compression Techniques...** compresses dialogue with importance and coherence signals plus dynamic budgets. Digest: [Developing Adaptive Context Compression Techniques for Large Language Models (LLMs) in Long-Running Interactions](../ml_research_analysis_2026/2603.29193_developing-adaptive-context-compression-techniques-for-large-language-models-llms-in-long-running-interactions_20260405_003852.md) (arXiv:2603.29193).

**Takeaway:** for code, the compressed memory should preserve artifacts like file paths, diffs, stack traces, test outputs, and “what failed / what was tried,” not just prose.

## Practical recommendation

### Use agentic compaction for:
- long debugging sessions
- refactors with clear milestone boundaries
- exploratory work where the agent can tell when a subtask is complete

### Use non-agentic compaction for:
- cache/KV reduction
- predictable cost control
- tasks where you need hard limits and stable behavior

### Best default for coding agents
A **hybrid stack**:
1. The agent decides *when* to compact.
2. A structured compressor decides *what* to preserve.
3. The retained state is schema-like, not just prose.

That aligns with the strongest themes in the papers above: agentic timing helps, but the retained representation must still be disciplined.

## Short conclusion
If you force me to choose, I would say:

- **Agentic compaction** is best for **timing**.
- **Non-agentic compaction** is best for **format and budget control**.
- **Hybrid** is best for real coding systems.

## References
1. [Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers](../ml_research_analysis_2026/2603.07670_memory-for-autonomous-llm-agents-mechanisms-evaluation-and-emerging-frontiers_20260402_083855.md) — arXiv:2603.07670
2. [Active Context Compression: Autonomous Memory Management in LLM Agents](../ml_research_analysis_2026/2601.07190_active-context-compression-autonomous-memory-management-in-llm-agents_20260210_161827.md) — arXiv:2601.07190
3. [Building Effective AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned](../ml_research_analysis_2026/2603.05344_building-effective-ai-coding-agents-for-the-terminal-scaffolding-harness-context-engineering-and-lessons-learned_20260403_112109.md) — arXiv:2603.05344
4. [Fast KV Compaction via Attention Matching](../ml_research_analysis_2026/2602.16284_fast-kv-compaction-via-attention-matching_20260402_233842.md) — arXiv:2602.16284
5. [COMI: Coarse-to-fine Context Compression via Marginal Information Gain](../ml_research_analysis_2026/2602.01719_comi-coarse-to-fine-context-compression-via-marginal-information-gain_20260210_114755.md) — arXiv:2602.01719
6. [Read As Human: Compressing Context via Parallelizable Close Reading and Skimming](../ml_research_analysis_2026/2602.01840_read-as-human-compressing-context-via-parallelizable-close-reading-and-skimming_20260210_054736.md) — arXiv:2602.01840
7. [PoC: Performance-oriented Context Compression for Large Language Models via Performance Prediction](../ml_research_analysis_2026/2603.19733_poc-performance-oriented-context-compression-for-large-language-models-via-performance-prediction_20260403_144018.md) — arXiv:2603.19733
8. [ContextEvolve: Multi-Agent Context Compression for Systems Code Optimization](../ml_research_analysis_2026/2602.02597_contextevolve-multi-agent-context-compression-for-systems-code-optimization_20260403_170529.md) — arXiv:2602.02597
9. [Developing Adaptive Context Compression Techniques for Large Language Models (LLMs) in Long-Running Interactions](../ml_research_analysis_2026/2603.29193_developing-adaptive-context-compression-techniques-for-large-language-models-llms-in-long-running-interactions_20260405_003852.md) — arXiv:2603.29193
