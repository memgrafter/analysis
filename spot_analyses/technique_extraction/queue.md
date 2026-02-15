# Technique Extraction Queue

Tracks planned, in-progress, and completed extractions.

References:
- Topic selection/scoping: [topic_selection_and_scoping.md](topic_selection_and_scoping.md)
- Execution procedure: [technique_extraction_procedure.md](technique_extraction_procedure.md)

---

## How this queue is structured (low-maintenance)

To minimize touchups when adding/removing topics:
- Use **priority tiers** (`P0`, `P1`, `P2`) instead of numbered lists.
- Keep each topic as **one table row**.
- Update only two fields when status changes: `status` and `tier`.

`status` values: `queued`, `in_progress`, `done`, `parked`

---

## Completed

| group | papers | status | summary | artifacts |
|---|---:|---|---|---|
| `test_time_compute_scaling` | 987 | done | 20 grep categories -> 3-tier taxonomy, 76.4% coverage | [analysis](test_time_compute_scaling.md), [papers](test_time_compute_scaling/papers.md), [option C](test_time_compute_scaling/option_c_extraction.md), [option A](test_time_compute_scaling/option_a_refinement.md) |
| `test_time_adaptation` | 284 | done | 41 grep patterns (Option C), 82.7% coverage, merged into 18-category final taxonomy | [analysis](test_time_adaptation.md), [papers](test_time_adaptation/papers.md), [option C](test_time_adaptation/option_c_extraction.md), [option A](test_time_adaptation/option_a_refinement.md) |
| `agentic_workflow_pipeline_design` | 92 | done | Reconciled into 8-category workflow-design taxonomy with standalone TTA-grade final doc | [analysis](agentic_workflow_pipeline_design.md), [papers](agentic_workflow_pipeline_design/papers.md), [option C](agentic_workflow_pipeline_design/option_c_extraction.md), [option A](agentic_workflow_pipeline_design/option_a_refinement.md) |
| `continual_online_tta` | 60 | done | Spin-off from TTA set; 58/60 grep-matched (96.7%), reconciled into 8-category continual/online TTA taxonomy | [analysis](continual_online_tta.md), [papers](continual_online_tta/papers.md), [option C](continual_online_tta/option_c_extraction.md), [option A](continual_online_tta/option_a_refinement.md) |
| `adaptive_compute_allocation` | 51 | done | Spin-off from TTC set; 51/51 grep-matched (100%), reconciled into 7-category adaptive compute-allocation taxonomy | [analysis](adaptive_compute_allocation.md), [papers](adaptive_compute_allocation/papers.md), [option C](adaptive_compute_allocation/option_c_extraction.md), [option A](adaptive_compute_allocation/option_a_refinement.md) |

---

## Active Queue (priority tiers)

### P0 (next)

| group | est papers | status | scope (1-line) | source/dependency | FlatAgents/FlatMachines relevance |
|---|---:|---|---|---|---|
| `multi_agent_debate` | ~170 | queued | Multi-LLM debate/deliberation frameworks for answer quality improvement | Decomposed from agentic umbrella; TTC-linked | Tests concrete orchestration motifs for `flatmachines` (parallel branches + reconciliation/voting) and execution strategies (`mdap_voting`-style semantics). |

### P1 (high value, after P0)

| group | est papers | status | scope (1-line) | source/dependency | FlatAgents/FlatMachines relevance |
|---|---:|---|---|---|---|
| `process_reward_models` | 80–150 | queued | Step-level reward signals for intermediate reasoning evaluation | TTC verification sub-cluster | Helps define verifier/judge role patterns (agent-as-judge, scoring loops) for `flatmachines` workflows and structured outputs in `flatagents`. |
| `reasoning_distillation` | 45–120 | queued | Compress expensive reasoning into smaller/faster models | TTC enabling technique | Informs cheap/expensive model cascade design and profile fallback policies across both SDKs. |
| `vlm_tta` | 80–150 | queued | Test-time adaptation for VLM/foundation models | TTA spin-off | Useful for multimodal agent pipeline design (tooling + adaptation hooks) and domain-specific model profile strategy. |

### P2 (needs tighter scoping or overlap control)

| group | est papers | status | scope (1-line) | source/dependency | note |
|---|---:|---|---|---|---|
| `prompt_based_tta` | 35–80 | queued | Prompt-centric test-time adaptation across modalities | TTA spin-off; overlaps `vlm_tta` | Run after `vlm_tta` or narrow aggressively to avoid duplication. |
| `training_free_tta` | 40–90 | queued | Gradient-free adaptation methods at deployment | TTA spin-off | Good candidate after `continual_online_tta`; likely cross-cuts multiple completed groups. |
| `agentic_workflow_design` | 150–300 | queued | Workflow graph design/generation/optimization for agent systems | Decomposed from umbrella | Broad; risks mixing mechanism papers with application/benchmark papers. Only run with stricter Phase 0 filters. |
| `multi_agent_collaboration` | 200–350 | queued | Specialized agent teams coordinating on tasks | Decomposed from umbrella | Bleeds into RL/robotics; requires stronger LLM-only scope gate first. |

---

## Parked / Needs Scoping

| topic | reason parked |
|---|---|
| Diffusion language models | Niche and vocabulary-fragmented in current corpus slices |
| Latent/implicit reasoning | Small but growing; may be better as sub-family until volume increases |
| Speculative decoding for reasoning | Cross-cuts TTC + efficiency; needs sharper boundary |
| RL for reasoning (broad) | Too large and conflates training-time vs inference-time techniques |
| Retrieval-augmented reasoning (broad) | High overlap with generic RAG/application papers |
| Code generation agents | Large domain-heavy cluster; likely requires decomposition first |
| Web/GUI agents | Large domain-heavy cluster; likely requires decomposition first |
| Scientific research agents | Domain-heavy; strong overlap with workflow/application papers |

---

## Notes for insertion/deletion

- Add new candidates as a single row in `P1` by default.
- Promote/demote by changing `tier` section only (no renumbering needed).
- When a topic is finished, move one row to **Completed** and attach artifact links.
