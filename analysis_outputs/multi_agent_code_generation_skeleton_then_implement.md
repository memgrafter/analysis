# Multi-Agent Code Generation: Skeleton-Then-Implement Pattern

## Research Question

Can one agent write the code skeleton (classes + function signatures) while follow-up agents implement each function? What arxiv papers from the analysis corpus validate this approach?

## Tier 1: Directly validates skeleton → implement pattern

| arxiv ID | Title | Key Pattern | Key Result |
|-----------|-------|-------------|------------|
| 2312.15960 | MoTCoder: Modular of Thought | Step 1: generate function headers + docstrings for sub-modules. Step 2: implement modules. | +12.9% APPS, +9.43% CodeContests |
| 2405.20092 | FUNCODER: Divide-and-Conquer Meets Consensus | Recursive decomposition into tree of sub-function stubs, bottom-up implementation with behavioral consensus selection | +9.8% Pass@1, 76.5% token reduction |
| 2310.08992 | CodeChain: Modular Code Generation | Generate modularized code with function headers + docstrings first, cluster and reuse representative sub-modules | +35% APPS, +76% CodeContests |
| 2602.03557 | Scaling Test-Driven Code Gen: Functions to Classes | Analyze intra-class method dependencies → topological schedule → implement each method incrementally with test feedback | +12-26 absolute pts class-level correctness |
| 2308.01861 | ClassEval: Class-Level Code Generation Benchmark | Tests holistic vs incremental vs compositional generation. Method-by-method generation beats holistic for most models. | Empirical validation of decomposed generation |
| 2501.06625 | Guided Code Generation: Multi-Agent Framework | Generalist Agent decomposes into tree of sub-problems → Code Agent implements bottom-up using only interface docs (no child implementations) → Critic + Tester validate | +23.79% Pass@1 (56.2% vs 45.4%), works with 8B model |
| 2510.23010 | TALM: Tree-Structured Multi-Agent with Long-Term Memory | Recursive tree decomposition with localized re-reasoning (only re-do affected subtrees on error) + long-term semantic memory for cross-task reuse | +1.3-6.2% Pass@1 over baselines, reduced tokens |

## Tier 2: Multi-agent frameworks with architect/implement separation

| arxiv ID | Title | Architecture | Key Result |
|-----------|-------|-------------|------------|
| 2308.00352 | MetaGPT | ProductManager → Architect → ProjectManager → Engineer → QA | 51.43% success rate, ~$1/project |
| 2312.13010 | AgentCoder | Programmer + Test Designer + Test Executor (role separation) | 96.3% HumanEval, 91.8% MBPP |
| 2405.11403 | MapCoder | Retrieval → Planning → Coding → Debugging | 93.9% HumanEval, 83.1% MBPP |
| 2602.01465 | AGYN | Coordinator → Researcher → Implementer → Reviewer | 72.2% SWE-bench 500 |
| 2502.05664 | CODESIM | Planning Agent (plan + simulate) → Coding Agent → Debugging Agent | 95.1% HumanEval, 90.7% MBPP |
| 2501.07811 | CodeCoR | Prompt Agent → Coding Agent → Test Agent → Repair Agent (multi-output pruning at each stage) | 77.8% avg Pass@1 across 4 datasets |
| 2510.19868 | KGACG | COPA (planning from SRS/ADD) → CA (coding with API knowledge) → TA (testing against requirements) | Application-level code from requirements docs |
| 2510.22626 | SwiftSolve | Planner → Static Pruner → Coder → Profiler → Complexity Analyst (efficiency-aware) | 61.5% pass@1, 80.8% solved@≤3 on competitive programming |

## Tier 3: Supporting evidence and scaling insights

| arxiv ID | Title | Relevance |
|-----------|-------|-----------|
| 2504.04220 | AdaCoder | Critical evaluation: existing multi-agent frameworks have high inference costs and poor generalizability. Proposes adaptive escalation. +27.69% Pass@1, 12x fewer tokens |
| 2509.17489 | MapCoder-Lite | Proves planning→coding decomposition can be distilled into single 7B model with LoRA adapters per role |
| 2307.05300 | Solo Performance Prompting (SPP) | Single LLM simulates multiple personas. Cognitive synergy only in GPT-4+ class models |
| 2307.15337 | Skeleton-of-Thought | Core mechanism: skeleton first, expand each point in parallel. Up to 2.39x speedup |
| 2307.13883 | ExeDec | Two models: subgoal predictor + code generator. 2-4x improvement on compositional generalization |
| 2504.15080 | DLCodeGen | Planning-guided generation for projects >300 lines. Planning critical for coherence |
| 2312.17025 | Experiential Co-Learning | Instructor + Assistant agents learn from past trajectories. Autonomy 0.33 → 0.71 |
| 2601.21469 | DebateCoder | Three agents debate plans before coding. Works even at 1B scale with confidence gating |
| 2412.11014 | CoopetitiveV | Multi-agent Verilog generation with competition mechanism. 99.2% pass@10. Pattern generalizes beyond Python |
| 2505.21898 | Co-Saving | Resource-aware MAS using "shortcuts" mined from successful trajectories to bypass redundant reasoning steps. 50.85% token reduction, +10.06% code quality vs ChatDev |

## Key Risks and Mitigations from the Literature

1. **Interface contract fidelity**: The skeleton agent may specify signatures that implementation agents cannot satisfy. FUNCODER mitigates this with behavioral consensus; CodeChain mitigates with clustering representative sub-modules.

2. **Cross-function state**: Skeleton may not communicate shared state or invariants. 2602.03557 shows class-level success lags method-level success even when all methods individually pass tests (the "composition gap").

3. **Inference cost**: Multi-agent systems are expensive. AdaCoder (2504.04220) shows 12x token reduction via adaptive escalation. MapCoder-Lite (2509.17489) shows distillation into a single model is viable.

4. **Error propagation**: Upstream skeleton errors cascade to all implementation agents. MetaGPT uses structured intermediate artifacts; CODESIM uses plan simulation/verification before coding.

5. **Dependency ordering**: Implementation agents must respect method dependencies. 2602.03557 uses topological sorting of the dependency graph; skeleton agent should output generation order.

6. **Over-decomposition**: Too-deep decomposition hurts. TALM (2510.23010) shows accuracy drops from 89.13% at depth 2 to 81.32% at depth 5. Optimal tree depth appears to be 2-3 levels for function-level tasks.

7. **Error recovery cost**: Linear pipelines require full restarts on upstream failure. TALM's localized re-reasoning confines error recovery to affected subtrees only, reducing correction cost. Guided Code Gen (2501.06625) uses interface-only composition so parent nodes never see child implementation details, limiting blast radius.

8. **Redundant agent steps**: Multi-agent pipelines waste tokens on boilerplate reasoning. Co-Saving (2505.21898) mines "shortcuts" from successful trajectories, achieving 50.85% token reduction by skipping redundant intermediate agent steps.

## Recommended Reading Order

1. 2312.15960 (MoTCoder) — exactly the two-step pattern
2. 2405.20092 (FUNCODER) — recursive version with consensus
3. 2602.03557 (Scaling TDD to Classes) — dependency-aware scheduling
4. 2308.01861 (ClassEval) — empirical evidence decomposition works
5. 2310.08992 (CodeChain) — modular generation with reuse
6. 2504.04220 (AdaCoder) — what breaks in multi-agent frameworks
7. 2501.06625 (Guided Code Gen) — tree decomposition with interface-only composition
8. 2510.23010 (TALM) — localized re-reasoning and long-term memory
