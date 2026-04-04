# Agent Memory Reading Plan for Multi-Agent Coding Systems

> **Goal**: Understand the 2026 agent memory landscape deeply enough to make informed architecture decisions and implement multiple memory systems for heterogeneous coding agents (Claude Code, Codex, FlatAgents, etc.) operating concurrently on the same repository with interdependent subtasks.
>
> **Key constraints**: No shims — agents share real files. Communication via pub/sub channels. Scale range 2–256 agents. Must support decision logs, intent broadcasting, sequential role selection, and cross-session persistence.
>
> **Generated**: 2026-04-03 from abstract review of 291 papers across memory, coordination, communication, coding agents, and world models.

---

## How to Use This Plan

Papers are organized into **8 thematic tracks** reflecting distinct memory subsystems you'll likely implement. Within each track, papers are tiered:

- **🔴 P0 — Must Read**: Directly addresses your use case or establishes a foundational concept you'll build on. Read the full paper.
- **🟠 P1 — High Value**: Strong relevance but may be domain-specific or partially overlapping with a P0. Read abstract + method + results.
- **🟡 P2 — Survey/Benchmark**: Useful for landscape awareness, evaluation methodology, or when you need to compare alternatives. Skim or read selectively.
- **⚪ P3 — Reference**: Tangentially relevant. Consult when a specific sub-question arises.

**Estimated total**: ~95 papers across all tracks. At P0 level: ~28 papers.

---

## 🎯 Start Here: The Three Most Directly Relevant Papers

If you read nothing else, read these three. Each addresses a different axis of your exact problem: async multi-agent SWE, immutable decision history, and memory portability across heterogeneous agents.

### 1. `2603.21489` — Effective Strategies for Asynchronous Software Engineering Agents

> *AI agents have become increasingly capable at isolated software engineering (SWE) tasks such as resolving issues on Github. Yet long-horizon tasks involving multiple interdependent subtasks still pose challenges both with respect to accuracy, and with respect to timely completion. A natural approach to solving these long-horizon tasks in a timely manner is asynchronous multi-agent collaboration, where multiple agents work on different parts of the task at the same time. But effective application...*

**Why this is #1**: This is literally your problem statement — multiple agents, interdependent subtasks, asynchronous execution, same repo. Covers the coordination strategies, failure modes, and what works vs. what doesn't when agents work in parallel on code. Your architectural starting point.

**Track**: 6 (Coding Agent-Specific) — P0

### 2. `2602.23193` — ESAA: Event Sourcing for Autonomous Agents in LLM-Based Software Engineering

> *Autonomous agents based on Large Language Models (LLMs) have evolved from reactive assistants to systems capable of planning, executing actions via tools, and iterating over environment observations. However, they remain vulnerable to structural limitations: lack of native state, context degradation over long horizons, and the gap between probabilistic generation and deterministic execution requirements. This paper presents the ESAA (Event Sourcing for Autonomous Agents) architecture, which...*

**Why this is #2**: Solves your "lost reasoning" problem at the architectural level. Every agent action becomes an immutable event with full provenance. You get a decision log, an audit trail, and native state management for free. This is your persistence backbone — it maps directly to FlatMachines' SQLite persistence + signals model, but for the memory layer. Also bridges the gap between LLM probabilistic outputs and the deterministic execution your git-based workflow requires.

**Track**: 3 (Decision Logs) + 6 (Coding Agents) — P0 in both

### 3. `2603.23234` — MemCollab: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation

> *Large language model (LLM)-based agents rely on memory mechanisms to reuse knowledge from past problem-solving experiences. Existing approaches typically construct memory in a per-agent manner, tightly coupling stored knowledge to a single model's reasoning style. In modern deployments with heterogeneous agents, a natural question arises: can a single memory system be shared across different models? We found that naively transferring memory between agents often degrades performance, as such memo...*

**Why this is #3**: You're running Claude Code + Codex + FlatAgents (with arbitrary model backends) on the same repo. They each reason differently. This paper proves naive memory sharing between heterogeneous agents *degrades* performance, then solves it via contrastive trajectory distillation — making memory portable across different reasoning styles without coupling to any single model. This is the bridge that makes a unified shared memory possible across your agent fleet.

**Track**: 1 (Shared & Cross-Agent Memory) — P0

---

## Track 1: Shared & Cross-Agent Memory

> *The core problem: multiple agents need to read/write a common memory substrate without memory homogenization or information overload.*

### 🔴 P0

| ID | Title | Why |
|----|-------|-----|
| `2602.03036` | **LatentMem: Customizing Latent Memory for Multi-Agent Systems** | Directly addresses multi-agent memory. Identifies two bottlenecks: (i) memory homogenization from lack of role-aware customization, (ii) information overload from fine-grained entries. Proposes latent compression with role-specific views. This is your central design tension. |
| `2603.23234` | **MemCollab: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation** | Exactly your problem: can heterogeneous agents share a single memory system? Finds naive transfer degrades performance. Proposes contrastive trajectory distillation to make memory portable across different models. Critical for Claude Code + Codex + FlatAgents interop. |
| `2602.05965` | **Learning to Share: Selective Memory for Efficient Parallel Agentic Systems** | Multiple agent teams running in parallel on similar sub-problems. Proposes selective memory sharing to avoid redundant computation. Directly applicable to parallel coding agents on interdependent subtasks. |
| `2602.13370` | **G2CP: A Graph-Grounded Communication Protocol for Verifiable and Efficient Multi-Agent Reasoning** | Replaces free-text agent communication with graph operations over a shared knowledge graph. Agents exchange traversal commands, subgraph fragments, and update operations. Strong candidate for your decision graph / intent broadcasting system. |
| `2601.08343` | **When KV Cache Reuse Fails in Multi-Agent Systems** | Reveals that KV cache sharing strategies effective for execution agents fail for judge-centric inference. Important practical warning for multi-agent architectures sharing compute resources. |
| `2602.01053` | **LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents** | Multi-agent systems sharing a backbone with lightweight adapters. Cache differences are dominated by LoRA deltas. Directly relevant if running multiple specialized agents on shared infrastructure. |

### 🟠 P1

| ID | Title | Why |
|----|-------|-----|
| `2602.00471` | **Dual Latent Memory for Visual Multi-Agent System** | Identifies "scaling wall" where more agent turns degrade performance due to text-centric communication bottleneck. Proposes latent-space alternatives. Applicable to your concern about coordination overhead. |
| `2601.20352` | **AMA: Adaptive Memory via Multi-Agent Collaboration** | Multi-agent collaboration for memory maintenance — addresses rigid retrieval granularity and coarse-grained update mechanisms. |
| `2601.22974` | **MiTa: Hierarchical Multi-Agent Collaboration with Memory-Integrated Task Allocation** | Manager-member hierarchy with integrated memory and task allocation. Directly relevant to your orchestrator design question. Addresses memory inconsistency and agent behavioral conflicts. |
| `2603.04428` | **Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM on Edge** | Practical engineering: persisting quantized KV caches to disk for multi-agent inference. Only 3 agents fit in FP16 at 8K context; Q4 compression enables 10+ agents. Real system constraints you'll hit. |
| `2601.21473` | **ScaleSim: Large-Scale Multi-Agent Simulation with Invocation-Distance Memory Management** | Scaling to many agents with sparse activation patterns. Uses invocation distance to predict which agent caches to keep resident. Relevant at your 256-agent scale. |
| `2602.00454` | **Cross-Modal Memory Compression for Efficient Multi-Agent Debate** | Compresses debate traces into compact image representations to reduce token explosion. Creative approach to the growing-context problem in multi-agent systems. |

### 🟡 P2

| ID | Title | Why |
|----|-------|-----|
| `2601.07978` | **Cost and Accuracy of Long-Term Memory in Distributed Multi-Agent Systems** | Empirical comparison of mem0 (vector) vs Graphiti (graph) under network constraints. Useful for your vector vs. graph memory decision. |
| `2602.15382` | **The Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems** | Latent state transfer without assuming homogeneous architectures. Relevant if you pursue sub-symbolic memory sharing between different agent types. |
| `2601.20465` | **BMAM: Brain-Inspired Multi-Agent Memory Framework** | Decomposes memory into functionally specialized subsystems. Addresses "soul erosion" — behavioral inconsistency across sessions. |
| `2602.00428` | **When Agents "Misremember" Collectively: Mandela Effect in LLM-based MAS** | Collective cognitive bias in multi-agent memory. Important failure mode to understand. |

---

## Track 2: Memory Architecture Fundamentals

> *The vocabulary and structural patterns you need before implementing anything.*

### 🔴 P0

| ID | Title | Why |
|----|-------|-----|
| `2603.07670` | **Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers** | The definitive 2026 survey. Structured account of how memory is designed, implemented, and evaluated. Read this first as your map of the landscape. |
| `2602.05665` | **Graph-based Agent Memory: Taxonomy, Techniques, and Applications** | Survey specifically on graph-structured memory. Covers relational dependencies, hierarchical organization, efficient retrieval. Essential if you go graph-based (you probably should for decision graphs). |
| `2603.21564` | **Toward a Theory of Hierarchical Memory for Language Agents** | Formalizes hierarchical memory via three operators: extraction (α), coarsening, and retrieval under token budget. Provides the mathematical framework for comparing design choices. |
| `2601.09913` | **Continuum Memory Architectures for Long-Horizon LLM Agents** | Defines CMA: persistent storage, selective retention, associative routing, temporal continuity. The class of systems you're building. |
| `2601.01885` | **Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management** | Unified LTM/STM framework integrated directly into agent behavior, not separate components. End-to-end optimization of memory management. |
| `2603.17244` | **Graph-Native Cognitive Memory: Formal Belief Revision Semantics for Versioned Memory** | Formal belief revision (AGM-style) applied to versioned memory graphs. The structural primitives (immutable revisions, mutable tag pointers, typed dependency edges, URI-based addressing) are identical to those for managing versionable code assets. *Directly maps to git-like version control for agent knowledge.* |

### 🟠 P1

| ID | Title | Why |
|----|-------|-----|
| `2601.03236` | **MAGMA: Multi-Graph Agentic Memory Architecture** | Multi-graph separation of temporal, causal, and entity information. Avoids monolithic store. Good architectural pattern for your decision log + intent + artifact separation. |
| `2601.02845` | **TiMem: Temporal-Hierarchical Memory Consolidation** | Temporal Memory Tree organizing conversations through hierarchical levels. Useful pattern for session-spanning coding projects. |
| `2601.15311` | **Aeon: High-Performance Neuro-Symbolic Memory Management** | Addresses "Vector Haze" (retrieval of disconnected fragments from flat RAG). Neuro-symbolic approach capturing hierarchical and temporal structure. |
| `2602.08369` | **MemAdapter: Fast Alignment Across Agent Memory Paradigms** | First step toward unifying heterogeneous memory paradigms (explicit, parametric, latent) in a single system via generative subgraph retrieval. Critical for supporting multiple memory backends. |
| `2603.01761` | **Modular Memory is the Key to Continual Learning Agents** | Strong position paper arguing in-context learning (ICL) memory beats in-weight learning (IWL) for continual operation. Directly relevant to your "fresh vs. cached context" decision. |

### 🟡 P2

| ID | Title | Why |
|----|-------|-----|
| `2601.14192` | **Toward Efficient Agents: Memory, Tool Learning, and Planning** | Efficiency-focused survey covering latency, tokens, steps. Read when optimizing costs. |
| `2602.11243` | **Evaluating Memory Structure in LLM Agents** | Analysis of what memory capabilities actually matter beyond simple fact retention. |
| `2601.02744` | **SYNAPSE: Episodic-Semantic Memory via Spreading Activation** | Cognitive science-inspired dynamic graph where relevance emerges from spreading activation. Interesting retrieval paradigm. |
| `2603.15280` | **Advancing Multimodal Agent Reasoning with Long-Term Neuro-Symbolic Memory** | Distinguishes neural (inductive) vs. symbolic (deductive) memory for different reasoning types. |

---

## Track 3: Decision Logs, Reasoning Traces & Intent Broadcasting

> *Your #1 failure mode: "lost reasoning." These papers address why decisions were made, how to share intent, and how to attribute outcomes to specific agent decisions.*

### 🔴 P0

| ID | Title | Why |
|----|-------|-----|
| `2601.10702` | **Grounding Agent Memory in Contextual Intent (STITCH)** | Indexes each trajectory step with structured retrieval cue and contextual intent. Retrieves by intent matching, not just semantic similarity. Exactly your "decision log that explains why" requirement. |
| `2603.06859` | **Contextual Counterfactual Credit Assignment for Multi-Agent RL in LLM Collaboration** | Isolates causal impact of each agent's decisions via counterfactual reasoning. Critical for understanding which agent's decision caused a downstream problem. |
| `2603.08754` | **Hindsight Credit Assignment for Long-Horizon LLM Agents (HCAPO)** | First framework integrating hindsight credit assignment into LLM agents. Addresses sparse rewards over long multi-step coding tasks. |
| `2602.23193` | **ESAA: Event Sourcing for Autonomous Agents in LLM-Based Software Engineering** | *Event sourcing* architecture for coding agents. Every state change is an immutable event. Provides native state, prevents context degradation over long horizons, and bridges probabilistic generation with deterministic execution. This is potentially your persistence backbone. |
| `2603.21692` | **Reasoning Provenance for Autonomous AI Agents** | Structured behavioral analytics beyond state checkpoints and execution traces. Addresses the provenance question: given an agent output, what reasoning chain produced it? |

### 🟠 P1

| ID | Title | Why |
|----|-------|-----|
| `2602.16165` | **HiPER: Hierarchical RL with Explicit Credit Assignment for LLM Agents** | Multi-scale credit assignment for long-horizon sparse-reward settings. Relevant to coding tasks where reward is "tests pass" at the end. |
| `2602.19225` | **Proximity-Based Multi-Turn Optimization: Credit Assignment for LLM Agent Training** | Distinguishes high-value signals from noise in multi-turn agents. Production-oriented. |
| `2601.12618` | **Disagreement as Data: Reasoning Trace Analytics in Multi-Agent Systems** | Treats reasoning traces as first-class data for understanding multi-agent behavior. |
| `2603.01481` | **Harmonizing Dense and Sparse Signals: Dual-Horizon Credit Assignment** | Balancing long-term objectives with turn-level constraints. Applicable to code quality vs. task completion tension. |
| `2602.12083` | **Differentiable Modal Logic for Multi-Agent Diagnosis, Orchestration and Communication** | Modal logic (knowledge, belief, obligation) formalized as differentiable neural networks for diagnosing semantic failures in multi-agent systems. Heavy but powerful. |

### ⚪ P3

| ID | Title | Why |
|----|-------|-----|
| `2601.21523` | **Explicit Credit Assignment through Local Rewards and Dependence Graphs in MARL** | Dependence graphs for separating agent contributions. Foundational MARL theory. |
| `2602.10863` | **ICA: Information-Aware Credit Assignment for Long-Horizon Information-Seeking Agents** | Visual-native search framework with fine-grained credit. Consult for credit assignment implementation details. |

---

## Track 4: Communication Protocols & Topology

> *How agents discover each other, what they say, when they say it, and how the communication graph evolves.*

### 🔴 P0

| ID | Title | Why |
|----|-------|-----|
| `2602.15055` | **Beyond Context Sharing: Unified Agent Communication Protocol (ACP)** | The most comprehensive treatment of cross-platform, decentralized agent-to-agent orchestration. Builds on MCP foundations. Directly addresses your protocol design question. |
| `2602.16424` | **Verifiable Semantics for Agent-to-Agent Communication** | Certification protocol ensuring agents share the same understanding of terms. Prevents semantic drift — critical when heterogeneous agents communicate about code structures. |
| `2602.11583` | **The Five Ws of Multi-Agent Communication: Who, Whom, When, What, Why** | Comprehensive survey organizing MA communication through the Five Ws. Your roadmap for communication design decisions. |
| `2603.19677` | **GoAgent: Group-of-Agents Communication Topology Generation** | Generates task-specific group structures for divide-and-conquer. Node-centric topology generation with explicit group formation. Directly relevant to organizing coding sub-teams. |
| `2602.03688` | **TodyComm: Task-Oriented Dynamic Communication for Multi-Round LLM-based MAS** | Dynamic communication topology that changes across rounds based on role changes and task progression. Your agents' communication needs will shift as coding progresses. |

### 🟠 P1

| ID | Title | Why |
|----|-------|-----|
| `2602.06039` | **DyTopo: Dynamic Topology Routing via Semantic Matching** | Manager-guided sparse directed communication graph reconstructed each round. Agents declare "needs" and "capabilities" for matching. Maps well to pub/sub discovery. |
| `2602.14681` | **ST-EVO: Generative Spatio-Temporal Evolution of Multi-Agent Communication Topologies** | Beyond spatial-only or temporal-only evolution — jointly evolves who talks to whom and when. |
| `2602.16873` | **AdaptOrch: Task-Adaptive Multi-Agent Orchestration** | Argues orchestration topology dominates individual model capability. Formal framework for topology adaptation. |
| `2601.10120` | **TopoDIM: One-Shot Topology Generation of Diverse Interaction Modes** | Evaluation + debate + generation modes in one topology. One-shot generation avoids sequential latency. |
| `2603.04833` | **SCoUT: Scalable Communication via Utility-Guided Temporal Grouping in MARL** | Addresses when and who to communicate with via temporal and agent abstraction. Key for scaling to many agents without communication explosion. |
| `2603.22823` | **Empirical Comparison of Agent Communication Protocols for Task Orchestration** | Head-to-head comparison of tool-integration protocol vs. inter-agent delegation protocol. Empirical data for your protocol choice. |
| `2602.11754` | **Cooperation Breakdown in LLM Agents Under Communication Delays** | FLCOA framework for how cooperation emerges under real-world computational and communication constraints. Important for distributed setups. |

### 🟡 P2

| ID | Title | Why |
|----|-------|-----|
| `2601.12886` | **Communication Methods in Multi-Agent Reinforcement Learning** | Survey of MARL communication methods. Background reading. |
| `2601.12518` | **Cooperative Multi-agent RL with Communication Constraints** | What happens when communication is limited and agents use outdated information. |
| `2603.07880` | **What Do AI Agents Talk About? Emergent Communication in AI-Only Social Network** | 47K agents, 361K posts, 2.8M comments. Characterizes emergent AI-to-AI discourse. Fascinating but tangential. |
| `2601.22041` | **Learning to Communicate Across Modalities in Multi-Agent Systems** | Perceptual heterogeneity in communication. Relevant if your agents have fundamentally different observation spaces. |
| `2603.16264` | **Adaptive Theory of Mind for LLM-based Multi-Agent Coordination** | Misaligned ToM depths impair coordination. Agents need calibrated reasoning about what others know. |
| `2603.25268` | **CRAFT: Grounded Multi-Agent Coordination Under Partial Information** | Agents with complementary but incomplete views must coordinate via natural language. Diagnostic framework for failure modes. |

---

## Track 5: Memory Lifecycle — Compression, Forgetting, Admission Control

> *You can't keep everything. These papers address what to store, what to compress, what to forget, and when.*

### 🔴 P0

| ID | Title | Why |
|----|-------|-----|
| `2603.13017` | **Structured Distillation for Personalized Agent Memory: 11× Token Reduction** | Compresses each exchange into 38-token compound objects with four fields including `files_touched`. Directly applicable to coding memory where you need to track what files were modified and why. |
| `2601.18642` | **FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory** | Adaptive decay replacing binary retention. Critical because coding projects accumulate stale context (old error messages, abandoned approaches) that must fade. |
| `2603.04549` | **Adaptive Memory Admission Control for LLM Agents** | Controlling what enters memory in the first place. Prevents accumulation of hallucinated or obsolete facts. Auditable admission policies. |
| `2601.07190` | **Active Context Compression: Autonomous Memory Management in LLM Agents (Focus)** | Agent-centric compression inspired by Physarum polycephalum. Specifically targets "Context Bloat" in long-horizon software engineering tasks. |

### 🟠 P1

| ID | Title | Why |
|----|-------|-----|
| `2603.01455` | **From Verbatim to Gist: Pyramidal Multimodal Memory via Semantic Information Bottleneck** | Verbatim → gist compression pipeline from fuzzy-trace theory. Applicable to compressing verbose tool outputs. |
| `2601.02553` | **SimpleMem: Efficient Lifelong Memory via Semantic Lossless Compression** | Three-stage pipeline for semantic lossless compression. Practical and efficient. |
| `2601.14287` | **Chain-of-Memory: Lightweight Memory Construction with Dynamic Evolution** | Finds complex construction (graphs) has marginal gains over lightweight approaches. Important counterpoint to graph-heavy designs. |
| `2603.15658` | **Did You Check the Right Pocket? Cost-Sensitive Store Routing** | Formulates memory retrieval as a store-routing problem. Oracle router achieves higher accuracy with fewer tokens than uniform retrieval. |

### 🟡 P2

| ID | Title | Why |
|----|-------|-----|
| `2601.04786` | **AgentOCR: Agent History via Optical Self-Compression** | Compresses agent history into visual format. Novel approach. |
| `2603.02473` | **Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory** | 3×3 study crossing write strategies with retrieval methods. Diagnostic framework for when your memory system underperforms. |

---

## Track 6: Coding Agent-Specific

> *Papers directly about coding agents, their architectures, and their specific memory/coordination needs.*

### 🔴 P0

| ID | Title | Why |
|----|-------|-----|
| `2603.05344` | **Building Effective AI Coding Agents for the Terminal (OPENDEV)** | Scaffolding, harness, context engineering for terminal-native coding agents. Lessons learned from production. Your closest architectural reference. |
| `2602.23193` | **ESAA: Event Sourcing for Autonomous Agents in LLM-Based Software Engineering** | Event sourcing for coding agents. Immutable event log, native state management, bridges probabilistic/deterministic gap. *Already listed in Track 3 but equally critical here.* |
| `2603.21489` | **Effective Strategies for Asynchronous Software Engineering Agents** | Directly addresses asynchronous multi-agent collaboration on interdependent subtasks in software engineering. This is your exact use case. |
| `2601.13295` | **CooperBench: Why Coding Agents Cannot Be Your Teammates Yet** | 600+ collaborative coding tasks. Tests coordination capabilities. Identifies what's missing for agents to be effective teammates. Your evaluation baseline. |
| `2602.01465` | **Agyn: Multi-Agent System for Team-Based Autonomous Software Engineering** | Models software engineering as collaborative team activity with role separation, communication, and review. Closest to your multi-agent coding setup. |

### 🟠 P1

| ID | Title | Why |
|----|-------|-----|
| `2602.05892` | **ContextBench: Benchmark for Context Retrieval in Coding Agents** | Process-oriented evaluation of how agents retrieve and use code context. 1,136 tasks across 66 repos, 8 languages. |
| `2601.10343` | **OctoBench: Scaffold-Aware Instruction Following in Repository-Grounded Agentic Coding** | Tests scaffold compliance across heterogeneous constraints. Relevant to your multi-agent harness design. |
| `2603.20432` | **Coding Agents are Effective Long-Context Processors** | Coding agents can externalize long-context processing into explicit file operations. Important insight for your architecture: agents can manage context via the filesystem itself. |
| `2603.03456` | **Asymmetric Goal Drift in Coding Agents Under Value Conflict** | How coding agents drift from explicit instructions over long contexts. A failure mode you must guard against. |
| `2603.26233` | **Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents** | When agents should ask for clarification vs. proceed. Critical for your multi-agent setup where one agent could ask another. |
| `2601.20789` | **SERA: Soft-Verified Efficient Repository Agents** | Training agents specialized to private codebases. Relevant if you specialize FlatAgents per-repo. |
| `2602.15763` | **GLM-5: from Vibe Coding to Agentic Engineering** | Production-scale coding agent with async RL infrastructure. Architectural reference. |
| `2602.19594` | **ISO-Bench: Can Coding Agents Optimize Real-World Inference Workloads?** | Tests agents on real codebase optimization (vLLM, SGLang). Closest to "agents working on a real repo" evaluation. |

### 🟡 P2

| ID | Title | Why |
|----|-------|-----|
| `2602.02262` | **OmniCode: Benchmark for Software Engineering Agents** | Broad SWE benchmark beyond narrow bug-fixing. |
| `2601.16443` | **Endless Terminals: Scaling RL Environments for Terminal Agents** | Procedural task generation pipeline. Useful for training. |
| `2603.03194` | **BeyondSWE: Can Code Agents Survive Beyond Single-Repo Bug Fixing?** | Cross-repo reasoning, dependency migration, full-repo generation. |
| `2602.01655` | **ProjDevBench: End-to-End Project Development** | End-to-end evaluation including architecture design and iterative refinement. |
| `2602.00592` | **DockSmith: Scaling Reliable Coding Environments via Agentic Docker Builder** | Environment construction as agentic capability. |

---

## Track 7: Self-Evolving Memory & Continual Learning

> *Agents that improve their memory systems over time — learning what to remember, how to organize it, and when to update.*

### 🔴 P0

| ID | Title | Why |
|----|-------|-----|
| `2602.02474` | **MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents** | Reframes memory operations as learnable, evolvable skills rather than fixed routines. Your memory system should get better at remembering over project lifetime. |
| `2602.02369` | **Live-Evo: Online Evolution of Agentic Memory from Continuous Feedback** | True online memory evolution under distribution shift, not static train/test. Your coding projects will shift focus over time. |
| `2603.10600` | **Trajectory-Informed Memory Generation for Self-Improving Agent Systems** | Extracts actionable learnings from execution trajectories. Agents learn from both successes and failures. Directly applicable to coding retrospectives. |
| `2601.08323` | **AtomMem: Learnable Dynamic Agentic Memory with Atomic Operations** | Reframes memory management as dynamic decision-making with atomic read/write primitives. Foundational primitive design. |

### 🟠 P1

| ID | Title | Why |
|----|-------|-----|
| `2602.07755` | **Learning to Continually Learn via Meta-learning Agentic Memory Designs** | Meta-learning the memory design itself. Addresses non-stationarity of real-world tasks. |
| `2601.07470` | **Learning How to Remember: Meta-Cognitive Management for Structured and Transferable Memory** | Memory abstraction as learnable cognitive skill. Handles distribution shift. |
| `2601.03192` | **MemRL: Self-Evolving Agents via Runtime RL on Episodic Memory** | Constructive episodic simulation — retrieving past experiences to synthesize solutions for novel tasks. |
| `2602.01966` | **Self-Consolidation for Self-Evolving Agents** | Learns from both successes AND failures (most systems only learn from success). |
| `2602.01869` | **ProcMEM: Reusable Procedural Memory from Experience** | Transforms episodic experience into reusable procedural skills without parameter updates. Directly applicable to coding agents learning repeated workflows. |

---

## Track 8: World Models, Context Engineering & Structured State

> *How agents maintain a model of the codebase, the project state, and each other's intentions.*

### 🔴 P0

| ID | Title | Why |
|----|-------|-----|
| `2603.09619` | **Context Engineering: From Prompts to Corporate Multi-Agent Architecture** | Establishes context engineering as a standalone discipline for multi-agent systems. Covers vendor architectures (Google ADK, Anthropic, LangChain). Your theoretical foundation. |
| `2602.05447` | **Structured Context Engineering for File-Native Agentic Systems** | 9,649 experiments across 11 models, 4 formats (YAML, MD, JSON, TOON), schemas 10–10K tables. Empirical guidance on how to structure context for programmatic agents. Directly applicable to codebase representation. |
| `2603.22083` | **Context Engineering Framework for Enterprise AI Agents via Digital-Twin MDP** | Lightweight, model-agnostic framework using offline RL. "Digital twin" of the environment for training. Applicable to simulating your repo state. |
| `2602.10090` | **Agent World Model: Infinity Synthetic Environments for Agentic RL** | Fully synthetic environment generation pipeline scaling to 1,000 environments. Agents interact with rich toolsets. Training infrastructure reference. |

### 🟠 P1

| ID | Title | Why |
|----|-------|-----|
| `2601.21557` | **Meta Context Engineering via Agentic Skill Evolution** | Agents learn to optimize their own context rather than using manually crafted harnesses. |
| `2601.06606` | **CEDAR: Context Engineering for Agentic Data Science** | Practical context engineering patterns. Structuring initial prompts with domain-specific input fields. |
| `2602.05842` | **Reinforcement World Model Learning for LLM-based Agents** | Self-supervised world model learning using sim-to-real gap rewards. |
| `2603.00808` | **MetaMind: General and Cognitive World Models via Meta-Theory of Mind** | Each agent learns to predict and plan over other agents' behavior through meta-ToM. Relevant to your sequential role selection requirement. |
| `2602.00785` | **World Models as an Intermediary between Agents and the Real World** | Position paper on using world models to bridge high-cost real environments. Applicable to "simulate the repo state before committing." |
| `2602.23997` | **Foundation World Models for Agents that Learn, Verify, and Adapt** | Vision for persistent, compositional world representations. Long-horizon architectural thinking. |
| `2603.20380` | **ALARA for Agents: Least-Privilege Context Engineering** | Least-privilege principle applied to agent context. Agents get only the context they need. Reduces noise and improves security. Applicable to your per-agent memory scoping. |

---

## Track Cross-Cutting: Benchmarks & Evaluation

> *When you need to evaluate your implementations.*

### 🟡 P2

| ID | Title | Why |
|----|-------|-----|
| `2602.16313` | **MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Tasks** | Closest to your use case: interdependent tasks requiring memory across sessions. |
| `2601.19935` | **Mem2ActBench: Evaluating Long-Term Memory Utilization in Task-Oriented Agents** | Tests proactive memory application, not just passive recall. |
| `2602.22769` | **AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications** | Agent-environment interaction memory (not dialogue-centric). |
| `2601.16690` | **EMemBench: Interactive Benchmarking of Episodic Memory for VLM Agents** | Template-based evaluation with verifiable ground truth from trajectories. |
| `2603.23840` | **VehicleMemBench: Multi-User Long-Term Memory** | Multi-user preference conflicts and temporal evolution. Applicable to multi-developer scenarios. |
| `2601.03515` | **Mem-Gallery: Benchmarking Multimodal Long-Term Conversational Memory** | How memory is preserved, organized, and evolved across trajectories. |
| `2602.03224` | **TAME: Trustworthy Test-Time Evolution of Agent Memory** | Safety alignment during memory evolution. Measures trustworthiness degradation. |
| `2602.23944` | **MemEmo: Evaluating Emotion in Memory Systems** | Edge case, but emotion/sentiment in commit messages and code review feedback is real signal. |

---

## Suggested Reading Order

### Phase 1: Landscape & Architecture (Week 1-2)
1. `2603.07670` — Survey: Memory for Autonomous LLM Agents *(map of everything)*
2. `2602.05665` — Survey: Graph-based Agent Memory *(if you go graph-based)*
3. `2602.11583` — Survey: Five Ws of Multi-Agent Communication *(communication landscape)*
4. `2603.21564` — Theory of Hierarchical Memory *(formal framework)*
5. `2601.09913` — Continuum Memory Architectures *(definitional)*
6. `2603.09619` — Context Engineering *(discipline overview)*

### Phase 2: Your Core Design Decisions (Week 2-4)
7. `2602.03036` — LatentMem *(shared memory design)*
8. `2603.23234` — MemCollab *(cross-agent portability)*
9. `2602.05965` — Learning to Share *(selective parallel sharing)*
10. `2602.13370` — G2CP *(graph-grounded communication)*
11. `2601.10702` — STITCH *(intent-grounded memory)*
12. `2603.17244` — Kumiho *(versioned memory with belief revision)*
13. `2602.23193` — ESAA *(event sourcing for coding agents)*

### Phase 3: Coding Agent Specifics (Week 3-5)
14. `2603.05344` — OPENDEV *(terminal coding agent architecture)*
15. `2603.21489` — Async SWE Agents *(your exact problem)*
16. `2602.01465` — Agyn *(team-based SWE)*
17. `2601.13295` — CooperBench *(evaluation baseline)*
18. `2603.13017` — Structured Distillation *(11× compression with files_touched)*

### Phase 4: Communication & Topology (Week 4-6)
19. `2602.15055` — ACP *(communication protocol)*
20. `2602.16424` — Verifiable Semantics *(prevent drift)*
21. `2603.19677` — GoAgent *(topology generation)*
22. `2602.03688` — TodyComm *(dynamic topology)*
23. `2602.16873` — AdaptOrch *(orchestration > model capability)*

### Phase 5: Lifecycle & Evolution (Week 5-7)
24. `2601.18642` — FadeMem *(forgetting)*
25. `2603.04549` — Admission Control *(what enters memory)*
26. `2602.02474` — MemSkill *(learnable memory operations)*
27. `2603.10600` — Trajectory-Informed Memory *(learning from execution)*
28. `2601.07190` — Focus *(context compression for SWE)*

### Phase 6: Deep Dives as Needed
Remaining P1 and P2 papers — pull from specific tracks as implementation questions arise.

---

## Implementation Roadmap (Derived from Papers)

Based on the landscape, here are the **distinct memory systems** to prototype:

1. **Decision Log** (Track 3): Event-sourced immutable log (`2602.23193` ESAA) with intent indexing (`2601.10702` STITCH). Every agent action is an event with reasoning provenance.

2. **Shared Knowledge Graph** (Track 1 + 2): Graph-native versioned memory (`2603.17244` Kumiho) with graph-grounded communication protocol (`2602.13370` G2CP). Agents exchange graph operations, not free text.

3. **Working Memory / Context Cache** (Track 5 + 8): Compressed per-agent KV cache (`2603.04428`) with adaptive admission control (`2603.04549`) and biological forgetting (`2601.18642` FadeMem).

4. **Cross-Agent Memory Bridge** (Track 1): Contrastive trajectory distillation (`2603.23234` MemCollab) enabling heterogeneous agents to share memory without performance degradation.

5. **Project Memory / Institutional Knowledge** (Track 7): Self-evolving procedural memory (`2602.01869` ProcMEM) extracting reusable skills from execution history, with meta-cognitive abstraction (`2601.07470`).

6. **Communication Topology Manager** (Track 4): Dynamic topology generation (`2603.19677` GoAgent) with utility-guided temporal grouping (`2603.04833` SCoUT) for scaling to many agents.

---

## Appendix: All Paper IDs by Track

<details>
<summary>Track 1: Shared & Cross-Agent Memory (16 papers)</summary>

```
2602.03036, 2603.23234, 2602.05965, 2602.13370, 2601.08343, 2602.01053,
2602.00471, 2601.20352, 2601.22974, 2603.04428, 2601.21473, 2602.00454,
2601.07978, 2602.15382, 2601.20465, 2602.00428
```
</details>

<details>
<summary>Track 2: Architecture Fundamentals (15 papers)</summary>

```
2603.07670, 2602.05665, 2603.21564, 2601.09913, 2601.01885, 2603.17244,
2601.03236, 2601.02845, 2601.15311, 2602.08369, 2603.01761, 2601.14192,
2602.11243, 2601.02744, 2603.15280
```
</details>

<details>
<summary>Track 3: Decision Logs & Reasoning Traces (12 papers)</summary>

```
2601.10702, 2603.06859, 2603.08754, 2602.23193, 2603.21692, 2602.16165,
2602.19225, 2601.12618, 2603.01481, 2602.12083, 2601.21523, 2602.10863
```
</details>

<details>
<summary>Track 4: Communication & Topology (18 papers)</summary>

```
2602.15055, 2602.16424, 2602.11583, 2603.19677, 2602.03688, 2602.06039,
2602.14681, 2602.16873, 2601.10120, 2603.04833, 2603.22823, 2602.11754,
2601.12886, 2601.12518, 2603.07880, 2601.22041, 2603.16264, 2603.25268
```
</details>

<details>
<summary>Track 5: Memory Lifecycle (10 papers)</summary>

```
2603.13017, 2601.18642, 2603.04549, 2601.07190, 2603.01455, 2601.02553,
2601.14287, 2603.15658, 2601.04786, 2603.02473
```
</details>

<details>
<summary>Track 6: Coding Agent-Specific (17 papers)</summary>

```
2603.05344, 2602.23193, 2603.21489, 2601.13295, 2602.01465, 2602.05892,
2601.10343, 2603.20432, 2603.03456, 2603.26233, 2601.20789, 2602.15763,
2602.19594, 2602.02262, 2601.16443, 2603.03194, 2602.01655, 2602.00592
```
</details>

<details>
<summary>Track 7: Self-Evolving Memory (9 papers)</summary>

```
2602.02474, 2602.02369, 2603.10600, 2601.08323, 2602.07755, 2601.07470,
2601.03192, 2602.01966, 2602.01869
```
</details>

<details>
<summary>Track 8: World Models & Context Engineering (11 papers)</summary>

```
2603.09619, 2602.05447, 2603.22083, 2602.10090, 2601.21557, 2601.06606,
2602.05842, 2603.00808, 2602.00785, 2602.23997, 2603.20380
```
</details>

<details>
<summary>Benchmarks & Evaluation (8 papers)</summary>

```
2602.16313, 2601.19935, 2602.22769, 2601.16690, 2603.23840, 2601.03515,
2602.03224, 2602.23944
```
</details>

---

*Total unique papers: ~95 (some appear in multiple tracks). P0 papers: ~28.*
