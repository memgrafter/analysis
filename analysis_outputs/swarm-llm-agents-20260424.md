Here’s a focused corpus read on **LLM swarms**, centered on your interests: basics, coding/shell agents, test-time compute vs capability-building, communication/persistence/tool use, and the Minsky angle.

## Executive take

In this corpus, **LLM swarms** are not one thing. They split into a few distinct families:

1. **Inference swarms**  
   Multiple agents/models collaborate at inference time to improve an answer.  
   Best examples:
   - `2510.24801` Fortytwo
   - `2510.10047` SwarmSys
   - `2603.12933` AMRO-S
   - `2512.13713` LoopBench

2. **Capability-construction swarms**  
   The swarm is used to **build or adapt** a better system/model/workflow, not just answer one query.  
   Best examples:
   - `2410.11163` Model Swarms
   - `2506.15672` SwarmAgentic
   - `2412.12839` HIVE
   - `2504.17967` PharmaSwarm

3. **Embodied / operational swarms**  
   LLMs coordinate robots, simulations, or labs.  
   Best examples:
   - `2509.16920` SwarmChat
   - `2401.17749` SwarmBrain
   - `2603.21344` AI Scientific Community

The biggest pattern is:

- **2024:** bridge papers, prototypes, graph/planning abstractions
- **2025:** real expansion
- **2026:** more routing / coordination / scaling refinements

And the most important conceptual split for your question is:

- **Swarm as test-time compute** = “spend more inference budget through parallel/iterated coordination”
- **Swarm as capability/outcome architecture** = “use multiple agents to create a better system, better design, or better workflow”

---

# 1) Who / What / When / Where / Why / How

## Who
The “who” in LLM swarms is usually one of these:

- **specialist role agents**
  - Explorer / Worker / Validator in `2510.10047`
  - Genomic / Literature / Market agents in `2504.17967`
  - Planning / Worker / Evaluation agents in `2603.21344`

- **peer nodes with voting/reputation**
  - Fortytwo’s 35-node decentralized inference swarm in `2510.24801`

- **experts treated as particles**
  - whole LLM experts as PSO particles in `2410.11163`

- **tool/model selectors and routers**
  - HIVE orchestration in `2412.12839`
  - ACO router in `2603.12933`

So “agent” can mean:
- an LLM with a role prompt
- a model expert
- a routing node
- a planning module
- or even a residue-position agent in protein design (`2511.22311`)

## What
An LLM swarm is a **multi-agent coordination system** where multiple language-model-driven units interact to produce better search, reasoning, routing, or design.

In this corpus, the swarm idea is usually one of:
- **parallel search**
- **debate / validation**
- **consensus**
- **routing**
- **shared-memory collaboration**
- **population-based optimization**

## When
In the corpus, the real LLM-swarm wave begins in **late 2024** and expands in **2025**.

Key chronology:
- `2410.11163` **Model Swarms**: strong early bridge from swarm intelligence to LLM expert adaptation
- `2412.12839` **HIVE**: planning/orchestration over multi-model systems
- `2025`: the field broadens into
  - agent generation
  - reasoning swarms
  - consensus swarms
  - drug discovery
  - robotic interaction
- `2026`: routing and virtual-lab concepts mature further

## Where
Where are swarms being used?

### Strongest “where” buckets
- **reasoning / inference**
  - `2510.10047`, `2510.24801`, `2512.13713`
- **planning / orchestration**
  - `2412.12839`, `2603.12933`
- **scientific discovery / design**
  - `2504.17967`, `2511.22311`, `2603.21344`
- **robotics / embodied systems**
  - `2509.16920`, `2401.17749`

## Why
Why use swarms instead of one big model?

The corpus gives a few recurring reasons:

1. **diversity of hypotheses**
   - multiple trajectories catch errors or cover more search space

2. **specialization**
   - different agents handle different subtasks or modalities

3. **validation / error correction**
   - critics, validators, peer ranking, or evaluators reduce single-path mistakes

4. **scalable test-time compute**
   - more parallel work without just making one chain-of-thought longer

5. **coordination under complexity**
   - routing, planning, and tool selection become explicit system problems

6. **memory distribution**
   - useful state can be spread across profiles, notes, pheromones, workspaces, or vector stores

## How
How do these swarms actually coordinate?

This is the most important engineering question, and the corpus shows several distinct mechanisms.

---

# 2) The main coordination patterns

## A. Central evaluator / ranking hub
This is the most common simple pattern.

Example:
- `2504.17967` **PharmaSwarm**

Pattern:
- specialized agents generate candidates
- shared memory aggregates
- an evaluator ranks outputs

Strengths:
- simple
- easy to reason about
- good for domain workflows

Weaknesses:
- not very decentralized
- evaluator becomes bottleneck / bias source

## B. Decentralized role cycle
Example:
- `2510.10047` **SwarmSys**

Pattern:
- Explorers decompose or scout
- Workers execute / reason / debate
- Validators check and terminate
- matching uses embeddings + ε-greedy exploration
- success reinforces future assignments via pheromone-like updates

This is one of the cleanest “true swarm” reasoning systems in the corpus.

## C. Peer-ranked consensus
Example:
- `2510.24801` **Fortytwo**

Pattern:
- many nodes answer
- nodes compare answers pairwise
- Bradley–Terry ranking aggregates the comparisons
- reputation weights who gets trusted more

This is swarm-as-consensus more than swarm-as-role-specialization.

## D. Pheromone-style routing
Example:
- `2603.12933` **AMRO-S**

Pattern:
- intent classifier picks layer/task type
- graph traversal is guided by pheromone matrices
- successful paths are reinforced asynchronously

This is especially relevant to your **breadcrumbs / queues / state** interest:
the pheromone matrix is essentially a persistent routing memory.

## E. Strategy-passing local memory
Example:
- `2512.13713` **LoopBench**

Pattern:
- each agent gets only local observations
- agents keep private strategy notes across rounds
- these notes act as consistent memory

This is very relevant to test-time scaling and distributed coordination:
the paper’s whole point is that swarm competence can emerge from **local observation + persistent local notes**.

## F. Graph-structured agent orchestration
Example:
- `2402.16823` **Language Agents as Optimizable Graphs**

Pattern:
- nodes = LLM calls, tool uses, function calls
- edges = communication paths
- optimize nodes/prompts and graph connectivity

This is not a swarm paper in the narrow naming sense, but it is one of the best abstractions for thinking about coding-agent swarms.

---

# 3) Test-time compute / test-time scaling vs capability/outcome swarms

This distinction matters a lot.

## Swarm as test-time compute
The swarm is mainly a way to spend more inference-time budget.

### Best corpus examples
- `2510.24801` **Fortytwo**
- `2510.10047` **SwarmSys**
- `2512.13713` **LoopBench**

### Adjacent contrast
- `2601.05593` **PaCoRe**  
  Not a swarm paper, but very useful as a comparison point.

PaCoRe says:
- fixed context limits sequential TTC
- solve by **parallel coordinated reasoning**
- compact conclusions into messages
- iterate

That is basically the cleanest non-swarm TTC framing in the corpus.

### My read
**Fortytwo** and **SwarmSys** are effectively swarm-shaped TTC systems:
- more agents = more search / validation / coverage
- gains come from inference coordination
- they are closest to “test-time scaling by swarm”

### What this buys
- better accuracy
- robustness to noisy or weak individual agents
- larger search surface at inference

### What it costs
- latency
- tokens
- communication overhead
- coordination failures
- risk of premature consensus

## Swarm as capability / outcomes
Here the swarm is not just spending more inference compute. It is **building or discovering a better artifact**.

### Examples
- `2410.11163` **Model Swarms**
  - swarm over weight space
  - adapts experts
- `2506.15672` **SwarmAgentic**
  - swarm over agent-system designs
- `2412.12839` **HIVE**
  - better orchestration over tools/models
- `2504.17967` **PharmaSwarm**
  - better domain workflow and hypothesis generation
- `2603.21344` **AI Scientific Community**
  - conceptual swarm of virtual labs

### My read
This second bucket is less about “scale inference” and more about:
- search over system configurations
- collaborative design
- modular specialization
- workflow discovery

So:

- **TTC swarms** optimize answers now
- **capability swarms** optimize systems / processes / outputs over time

---

# 4) Coding agents and shell/bash agents

## Short answer
In this corpus, **direct swarm papers on coding/shell/bash agents are still sparse**.

There are some **adjacent signals**, but not yet a clean canonical “shell-agent swarm” paper in the files I checked.

## What the corpus does show

### A. Coding is an evaluation domain, not yet the dominant swarm architecture
- `2510.24801` **Fortytwo** includes **LiveCodeBench**
- `2510.10047` **SwarmSys** evaluates on **scientific programming tasks**
- `2402.16823` **GPTSwarm** reports **HumanEval** and GAIA improvements

So coding is definitely part of the evaluation surface.

### B. Tool-using agent frameworks are the closest precursor
- `2407.10718` **Sibyl**
  - uses only two tools:
    - web browser
    - **computer terminal with Python interpreter**
  - global workspace + debate jury + compression
- `2412.12839` **HIVE**
  - plans across models/tools using C-KG and PDDL
  - stores execution code snippets in its graph
- `2401.17749` **SwarmBrain**
  - natural language strategy -> executable commands via `python-sc2`

### C. Best architectural bridge to coding swarms
- `2402.16823` **Language Agents as Optimizable Graphs**

Because it explicitly treats:
- nodes as LLM queries / tool usage / function calls
- edges as information flow
- composite graphs as swarms

That is very close to how I’d model a coding/shell swarm:
- planner node
- shell-executor node
- test-runner node
- critic node
- patch generator node
- memory node
- merge/rank node

## My assessment
For **coding agents**, the corpus suggests the field is moving toward swarms, but the cleanest evidence is still:
- graph-based orchestration
- debate/critic systems
- tool-using agents
- inference swarms evaluated on coding benchmarks

For **shell/bash agents specifically**:
- I do **not** see a core swarm paper yet that makes shell execution the central swarm primitive
- the strongest adjacent signal is Sibyl’s terminal tool and general tool-using multi-agent systems

So if your interest is practical coding/shell swarms, the corpus is more useful for **patterns** than for direct exemplars.

---

# 5) Communication breadcrumbs, pub/sub, persistence, shared state

This is one of the richest parts of the analysis.

## Strongest explicit pattern: MQTT + vector DB
Not branded as a swarm paper, but extremely relevant:

- `2508.19042` **A Concurrent Modular Agent: Framework for Autonomous LLM Agents**

This paper is almost tailor-made for your interest:
- modules are async Python functions
- communicate through **MQTT publish/subscribe**
- share memory through **ChromaDB**
- use shared global state without centralized synchronous control
- explicitly tied to **Minsky’s Society of Mind**

This is the clearest corpus example of:
- pub/sub
- persistence
- breadcrumbs
- asynchronous modular cognition

If you care about actual systems design, this is one of the most useful papers adjacent to swarms.

## Shared memory
- `2504.17967` **PharmaSwarm**
  - specialized agents + **shared memory architecture**
- `2407.10718` **Sibyl**
  - **global workspace**
- `2508.19042` **CMA**
  - shared vector DB / global state

## Persistent local memory
- `2512.13713` **LoopBench**
  - private strategy notes across rounds
  - one of the clearest “persistent breadcrumb” mechanisms

## Distributed profile memory
- `2510.10047` **SwarmSys**
  - adaptive agent and event profiles
  - embedding-based matching
  - reinforcement through profile updates

## Routing memory
- `2603.12933` **AMRO-S**
  - task-specific pheromone matrices
  - probably the cleanest “breadcrumb routing” design in the swarm subset

## Message compaction / summarized persistence
- `2601.05593` **PaCoRe**
  - trajectory conclusions become compact messages
  - these persist across rounds as guidance

That’s not exactly pub/sub, but it is a strong TTC-era memory pattern.

## My synthesis
There are at least **five different persistence styles** in the corpus:

1. **shared workspace**  
2. **vector-store memory**  
3. **private strategy notes**  
4. **pheromone / routing state**  
5. **compressed inter-round messages**

If you’re designing real swarms, this matters more than “how many agents?”  
The architectural question becomes: **what is the persistence substrate?**

---

# 6) Tool use in swarms

Tool use is present, but uneven.

## Best explicit tool-use adjacent paper
- `2407.10718` **Sibyl**
  - browser
  - terminal / Python interpreter

## Best planning/orchestration tool paper
- `2412.12839` **HIVE**
  - capability graph
  - PDDL planner
  - model/tool orchestration
  - execution code snippets

## Best abstraction for tool-using swarms
- `2402.16823` **Language Agents as Optimizable Graphs**
  - tool usage is a first-class node type

## In the actual “LLM swarm” papers
Tool use is less mature than:
- consensus
- routing
- memory
- role specialization

So today’s corpus says:
- **tool-using agents are real**
- **tool-using swarms are emerging**
- but the strongest current work is still in **reasoning/routing/memory**, not shell-tool swarms specifically

---

# 7) The Minsky angle

You asked not to over-color the search with Minsky, and that’s the right instinct.

## Bottom line
**Minsky is more explicit in adjacent modular-agent papers than in the swarm papers themselves.**

### Explicitly relevant adjacent papers
- `2508.19042` **Concurrent Modular Agent**
  - explicitly says emergent behavior supports **Minsky’s Society of Mind**
  - async modules + shared state + natural language coordination
- `2407.10718` **Sibyl**
  - explicitly grounded in **Society of Mind theory**
  - actor/critic jury + global workspace
- `2402.16823` **Language Agents as Optimizable Graphs**
  - explicitly invokes the “society of mind” principle
- `2401.05799` **Designing Heterogeneous LLM Agents for Financial Sentiment Analysis**
  - explicitly references Minsky’s theory of mind / specialization

## In the swarm papers proper
The dominant inspirations are more often:
- PSO
- ACO / pheromones
- stigmergy
- collective search
- bandit-style exploration/exploitation
- consensus / ranking

So the Minsky connection is mostly:
- **modular specialists**
- **distributed cognition**
- **coordination without a single monolithic thinker**

That part is highly relevant to your interests.

My read is:
- **Minsky gives the cognitive composition story**
- **swarm papers give the coordination/control story**

Those are complementary, not identical.

---

# 8) What seems most relevant to your interests specifically

## If you care about coding/shell agents
Read:
1. `2402.16823` Language Agents as Optimizable Graphs
2. `2407.10718` Sibyl
3. `2412.12839` HIVE
4. `2510.24801` Fortytwo
5. `2510.10047` SwarmSys

Why:
- graph composition
- tool use
- terminal/browser execution
- coding benchmark relevance
- reasoning swarm design

## If you care about test-time scaling
Read:
1. `2510.24801` Fortytwo
2. `2510.10047` SwarmSys
3. `2512.13713` LoopBench
4. `2601.05593` PaCoRe

Why:
- consensus TTC
- role-based TTC
- local-memory coordination
- clean contrast with parallel coordinated reasoning

## If you care about communication / persistence
Read:
1. `2603.12933` AMRO-S
2. `2512.13713` LoopBench
3. `2504.17967` PharmaSwarm
4. `2508.19042` CMA
5. `2407.10718` Sibyl

Why:
- pheromone routing memory
- persistent local notes
- shared memory
- pub/sub + vector db
- global workspace

## If you care about “swarms that build systems”
Read:
1. `2410.11163` Model Swarms
2. `2506.15672` SwarmAgentic
3. `2412.12839` HIVE
4. `2603.21344` AI Scientific Community

---

# 9) My overall conclusions

## 1. LLM swarms are real, but the design space is fragmented
The corpus does not support a single canonical LLM-swarm architecture.  
It supports several families:
- consensus swarms
- routing swarms
- role-specialized reasoning swarms
- design/build swarms
- orchestration swarms

## 2. Test-time scaling is one of the strongest emerging meanings of “swarm”
Especially in:
- Fortytwo
- SwarmSys
- LoopBench
- and by contrast PaCoRe

## 3. Coding/shell swarms are not yet the clean center of the literature here
You can see the pieces:
- coding benchmarks
- tool nodes
- terminal tools
- graph orchestration
But not yet a canonical “swarm of bash agents” paper.

## 4. Persistence substrate is a first-class design choice
The corpus strongly suggests you should think about:
- shared memory
- local notes
- routing traces
- vector stores
- pub/sub buses
- compacted inter-round messages

not just agent count.

## 5. Minsky is not central to swarm branding, but very relevant architecturally
His strongest contribution here is as a lens for:
- modularity
- specialization
- composition
- emergent system-level competence from simpler parts

That lens fits very well with adjacent agent-framework papers.

If you want, next I can do either of these:

1. make you a **tight reading list just for coding/shell/tool-using agent swarms**, or  
2. write a **design memo for a practical coding-agent swarm architecture** using the corpus patterns:
   - planner
   - shell executor
   - test runner
   - critic
   - shared workspace / pub-sub / memory
   - TTC vs capability modes
