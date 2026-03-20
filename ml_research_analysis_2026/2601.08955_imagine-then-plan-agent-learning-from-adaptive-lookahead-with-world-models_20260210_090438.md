---
ver: rpa2
title: 'Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models'
arxiv_id: '2601.08955'
source_url: https://arxiv.org/abs/2601.08955
tags:
- uni00000013
- agent
- world
- lookahead
- uni00000011
- transformers
- self-attention
- retrieval-augmented-generation
- instruction-tuning
- parameter-efficient-finetuning
- mixture-of-experts
- chain-of-thought
core_contribution: The paper proposes Imagine-then-Plan (ITP), a unified framework
  that equips LLM-based agents with adaptive lookahead imagination via learned world
  models. By extending the standard POMDP to a Partially Observable and Imaginable
  MDP (POIMDP), ITP enables agents to explicitly reason over both the observable present
  and imagined future trajectories.
---

# Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models

## Quick Facts
- **arXiv ID**: 2601.08955
- **Source URL**: https://arxiv.org/abs/2601.08955
- **Reference count**: 38
- **Primary result**: ITP with adaptive lookahead imagination via learned world models significantly outperforms competitive baselines on ALFWorld and ScienceWorld benchmarks.

## Executive Summary
The paper introduces Imagine-then-Plan (ITP), a unified framework that equips LLM-based agents with adaptive lookahead imagination via learned world models. By extending the standard POMDP to a Partially Observable and Imaginable MDP (POIMDP), ITP enables agents to explicitly reason over both the observable present and imagined future trajectories. The core innovation is an adaptive lookahead mechanism that dynamically scales the imagination horizon by trading off the ultimate goal and estimated task progress, allowing the agent to optimize the balance between foresight reliability and computational efficiency. Extensive experiments demonstrate that ITP significantly outperforms competitive baselines, with ITPI achieving strong zero-shot planning performance and ITPR achieving the highest success rates across all tested backbone models.

## Method Summary
ITP operates by first training a world model to predict next states given current state and action. The framework is instantiated into two variants: ITPI, a training-free method that uses prompt-based reflection for adaptive lookahead, and ITPR, a reinforcement-trained method that learns optimal horizon selection via A2C. At each step, the agent selects an imagination horizon K_t adaptively, generates K_t-step imagined trajectories using the world model, and conditions its action selection on both the current state and the imagined future. ITPR adds a K-head predictor to learn when to imagine, optimized via a three-stage process: pseudo-label generation from expert trajectories, warm-up with supervised fine-tuning, and online reinforcement learning with A2C.

## Key Results
- ITPI substantially enhances zero-shot planning performance, demonstrating that adaptive lookahead can be implemented without additional training.
- ITPR achieves the highest success rates across all tested backbone models (Qwen2.5-7B, Qwen3-8B, Llama3.1-8B) on both ALFWorld and ScienceWorld benchmarks.
- The adaptive horizon selection mechanism provides a better success rate vs. computational cost trade-off compared to fixed-horizon baselines.

## Why This Works (Mechanism)

### Mechanism 1: Adaptive Lookahead with World Model Rollouts
Dynamically adjusting imagination horizon based on task progress improves planning quality while controlling computational cost. At each step t, the agent uses a K-head predictor P_θ(K_t|s_t) to select horizon K_t ∈ {0,1,...,K_max}, then performs K_t-step rollouts in the world model to generate imagined trajectory τ̂_t. This trajectory provides signals about goal progress and potential conflicts before real execution.

### Mechanism 2: POIMDP Formulation for Dual-Stream Conditioning
Conditioning actions on both observed present and imagined future creates richer decision context than observation-only policies. The policy π_θ(·|s_t, τ̂_t^(K_t)) receives both current state s_t and imagined future τ̂_t, enabling detection of bottlenecks and self-correction before irreversible execution. This extends POMDP to POIMDP (Partially Observable and Imaginable MDP).

### Mechanism 3: Reinforcement-Trained Horizon Selection (ITPR)
Jointly optimizing lookahead horizon selection and action policy via A2C learns strategic "when to imagine." Stage 1 generates pseudo-labels K̃_t by scoring expert-action likelihood under different horizons. Stage 2 warms up policy and K-head predictor. Stage 3 uses online A2C with reward r_t+1 = r_env - λ_K·K_t - λ_step to balance task success against imagination cost.

## Foundational Learning

- **World Models as Learned Simulators**: Why needed: ITP requires M_φ that predicts s' given (s, a) for mental rehearsal. Quick check: Given state "holding apple, near microwave" and action "heat apple," what next state should the world model predict?
- **Partially Observable MDPs (POMDPs)**: Why needed: The baseline formalization that POIMDP extends. Agents only receive observations o_t, not true states s_t, and must maintain history h_t. Quick check: Why can't a POMDP agent directly condition on s_t, and how does adding τ̂_t help?
- **Advantage Actor-Critic (A2C)**: Why needed: ITPR uses A2C for online optimization with advantage estimates A_t = Q(s,a) - V(s) to reduce variance. Understanding the actor-critic split is essential for debugging training dynamics. Quick check: What happens to learning if the value head V_θ(s_t) consistently underestimates returns?

## Architecture Onboarding

- **Component map**: World Model M_φ → K-head Predictor P_θ → Agent Policy π_θ → Value Head V_θ
- **Critical path**: 
  1. World Model Training: Fine-tune on D_WM = D_exp ∪ D_roll via next-token prediction
  2. Stage 1 (ITPR only): pseudo-label K_t via teacher-forced expert rollouts on world model
  3. Stage 2 (ITPR only): warm-up SFT with lookahead-conditioned policy + K-head predictor
  4. Stage 3 (ITPR only): online A2C with environment rollouts and reward penalty
- **Design tradeoffs**: ITPI vs. ITPR: ITPI requires no training, uses prompt-based reflection; ITPR adds ~2-3x training overhead but learns optimal horizon selection and achieves higher peak SR
- **Failure signatures**: World model collapse: τ̂_t becomes repetitive or nonsensical → check world model loss on held-out transitions; K-head degeneracy: Always predicts K=0 or K=K_max → check pseudo-label distribution and entropy bonus β
- **First 3 experiments**:
  1. Sanity check: Train world model on D_exp only, verify it predicts next observations with >80% token-level accuracy on held-out transitions
  2. ITPI baseline: Run ITPI with fixed K=1,2,3 on 20 ALFWorld tasks; compare SR and token cost to establish fixed-horizon performance curve
  3. ITPR ablation: Train ITPR with and without Stage 3 (online A2C); plot SR gap to quantify value of learned adaptive lookahead vs. warm-up only

## Open Questions the Paper Calls Out
- **Multimodal environments**: How does adaptive lookahead perform in multimodal environments with high-dimensional visual or sensorimotor states compared to textual benchmarks? The current evaluation does not fully capture challenges of multimodal environments, open-world tool-use, or real-world robotic control.
- **Computational efficiency**: Can techniques like world model distillation or speculative decoding effectively mitigate the inference-time computational overhead associated with the ITP framework? The framework introduces higher inference-time overhead compared to purely reactive agents.
- **Model error compounding**: How does compounding error in the learned world model affect the reliability of long-horizon planning within the POIMDP formulation? Excessive rollouts can amplify model errors, but the specific failure rate caused by world model hallucination versus agent policy error is not quantified.

## Limitations
- The current evaluation focuses on textual environments and does not address multimodal settings with visual or sensorimotor inputs where world model fidelity may be more challenging.
- The method introduces higher inference-time overhead compared to purely reactive agents, which may limit real-time applicability without further optimization.
- The POIMDP formulation lacks extensive theoretical grounding for why conditioning on imagined futures should consistently improve planning quality over observed-only policies.

## Confidence
- **High confidence**: The empirical demonstration that adaptive lookahead outperforms fixed-horizon baselines on ALFWorld and ScienceWorld benchmarks
- **Medium confidence**: The claim that ITPI's training-free approach can match or exceed specialized trained agents, given that this requires strong world model generalization
- **Low confidence**: The scalability claims beyond the tested environments, as the method's performance may degrade when world model prediction accuracy drops below critical thresholds

## Next Checks
1. Test world model failure modes by deliberately corrupting the training distribution and measuring impact on ITP performance degradation curves
2. Compare ITP's computational efficiency against oracle adaptive methods that know the true task progress to quantify the value of the learned estimation approach
3. Evaluate ITP in environments with stochastic dynamics where world model predictions have inherent uncertainty to assess robustness to model error