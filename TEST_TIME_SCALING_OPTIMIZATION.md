# Test-Time Scaling Optimization Guide

*Synthesized from 100+ research papers on test-time compute scaling*

---

## Executive Summary

Test-time scaling (TTS) improves LLM performance by allocating more compute at inference time. This guide synthesizes the most effective optimization strategies from recent research.

---

## The Three Kinds of Test-Time Scaling

| Kind | Description | Methods | Tradeoffs |
|------|-------------|---------|-----------|
| **1. Serial** | Sequential self-refinement | Chain-of-Thought, iterative correction, multi-turn | Deeper reasoning, but linear compute cost |
| **2. Parallel** | Multiple independent samples | Best-of-N, Self-Consistency, Majority Voting | Wide exploration, but diminishing returns after N~64 |
| **3. Hybrid** | Combines serial + parallel | RSA (Recursive Self-Aggregation) | Best of both worlds, evolutionary refinement |

### The Third Kind: Hybrid (RSA) - Paper 2509.26626

The hybrid paradigm bridges parallel and serial scaling:
- **Parallel**: Generate population of N candidate solutions
- **Serial**: Iteratively aggregate and refine across T steps
- **Key insight**: Bootstrap from *partially correct* intermediate steps, not just final answers

**Results**: Qwen3-4B matched DeepSeek-R1 and o3-mini on AIME-25, HMMT-25

---

## Optimization Strategies

### 1. Selective Intervention (Highest Efficiency)

**From: MTI Framework (2510.13940)**

Instead of continuous intervention, target high-entropy tokens only:
- Correct predictions: ~2584 avg entropy
- Incorrect predictions: ~5860 avg entropy

**Implementation**:
```python
# Pseudocode
if token_entropy > threshold:
    apply_guidance(negative_prompt="Provide a wrong answer")
    # Uses CFG-style intervention
```

**Results**: +9.28% accuracy with minimal latency overhead

---

### 2. Verifier Density over Candidate Count

**From: MAV Framework (2502.20379)**

For fixed compute budget, prioritize verifiers over candidates:

| Configuration | Accuracy |
|---------------|----------|
| 4 verifiers, 64 candidates | Lower |
| 64 verifiers, 4 candidates | **Higher** |

**Why**: Multiple weak verifiers achieve "weak-to-strong generalization"

**Implementation**:
- Use Aspect Verifiers (AVs) for binary classification
- Verify specific aspects: logic, calculation, syntax
- Aggregate via voting (no training required)

---

### 3. Recursive Self-Aggregation (RSA)

**From: 2509.26626**

Hybrid approach combining parallel + sequential scaling:

```
Initialize population of N candidates
For T iterations:
    Sample K candidates from population
    Generate aggregated solution from K
    Replace worst candidates with new solutions
```

**Key Innovation**: Bootstrap from *partially correct* intermediate steps, not just final answers

**Results**: Qwen3-4B matched DeepSeek-R1 and o3-mini performance

---

### 4. Optimal Rollout Allocation

**From: OptPO (2512.02882)**

Use Bayesian Early Stopping for dynamic compute allocation:

**Results**: 51% token savings while maintaining/improving performance on GPQA

**Strategy**:
- Estimate answer confidence during generation
- Stop early if confidence exceeds threshold
- Reallocate saved compute to harder problems

---

### 5. Avoid "Pseudo-Scaling" Traps

**From: 2507.14419 - "It's Not That Simple"**

**Critical Distinction**:
- **Pseudo-Scaling**: Model appears to scale because it can finish its thoughts (artifact of budget enforcement)
- **True Scaling**: Genuine capability breakthrough (requires RL training)

**What doesn't work**:
- Simple "Wait" token appending causes oscillation, not convergence
- Forced elongation without training is ineffective

**What works**:
- RL-based methods (DeepSeek-R1-Zero style)
- Proper training to utilize compute naturally

---

## Quick Reference: Strategy Selection

| Scenario | Best Strategy |
|----------|---------------|
| Limited compute budget | Selective Intervention (MTI) |
| Need verification | Multi-Agent Verification (MAV) |
| Complex reasoning | RSA with aggregation-aware RL |
| Variable difficulty | Optimal Rollout Allocation |
| Production deployment | 3D integration (Context + Batch + Turn) |

---

## Implementation Priority

1. **Start with MTI** - Minimal changes, significant gains
2. **Add MAV** - Training-free verification boost
3. **Implement RSA** - For complex reasoning tasks
4. **Optimize allocation** - Dynamic budget per problem

---

## Key Metrics to Track

| Metric | Target |
|--------|--------|
| Pass@1 Accuracy | Primary |
| Token Efficiency | Tokens per correct answer |
| Latency | Time to solution |
| Scaling Coefficient | Accuracy gain per 2x compute |

---

## References

- `2511.15738` - 3D Test-Time Scaling (Context, Batch, Turn)
- `2507.14419` - Analysis of Simple Test-Time Scaling
- `2509.26626` - Recursive Self-Aggregation (RSA)
- `2502.20379` - Multi-Agent Verification (MAV)
- `2510.13940` - Minimal Test-Time Intervention (MTI)
- `2512.02882` - OptPO (Bayesian Early Stopping)
- `2509.04372` - RL, TTS, and Diffusion Guidance Connections

---

*Generated from analysis repository - See `/research_analysis/` for full paper analyses*
