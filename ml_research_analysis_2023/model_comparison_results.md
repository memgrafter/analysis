# Model Comparison Results: Trinity vs Step-3.5-Flash

**Date:** 2026-03-30  
**Pipeline:** research_paper_analysis_v2 (FlatMachine)  
**Evaluator:** Claude (blind eval against source papers)  
**Papers:** 10 arXiv papers processed through identical pipeline  

---

## Summary Table

| # | Paper | Trinity (/60) | Step-3.5-Flash (/60) | Gap |
|---|-------|:------------:|:--------------------:|:---:|
| 1 | NorwAI LLMs (2601.03034) | 35 | 47 | +12 |
| 2 | F2LLM-v2 Multilingual (2603.19223) | 31 | 44 | +13 |
| 3 | Coordinated Semantic RAG (2603.04647) | 37 | 45 | +8 |
| 4 | AgentDropoutV2 (2602.23258) | 35 | 44 | +9 |
| 5 | Rodent-Bench (2602.18540) | 22 | 44 | +22 |
| 6 | CCCaption (2602.21655) | 31 | 46 | +15 |
| 7 | FastBUS (2603.00517) | 25 | 43 | +18 |
| 8 | Token Compressor (2603.25340) | 28 | 45 | +17 |
| 9 | QuePT Quantization (2602.12609) | 32 | 45 | +13 |
| 10 | Sympformer (2603.16535) | 33 | 47 | +14 |
| **Grand Total** | **309/600** | **450/600** | **+141** |
| **Average per paper** | **30.9/60** | **45.0/60** | **+14.1** |

---

## Per-Section Averages (across all 10 papers)

| Section | Trinity Avg | Step-3.5-Flash Avg | Notes |
|---------|:-----------:|:-------------------:|-------|
| Key Outcome | 5.9 | 8.0 | Step cites specific numbers; Trinity paraphrases abstracts |
| Why & Hypothesis | 5.1 | 8.3 | Step decomposes mechanisms with equation/section citations |
| Open Questions | 4.6 | 7.9 | Trinity outputs "None" on 2 papers; Step grounds in stated limitations |
| Reproduction Notes | 5.4 | 8.2 | Step includes hyperparameters, failure signatures, concrete experiments |
| Limits & Confidence | 5.4 | 7.7 | Step calibrates per-mechanism; Trinity over-claims or stays surface-level |
| Domain Tags | 4.0 | 4.9 | Both suffer from pipeline-injected irrelevant tags; Trinity sometimes empty |

---

## Key Findings

### Step-3.5-Flash Strengths
- **Quantitative grounding:** Consistently cites specific numbers, table references, equation numbers from papers
- **Mechanistic decomposition:** Breaks methods into distinct mechanisms with evidence anchors and break conditions
- **Critical calibration:** Assigns appropriate confidence levels per-claim with explicit rationale
- **Actionable reproduction:** Includes hyperparameters, expected baselines, failure signatures
- **Paper-grounded questions:** Open questions trace to actual gaps the authors acknowledge

### Trinity Weaknesses
- **Surface-level paraphrasing:** Reads like abstract rewrites rather than critical analysis
- **Missing sections:** Outputs "None" for Open Questions on 2/10 papers despite clear material in source
- **Over-confident:** Assigns "High confidence" to claims backed only by limited internal ablation
- **Vague evidence anchors:** Uses "[corpus]" or "section notes..." instead of specific citations
- **Empty domain tags:** Fails to populate tags on 3/10 papers

### Shared Pipeline Issues
- **Hallucinated domain tags:** Both models generate irrelevant tags (RAG, MoE, chain-of-thought, instruction-tuning) regardless of paper content — this is a pipeline prompt issue, not model-specific
- **Tag template bleeding:** The tag list appears partially hardcoded in the pipeline prompt

---

## Recommendation

**Step-3.5-Flash is the clear winner for production use.** It outperforms Trinity on every paper (avg +14.1 points/paper) and every section category. The advantage is most pronounced on:
- Papers requiring deep technical extraction (Rodent-Bench: +22, FastBUS: +18)
- Sections requiring critical judgment (Why & Hypothesis, Open Questions)

### Action Items
1. **Switch production model to Step-3.5-Flash** (`openrouter/stepfun/step-3.5-flash:free`)
2. **Fix domain tags pipeline prompt** — remove or rework the tag suggestion list to prevent hallucinated irrelevant tags
3. **Monitor rate limits** — Step-3.5-Flash may have different RPM/TPM limits than Trinity on OpenRouter free tier
4. **Re-evaluate periodically** — free model availability changes; keep Trinity as fallback

---

## Raw Evaluations

### Paper 1: NorwAI LLMs (2601.03034)
| Section | Trinity | Step-3.5-Flash | Notes |
|---------|:-------:|:--------------:|-------|
| Key Outcome | 6 | 8 | Step more precise (GPT-4 + journalist comparison); Trinity overstates |
| Why & Hypothesis | 5 | 8 | Step has specific evidence anchors, testable assumptions |
| Open Questions | 6 | 8 | Step questions more paper-specific and actionable |
| Reproduction Notes | 6 | 8 | Step includes specific details (5k samples, NorBERT) |
| Limits & Confidence | 5 | 8 | Step identifies sharper limitations (reward hacking, confounded causality) |
| Domain Tags | 7 | 7 | Identical tag sets |
| **Total** | **35** | **47** | |

### Paper 2: F2LLM-v2 (2603.19223)
| Section | Trinity | Step-3.5-Flash | Notes |
|---------|:-------:|:--------------:|-------|
| Key Outcome | 6 | 7 | Step cites specific numbers but misnames comparison models |
| Why & Hypothesis | 5 | 8 | Step grounds mechanisms in specific sections/figures |
| Open Questions | 5 | 8 | Step traces to paper's own admissions |
| Reproduction Notes | 5 | 8 | Step includes hyperparameters and experiment thresholds |
| Limits & Confidence | 4 | 7 | Trinity makes factual error (claims no MRL ablation) |
| Domain Tags | 6 | 6 | Identical |
| **Total** | **31** | **44** | |

### Paper 3: Coordinated Semantic RAG (2603.04647)
| Section | Trinity | Step-3.5-Flash | Notes |
|---------|:-------:|:--------------:|-------|
| Key Outcome | 7 | 8 | Step names all five baselines |
| Why & Hypothesis | 7 | 8 | Step adds break conditions, cites equations |
| Open Questions | 6 | 8 | Step grounded in paper's stated future work |
| Reproduction Notes | 6 | 8 | Step flags concrete missing details |
| Limits & Confidence | 6 | 8 | Step identifies absence of ablation studies |
| Domain Tags | 5 | 5 | Both include unsupported tags |
| **Total** | **37** | **45** | |

### Paper 4: AgentDropoutV2 (2602.23258)
| Section | Trinity | Step-3.5-Flash | Notes |
|---------|:-------:|:--------------:|-------|
| Key Outcome | 6 | 8 | Step cites specific deltas (AIME25 +6.67%) |
| Why & Hypothesis | 6 | 8 | Step anchors to Table 1 ablation numbers |
| Open Questions | 6 | 8 | Step surfaces computational overhead concerns |
| Reproduction Notes | 6 | 8 | Step includes expected ablation baselines |
| Limits & Confidence | 6 | 7 | Both identify prompt template gaps |
| Domain Tags | 5 | 5 | Both include irrelevant tags |
| **Total** | **35** | **44** | |

### Paper 5: Rodent-Bench (2602.18540)
| Section | Trinity | Step-3.5-Flash | Notes |
|---------|:-------:|:--------------:|-------|
| Key Outcome | 5 | 8 | Trinity omits all quantitative results |
| Why & Hypothesis | 1 | 8 | Trinity outputs "None" |
| Open Questions | 1 | 8 | Trinity outputs "None" |
| Reproduction Notes | 5 | 8 | Step adds frame-sampling tradeoffs, expected values |
| Limits & Confidence | 6 | 8 | Step flags missing API hyperparameters |
| Domain Tags | 4 | 4 | Both have irrelevant tags |
| **Total** | **22** | **44** | |

### Paper 6: CCCaption (2602.21655)
| Section | Trinity | Step-3.5-Flash | Notes |
|---------|:-------:|:--------------:|-------|
| Key Outcome | 5 | 8 | Trinity omits all specific numbers |
| Why & Hypothesis | 5 | 9 | Step gives 3 mechanisms with evidence anchors |
| Open Questions | 6 | 8 | Step cites Conclusion directly |
| Reproduction Notes | 5 | 9 | Step includes α=0.05, lr=1e-6, 20 epochs, reward formula |
| Limits & Confidence | 6 | 8 | Step pinpoints undisclosed template details |
| Domain Tags | 4 | 4 | Both carry irrelevant tags |
| **Total** | **31** | **46** | |

### Paper 7: FastBUS (2603.00517)
| Section | Trinity | Step-3.5-Flash | Notes |
|---------|:-------:|:--------------:|-------|
| Key Outcome | 6 | 8 | Step names 10+ settings, specific baselines |
| Why & Hypothesis | 5 | 9 | Step decomposes §4.1–4.4 with break conditions |
| Open Questions | 2 | 8 | Trinity outputs "None" — paper has clear open questions |
| Reproduction Notes | 4 | 8 | Step provides component map, hyperparameter knobs |
| Limits & Confidence | 5 | 7 | Step adds GBP construction gaps |
| Domain Tags | 3 | 3 | Both have irrelevant pipeline-injected tags |
| **Total** | **25** | **43** | |

### Paper 8: Token Compressor (2603.25340)
| Section | Trinity | Step-3.5-Flash | Notes |
|---------|:-------:|:--------------:|-------|
| Key Outcome | 5 | 8 | Step cites BLEU-4 99.31@4x, F1 33.35 |
| Why & Hypothesis | 5 | 8 | Step references LoRA config, polysemy metric |
| Open Questions | 6 | 7 | Both reasonable; Step slightly tighter |
| Reproduction Notes | 5 | 8 | Step details loss formula, Z-vocab size |
| Limits & Confidence | 5 | 8 | Step scopes validation to Qwen3 ≤4B |
| Domain Tags | 2 | 6 | Trinity has zero tags |
| **Total** | **28** | **45** | |

### Paper 9: QuePT (2602.12609)
| Section | Trinity | Step-3.5-Flash | Notes |
|---------|:-------:|:--------------:|-------|
| Key Outcome | 7 | 8 | Step cites 1/26× PTMQ speedup |
| Why & Hypothesis | 6 | 8 | Step references specific equations |
| Open Questions | 6 | 8 | Step captures stated conclusion limitations |
| Reproduction Notes | 6 | 8 | Step includes λ-weighted fusion, ablation tables |
| Limits & Confidence | 5 | 8 | Trinity over-calibrates confidence |
| Domain Tags | 2 | 5 | Trinity empty tags |
| **Total** | **32** | **45** | |

### Paper 10: Sympformer (2603.16535)
| Section | Trinity | Step-3.5-Flash | Notes |
|---------|:-------:|:--------------:|-------|
| Key Outcome | 7 | 9 | Step adds 25% improvement calc and wall-clock tradeoff |
| Why & Hypothesis | 6 | 9 | Step cites Eq 22, 29, Prop 4.1, 4.3, 5.1 |
| Open Questions | 6 | 8 | Step matches Section 7 stated future work |
| Reproduction Notes | 6 | 9 | Step names specific learned parameters |
| Limits & Confidence | 6 | 8 | Step identifies implementation ambiguities |
| Domain Tags | 2 | 4 | Trinity empty; Step hallucinates some |
| **Total** | **33** | **47** | |
