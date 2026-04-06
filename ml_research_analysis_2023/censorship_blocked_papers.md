# StepFun Censorship-Blocked Papers (Active Queue)

These 8 papers are stuck in the pipeline. StepFun returns HTTP 451 (`censorship_blocked`) for medical/sensitive content.

| arxiv_id | Title | Topic | Had 451? |
|----------|-------|-------|----------|
| 2601.17284 | Mind the Ambiguity: Aleatoric Uncertainty Quantification in LLMs for Safe Medical Question Answering | Medical AI safety | Likely |
| 2602.07319 | Beyond Accuracy: Risk-Sensitive Evaluation of Hallucinated Medical Advice | Medical AI hallucination | Likely |
| 2602.08373 | Grounding Generative Planners in Verifiable Logic: A Hybrid Architecture for Trustworthy Embodied AI | Trustworthy AI | **Yes** (confirmed in logs) |
| 2602.17911 | Condition-Gated Reasoning for Context-Dependent Biomedical Question Answering | Biomedical QA | Likely |
| 2603.01625 | Measuring What VLMs Don't Say: Validation Metrics Hide Clinical Terminology Erasure in Radiology Report Generation | Radiology / Clinical NLP | Likely |
| 2603.16077 | MDM-Prime-v2: Binary Encoding and Index Shuffling Enable Compute-optimal Scaling of Diffusion Language Models | Diffusion models | Unlikely |
| 2603.22327 | AgentSLR: Automating Systematic Literature Reviews in Epidemiology with Agentic AI | Epidemiology AI | Likely |
| 2604.00195 | Lévy-Flow Models: Heavy-Tail-Aware Normalizing Flows for Financial Risk Management | Financial risk modeling | Possible |

## Notes
- StepFun is a Chinese LLM provider that censors medical, health, and potentially financial content.
- 5 of 8 are clearly medical/clinical — will never pass StepFun's content filter.
- 2602.08373 ("Trustworthy AI") was confirmed blocked via 451 in pipeline logs.
- These papers need an alternative model (e.g. Trinity, or a non-Chinese provider).
