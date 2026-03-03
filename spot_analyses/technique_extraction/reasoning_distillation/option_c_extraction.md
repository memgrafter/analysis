# Reasoning Distillation — Option C (Grep-Seeded)

**Date:** 2026-02-16  
**Group:** `reasoning_distillation` (86 papers)  
**Coverage:** 86/86 matched (100%)

---

## Group Definition Used (Phase 0)

2025/2026-only papers where title/core contribution jointly indicate:

- distillation (`distill*`)
- reasoning-style supervision (`reasoning` / `chain-of-thought` / `rationale`)
- transfer/compression framing (`teacher` / `student` / `small language model` / `compact` / `compression`)

---

## Grep-Seeded Categories

### 1) Teacher-Student Reasoning Transfer (72 papers)
Patterns: `teacher|student|knowledge distillation|student model|small model|SLM`.

Most papers explicitly frame reasoning transfer from stronger teachers to cheaper students.

### 2) Trace/Rationale-Centric Objectives (40 papers)
Patterns: `chain-of-thought|CoT|reasoning trace|rationale|step-by-step|reasoning trajectory`.

Supervision is applied at intermediate reasoning-trace level instead of final-answer imitation only.

### 3) Domain-Specialized Reasoning Distillation (30 papers)
Patterns: `code generation|text-to-sql|retrieval|RAG|reranking|e-commerce|asset health|causal reasoning`.

Reasoning distillation adapted to task-specific constraints and outputs.

### 4) Compression & Length-Control Distillation (26 papers)
Patterns: `compression|compress|pruning|token-level|prefix|truncate|length|efficiency`.

Methods that preserve reasoning quality while shortening/compressing traces.

### 5) Data Curation / Rationale Selection (24 papers)
Patterns: `dataset|augmentation|selection|difficulty|quality|filter|synthetic|data-efficient`.

Focus on selecting better teacher traces rather than scaling raw distillation volume.

### 6) Multimodal Reasoning Distillation (19 papers)
Patterns: `vision|multimodal|speech|audio|document VQA|VLM|VLA|driving`.

Transfer of reasoning behavior to multimodal students and embodied settings.

### 7) RL / Preference-Enhanced Distillation (15 papers)
Patterns: `reinforcement learning|RL|DPO|preference optimization|reward-guided|on-policy`.

Distillation combined with optimization over reward/preference signals.

### 8) Safety/Alignment-Aware Distillation (14 papers)
Patterns: `safety|alignment|refusal|jailbreak|harmful`.

Papers explicitly balancing reasoning transfer with alignment constraints.

### 9) Agentic/Social Distillation Routes (8 papers)
Patterns: `multi-agent|debate|Socratic|graph distillation|agent`.

Distill collaborative or multi-agent reasoning dynamics into smaller students.

### 10) Multi-Teacher Conflict Handling (5 papers)
Patterns: `multi-teacher|multiple teacher|knowledge conflicts|drifting teachers`.

Methods for combining inconsistent teacher signals during distillation.

---

## Overlap Notes

- Heavy overlap between **teacher-student transfer**, **trace objectives**, and **data curation**.
- **Domain-specialized** and **multimodal** buckets often co-occur with compression or RL mechanisms.
- **Multi-teacher** is small but methodologically distinct (teacher-conflict management).

---

## Coverage Summary

- Total papers: **86**
- Matched by grep categories: **86 (100%)**
- Dominant core: teacher-student transfer + trace-level supervision + data/efficiency controls
- Boundary tail: domain-heavy papers where reasoning distillation is secondary to application goals
