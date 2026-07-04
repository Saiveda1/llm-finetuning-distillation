# Fine-Tuning & Distillation Project Document

**Prepared For:** Sai Veda  
**GitHub Publishing Account:** Nikeshk834  
**Repository Slug:** `08-llm-finetuning-distillation`  
**Verified Test Count From Portfolio Index:** 29  

## Background

A fully-runnable, **GPU-free** pipeline that implements the core techniques used to
adapt and compress large language models — **full fine-tuning**, **LoRA** low-rank
adaptation, **knowledge distillation**, an automated **eval/promotion gate**, and a
**synthetic-data scaling study** — end to end, deterministically, on a CPU in a few
minutes.

> ### Honest framing: surrogate model, real techniques
> There is no GPU and no real LLM here. Every technique is implemented **for real**
> — real gradients, a real train/eval loop, a real low-rank adapter, a real KD loss
> — but on a small **surrogate model**: a 1-hidden-layer NumPy MLP over hashed
> instruction-text features, trained on a synthetic instruction-tuning dataset.
> **I did not train a 70B model.** The point is that fine-tuning, LoRA, and
> distillation are parameterization/optimization techniques whose correctness and
> trade-offs are *model-agnostic*: give them a weight matrix, a differentiable
> loss, and a training loop and they behave exactly as they do on a transformer.
> [`ARCHITECTURE.md`](./ARCHITECTURE.md) maps every component 1:1 onto the real
> stack (QLoRA + 🤗 Transformers `Trainer` + vLLM), and the last README section
> spells out how to scale it up.

## Headline results

All numbers below are produced by `python scripts/run_pipeline.py` and stored in
[`benchmarks/results.json`](./benchmarks/results.json) / `benchmarks/results.md`.

| Technique | Result |
|---|---|
| **LoRA vs full fine-tune** | Both reach **0.934** on the shifted target domain; LoRA trains **2,889** params vs **33,417** → **91.4% fewer trainable parameters**. Adapter merges exactly (max logit diff `0.0e+00`). |
| **Knowledge distillation** | Teacher ensemble **1.000**; a **17.9× smaller** student recovers **100%** of it (**1.000**) with KD, vs **0.844** without — a **+0.156** lift from dark knowledge alone. |
| **Eval gate** | Good model (val **0.998**) → **APPROVED**; corrupted-label model (val **0.138**) → **BLOCKED**. |
| **Data scaling** | Held-out accuracy climbs **0.599 → 0.857** as training data grows 100 → 4,000 examples. |
| **Data generator** | **23,023 rows/s**, 1,000,000 rows in **43.4 s** at bounded memory (100k-row chunks) → ~**12 h** for 1e9 single-process, embarrassingly parallel. |

## Project Purpose

This repository is part of the AI engineering portfolio and focuses on the following problem space:

- LoRA + knowledge distillation, from scratch
- Headline result from the portfolio index: LoRA **−91% params**; KD recovers **100%** of teacher

## What This Project Solves

This project provides a production-style implementation with benchmark evidence and operational checks committed into the repository.

## Technical Approach

# Architecture & Design Decisions

## The core idea: surrogate model, real techniques

Fine-tuning, LoRA, and distillation are **optimization/parameterization
techniques**. Their correctness and their trade-offs (trainable-parameter
reduction, dark-knowledge transfer, exact adapter merge, promotion gating) do
**not** depend on the model being a 70B transformer — they depend on there being
(a) a weight matrix to adapt, (b) a differentiable loss, and (c) a train/eval
loop. This repo provides all three with a small NumPy model so the *entire*
pipeline runs deterministically on a CPU in minutes, and every claim is backed by
a number the code actually produced.

```
instruction text ──▶ HashingVectorizer ──▶ x ∈ R^D
                                              │
                        ┌─────────────────────┴───────────────────────┐
                        ▼                                              ▼
                   W1 (D×H) ── ReLU ── W2 (H×C) ── softmax        (the "LLM" surrogate)
                        ▲
        LoRA:  W1_eff = W1 + (α/r)·B·A   (freeze W1, train B,A + head)
```

| Real LLM stack | This repo's surrogate |
|---|---|
| Token embeddings + attention | Hashing-trick features (`features.py`) |
| Transformer block weight `W ∈ R^{d×d}` | First linear layer `W1 ∈ R^{D×H}` |
| LoRA on q/k/v/o projections | LoRA on `W1` (`lora.py`) |
| Instruction-tuning corpus (FLAN/Alpaca) | Synthetic instruction generator (`data.py`) |
| Teacher = large model / ensemble | Bagged MLP ensemble (`distill.py`) |
| KD soft-label loss (Hinton et al.) | Same loss, same `T²` scaling |
| CI eval gate before deploy | `gate.py` |

## Component decisions

**Featurizer — hashing trick, not a learned vocab.** Stateless and deterministic:
train/val/test and any new text featurize identically with zero fitted state, so
reproducibility is trivial and there is no vocabulary drift. Signed hashing
(Weinberger 2009) halves collision bias. The techniques downstream are
feature-agnostic, so a simple featurizer is the right call.

**Model — 1-hidden-layer MLP with explicit forward/backward.** Written by hand so
the LoRA adapter and the KD student reuse the *exact* same math and differ only in
which parameters get gradients and what loss is applied at the logits. Adam is
shared across all trainers (`optim.py`) so results reflect the technique, not the
optimizer.

**Domain shift for t

## Benchmark And Validation Evidence

The portfolio root documents **29 passing tests** for this project, and the repo quickstart uses `make test` as the standard validation path. The benchmark outputs committed in `benchmarks/` and the generated visuals in `assets/` are the evidence package for this delivery.

### results.md

# Benchmark Results

_Surrogate MLP over hashed instruction-text features. Numbers produced by `scripts/run_pipeline.py`._

## LoRA vs full fine-tuning (domain adaptation)

| Method | Target-domain accuracy | Trainable params |
|---|---|---|
| Full fine-tune | 0.934 | 33,417 |
| **LoRA (r=4)** | **0.934** | **2,889** |

**Trainable-parameter reduction: 91.4%** at matching accuracy. Adapter merges exactly (max logit diff 0.0e+00).

## Knowledge distillation

| Model | Accuracy | Params |
|---|---|---|
| Teacher (ensemble) | 1.000 | 38,331 |
| **Student + KD** | **1.000** | **2,137** |
| Student, no KD | 0.844 | 2,137 |

**KD accuracy recovery: 100.0%** of the teacher at **17.9x** compression (T=4.0, alpha=0.2). KD lifts the same-size student by +0.156.

## Eval / promotion gate

- Good model: val acc 0.998 -> **APPROVED**
- Deliberately-bad model: val acc 0.138 -> **BLOCKED**

## Visual Artifacts Reviewed

- `assets/01_kd_curves.png`: Knowledge distillation: dark knowledge beats hard labels.
- `assets/02_param_pareto.png`: Distillation efficiency (parameters vs accuracy).
- `assets/03_data_scaling.png`: Synthetic-data scaling.
- `assets/04_lora_and_gate.png`: LoRA parameter efficiency + promotion-gate scorecard.

## Engineering Notes

The primary design and scale decisions are documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md). The benchmark markdown in [`benchmarks/`](./benchmarks) and the generated figures in [`assets/`](./assets) should be read together: the markdown gives the measured numbers, and the screenshots make those results easier to inspect quickly during review.

## Files Included In This Repo

- [`README.md`](./README.md) for project overview, quickstart, and headline results
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) for system design and scaling choices
- [`benchmarks/`](./benchmarks) for measured results from the committed runs
- [`assets/`](./assets) for generated screenshots and dashboards
- [`tests/`](./tests) for the automated validation suite

## Delivery Summary

This project document was prepared for **Sai Veda** so the repository reads like a real project handoff: what the system is for, what problem it solves, what evidence supports it, and where the benchmark and test artifacts live inside the repo.
