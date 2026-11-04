# Evaluation Report - Ml Graph Compiler

> Reproduce everything here with `make evaluation`. Report mean ± 95% confidence interval over N seeds.

## Setup
- **Task / dataset:** name, size, split.
- **Metrics:** definition of each metric and why it's the right one.
- **Graders:** exact-match / model-graded (which model, prompt) / human.
- **Baselines:** what you compare against.
- **Seeds:** list. **Hardware:** e.g., 1×A100 80GB. **Cost:** $X. **Runtime:** Yh.

## Headline results

| Metric | This project | Baseline | SOTA / ref | Δ vs baseline |
|---|---:|---:|---:|---:|
| Primary (↑) | **0.00 ± 0.00** | 0.00 | 0.00 | +0.00 |
| Latency p95 (ms, ↓) | **0** | 0 | - | - |
| Cost / 1k ($, ↓) | **0.00** | 0.00 | - | - |

## Ablations

| Variant | Metric | Δ | Notes |
|---|---:|---:|---|
| Full | 0.00 | - | reference |
| - component X | 0.00 | −0.00 | shows X matters |

## LLM-as-judge validation (if used)
- Agreement with human labels (Cohen's κ / accuracy), bias controls, sample size.

## Error analysis
- Top failure categories with counts and representative examples.

## Reproducibility
- Exact command, configuration hash, dataset version, W&B/MLflow run URL.
