# Void Models Research

Should a capital-constrained startup train its own small, specialized AI models, or build on frontier model APIs? This repository holds a research report and an interactive decision tool that answer that question for a student founder in India, using prices and benchmarks read on 23 October 2026.

**Live site:** https://void-models-research.vercel.app

## The verdict

- **Model-first company: decline.** Fine-tuned weights lose their edge within months, and revenue is likely a year away.
- **Software-first, verifier-centric workflow product: back in tranches**, with small-model training kept as an option that is funded only after a pre-registered experiment shows it pays.

Four numbers drive it:

| Figure | What it means |
| --- | --- |
| $0.10 / $0.50 per 1M tokens | Input / output price of the cheapest frontier tier (Claude Haiku 5.5, GPT-6 Luna), a 10x cut in one year |
| 7 months | How long open coding models of 32B or less took to match the frontier's SWE-bench Verified score |
| Rs 650 to Rs 1,700 | GPU cost of one LoRA fine-tuning run on an 8B model |
| Under Rs 1 lakh | Spend needed to reach the go/no-go decision on training |

## What is in the repository

| Path | Contents |
| --- | --- |
| `index.html` | The interactive report: 18 sections, calculators for training cost, evaluation cost, serving break-even, experiment sample size and unit economics, sortable and filterable tables, a benchmark chart and go/no-go gate checklists. One self-contained file. |
| `REPORT.md` | The full written report with every source, evidence grade and formula. |
| `analysis/training_estimates.py` | VRAM and GPU-time estimates behind the training cost tables. |
| `analysis/unit_economics.py` | Bottom-up unit economics for the four business-model scenarios. |

## Method

- Model prices for 468 models from the OpenRouter models API, and GPU prices from provider pages, all read on 23 October 2026. Currency conversion at Rs 95 per USD.
- Every claim carries an evidence grade: A (peer-reviewed or reproduced), B (technical report with released artifacts), C (company-reported), D (estimate or own calculation), S (speculation about closed pipelines).
- Costs such as VRAM, GPU-hours and unit economics are calculations with their formulas and assumptions shown, not measurements.
- The draft verdict was reviewed adversarially by two other model families, and the accepted corrections are listed in the report.

## How it was made

The research, calculations, writing and web page were produced with Claude, Anthropic's AI assistant, working from a research brief I wrote: the questions, constraints, budgets and decision criteria. Treat grade C and D figures as directional, and re-check prices before relying on them; they change often.

## Run it locally

Open `index.html` in any modern browser. No build step or server is needed.

```bash
python analysis/training_estimates.py
python analysis/unit_economics.py
```

## Next step

The report's "Experiment design" section specifies a fair, pre-registered test of whether a small specialized model can match a frontier model on one narrow task. Running that test is the planned follow-up.
