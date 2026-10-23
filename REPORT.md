# Void Models — Deep Research Report and Investment Verdict

Oct 23, 2026 · Prepared for Yashwanth Cherukuru

## How to read this report

This report answers one question: should a capital-constrained founder in India spend scarce money training specialized model weights, or on something else? Every claim is tagged with how much it can be trusted, and every cost is shown as a formula you can re-run with your own numbers.

**Evidence grades used throughout**

| Grade | Meaning | How much weight to put on it |
| --- | --- | --- |
| A | Peer-reviewed paper, or result independently reproduced by others | Strong; still check it applies to your setting |
| B | Technical report or preprint from a lab with released weights, code or data | Good; numbers are self-reported but checkable |
| C | Company-reported benchmark or blog with no released artifacts | Directional only; treat as marketing until reproduced |
| D | Analyst estimate, survey, or my own calculation | Explicit assumptions shown; re-run them yourself |
| S | Speculation about proprietary pipelines | Labeled as such; never used as a load-bearing fact |

**Method.** Web research run on 23 October 2026 across papers, model cards, leaderboards, provider price pages and industry reports, deliberately searching for evidence against the Void Models hypothesis as well as for it. Live prices for 468 models were read from the OpenRouter models API on that date; GPU and fine-tuning prices from provider pages. Costs I calculated (VRAM, FLOPs, GPU-hours, unit economics) are grade D and show the formula and assumptions. A condensed draft was then sent for adversarial review to two other model families (GPT-6.1 Sol and Gemini 4 Argon via OpenRouter, total cost about $0.06); accepted corrections include the non-inferiority sample sizes and test scope in section 8, a privacy path through Gate 2, and margin caveats in section 10. Probabilities in section 15 are subjective and labeled as such.

**Limits.** Proprietary training pipelines (Claude, GPT, Gemini) are not public; anything said about them beyond their published reports is grade S. Benchmark scores are a moving target month to month; the conclusions here are built to survive a new model release, not to depend on one leaderboard snapshot.

## 1. Executive investment verdict

**Void Models is not viable as a model-first company, but it is viable as a software-first, verifier-centric vertical AI company that keeps small-model training as a gated option.** Training specialized weights is cheap. What is expensive and slow is selling them, and they lose value within months.

Five facts decide it:

1. **The cost case has collapsed.** Claude Haiku 5.5 and GPT-6 Luna list at $0.10 input / $0.50 output per 1M tokens. A year earlier, Haiku 4.5 listed at $1 / $5. Self-hosted small models now win on price only at high sustained volume, on-premises, or under tight latency.
2. **Weight-level leads expire.** Open coding models of 32B or less reached the frontier's SWE-bench score about 7 months after the frontier did (section 11).
3. **Compute is not the constraint.** An 8B LoRA SFT run costs ₹650 to ₹1,700, and a GRPO run ₹5,000 to ₹1 lakh. Data, evaluation and sales are the real costs.
4. **Small specialists win on narrow, checkable, stable tasks, mostly on cost-adjusted terms.** A strict accuracy win over today's best frontier system with equal tools is uncommon (subjective 15% to 25%) and least likely in coding.
5. **Buyers pay for outcomes, not models.** Standalone fine-tuning platforms were absorbed by larger companies (OpenPipe into CoreWeave, Predibase into Rubrik).

| Question | Answer |
| --- | --- |
| Technically feasible? | Yes for a cost-adjusted win on a narrow verifiable task (subjective 30% to 45% under strict rules); rarely for beating the frontier on accuracy |
| Financially viable as a model company? | No: low odds of revenue within a year; the asset depreciates in months |
| Financially viable as a vertical AI workflow company? | Yes, if priced on outcomes and kept narrow (section 10) |
| Capital needed to reach the training decision (Gate 2) | Under ₹1 lakh |
| Time to first revenue, recommended path | 1 to 3 months |

**Do next:** run 10+ buyer calls in one workflow with checkable outputs (Gate 0, two weeks, ₹10,000 cap). Then sell a concierge pilot built on APIs plus validators (Gate 1). Fund the section 8 experiment only on that customer's task.

## 2. How elite models actually become capable

Capability is mostly bought with data quality and compute, then *elicited* by post-training; architecture and parameter count set the ceiling but rarely explain the gap between two models of the same size. That single fact drives most of this report: a startup cannot win the pretraining race, but it can sometimes win the elicitation race on one narrow distribution.

### 2.1 The mechanics, in one pass

- **Tokenization.** Text becomes integer IDs via byte-pair encoding. This is not neutral: the same sentence can cost up to \~15x more tokens in some languages than English ([Petrov et al. 2023](https://arxiv.org/abs/2305.15425), grade A), which matters for Indian-language products because cost and context scale with tokens.
- **Residual stream.** Each token becomes a vector `x`; every layer *adds* to it rather than replacing it. Attention moves information between positions; the MLP transforms information at one position ([Elhage et al. 2021](https://transformer-circuits.pub/2021/framework/index.html), grade B).
- **Attention.** Routes information: "which earlier tokens matter for this one". Induction heads, a two-layer attention circuit that copies patterns seen earlier in context, appear abruptly in training and track the onset of in-context learning ([Olsson et al. 2022](https://arxiv.org/abs/2209.11895), grade B).
- **MLP layers.** Behave like key-value memories over patterns ([Geva et al. 2021](https://arxiv.org/abs/2012.14913), grade A); factual associations can be located and edited mostly in mid-layer MLPs ([Meng et al. 2022](https://arxiv.org/abs/2202.05262), grade A).
- **Normalization** (RMSNorm/LayerNorm) keeps activations in range so deep stacks train stably; it is an enabler, not a capability source.
- **Superposition.** Models store far more features than they have dimensions by overlapping them ([Elhage et al. 2022](https://arxiv.org/abs/2209.10652), grade B). This is why capacity is shared: training hard on one domain competes for the same directions other skills use, the mechanical root of forgetting.

```latex
h_{\ell+1} = h_\ell + \mathrm{Attn}_\ell(\mathrm{Norm}(h_\ell)) + \mathrm{MLP}_\ell\big(\mathrm{Norm}(h_\ell + \mathrm{Attn}_\ell(\mathrm{Norm}(h_\ell)))\big), \qquad \mathcal{L} = -\sum_t \log p_\theta(x_t \mid x_{<t})
```

### 2.2 Why next-token prediction produces skills

To predict the next token of a Python file well, a model must track scope, types, imports and what a function is meant to do; minimizing loss on enough code forces internal models of syntax and semantics. The same pressure on worked solutions yields arithmetic and algorithm-following, though less reliably than on text that is pattern-rich.

- **What parameters store.** Controlled experiments estimate roughly 2 bits of factual knowledge per parameter, independent of size, when facts are seen enough times ([Allen-Zhu and Li 2024](https://arxiv.org/abs/2404.05405), grade B). A 7B model therefore holds on the order of 14 Gbit of facts. That is ample for one domain's procedures, small for "all of medicine plus all of law".
- **Why specialization is possible.** A general model spends capacity on everything; a narrow task uses a small subset. Fine-tuning reallocates attention to that subset and fixes output format, but it cannot cheaply add broad reasoning the base never learned.
- **Why it is limited.** Fine-tuning on facts the base does not already know is learned slowly and *increases* hallucination ([Gekhman et al. 2024](https://arxiv.org/abs/2405.05904), grade A). For new knowledge, retrieval beats fine-tuning ([Ovadia et al. 2023](https://arxiv.org/abs/2312.05934), grade B).

### 2.3 Why parameter count alone does not determine quality

| Factor | Evidence | Size of effect |
| --- | --- | --- |
| Tokens trained on | Chinchilla compute-optimal is about 20 tokens per parameter ([Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)); modern small models are deliberately "over-trained" far past it, e.g. Qwen3 on \~36T tokens at every size ([Qwen3 report](https://arxiv.org/abs/2505.09388)) | Large: today's 8B beats 2023's 70B on most benchmarks |
| Data quality | phi-1: 1.3B params, 7B tokens of filtered and synthetic "textbook" code reached 50.6% HumanEval ([Gunasekar et al. 2023](https://arxiv.org/abs/2306.11644)); filtered web data (FineWeb-Edu, DCLM) beats raw web at equal compute ([Penedo et al. 2024](https://arxiv.org/abs/2406.17557), [Li et al. 2024](https://arxiv.org/abs/2406.11794)) | Large, and the cheapest lever a startup controls |
| Capability density over time | "Densing law": capability per parameter has been doubling roughly every few months ([Xiao et al. 2024](https://arxiv.org/abs/2412.04315), grade B) | Large, and it is the threat: next year's base model may beat this year's fine-tune |
| Architecture tweaks | SwiGLU, RoPE, GQA, MoE give constant-factor efficiency gains at fixed compute | Moderate; no evidence a startup's novel architecture beats a tuned transformer at small scale |
| Post-training | Same base, different post-training, large swings in usefulness ([InstructGPT](https://arxiv.org/abs/2203.02155): a 1.3B tuned model preferred over 175B untuned) | Large for usability; small for raw knowledge |

### 2.4 The training stages are not interchangeable

| Stage | What it optimizes | Typical data | Typical compute | What it really changes |
| --- | --- | --- | --- | --- |
| Pretraining from scratch | Next-token loss on everything | 10T to 40T tokens | 10^23 to 10^25 FLOPs | Knowledge, language, latent skills |
| Continued (domain-adaptive) pretraining | Same loss, domain corpus | 0.1B to 100B tokens | 10^19 to 10^22 | Domain vocabulary, style, some knowledge ([Gururangan et al. 2020](https://arxiv.org/abs/2004.10964)) |
| Supervised fine-tuning (SFT) | Loss on target responses only | 1k to 1M examples | 10^16 to 10^20 | Format, task framing, imitation of a teacher; 1,000 curated examples can suffice for style ([LIMA](https://arxiv.org/abs/2305.11206)) |
| Preference optimization (RLHF, DPO) | Prefer chosen over rejected outputs | 10k to 1M pairs | Similar to SFT | Tone, helpfulness, refusal behaviour ([DPO](https://arxiv.org/abs/2305.18290)) |
| RL with verifiable rewards (RLVR) | Expected reward from a checker | Prompts plus a verifier | Dominated by sampling | Reliability on checkable tasks, longer reasoning ([DeepSeek-R1](https://arxiv.org/abs/2501.12948)) |

### 2.5 How each capability is built, measured and improved

| Capability | Mainly developed by | Measured by | Improved most by |
| --- | --- | --- | --- |
| Knowledge | Pretraining scale | Closed-book QA, held-out facts | Retrieval, not fine-tuning |
| Instruction following | SFT plus preference tuning | IFEval-style constraint checks, human preference | Diverse, high-quality instruction data |
| Reasoning (math, code) | Pretraining on reasoning-rich text, then RLVR | pass@1 on fresh, time-split problems | RLVR, distillation of long reasoning traces, test-time sampling |
| Tool use | SFT on tool traces, then RL in environments | Task success in a sandbox (tau-bench, Terminal-Bench) | Realistic environments plus reward |
| Unseen-problem solving | All of the above | Performance on post-cutoff, private tasks | Breadth of base model; you cannot fine-tune it in cheaply |

### 2.6 Telling generalization from memorization and contamination

Benchmark scores overstate ability whenever test items leaked into training data. Fresh GSM8K-style problems (GSM1k) caused drops of up to \~13 points for some model families ([Zhang et al. 2024](https://arxiv.org/abs/2405.00332), grade A). On SWE-bench Verified, models identified the buggy file path from the issue text alone far more often than on unseen repositories, a memorization signal ([Liang et al. 2025](https://arxiv.org/abs/2506.12286), grade B).

Practical tests you should run on any model you train or compare:

1. **Time split.** Evaluate only on items created after every model's training cutoff, as LiveCodeBench does ([Jain et al. 2024](https://arxiv.org/abs/2403.07974)).
2. **Perturbation.** Rename identifiers, reorder clauses, change constants; a memorizer degrades sharply, a generalizer does not.
3. **Private holdout.** Keep a test set no model or teacher has ever seen, never sent to an API used for data generation.
4. **Overlap audit.** n-gram and embedding overlap between training data and tests; drop near-duplicates.
5. **Out-of-distribution slice.** A second customer, framework version or repository the model never trained on.

### 2.7 What has strong evidence, and what does not

**Strong evidence:** more and better data; compute; RL with verifiable rewards on checkable tasks; distillation from a stronger model; repeated sampling plus a verifier ([Brown et al. 2024](https://arxiv.org/abs/2407.21787)); retrieval for knowledge.

**Weak or contested:**

- "Emergent abilities are sudden phase changes": much of the sharpness is an artifact of all-or-nothing metrics ([Schaeffer et al. 2023](https://arxiv.org/abs/2304.15004)).
- "Chain-of-thought shows the model's true reasoning": explanations can be unfaithful ([Turpin et al. 2023](https://arxiv.org/abs/2305.04388)).
- "RL teaches fundamentally new skills": contested. At large k, base models often solve as many problems as RL-tuned ones ([Yue et al. 2025](https://arxiv.org/abs/2504.13837)); prolonged RL can expand the solvable set ([Liu et al. 2025, ProRL](https://arxiv.org/abs/2505.24864)); and random rewards improved one model family on math, so some "RL gains" are elicitation of pre-existing behaviour ([Shao et al. 2025](https://arxiv.org/abs/2506.10947)).
- "A long context window means the model uses it": accuracy drops for information in the middle of long inputs ([Liu et al. 2023](https://arxiv.org/abs/2307.03172)).

## 3. How elite coding models are actually built

Elite coding ability now comes from reinforcement learning in executable environments on top of very large, code-heavy pretraining, not from clever fine-tuning of small models. A small model can beat a frontier model only on narrow, verifiable, high-volume micro-tasks, and even then the usual win is equal accuracy at a fraction of the cost and latency, not higher accuracy.

### 3.1 Where the frontier is (October 2026)

| Model | SWE-bench Verified | Size | Weights |
| --- | --- | --- | --- |
| Claude Fable 5 | 95.0% | Not disclosed | Closed |
| Claude Opus 4.8 | 88.6% | Not disclosed | Closed |
| Reflection Beam 1.0 | 81.6% | 501B | Open |
| DeepSeek-V4-Pro-Max | 80.6% | 1.6T | Open |
| Qwen3.6-27B | 77.2% | 28B dense | Open, Apache-2.0 |
| Meta Muse Glimmer-30B | 76.0% | 30B | Open |
| Devstral Small 2 | \~68% (vendor) | 24B | Open, Apache-2.0 |

Source: [llm-stats leaderboard](https://llm-stats.com/benchmarks/swe-bench-verified), page dated 23 October 2026; scores are mostly vendor-reported with different scaffolds (grade C). Two cautions matter more than the ranking. First, Verified is saturating: across 254 submissions, exact paired tests could not separate any of the 29 adjacent top-thirty pairs, and scaffold choice moved one model by 29.8 points versus an 8.8-point spread across the top thirty ([Liu et al. 2026](https://arxiv.org/abs/2609.17394), grade B). Second, memorization inflates it (section 2.6). On the harder, less contaminated SWE-bench Pro commercial subset, no model exceeded \~51.5% in a September 2026 snapshot ([Morph tracker](https://www.morphllm.com/swe-bench-pro), grade C; original benchmark [Deng et al. 2025](https://arxiv.org/abs/2509.16941)).

The useful reading: a dense \~27B open model now sits roughly 10 to 18 points behind the best closed systems on repository-level repair, and the gap widens on harder, fresher tasks.

### 3.2 The pipeline, separating verified facts from speculation

1. **Collect code legally.** Permissively licensed GitHub code with opt-out, deduplicated: The Stack v2 covers 619 languages and was used for StarCoder2 ([Lozhkov et al. 2024](https://arxiv.org/abs/2402.19173), grade B). Licensing risk is real: copyleft and unlicensed code, secrets and PII must be filtered; litigation over training on public code is ongoing in several jurisdictions.
2. **Filter and deduplicate aggressively.** OpenCoder's fully open ablations showed file-level dedup and quality filtering matter more than raw volume ([Huang et al. 2024](https://arxiv.org/abs/2411.04905), grade B). phi-1 showed curated "textbook" code beats 100x more raw code at small scale ([Gunasekar et al. 2023](https://arxiv.org/abs/2306.11644)).
3. **Pretrain or continue pretraining at enormous scale.** Qwen2.5-Coder used \~5.5T code-related tokens ([Hui et al. 2024](https://arxiv.org/abs/2409.12186)); DeepSeek-Coder-V2 continued from a general checkpoint with 6T more tokens ([DeepSeek-AI 2024](https://arxiv.org/abs/2406.11931)). Repository-level packing (files ordered by dependency) and fill-in-the-middle objectives teach cross-file reasoning and editing.
4. **Synthetic instruction data.** OSS-Instruct seeds problems from real code snippets and measurably improves small models ([Wei et al. 2023](https://arxiv.org/abs/2312.02120), grade B). Synthetic data helps most when it is *executed and filtered*; unverified synthetic code mostly teaches the teacher's mistakes.
5. **Execution feedback and RL.** Rewards from unit tests and compilers: early CodeRL ([Le et al. 2022](https://arxiv.org/abs/2207.01780)), then large-scale RLVR. DeepCoder-14B reached 60.6% on LiveCodeBench, near o3-mini-low, trained with RL on \~24k verified problems using 32 H100s for 2.5 weeks ([Agentica/Together 2025](https://www.together.ai/blog/deepcoder), grade B).
6. **Agentic, repository-level RL.** Models learn to navigate repos, run tests and edit across files in sandboxes. Open recipes: SWE-Gym ([Pan et al. 2024](https://arxiv.org/abs/2412.21139)), SWE-smith's 50k synthesized task instances ([Yang et al. 2025](https://arxiv.org/abs/2504.21798)), R2E-Gym ([Jain et al. 2025](https://arxiv.org/abs/2504.07164)), Meta's SWE-RL ([Wei et al. 2025](https://arxiv.org/abs/2502.18449)). Qwen3-Coder's team described running \~20,000 parallel environments for agent RL (Qwen blog, July 2025, grade C).
7. **What frontier labs do beyond this (grade S).** Claude, GPT and Gemini pipelines are not public. Their reports and system cards confirm RL on real-world coding tasks in many environments; scale of environments, reward models and human data is speculation.

A telling industry data point: Cursor's Composer 2 (March 2026) was built on an open-weight base, Kimi K2.5 (\~1T parameters), with continued pretraining and a large RL scale-up; Cursor said roughly three-quarters of total compute was its own ([DataCamp summary](https://www.datacamp.com/blog/composer-2), grade C). Even a richly funded coding company specializes a *very large* open base rather than pretraining or using a 7B model.

### 3.3 Why benchmark wins do not equal developer productivity

In a randomized trial, 16 experienced open-source developers took 19% *longer* with early-2025 AI tools while believing they were \~20% faster ([METR 2025](https://arxiv.org/abs/2507.09089), grade A). METR's 2026 follow-up was judged unreliable because developers increasingly refused to work without AI, biasing results; METR believes speedups are now larger but cannot size them ([METR, 2026](https://metr.org/), grade B). Reasons benchmarks mislead: tasks are pre-specified with tests, real work is under-specified; benchmarks reward a single patch, real work rewards maintainability and review cost; scaffolding and repo context dominate outcomes.

### 3.4 Can a small specialist beat the frontier in a coding domain?

| Domain | Verifier available | Beatable on accuracy? | Beatable on cost/latency at equal accuracy? | Verdict |
| --- | --- | --- | --- | --- |
| Compiler and build error repair (one toolchain) | Strong: compiles, tests pass | Plausible | Yes | Best technical candidate |
| Proprietary DSL or internal framework code | Compiler, simulator, customer tests | Yes, if customer data exists | Yes | Best commercial candidate, but needs customer data |
| Narrow API migration (one library version jump) | Tests, type-check | Plausible | Yes | Good, competes with codemods (OpenRewrite) and frontier agents |
| Text-to-SQL on one known schema | Execution result match | Plausible with schema data | Yes | Good; frontier plus schema retrieval is a strong baseline |
| Test generation | Compiles, passes, mutation score, coverage | Unlikely | Yes, inside an execution loop | System beats weights; Meta's TestGen-LLM shows the filter pattern ([Alshahwan et al. 2024](https://arxiv.org/abs/2402.09171)) |
| PostgreSQL query optimization | EXPLAIN cost, runtime, result equivalence | Unlikely | Partly | Tool loop matters more than weights |
| Fast apply / next-edit prediction | Diff applies, matches target | Yes, proven | Yes, proven | Real but owned by IDE vendors (Cursor Tab, Morph: \~10,500 tok/s claimed, [Morph](https://www.morphllm.com/comparisons), grade C) |
| Security-focused code review | Weak: no ground truth for missed bugs | No | Only as a cheap pre-filter | Pair frontier with Semgrep/CodeQL instead |
| Django, FastAPI, Java debugging, React/Next.js feature work | Partial: tests exist but tasks are open-ended | No | No | Frontier wins; do not train for this |

**What makes a comparison fair** (section 8 operationalizes this): same repository context and retrieval, same tools (compiler, test runner) for both sides, same attempt budget (pass@1 versus pass@1, or equal cost), a contamination-free test set built after every model's cutoff, and current frontier models at reasonable effort settings, not last year's or a crippled prompt. Beating GPT-4o, a zero-shot prompt, or a model denied tools proves nothing about today's frontier.

**What is unlikely to work:** training a small model to be "the best Django model" or "the best React model". Frameworks are well covered in frontier pretraining, tasks are open-ended, and frontier vendors improve on exactly these every quarter.

## 4. Training-method comparison: what works, when, and why

For a startup on a single rented GPU, the winning stack is data work plus cheap weight updates: verifier-filtered distillation into a LoRA adapter, optionally followed by small-scale RL with a verifiable reward, wrapped in retrieval and tools. Pretraining from scratch and continued pretraining are the methods most likely to burn money without beating a newer open base model.

### 4.1 The memory and compute arithmetic behind every row below

Training memory has five parts: weights, gradients, optimizer states, activations, and temporary buffers. For P parameters with AdamW in mixed precision:

```latex
\text{Full FT: } \underbrace{2P}_{\text{bf16 weights}} + \underbrace{2P}_{\text{bf16 grads}} + \underbrace{4P}_{\text{fp32 master}} + \underbrace{8P}_{\text{Adam } m, v} = 16P \text{ bytes} \;\;(\div N \text{ GPUs with ZeRO-3/FSDP})
```

```latex
\text{LoRA: } 2P + 16\,P_{\text{LoRA}} + \text{act}, \qquad \text{QLoRA: } \approx 0.52P + 16\,P_{\text{LoRA}} + \text{act}, \qquad P_{\text{LoRA}} = r\,(d_{in} + d_{out}) \text{ per adapted matrix}
```

Activations per layer without recomputation are about s·b·h·34 bytes once FlashAttention removes the quadratic attention term ([Korthikanti et al. 2022](https://arxiv.org/abs/2205.05198)). For an 8B-class model (h = 4096, 36 layers) at 4,096 tokens and batch 1, that is \~20 GB; with activation checkpointing it drops to \~1.2 GB of saved layer inputs plus one layer recomputed, at the price of \~33% more compute.

```latex
C_{\text{full}} \approx 6ND\;(8ND \text{ with checkpointing}), \qquad C_{\text{LoRA}} \approx 4ND\;(6ND), \qquad t = \frac{C}{\text{peak FLOP/s} \times \text{MFU}}
```

N = parameters, D = training tokens. LoRA skips weight gradients for frozen weights but still back-propagates through them, so it saves roughly a third of compute, not 99%. Worked example (grade D): 90M training tokens of LoRA SFT on an 8B model is \~4.3×10^18 FLOPs, about 3.5 hours on one H100 at an assumed 35% MFU, or \~21 hours on one RTX 4090. The GPU bill for a typical SFT run is single-digit dollars; iteration, data and evaluation are where money goes.

Unsloth's measured minimums for adapter training (batch 1, short context): QLoRA 6 GB / LoRA 22 GB at 8B; 8.5 / 33 GB at 14B; 22 / 64 GB at 27B; 26 / 76 GB at 32B ([Unsloth requirements](https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements), grade B). Longer contexts and larger batches need more.

### 4.2 Ranking table

Costs are indicative for an 8B-class base unless noted, using October 2026 prices: Runpod community H100 SXM $2.69/hr, Fireworks LoRA SFT $0.50 per 1M training tokens (≤16B), Tinker Qwen3-8B $0.44 train / $0.60 sample per 1M tokens. ₹ at ₹95/USD.

| Method | Typical cost (USD) | Complexity | Quality potential on a narrow task | Evidence | Startup suitability |
| --- | --- | --- | --- | --- | --- |
| Retrieval (RAG) + structured context | $0 to $200 setup | Low | High for knowledge tasks | A | Excellent, do first |
| Tools + execution + verifier (no training) | $0 to $500 | Medium | High for checkable tasks | A | Excellent, do first |
| Test-time scaling: best-of-N with verifier | Per-query cost x N | Low | High where a verifier exists | A | Excellent |
| Hybrid routing to a frontier fallback | Per-query | Medium | Highest system reliability | B | Excellent |
| Teacher-distilled SFT via LoRA/QLoRA | $10 to $300 compute + $50 to $2k teacher data | Medium | High for format, style, narrow skills | A/B | Excellent |
| Rejection-sampling fine-tuning (RFT/STaR) | 2x to 5x SFT | Medium | High where a verifier exists | A | Very good |
| RL with verifiable rewards (GRPO), LoRA | $50 to $2,000 per run | High | High on checkable tasks; contested beyond elicitation | B | Good, after SFT works |
| DPO / preference optimization | $20 to $500 + labeling | Medium | Moderate (tone, preferences) | A | Moderate |
| Model merging (TIES, DARE) | \~$0 compute | Low | Low to moderate, unpredictable | B | Cheap experiment only |
| Full-parameter SFT | 5x to 20x LoRA (multi-GPU) | High | Slightly higher than LoRA on large datasets | A | Rarely justified early |
| Continued pretraining | $150 to $2k per 1B to 10B tokens (8B) | High | Moderate; only for genuinely new vocabulary/corpora | A | Poor early; maybe later |
| On-policy distillation (logit-level, open teacher) | Teacher inference on student samples | High | High, sample-efficient | B | Good once you self-host a teacher |
| Mixture-of-experts training from scratch | Pretraining-scale | Very high | Not reachable | A | No |
| Pretraining from scratch | $11k+ for 1B on 1T tokens; \~$200k for 7B on 2T | Very high | Below existing open bases | A | No |

### 4.3 Method cards

**Pretraining from scratch.** Minimizes next-token loss over trillions of tokens; updates all weights. It builds knowledge and latent skills and nothing else does that. A 1B model on 1T tokens needs about 6×10^21 FLOPs, roughly 4,200 H100-hours (about $11k to $17k, ₹10 to 16 lakh) at 40% MFU, and would still trail open models of the same size trained on about 36T tokens ([Qwen3 report](https://arxiv.org/abs/2505.09388)). Failure mode: you build a worse copy of something free. Suitability: none.

**Continued / domain-adaptive pretraining (DAPT).** Same loss on a domain corpus; updates all weights (or high-rank adapters). Helps when the domain has vocabulary and patterns absent from general data, e.g. biomedical text or a proprietary language ([Gururangan et al. 2020](https://arxiv.org/abs/2004.10964)). Risks: forgetting general and instruction-following ability unless you replay general data and re-do instruction tuning; mitigations include learning-rate re-warming and replay ([Ibrahim et al. 2024](https://arxiv.org/abs/2403.08763)). Needs 100M+ clean domain tokens to matter. Cost: an 8B model on 1B tokens is about 6.4×10^19 FLOPs, about 45 H100-hours, or $120 to $180. Evidence A for domain-text tasks; weak for "make it smarter".

**Full-parameter SFT.** Cross-entropy on responses only; updates all weights. Needs 16 bytes/param plus activations, so an 8B model needs \~2 to 4 x 80 GB GPUs with FSDP. Gains over LoRA appear mainly with large datasets ([Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/), grade B). Forgets more than LoRA ([Biderman et al. 2024](https://arxiv.org/abs/2405.09673), grade A).

**LoRA.** Freezes W and learns a low-rank update ΔW = BA with rank r; only A and B (typically 0.1% to 2% of parameters) are trained, not every weight. Matches full fine-tuning on small-to-medium SFT datasets and in RL even at very low rank, but underperforms on large datasets; apply it to all layers including MLPs, and use \~10x the full-FT learning rate ([Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/), grade B; [Hu et al. 2021](https://arxiv.org/abs/2106.09685), grade A). LoRA "learns less and forgets less" than full FT on code and math ([Biderman et al. 2024](https://arxiv.org/abs/2405.09673)). Fits one 24 GB GPU at 8B in bf16 with checkpointing.

**QLoRA.** Base weights quantized to 4-bit NF4 and frozen; LoRA adapters trained in bf16 through the dequantized weights. Enabled a 65B fine-tune on one 48 GB GPU ([Dettmers et al. 2023](https://arxiv.org/abs/2305.14314), grade A). Slightly slower per step than LoRA and a small quality cost in some settings; ideal on 16 to 24 GB consumer cards.

**Other PEFT (DoRA, IA3, prefix/prompt tuning, GaLore).** DoRA decomposes magnitude and direction and sometimes edges LoRA ([Liu et al. 2024](https://arxiv.org/abs/2402.09353)); prompt tuning is much weaker at small scale; GaLore projects gradients to cut optimizer memory for full-rank training ([Zhao et al. 2024](https://arxiv.org/abs/2403.03507)). None changes the strategic picture; LoRA is the default.

**Knowledge distillation (logit-level).** Student minimizes KL divergence to the teacher's token distribution ([Hinton et al. 2015](https://arxiv.org/abs/1503.02531)). Richer signal than hard labels, but requires the teacher's logits, so the teacher must be an open model you run. On-policy variants train on the student's own samples scored by the teacher, fixing exposure bias and improving sample efficiency ([Agarwal et al. 2024, GKD](https://arxiv.org/abs/2306.13649), grade B).

**Sequence-level distillation / teacher-generated synthetic data.** Train on teacher outputs as targets ([Kim and Rush 2016](https://arxiv.org/abs/1606.07947)). DeepSeek-R1's distilled 14B and 32B Qwen models beat o1-mini on AIME 2024 from \~800k teacher samples ([DeepSeek-AI 2025](https://arxiv.org/abs/2501.12948), grade A). **Legal constraint:** Anthropic's terms forbid using outputs to train models that compete with Anthropic's, listing "using Outputs as training targets" as an example ([Anthropic, March 2026](https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model)); OpenAI's terms have a similar no-competing-models clause. A model you sell as a coding or general assistant is plausibly "competing". Use permissively licensed open teachers (check each model's license file) or the customer's own data.

**Rejection sampling and best-of-N fine-tuning.** Sample N candidates, keep those a verifier accepts, fine-tune on them; iterate (STaR, [Zelikman et al. 2022](https://arxiv.org/abs/2203.14465); RFT, [Yuan et al. 2023](https://arxiv.org/abs/2308.01825)). This is the single most cost-effective method when a verifier exists, because the verifier converts cheap samples into clean data.

**DPO and relatives (IPO, KTO, ORPO, SimPO).** Optimize a closed-form objective so chosen responses become more likely than rejected ones relative to a reference model ([Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)). Good for tone, preference and refusal behaviour; weak for correctness unless pairs come from a verifier. Prone to length exploitation and over-optimization.

**RL with verifiable rewards; PPO and GRPO.** PPO uses a learned value function ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)); GRPO drops it, normalizing each sample's reward against a group of samples for the same prompt ([Shao et al. 2024](https://arxiv.org/abs/2402.03300)), with fixes in DAPO ([Yu et al. 2025](https://arxiv.org/abs/2503.14476)) and Dr. GRPO ([Liu et al. 2025](https://arxiv.org/abs/2503.20783)).

```latex
\hat A_i = \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})}, \qquad \mathcal{J} = \mathbb{E}\Big[\tfrac{1}{G}\textstyle\sum_i \min\big(\rho_i \hat A_i,\ \mathrm{clip}(\rho_i, 1\pm\epsilon)\hat A_i\big)\Big] - \beta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})
```

Cost is dominated by sampling. Worked example on Tinker (grade D): 300 steps x 64 prompts x 8 samples x 3k tokens = \~460M sampled tokens; on Qwen3-8B at $0.60 sample + $0.44 train per 1M that is \~$480 (₹46k); on Qwen3.5-9B \~$1,040 ([Tinker pricing, 19 Oct 2026](https://thinkingmachines.ai/tinker/pricing/)). Failure modes: reward hacking (tests that pass trivially), entropy collapse, gains that are elicitation rather than new skill ([Yue et al. 2025](https://arxiv.org/abs/2504.13837)), and spurious-reward effects on some model families ([Shao et al. 2025](https://arxiv.org/abs/2506.10947)). RL forgets less than SFT at matched gains ([Shenfeld et al. 2025](https://arxiv.org/abs/2509.04259), grade B).

**Curriculum learning and multi-stage training.** Order data easy to hard, or run SFT then RL. Multi-stage (SFT for format, RL for reliability) is standard and well-supported; fine-grained curricula show inconsistent gains. Cheap to try once the pipeline exists.

**Data-mixture optimization.** Choose domain weights to minimize loss on a target (DoReMi, [Xie et al. 2023](https://arxiv.org/abs/2305.10429); RegMix, [Liu et al. 2024](https://arxiv.org/abs/2407.01492)). In fine-tuning, the practical version is: always mix 10% to 30% general instruction data to limit forgetting, then tune the ratio against your eval.

**Self-training / iterative improvement.** Model generates, verifier filters, model retrains, repeat. Works with a strong external verifier; without one, models amplify their own errors and diversity collapses.

**Model merging.** Average or combine fine-tuned weight deltas (task arithmetic, [Ilharco et al. 2022](https://arxiv.org/abs/2212.04089); TIES, [Yadav et al. 2023](https://arxiv.org/abs/2306.01708); DARE, [Yu et al. 2023](https://arxiv.org/abs/2311.03099)). Zero compute, useful for combining a domain adapter with general skills; results are unpredictable and must be re-evaluated every time.

**Mixture-of-experts.** Routes each token to a few expert MLPs so total parameters grow without per-token compute ([Fedus et al. 2021](https://arxiv.org/abs/2101.03961); [Mixtral](https://arxiv.org/abs/2401.04088)). For you this is a *base-model choice*, not a training project: MoE bases like Qwen3.6-35B-A3B serve fast because only \~3B parameters are active, but fine-tuning them needs memory for all experts.

**Retrieval-augmented generation.** Injects knowledge at inference ([Lewis et al. 2020](https://arxiv.org/abs/2005.11401)); beats fine-tuning for adding facts ([Ovadia et al. 2023](https://arxiv.org/abs/2312.05934)). Updates no weights; cost is engineering plus a vector or keyword index.

**Tool-augmented models.** The model calls compilers, databases, search, calculators ([Toolformer](https://arxiv.org/abs/2302.04761), [ReAct](https://arxiv.org/abs/2210.03629)). In coding, the scaffold can move results by tens of points (section 3.1), more than most fine-tunes.

**Test-time scaling and search.** Spend more inference compute per query: repeated sampling raised SWE-bench Lite coverage from 15.9% with one sample to 56% with 250 ([Brown et al. 2024](https://arxiv.org/abs/2407.21787), grade A); compute-optimal test-time scaling can let a smaller model match a \~14x larger one on problems of moderate difficulty ([Snell et al. 2024](https://arxiv.org/abs/2408.03314), grade B). The catch: coverage gains only become accuracy if a verifier can pick the right sample.

**Small model plus verifier, and hybrid systems.** A small model proposes; tests, compilers, schema validators or a judge accept or reject; failures escalate to a frontier model. This is where small models earn their keep, and it is how you separate weight gains from system gains (section 12). NVIDIA argues many agent sub-calls are repetitive enough for small models ([Belcak et al. 2025](https://arxiv.org/abs/2506.02153), position paper, grade C).

**How methods combine.** Strong combinations: RAG + SFT (format) + verifier; rejection-sampled SFT then GRPO on the same verifier; LoRA adapters merged or hot-swapped per customer. Weak combinations: DAPT then little instruction data (breaks chat ability); DPO on unverified pairs; RL without a held-out eval (reward hacking goes unseen).

## 5. Specialized-model performance limits

On a narrow task, post-training mostly converts what a base model *can sometimes do* into what it *reliably does*; it rarely adds what the base cannot do at all. The single most predictive measurement is the base model's pass@k on your task with a verifier: if the base never solves an item in 64 tries, cheap training will almost never make it solve that item on the first try.

### 5.1 What is demonstrated, plausible, or unsupported

| Question | Demonstrated | Plausible | Unsupported |
| --- | --- | --- | --- |
| Does capacity limit a specialist? | Knowledge storage is bounded (about 2 bits/param, [Allen-Zhu and Li 2024](https://arxiv.org/abs/2404.05405)); small models trail badly on long-horizon agentic repair (section 3.1) | A narrow procedure (schema extraction, one compiler's errors) needs a small fraction of a 7B model's capacity | A 3B to 14B model matching frontier *breadth* inside a whole profession |
| Can training recover missing abilities? | Format, schema adherence, domain style, and transferred reasoning traces from a stronger teacher ([DeepSeek-R1 distills](https://arxiv.org/abs/2501.12948)) | Domain procedures the base saw rarely but did see | Injecting large new factual knowledge by SFT; it raises hallucination ([Gekhman et al. 2024](https://arxiv.org/abs/2405.05904)) |
| Can LoRA approach full fine-tuning? | Yes for RL and small-to-medium SFT; gap on large datasets ([Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) | Gap is small below about 100M training tokens | LoRA equals full FT for continued pretraining on billions of tokens ([Biderman et al. 2024](https://arxiv.org/abs/2405.09673) shows it does not) |
| When is continued pretraining necessary? | New vocabulary or script, proprietary languages, domain corpora with 100M+ tokens ([Gururangan et al. 2020](https://arxiv.org/abs/2004.10964)) | Low-resource Indian languages; a customer's internal DSL | For a well-covered framework such as Django or React |
| Does RL create new strategies? | It reliably raises pass@1 and response length on checkable tasks | Long RL can expand the solvable set ([ProRL](https://arxiv.org/abs/2505.24864)) | That a few hundred GRPO steps on a 7B model teach abilities absent from its pass@64; at large k, base models often match RL models ([Yue et al. 2025](https://arxiv.org/abs/2504.13837)) |
| Can verifiers and tools compensate for capacity? | Yes: repeated sampling plus a checker multiplies solved problems ([Brown et al. 2024](https://arxiv.org/abs/2407.21787)); scaffolds move coding scores by up to about 30 points ([Liu et al. 2026](https://arxiv.org/abs/2609.17394)) | A 14B model plus strong verifier matching a frontier model on a checkable task at lower cost | Tools rescuing a model on tasks with no checkable output |
| Does inference-time search change the ceiling? | A smaller model with compute-optimal test-time search can beat a roughly 14x larger one on easy and medium problems, not on the hardest ([Snell et al. 2024](https://arxiv.org/abs/2408.03314)) | Best-of-8 with a verifier closes most of the gap on routine tasks | Search fixing hard reasoning failures |
| Forgetting and negative transfer | Fine-tuning erodes general skills ([Luo et al. 2023](https://arxiv.org/abs/2308.08747)); LoRA and RL forget less ([Biderman et al. 2024](https://arxiv.org/abs/2405.09673), [Shenfeld et al. 2025](https://arxiv.org/abs/2509.04259)) | Mixing 10% to 30% general data limits damage | Forgetting being irrelevant because "we only need one task": customers send off-distribution inputs |
| Diversity versus specialization | Narrowly tuned models learn shortcuts: a LoRA-tuned 3B classifier was the most keyword-dependent system tested, while the frontier zero-shot model was the least ([Huynh and Pham 2026](https://arxiv.org/abs/2609.20847)) | Diverse paraphrased training data reduces shortcutting | That high in-distribution accuracy predicts production accuracy |
| Diminishing returns | SFT accuracy grows roughly log-linearly in data size on reasoning tasks ([Yuan et al. 2023](https://arxiv.org/abs/2308.01825)) | Each doubling of clean data buys a roughly constant number of points until a plateau | Linear returns to more data or compute |

### 5.2 Where specialization clearly works, with numbers

- A fine-tuned Qwen3.5-27B scored 95.0% on a 51-subcategory security-document taxonomy versus 75.4% to 79.9% for unnamed commercial models under the same prompt, trained on 78,358 examples ([arXiv 2605.20368](https://arxiv.org/abs/2605.20368), May 2026, grade C because comparators and prompts are not disclosed). The pattern: a fixed, fine-grained label scheme the frontier model was never told in enough detail.
- 310 LoRA fine-tunes across 31 tasks beat their bases by 34 points and GPT-4 by 10 points on average ([Zhao et al. 2024, LoRA Land](https://arxiv.org/abs/2405.00732), grade B). These were classic NLP tasks against a 2023-era model; frontier zero-shot today would close much of that gap.
- Counter-evidence that must be weighed equally: GPT-4 with prompting alone beat the specialized Med-PaLM 2 on MedQA, 90.2% versus 86.5% ([Nori et al. 2023](https://arxiv.org/abs/2311.16452), grade B), and GPT-4 outperformed the domain-pretrained BloombergGPT on most financial tasks ([Li et al. 2023](https://arxiv.org/abs/2305.05862), grade B).

The rule that reconciles these: specialization wins when the target is a **narrow, stable, labelable decision space with abundant in-distribution examples**; generalists win when the task needs **broad knowledge, open-ended judgment, or adapts faster than you can collect labels**.

### 5.3 A pipeline that approaches the useful ceiling

1. **Freeze three test sets first:** in-distribution test, private holdout never sent to any API, and an out-of-distribution slice (new customer, new library version).
2. **Measure baselines:** base model prompted; current frontier prompted with the same context; your no-training system (retrieval + tools + verifier).
3. **Measure the base's pass@k curve** for k = 1, 4, 16, 64. The gap between pass@1 and pass@64 is your realistic headroom for training.
4. **SFT via LoRA** on verifier-filtered data distilled from a permissively licensed teacher, at 1k, 2k, 4k, 8k, 16k examples, to see the data-scaling slope.
5. **Rejection-sampling rounds** on the student's own outputs (on-policy data), one to three rounds.
6. **GRPO with the same verifier** only if pass@16 is well above pass@1 after SFT.
7. **Test-time layer:** best-of-N with the verifier, then escalation to a frontier model on failure or low confidence.
8. **Regression checks** on general skills and the OOD slice after every stage.

### 5.4 Evidence that more training is no longer worth paying for

Stop investing in weights when any two of these hold:

- The last doubling of clean training data added less than 1 point on the private holdout, inside the confidence interval.
- pass@1 is within 2 to 3 points of the base's pass@16: RL headroom is used up.
- Dev-set gains no longer show up on the OOD slice.
- A newly released open base of the same size matches your fine-tune zero-shot or with a prompt.
- The cost per point exceeds the value per point:

```latex
\text{Monthly value of +1 point} = V \times 0.01 \times c_{\text{fail}}, \qquad \text{payback (months)} = \frac{\text{experiment cost}}{V \times 0.01 \times c_{\text{fail}}}
```

V is monthly task volume and c\_fail the cost of a failure (an escalation to a frontier model, a human review, or a lost customer). At 100,000 tasks a month and a $0.05 escalation cost, a point is worth $50 a month; a $500 experiment that buys one point pays back in 10 months, which is usually too slow for a startup.

## 6. Most promising base models (October 2026)

Start from a permissively licensed Qwen3.5/3.6 model at 9B or 27B, with Gemma 4, Granite 4.2 and Meta Muse Glimmer as alternates; the license and your eval should decide, not leaderboard rank. Most published numbers below are vendor-reported (grade C); differences under 3 to 5 points are noise on most benchmarks.

| Model | Total / active params | License | Released | Reported evidence | Adapter training on one GPU | Hosted price, $ per 1M in/out |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | 4B dense | Apache-2.0 | Feb 2026 | Family release notes ([guide](https://www.verdent.ai/guides/model/qwen-3-5)) | QLoRA on 8 GB laptop GPU | Tinker train $0.74 |
| Gemma 4 E4B | Small, on-device | Apache-2.0 | 2026 | Built for phones and edge ([The Decoder](https://the-decoder.com/googles-gemma-4-is-now-available-with-apache-2-0-licensing-for-the-first-time/)) | QLoRA on 8 GB | Not listed |
| IBM Granite 4.2 8B | 8B | Apache-2.0 | 25 Aug 2026 | Agentic RL applied ([Crypto Briefing](https://cryptobriefing.com/ibm-granite-4-2-models-local-deployment/)) | QLoRA 6 GB / LoRA 22 GB | 0.06 / 0.25 |
| Ministral 3 8B / 14B | 8B / 14B | Apache-2.0 | Dec 2025 | Mistral release | LoRA on 24 to 40 GB | 0.15 / 0.15 and 0.20 / 0.20 |
| Qwen3.5-9B | 9B dense | Apache-2.0 | Feb 2026 | Family release notes | QLoRA 6.5 GB / LoRA 24 GB | 0.17 / 0.25 |
| gpt-oss-20b | 21B / 3.6B MoE | Apache-2.0 | Aug 2025 | OpenAI model card | QLoRA on 24 GB | 0.018 / 0.09 |
| Devstral Small 2 | 24B dense | Apache-2.0 | Dec 2025 | SWE-bench Verified about 68% (vendor) | QLoRA on 24 GB | Not listed |
| Gemma 4 26B-A4B | 25.2B / 3.8B MoE | Apache-2.0 | 2026 | GPQA Diamond and AIME figures (vendor) | QLoRA on 24 to 32 GB | 0.09 / 0.30 |
| Qwen3.6-27B | 28B dense | Apache-2.0 | 21 Apr 2026 | SWE-bench Verified 77.2%, Terminal-Bench 2.0 59.3% (vendor) | QLoRA 22 GB; LoRA needs 64 GB+ | 0.25 / 2.00 |
| Qwen3.8-27B | 27B dense | Unclear; reports conflict | 14 Aug 2026 | SWE-bench Pro 61.7% (vendor, unreproduced) | QLoRA 22 GB | 0.20 / 2.50 |
| Gemma 4 31B | 31B dense | Apache-2.0 | 2026 | GPQA Diamond 85.7% (independent, Artificial Analysis) | QLoRA 26 GB | 0.20 / 0.40 |
| Meta Muse Glimmer-30B | 30B | Apache-2.0 | 10 Aug 2026 | SWE-bench Verified 76.0% (leaderboard) | QLoRA on 24 to 32 GB | Not listed |
| Qwen3.6-35B-A3B | 35B / 3B MoE | Apache-2.0 | Apr 2026 | Family release notes | Needs memory for all experts: 40 GB+ | 0.10 / 0.95 |

Sources: model sizes and licenses from release coverage linked above and [Muse Glimmer coverage](https://www.implicator.ai/meta-releases-30b-open-weight-muse-glimmer-and-promises-spark-1-2-weights/); SWE-bench from [llm-stats](https://llm-stats.com/benchmarks/swe-bench-verified); prices from the OpenRouter models API and [Tinker pricing](https://thinkingmachines.ai/tinker/pricing/), both read 23 October 2026; VRAM from [Unsloth](https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements).

### 6.1 How to choose

- **Laptop or on-device product:** Qwen3.5-4B or Gemma 4 E4B, quantized to 4-bit. Expect narrow skills only.
- **Default for a narrow specialist on one rented GPU:** Qwen3.5-9B or Granite 4.2 8B. Cheap to train (minutes to hours), cheap to serve, and hosted platforms serve LoRA variants at base-model price ([Fireworks](https://fireworks.ai/pricing)).
- **Coding micro-tasks where quality matters more than cost:** Qwen3.6-27B or Muse Glimmer-30B. These are within roughly 10 to 18 points of the best closed systems on SWE-bench Verified and are the only sizes in the brief's 3B to 30B range with credible repository-level numbers.
- **Latency-critical serving:** MoE bases with 3B to 4B active parameters (Gemma 4 26B-A4B, Qwen3.6-35B-A3B) generate fast but are more awkward to fine-tune.
- **Teachers for distillation:** large open models with permissive licenses (GLM-5.x reported MIT, Qwen3.5-397B Apache-2.0, Kimi K2.6 under a modified MIT license, DeepSeek V4 reported MIT). Read each LICENSE file yourself before generating training data; closed-model outputs are restricted (section 4.3).
- **Research and contamination audits:** fully open models with published data (the OLMo family) let you check whether a test item was in pretraining.

### 6.2 The selection experiment, before any training

Pick three candidates, run each on the same 200-item private dev set with the same prompt, retrieval and tools, and measure pass@1, pass@16, latency and cost per successful task. Choose the one with the best pass@16 at acceptable serving cost: pass@16 predicts trainable headroom better than pass@1 (section 5). Budget: under $20 of hosted inference.

## 7. Training infrastructure and cost estimates

GPU compute is the cheapest line in a narrow-specialist budget: a full LoRA SFT run on an 8B model costs $7 to $18 (₹650 to ₹1,700) on rented hardware, and an RL run $70 to $1,100. The expensive lines are iteration count, teacher-generated data, frontier evaluation baselines, and your own time. FX used throughout: ₹95 per USD (mid-market 94.90 on 23 October 2026, [open.er-api](https://open.er-api.com/v6/latest/USD); ECB 94.82 on 22 October via Frankfurter). Prices exclude 18% GST and card forex markup unless noted.

### 7.1 Hardware compared

| GPU | VRAM | Dense BF16 tensor TFLOPS (approx.) | Memory bandwidth | Cheapest on-demand found, $/hr | Where |
| --- | --- | --- | --- | --- | --- |
| RTX 3090 | 24 GB | 71 | 936 GB/s | 0.22 | [Runpod community](https://www.runpod.io/pricing) |
| RTX 4090 | 24 GB | 165 | 1,008 GB/s | 0.33 to 0.34 | [aggregator](https://www.amcompute.com/gpu-cloud-prices), Runpod |
| RTX 5090 | 32 GB | 209.5 | 1,792 GB/s | 0.50 to 0.69 | aggregator, Runpod |
| L4 | 24 GB | 121 | 300 GB/s | 0.43 to 0.44 | [Jarvislabs (India)](https://jarvislabs.ai/pricing), Runpod |
| A100 40 GB | 40 GB | 312 | 1,555 GB/s | 0.89 | Jarvislabs |
| A100 80 GB | 80 GB | 312 | 2,039 GB/s | 0.89 to 1.19 | aggregator, Runpod |
| H100 SXM | 80 GB | 989 | 3.35 TB/s | 1.70 to 2.69 | aggregator, Runpod |
| H200 | 141 GB | 989 | 4.8 TB/s | 2.09 to 3.59 | aggregator, Runpod |
| B200 | 180 GB | about 2,250 | 8 TB/s | 3.69 to 5.98 | aggregator, Runpod |

TFLOPS are NVIDIA datasheet dense figures with FP32 accumulation; the RTX 5090 figure comes from secondary sources ([NVIDIA forum](https://forums.developer.nvidia.com/t/rtx-5090-peak-bf16-tensor-tflops/350543)) and is approximate. Prices: Runpod page updated 8 October 2026; aggregator updated 17 October 2026; Lambda lists H100 SXM at $3.99 to $4.29 and A100 80 GB at $2.79 ([Lambda](https://lambda.ai/pricing)). Indian providers: Jarvislabs lists H100 SXM $3.49, A100 80 GB $1.49, L4 $0.44; E2E Networks lists H100 at ₹362/hr before GST on one page and different figures elsewhere ([E2E](https://www.e2enetworks.com/gpus/nvidia-h100.md)), so get a quote.

### 7.2 Memory and time by model size

Standard job: LoRA SFT on 30M unique tokens for 3 epochs (90M training tokens), 4,096-token sequences, batch 1 with gradient accumulation, activation checkpointing, AdamW on adapter weights only, LoRA rank 16 on all linear layers (about 0.6% of parameters). Peak VRAM below is my estimate (weights + adapter states + checkpointed activations + 1.5 GB overhead); Unsloth's measured minimums at short context are lower ([Unsloth](https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements)). Times assume 25% MFU for QLoRA and 35% for LoRA. **These are calculations, not measured throughput (grade D).** Measure your own tokens per second in the first 10 minutes of any run and rescale.

| Model | QLoRA peak VRAM | LoRA bf16 peak VRAM | Full FT optimizer state (needs multi-GPU) | Best value setup | Time | Cost USD | Cost INR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3B | about 4 GB | about 9 GB | 48 GB + activations | LoRA on RTX 4090 | 7.8 h | $2.65 | ₹250 |
| 8B | about 8 GB | about 20 GB | 128 GB (2 x 80 GB with FSDP) | LoRA on RTX 4090, or H100 for speed | 20.8 h / 3.5 h | $7.06 / $9.33 | ₹670 / ₹890 |
| 14B | about 13 GB | about 34 GB | 224 GB (4 x 80 GB) | QLoRA on RTX 4090, or LoRA on H100 | 50.9 h / 6.1 h | $17.31 / $16.32 | ₹1,640 / ₹1,550 |
| 27B to 30B dense | about 24 GB | about 68 GB | 480 GB (8 x 80 GB) | LoRA on one H100 80 GB | 13.0 h | $34.97 | ₹3,320 |

Slowest-to-fastest spread for the 8B job across GPUs: 48 h on an RTX 3090 ($10.62) to 3.5 h on an H100 ($9.33). The cost is nearly flat; you pay for speed of iteration, which is worth paying for once you run more than a few experiments a week.

Two caveats change these numbers for real tasks. Long inputs: CI-repair trajectories and multi-page documents often run 8k to 32k tokens, which roughly quadruples activation memory and step time per sequence versus 4k, so a 24 GB card may force QLoRA or a 40 to 80 GB GPU. Environments: in RL on CI tasks, running builds and tests in sandboxes for every rollout can cost more CPU time and orchestration work than the GPU tokens.

### 7.3 Other jobs

| Job | Assumptions | Estimated compute | Cost |
| --- | --- | --- | --- |
| Managed LoRA SFT (Fireworks) | 90M training tokens, model 16B or smaller, $0.50 per 1M ([Fireworks](https://fireworks.ai/pricing)) | Managed | $45 (₹4,300) |
| Managed LoRA SFT (Tinker, Qwen3-8B) | 90M tokens at $0.44 per 1M ([Tinker](https://thinkingmachines.ai/tinker/pricing/)) | Managed | $40 (₹3,800) |
| GRPO run, managed (Tinker, Qwen3-8B) | 300 steps x 64 prompts x 8 samples x 3k tokens = 460M sampled tokens | Managed | about $510 (₹48,000) incl. prefill |
| GRPO run, managed (Tinker, Qwen3.5-9B) | Same | Managed | about $1,090 (₹1.04 lakh) |
| GRPO run, self-hosted 8B on 1 x H100 | Same; assumed 5k to 15k tok/s aggregate vLLM decode (assumption, not measured) + LoRA update | 26 to 43 H100-hours | $70 to $116 plus engineering time |
| Small GRPO pilot | 100 steps x 32 prompts x 8 samples x 2k tokens = 51M tokens, Tinker Qwen3-8B | Managed | about $55 (₹5,200) |
| Continued pretraining, 8B, 1B tokens | Full FT, 8ND with checkpointing, 8 x H100 at 40% MFU | 45 H100-hours | $121 to $180 |
| Continued pretraining, 8B, 10B tokens | Same | 450 H100-hours | $1,200 to $1,800 |
| Teacher data generation | 20k prompts x 4 samples x 2k output tokens from an open teacher at $0.60 per 1M output (e.g. DeepSeek V4.1 Flash on OpenRouter) | API | about $100 plus input tokens |
| Storage and checkpoints | LoRA adapter for 8B rank 16: about 0.1 GB; full 8B checkpoint 16 GB, 128 GB with optimizer; Tinker storage $0.10 per GB-month | 200 GB working set | $15 to $20 per month |

### 7.4 Evaluation is often the biggest line item

Frontier baselines are expensive because you must run them several times, on the same tasks, at realistic effort. Using OpenRouter list prices on 23 October 2026 (Claude Opus 5.5 $4/$20, Sonnet 5.5 $2/$10, Haiku 5.5 $0.10/$0.50, GPT-6.1 Sol $2/$10 per 1M input/output):

| Eval type | Volume | Opus 5.5 | Sonnet 5.5 or GPT-6.1 Sol | Haiku 5.5 |
| --- | --- | --- | --- | --- |
| Single-turn tasks | 500 items x 3 runs x (8k in + 1.5k out) | $93 | $47 | $2.30 |
| Agentic coding tasks | 200 tasks x 3 runs x (400k in + 20k out), before caching | $1,200 | $600 | $30 |

Batch tiers halve these where latency does not matter; prompt caching cuts repeated agent context substantially. Budget at least as much for evaluation as for training.

### 7.5 Serving: when does self-hosting pay?

The cost comparison that matters is not "my 8B versus Opus"; it is "my 8B versus the cheapest acceptable hosted model". Cheap frontier-lab models now list at $0.10 input / $0.50 output per 1M (Claude Haiku 5.5, GPT-6 Luna), and hosted 8B-class open models at $0.06 to $0.25.

```latex
\text{Break-even tokens per month} = \frac{\text{GPU \$/hr} \times 730}{\text{hosted \$ per token (blended)}}
```

A dedicated L4 at $0.44/hr costs about $321 a month. Against Haiku 5.5 at a blended $0.20 per 1M (3:1 input:output), break-even is about 1.6B tokens a month, an average of about 620 tokens per second, around the clock. Against Sonnet 5.5 at a blended $4 per 1M it is only 80M tokens a month. Conclusion: self-hosting a small model beats mid-tier frontier pricing easily, but beats the cheapest frontier tier only at sustained high utilization. Privacy, on-premises requirements and latency, not price, are the stronger reasons to self-host. For fine-tuned models with bursty traffic, serverless multi-LoRA hosting at base-model price avoids idle GPUs ([Fireworks](https://fireworks.ai/pricing)).

### 7.6 Can consumer GPUs contribute?

Yes, for exactly the work a capital-constrained founder should be doing: QLoRA and LoRA up to 14B on 24 GB, QLoRA of 27B to 32B models on 24 to 32 GB at short context, local evaluation, and quantized inference. No, for full fine-tuning above about 3B, continued pretraining, or RL at useful scale. Buying a card only beats renting with heavy use: at a $0.34/hr rental rate, a card costing ₹2.5 lakh (check current Indian retail) equals about 7,700 rental hours.

### 7.7 Access and payment from India

| Option | What you get | Practicality | Caveats |
| --- | --- | --- | --- |
| Runpod, Vast.ai, Lambda | Cheapest global on-demand GPUs, USD billing | Works with an internationally enabled card; Runpod and Vast also accept crypto | Card forex markup of 0 to 3.5% depending on card; GST may apply; spot/community capacity can vanish mid-run |
| Jarvislabs, E2E Networks, Yotta, Neysa | Indian data centres, INR invoices with GST | Easiest for a GST-registered company; data stays in India | Listed prices often above global spot; quotes vary |
| IndiaAI compute portal | Subsidized GPUs, reported at ₹65 to ₹150 per GPU-hour | Application-gated for registered startups, researchers and public institutions; 93.18 lakh GPU-hours sanctioned to 237 projects by August 2026 ([Communications Today, 1 Oct 2026](https://www.communicationstoday.co.in/indiaai-mission-faces-gap-between-capacity-and-deployed-gpus/)) | Not self-serve; demand exceeds supply; plan months ahead |
| Managed fine-tuning APIs (Tinker, Fireworks, Together) | Pay per token, no GPU management | Fastest path from zero to a trained adapter | You still own the LoRA weights on Tinker; check export terms on each platform |
| Kaggle | About 30 free GPU-hours a week on T4 x2 or P100 | Good for QLoRA experiments up to 8B | Sessions capped at roughly 9 to 12 hours; old GPUs |
| Google TPU Research Cloud | Free TPUs for accepted researchers and individuals | Real but competitive; expects public output | JAX/TPU tooling friction; program repo archived April 2026, verify status ([TRC](https://sites.research.google/trc/about/)) |
| Cloud startup credits | Google Start tier about $2k, Microsoft entry $200 rising with milestones, AWS Activate Founders $1k to $5k (sources conflict) | Real but small without an investor | Large AI tiers ($100k to $350k) need institutional funding ([comparison](https://betterstack.com/community/comparisons/best-startup-cloud-credit-programs/)) |

Marketing versus reality: "up to $350,000 in credits" is real for funded AI startups and irrelevant to a bootstrapped student until an investor or accelerator signs on. The genuinely accessible free compute is Kaggle, small cloud entry tiers, managed-API trial credits, and (with patience) IndiaAI and TRC.

## 8. Experiment design: a fair test of the central hypothesis

Run the test on the task your first paying customer brings (Gate 1), under rules that favour neither side, and size it to what one founder can build. Two worked candidates follow: **Indian document extraction and reconciliation**, where frontier models have seen the least of the local formats and a strict accuracy win is likeliest, and **CI build-break repair**, which has the cleanest verifier but is the domain frontier labs optimize hardest. Budget roughly $300 to $1,500 (₹28,000 to ₹1.4 lakh) and 6 to 12 weeks part-time; start with a 300-task minimum eval and expand only if the result is close.

### 8.1 Choosing the test domain

| Criterion | Document extraction and reconciliation (India) | CI build-break repair |
| --- | --- | --- |
| Verifier | Schema validation, totals, GSTIN and tax checks, match against the ledger; some judgment remains | Compiler plus tests; still needs hidden tests and forbidden-change checks |
| Frontier prior | Weaker on messy scans, local formats, bilingual text | Strong: coding is the most optimized frontier domain |
| Data rights | Needs written customer consent; holding client documents is not a right to train across clients | Permissively licensed public repositories |
| Hidden build cost | Labeling 300 to 1,000 real documents | One reproducible container per task, often weeks of work; sandbox runs also dominate RL cost |
| Likely result (subjective) | Best chance of a strict accuracy win | Best chance of a cost-only win |

Either way, a win says nothing about open-ended feature work (section 3.4). The CI version is used as the worked example below; document equivalents are noted where they differ.

### 8.2 Systems compared

Every system gets the same inputs and tools: for CI repair, the failing file, retrieved related files, the compiler output and up to 3 compile-and-test iterations; for documents, the page images or text, the target schema, validators and up to 3 validation retries. Run two comparisons: a shared scaffold (for attribution) and each side's reasonably optimized system under the same cost budget (for buying decisions), because a shared scaffold can favour one model.

| ID | System | Purpose |
| --- | --- | --- |
| N0 | Non-LLM baseline: rules, templates, compiler quick-fixes, a small encoder or gradient-boosted classifier | Tests whether an LLM is needed at all |
| B0 | Small base model (e.g. Qwen3.5-9B, Qwen3.6-27B), prompted, scaffold | Floor; measures pass@k headroom |
| B1 | Frontier models with scaffold: Claude Opus 5.5, Sonnet 5.5, GPT-6.1 Sol, Gemini 4 Argon, at default or medium effort | The bar to beat |
| B2 | Cheap frontier tier with scaffold, prompt caching and batch pricing where applicable: Claude Haiku 5.5, GPT-6 Luna | The real cost competitor |
| B3 | Small base + full system: retrieval, iterative tool loop, verifier, best-of-4 | Isolates the value of the system without training |
| T1 | B3 + LoRA SFT on verifier-filtered distilled data | Value of SFT |
| T2 | T1 + 2 rounds of rejection sampling on own outputs | Value of on-policy data |
| T3 | T2 + GRPO with the verifier as reward | Value of RL |
| H | T3 with escalation to the best B1 model when the verifier fails | The product configuration |

### 8.3 Data: how much, from where

- **CI repair training: 5,000 to 20,000 verified repair pairs.** Sources: (a) public commits on MIT/Apache/BSD repositories where CI went red then green with a small diff; (b) error injection into compiling code (rename a symbol, change a type, drop an import, bump a library version), keeping the original as the target, verified by compile and tests; (c) repairs sampled from a permissively licensed open teacher and kept only if they compile and pass tests. Record licenses per file; exclude copyleft and unlicensed code.
- **Document training: 1,000 to 5,000 labeled pages.** From pilot customers with written consent for training, plus synthetic documents rendered from real templates; teacher labels from an open model are kept only when validators pass and a human spot-check agrees.
- **Legal guardrails:** no outputs from Claude, GPT or Gemini as training targets (section 4.3); frontier models are evaluated, never distilled. Note the consequence: a teacher weaker than the frontier cannot by itself produce a student that beats the frontier, so any strict win must come from verification, RL against the checker, or proprietary customer data.

### 8.4 Contamination-resistant test sets (built before any training)

| Set | Minimum viable eval | Full eval (only if the gap is within 10 points) | Construction | What it guards against |
| --- | --- | --- | --- | --- |
| Fresh | 100 | 400 | Items created after every evaluated model's training cutoff | Memorization by any model |
| Private | 100 | 300 | Repositories or customers never used for training, never sent to a data-generation API | Leakage through your own pipeline |
| Out-of-distribution | 60 | 150 | A different framework version, document template or customer | Overfitting |
| Robustness | 40 | 150 | Fresh items with identifiers renamed, reformatted, or rescanned | Surface-pattern matching |
| **Total** | **300** | **1,000** |  |  |

Split by repository or customer, never by individual item, because items from one source are correlated; hash and store the sets; never tune prompts on them; use a separate 200-task dev set for all iteration.

### 8.5 Metrics

```latex
\text{pass@}k = \mathbb{E}_{\text{tasks}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right], \quad n = 5 \text{ samples per task}, \; c = \text{samples that pass}
```

- **Task success:** compiles, all original tests pass, a hidden test set added by you passes, no test was deleted or weakened, no new `any`, `# type: ignore` or `@ts-ignore` (these are reward hacks).
- **Severity-weighted failure rate (SWFR):** silent wrong fix that compiles but fails hidden tests = 10; reward hack = 5; still fails to compile = 1; explicit abstain or escalate = 0.5. SWFR = sum of weights / number of tasks.
- **Cost per successful task:** (inference + verifier compute + escalation cost + training cost amortized over 12 months of projected volume) / successes.
- **Latency:** p50 and p95 wall-clock including compile loops.
- **Reliability:** run-to-run standard deviation over 5 seeds; calibration of confidence (does a high-confidence answer pass more often?); abstention precision.

### 8.6 Statistics

All comparisons are paired on the same tasks. Use exact McNemar tests on pass/fail per task, a cluster bootstrap (resampling repositories or customers, 10,000 resamples) for confidence intervals, and Holm correction across the systems compared. Paired tasks needed to show a **difference** with 80% power at two-sided alpha 0.05:

| Discordant share | Detect 3 points | Detect 5 points | Detect 10 points |
| --- | --- | --- | --- |
| 15% | 1,306 | 469 | 116 |
| 20% | 1,742 | 626 | 155 |
| 30% | 2,614 | 940 | 234 |

Showing **non-inferiority** (that the small system is no more than a margin worse, assuming no true difference, one-sided alpha 0.05, 80% power) needs more: at 20% discordance about 3,100 tasks for a 2-point margin, 1,370 for 3 points, 770 for 4 points, 500 for 5 points. So the 300-task minimum eval resolves about 10-point differences, and the 1,000-task full eval supports a 4-point non-inferiority claim, not 2. Claims finer than the design supports are exactly the trap the SWE-bench audit warns about ([Liu et al. 2026](https://arxiv.org/abs/2609.17394)).

### 8.7 Pre-registered thresholds

Written down before running, judged on fresh + private sets combined:

| Outcome | Rule |
| --- | --- |
| **Strong win** | T3 pass@1 above the best B1 with the 95% CI excluding zero; SWFR no worse; cost per accepted result no higher; p95 latency no worse |
| **Commercial (Pareto) win** | T3 non-inferior to the best B1 within 4 points on the full eval (provisionally within 10 points on the minimum eval); full cost per accepted result (tokens, retries, tool runs, escalations, human review) at most one third of the best B1 and no more than B2's; SWFR no worse; OOD drop no larger than 1.5x the frontier's |
| **Privacy-required path** | When the customer cannot send data to external APIs, the bar is the customer's own accuracy bar at an acceptable cost; beating API prices is not required |
| **Training adds nothing** | T3's gain over B3 is below the resolvable margin and its upper confidence bound excludes a gain worth its cost: ship B3, or B2 with the system |
| **Hypothesis rejected** | T3 worse than the best B1 by more than the margin, or not cheaper per accepted result than B2 with the same system, or OOD drop more than 2x the frontier's |
| **No LLM needed** | N0 within the margin of every LLM system: build conventional software |

### 8.8 Reliability outside the benchmark

After the offline result, shadow-run H on 2 to 3 real teams' CI for 4 weeks: log every failure, measure the share of fixes merged without edits, reverted within 7 days, and developer-reported time saved. A system that wins offline but is merged less than 50% of the time in shadow mode is not a product.

## 9. Best commercial opportunities, ranked

The best near-term opportunities are those where the output is checkable, the buyer already pays humans to do the work, and a founder can reach buyers without a sales team; in most of them fine-tuning is an optional cost optimization, not the product. Several exciting domains (Indian-language consumer apps, education, legal, generic enterprise assistants) are poor first businesses for this founder.

### 9.1 Ranking for an India-based student founder with strong software skills and little capital

| Rank | Opportunity | Who pays, and evidence of price | Frontier shortfall | Does fine-tuning differentiate? | Main obstacle |
| --- | --- | --- | --- | --- | --- |
| 1 | Productized AI automation for global SMBs and agencies (workflow builds, retainers) | Owners pay for hours saved; freelance AI/RAG work bills about $40 to $100/hr ([rate guide](https://adsnipper.com/blog/cost-to-hire-ai-developer/), grade C) | Small: frontier APIs suffice | No | Low defensibility; founder time caps revenue |
| 2 | Verifiable document workflows in one vertical (invoice, GST, bank-statement extraction and reconciliation for CA firms, mid-size finance teams, logistics) | Indian CA tools sell at about ₹10k to ₹50k a year ([ICAI use-case page](https://ai.icai.org/usecases_details.php?id=250)); extraction APIs at $0.015 to $0.30 per page ([comparison](https://imagetotable.ai/blog/document-extraction-pricing-breakdown-2026)) | Moderate: Indian formats, reconciliation logic, data residency | At high volume or on-premises, yes; otherwise routing to cheap hosted models suffices | Low Indian SMB price points; many incumbents |
| 3 | Narrow developer automation (CI build-break repair, version migrations, test generation) for global software teams | Code review seats sell at $24 to $72 per developer-month ([CodeRabbit pricing tracker](https://www.usagepricing.com/blueprint/coderabbit)) | Small to moderate; frontier agents already do this | Yes for cost and latency on verifiable sub-tasks (section 3.4) | Competing with Claude Code, Codex, Cursor and GitHub; buyers can do it themselves |
| 4 | Healthcare revenue-cycle admin (coding, denials, prior auth) via Indian RCM firms | India RCM market about $4B in 2025 per Ken Research; firms such as IKS, AGS, Omega are automating ([Ken Research](https://www.kenresearch.com/india-revenue-cycle-management-market)) | Moderate: payer rules, code sets, audit trails | Yes: private, high volume, narrow labels | HIPAA/BAA, PHI access, domain experts, long sales cycles |
| 5 | Defensive security triage (alert dedup, log summarization, phishing classification) | Security budgets are real; buyers demand trust and references | Moderate | Yes for on-premises and volume | Needs security expertise and credibility |
| 6 | Indic-language customer support (text and voice) | Enterprises pay; outcome pricing benchmarks: Intercom Fin $0.99 per resolution ([Intercom pricing coverage](https://www.gleap.io/blog/intercom-fin-ai-pricing-2026)) | Moderate on code-mixed Indic and latency | Yes for latency, cost, Indic tokens | Crowded (Sarvam, Gnani, Yellow.ai, Haptik); speech stack |
| 7 | Analytics and text-to-SQL for one vertical | Teams pay for answers; BI incumbents bundle AI | Small: top BIRD systems reach about 80% to 82% with frontier models ([Huawei result coverage](https://technologymagazine.com/news/huawei-datagallery-tops-benchmark-for-enterprise-text2sql-ai)) | Partly, on one customer's schema | Generic tools and BI vendors |
| 8 | Financial document processing for lenders | NBFCs and banks pay per file | Moderate | Yes, at volume | Regulated buyers, entrenched vendors (account aggregator and bank-statement analyzers) |
| 9 | Legal document workflows | Large firms pay; Indian SMB legal pays little | Small for drafting, real for citation accuracy | Rarely | Liability, Harvey-class competitors |
| 10 | Enterprise internal knowledge assistants | Bundled into Microsoft, Google, Glean, ChatGPT Enterprise | None | No | Platform bundling |
| 11 | Indian-language consumer apps | Consumers rarely pay; frontier apps are free | Shrinking: Sarvam claims Indic wins on its own benchmark ([Sarvam 105B coverage](https://www.buildfastwithai.com/blogs/sarvam-105b-india-s-open-source-llm-for-22-indian-languages-2026), grade C) | Yes technically, but monetization fails | Willingness to pay |
| 12 | Education | Low WTP, free frontier tutors, children's-data rules | None for general tutoring | No | Price and trust |

### 9.2 What the ranking turns on

- **Willingness to pay beats technical elegance.** Indian SMB software price points (₹2k to ₹50k a year for compliance tools) leave almost no room for support and acquisition costs (section 10). Mid-market and foreign buyers pay 5x to 20x more for the same hours saved.
- **Checkable outputs are the only place where small models reliably win** (sections 3 and 5). Document reconciliation, CI repair and revenue-cycle coding all have checks; tutoring, legal advice and open-ended support mostly do not.
- **Retrieval and ordinary software solve most of the problem cheaply.** In documents, a schema, validators and reconciliation rules do more than a fine-tune; in support, a good knowledge base does more than a model.
- **Compliance is a moat only once you can afford it.** India's DPDP Rules were notified on 13 November 2025 with most data-fiduciary duties effective from 13 May 2027 ([Mondaq summary](https://www.mondaq.com/india/data-protection/1708164/digital-personal-data-protection-rules-2025-notified)); US healthcare work needs HIPAA business-associate agreements. Data-residency and on-premises demands from these rules are where a private small model becomes a selling point.
- **Defensibility against cheaper frontier models** is strongest where you hold workflow integration and proprietary labeled data (healthcare admin, a vertical document workflow) and weakest where the frontier vendor ships the same feature (coding assistants, enterprise search).

### 9.3 Domains to avoid as a first business

Absent a specific distribution advantage, avoid as a first business: generic coding copilots, enterprise knowledge assistants, education apps, Indian-language consumer chat, and legal drafting for Indian SMBs. Each has at least one serious problem for a bootstrapped founder: free frontier substitutes, platform bundling, or buyers who do not pay. These are broad markets, not verdicts on every workflow inside them; privileged access to a sharply defined workflow (for example through a family trade network) can reverse the ranking.

## 10. Monetization and unit economics

Selling model access is the weakest way for Void Models to make money: cheap frontier tiers list at $0.10 / $0.50 per 1M tokens and hosted 8B open models at $0.06 to $0.25, so a model API has no pricing room. The viable models sell an outcome (a reconciled invoice set, a fixed build, a migrated module) or a private deployment, and the numbers below show that price point and who hosts the GPUs matter more than model quality.

### 10.1 Business models compared

| Business model | Who pays, for what | Realistic price | Gross margin | Sales cycle / CAC | Verdict |
| --- | --- | --- | --- | --- | --- |
| Model API subscription or usage API | Developers, for tokens | At or below $0.06 to $0.25 per 1M (hosted 8B market) | Thin; depends on GPU utilization | Self-serve; high marketing cost | No: commodity, frontier tiers undercut |
| Coding copilot for one framework | Developers, for productivity | $10 to $40 per seat-month | 40% to 60% after inference | Self-serve, crowded | No: Copilot, Cursor, Claude Code |
| Domain-specific AI SaaS | A vertical's operators, for hours saved | ₹1.5k to ₹10k per month in India; $49 to $500 per month abroad | 45% to 65% at scale | Weeks; CAC ₹3k to ₹15k (India SMB), $100 to $500 (global self-serve) | Yes: core product path |
| Outcome-based automation | Ops teams, per completed item | Benchmarks: $0.99 per resolved support ticket ([Intercom Fin](https://www.gleap.io/blog/intercom-fin-ai-pricing-2026)); per invoice or per fix | 50% to 80% if success is measurable | Pilot-led, 1 to 3 months | Yes: aligns price with value |
| Productized AI services | SMB owners and agencies, per project plus retainer | $1.5k to $5k per project, $200 to $500 per month upkeep | 70% to 90% before founder time | Days to weeks; referrals and outbound | Yes: fastest cash, weak moat |
| Dedicated enterprise deployment | Regulated mid-size firms, for privacy | ₹15L to ₹40L per year (assumption) | About 34% if you host GPUs; about 79% if the customer's cloud hosts them | 3 to 9 months; ₹3L to ₹8L per logo | Later, after references |
| On-premises licensing | Banks, hospitals, government | License plus support | High, but support-heavy | 6 to 18 months, tenders | Later |
| White-label models | Systems integrators, resellers | Low per unit | Moderate | Partner-dependent | Opportunistic only |
| Fine-tuning as a service / private customization | Teams with data | $2k to $20k per project | Services margin | Weeks | Only as services; platforms consolidated (OpenPipe to CoreWeave, [Sept 2025](https://coreweave.com/news/coreweave-to-acquire-openpipe-leader-in-reinforcement-learning); Predibase to Rubrik, June 2025) and Tinker/Fireworks commoditize it |
| Datasets and evaluation products | AI teams and vendors | $5k to $30k per custom eval suite (assumption) | High | Slow, trust-based | Possible side line |
| Inference cost optimization | Companies with big LLM bills | Share of savings | High if savings are real | Medium | Shrinking: cheap frontier tiers erase most savings |

### 10.2 Bottom-up scenarios

All figures are grade D calculations; change the assumptions, not just the outputs. Revenue, gross profit, operating profit before and after salaries are kept separate. ₹ in lakh (L) at ₹95/USD.

**A. India SMB vertical SaaS (document workflow).** Price ₹1,500 per customer-month; 2,000 documents per customer-month; inference ₹0.15 per document (90% on a $0.10/$0.50 model, 10% escalated to a $2/$10 model); support ₹250 per customer-month; hosting ₹5k to ₹80k per month fixed plus ₹50 per customer; payment fees 2.36%; churn 4% a month; CAC ₹6,000.

| Paying customers | Revenue / yr | Gross profit (margin) | CAC + other opex | Operating profit before salaries | Team and founder pay | After salaries | One-time CAC to reach base |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | ₹1.8L | ₹0.4L (24%) | ₹2.3L | -₹1.9L | ₹0 | -₹1.9L | ₹0.6L |
| 100 | ₹18.0L | ₹8.5L (47%) | ₹8.9L | -₹0.3L | ₹6.0L | -₹6.3L | ₹6.0L |
| 1,000 | ₹180L | ₹93.8L (52%) | ₹58.8L | ₹35.0L | ₹90L | -₹55L | ₹60L |

At ₹1,500 a month this business never pays a team. At ₹4,000 a month the 100-customer case flips to about ₹28.9L before salaries and ₹22.9L after paying the founder ₹50k a month. Price, not model quality, decides it.

**B. Global developer tool (CI build-break repair, self-serve).** $49 per team-month; inference $14 per team-month (about 300 fixes, small model first, 20% escalated to a $2/$10 model); support $3; hosting $200 to $4,000 per month fixed; merchant-of-record fees 5%; churn 3% a month; CAC $150.

| Paying teams | Revenue / yr | Gross profit (margin) | Operating profit before salaries | After salaries | One-time CAC |
| --- | --- | --- | --- | --- | --- |
| 10 | $5,880 (₹5.6L) | $1,026 (17%) | -$1,619 (-₹1.5L) | -₹1.5L | $1,500 |
| 100 | $58,800 (₹55.9L) | $27,060 (46%) | $15,344 (₹14.6L) | ₹8.6L | $15,000 |
| 1,000 | $588,000 (₹559L) | $294,600 (50%) | $209,021 (₹199L) | ₹109L | $150,000 |

Break-even with the founder paid ₹50k a month: about 69 teams. Inference is 29% of price here, so this is the one scenario where a cheaper fine-tuned small model moves margin.

**C. Enterprise private deployment.** ACV ₹20L plus ₹5L setup; support and customization ₹4.3L per customer-year; CAC ₹5L; 15% logo churn; GPU hosting ₹9L per customer-year *only if you host*.

| Enterprise customers | Recurring revenue / yr | Gross margin, customer hosts GPUs | After salaries, customer hosts | After salaries, you host GPUs | One-time CAC |
| --- | --- | --- | --- | --- | --- |
| 1 | ₹20L (+₹5L setup) | 79% | ₹11.9L (no hires) | ₹3.0L | ₹5L |
| 5 | ₹100L | 79% | ₹25.9L (2 engineers) | -₹19.1L | ₹25L |
| 20 | ₹400L | 79% | ₹85L (10 staff) | -₹95L | ₹100L |

Lesson: never bundle dedicated GPUs into a fixed enterprise price; deploy into the customer's cloud or pass infrastructure through at cost.

Caveat: Indian enterprise deals at this price often need forward-deployed integration and on-site support. At 0.5 FTE per customer (₹7 to ₹9 lakh a year) instead of ₹4.3 lakh, customer-hosted gross margin falls to roughly 55% to 65%.

**D. Productized services (global SMB).** About 60 founder-hours per project.

| Volume per year | Revenue | Gross margin (founder time excluded) | Founder hours |
| --- | --- | --- | --- |
| 6 projects at $2,500 + 3 retainers | $25,800 (₹24.5L) | 87% | about 540 |
| 12 projects at $3,000 + 8 retainers | $64,800 (₹61.6L) | 88% | about 1,200 |
| 24 projects at $3,500 + 20 retainers (contractors) | $156,000 (₹148L) | 69% | capped by delivery |

Services out-earn every product scenario in year one, at the cost of no compounding asset unless you deliberately keep the data, evals and components.

The 87% is a services margin that excludes your labour: the first row pays about $25,800 / 540 hours, roughly $48 (₹4,500) an hour. Keep services inside the vertical you intend to productize; generic SMB projects pay the bills but do not build the data, evals and integrations a product needs.

### 10.3 How to get the first paying customer without building a model

1. **Pick one workflow and one buyer type you can reach in two weeks:** CA firms through your network, a software agency, or US SMBs via cold outreach.
2. **Run 15 discovery calls.** Quantify hours per month spent, error cost, and what they pay today. Collect sample inputs with permission.
3. **Deliver a concierge pilot in week 3:** frontier API + rules + validators + you as the human in the loop. Charge for it (₹10k to ₹25k, or $500 to $1,500); free pilots teach you nothing about willingness to pay.
4. **Build the eval set from their real data** (with consent and a DPA). This, not the model, is the first proprietary asset.
5. **Convert to a monthly contract** priced on outcome or volume.
6. **Train weights only when** volume, latency or an on-premises requirement makes the section 7.5 break-even favourable, and the section 8 rules say training beats the system.

## 11. Failure analysis and defensibility

The strongest case against a model-centric Void Models is that weight-level advantages expire within months: open coding models of 32B or less have reached the frontier's SWE-bench score roughly seven months after the frontier did, Anthropic cut its cheapest tier's price 10x in a year, and the best-known fine-tuning platforms were absorbed into larger companies. What compounds is data rights, evaluation assets, workflow integration and distribution.

*Chart: open coding models of 32B or less versus the best Claude release on SWE-bench Verified. Interactive version in the "Risks and moats" section of the web page.*

Qwen3.6-27B reached 77.2% in April 2026, the score Claude Sonnet 4.5 posted in September 2025; any fine-tune lead you build on today's base faces the same clock.

### 11.1 The strongest arguments against, steelmanned

1. **Frontier price collapse.** Claude Haiku 4.5 listed at $1 / $5 per 1M tokens in October 2025; Claude Haiku 5.5 lists at $0.10 / $0.50 in October 2026, and GPT-6 Luna matches it (OpenRouter models API, read 23 October 2026). The "10x cheaper than frontier" pitch now only holds against mid-tier models.
2. **Open-weight release cadence.** Qwen shipped the 3.5, 3.6, 3.7 and 3.8 families between February and September 2026 (OpenRouter listing dates). Each release can match or beat last quarter's fine-tune zero-shot, resetting your lead.
3. **Enterprises buy closed models.** Open-source share of enterprise LLM usage fell from 19% to 11% during 2025, and Anthropic held an estimated 54% of enterprise coding spend ([Menlo Ventures, Dec 2025](https://menlovc.com/perspective/2025-the-state-of-generative-ai-in-the-enterprise/), grade C survey).
4. **Fine-tuning as a standalone business did not survive.** OpenPipe was acquired by CoreWeave in September 2025 ([CoreWeave](https://coreweave.com/news/coreweave-to-acquire-openpipe-leader-in-reinforcement-learning)); Predibase by Rubrik in June 2025. Managed APIs such as Tinker and Fireworks now sell the same capability per token.
5. **The bar for "our own model" is set by companies with distribution.** Cursor's Composer 2 sits on a roughly 1T-parameter open base with large-scale continued pretraining and RL (section 3.2). A student startup cannot outspend that; it can only out-focus it.
6. **Benchmark wins mislead.** Fine-tuned small models can post high in-distribution scores by learning keywords ([Huynh and Pham 2026](https://arxiv.org/abs/2609.20847)); developers misjudge their own AI speedups by about 40 points ([METR 2025](https://arxiv.org/abs/2507.09089)).
7. **The best teachers are off-limits.** Distilling Claude or GPT outputs into a competing model breaches their terms (section 4.3), so your teacher is at best the strongest permissively licensed open model.
8. **Buyers pay for outcomes, not models.** No evidence found that customers pay a premium for "a specialized model" as such; they pay for hours saved, error rates and compliance.
9. **Founder constraints.** Enterprise cycles of 3 to 9 months, procurement and security reviews, and a credibility gap for a student founder all lengthen time to revenue.

### 11.2 What happens if

| Scenario | Model-centric Void Models | Software-centric Void Models | Mitigation |
| --- | --- | --- | --- |
| Frontier prices fall another 10x | Cost pitch disappears | Gross margin rises | Price on outcomes, not tokens |
| A new open base beats your fine-tune | Weights become worthless | Swap base, keep the system | Treat weights as disposable; make re-training a one-week pipeline |
| Enterprises deploy open models themselves | Nothing left to sell | Sell integration, evals, operations | Own the workflow and the eval suite |
| Users stay with ChatGPT, Claude, Copilot | Standalone model ignored | Integrate inside the tools they use | Ship as an add-on to Tally, GitHub, Slack, ERP |
| Privacy or data-residency rules tighten (DPDP from May 2027) | Private small model gains value | Private deployment option becomes a feature | Keep an on-premises path ready, not built |

### 11.3 Temporary versus compounding advantages

| Advantage | Lasts | Why |
| --- | --- | --- |
| A fine-tuned model's benchmark lead | Often under a year, sometimes zero | Next open base or price cut erases it (chart above) |
| Cost advantage versus the frontier | Months | Cheap tiers fell 10x in a year |
| Prompt and scaffold tricks | Weeks to months | Copied, or absorbed into vendor products |
| Proprietary labeled data from real workflows (with contractual rights) | Compounds | Grows with every customer; rivals cannot download it |
| Evaluation sets built from real failures | Compounds | Lets you adopt each new base faster and prove quality to buyers |
| Integrations into customer systems | Compounds | Switching costs rise with each connected system |
| Compliance posture (DPAs, audits, data residency) | Compounds slowly | Expensive to copy, required by regulated buyers |
| Distribution: channel partners, community, references | Compounds | Lowers CAC every year |
| A re-training pipeline that ports to any new base in days | Compounds | Turns new open releases from a threat into an upgrade |

### 11.4 When to abandon the model-centric strategy

Switch to a software-centric company if any two of these are true:

1. The section 8 experiment ends in "training adds nothing" or "hypothesis rejected".
2. Fewer than 20% of interviewed buyers name cost, latency or privacy among their top three purchase criteria.
3. Inference is below 15% of revenue at projected volume, so a cheaper model cannot move margin much.
4. A new open base or cheap frontier tier matches your fine-tune within 2 points within six months of release.
5. You cannot obtain contractual rights to train on customer data.
6. Three consecutive training iterations each add less than 1 point on the private holdout.

## 12. Recommended technical architecture

Build a verifier-centred system in which the model is a swappable part: structured context, a cheap worker model, deterministic tools and checks, retries, and escalation to a frontier model. Train your own LoRA adapter only when the logs show it beats the cheapest acceptable hosted model on cost per verified success.

*Diagram: request and context, router, worker model, tools, verifier; retries, escalation to a frontier model and human review; logs feeding eval sets and offline training. Interactive version in the "Architecture" section of the web page.*

The runtime path runs top to bottom in the centre; anything the verifier rejects is retried, then escalated, then sent to a human; every verdict feeds the eval and training sets on the left, and a new adapter replaces the worker only if it passes the eval gates.

### 12.1 Components, by strength of evidence

| Component | Status | Evidence | Build when |
| --- | --- | --- | --- |
| Structured context and retrieval | Demonstrated | Retrieval beats fine-tuning for knowledge ([Ovadia et al. 2023](https://arxiv.org/abs/2312.05934)) | Day 1 |
| Deterministic verifier (schema, totals reconcile, compiles, tests pass) | Demonstrated | Verifiers turn sampling into accuracy ([Brown et al. 2024](https://arxiv.org/abs/2407.21787)) | Day 1 |
| Tools and executor | Demonstrated | Scaffold effects up to about 30 points in coding ([Liu et al. 2026](https://arxiv.org/abs/2609.17394)) | Day 1 |
| Retries and best-of-N against the verifier | Demonstrated | Same as above | Day 1 |
| Escalation to a frontier model | Demonstrated | Learned routers cut cost more than 2x at near-equal quality ([Ong et al. 2024, RouteLLM](https://arxiv.org/abs/2406.18665), grade B) | Day 1, with rules first |
| Logging and eval store | Engineering necessity | Everything else depends on it | Day 1 |
| Human review for high-severity outputs | Required in regulated work | Liability, not ML | When outputs carry money or compliance risk |
| LoRA SFT on verified data | Demonstrated | Sections 4 and 5 | When volume, latency or on-premises needs pass the section 7.5 break-even |
| Rejection-sampling rounds | Demonstrated where a verifier exists | [Yuan et al. 2023](https://arxiv.org/abs/2308.01825) | With SFT |
| GRPO / RLVR | Plausible for your task, contested beyond elicitation | [Yue et al. 2025](https://arxiv.org/abs/2504.13837) | Only if pass@16 is well above pass@1 after SFT |
| Confidence calibration | Partly demonstrated | Agreement across samples (self-consistency, [Wang et al. 2022](https://arxiv.org/abs/2203.11171)) beats verbalized confidence, which is overconfident ([Xiong et al. 2023](https://arxiv.org/abs/2306.13063)) | Use verifier outcome plus sample agreement |
| Learned dynamic router | Plausible | RouteLLM | After 10k+ logged requests |
| On-policy distillation from an open teacher | Plausible | [Agarwal et al. 2024](https://arxiv.org/abs/2306.13649) | After SFT plateaus |
| Continued pretraining | Unjustified for now | Only for new languages or DSLs with 100M+ tokens | Probably never at this stage |
| Pretraining, MoE training, full fine-tuning, multi-node clusters | Unjustified | Section 4 | Not at this stage |
| Model merging | Cheap experiment only | Unpredictable results | Optional |
| Owning GPU servers | Unjustified | Section 7.6 | Only at sustained high utilization |

### 12.2 Separating gains from better weights versus better system design

Run a 2x2 on the same tasks: base or tuned weights, each with a bare prompt and with the full system. With accuracies B0 (base, bare), BS (base, system), T0 (tuned, bare), TS (tuned, system):

```latex
\Delta_{\text{weights}} = \tfrac{1}{2}\big[(T_S - B_S) + (T_0 - B_0)\big], \quad \Delta_{\text{system}} = \tfrac{1}{2}\big[(B_S - B_0) + (T_S - T_0)\big], \quad \text{interaction} = T_S - B_S - T_0 + B_0
```

A negative interaction is the common case: tuning helps less once the system is good. Then add one more arm, the best frontier model inside your system. If it beats your tuned model by more than your cost saving is worth, the system is the asset and the weights are optional.

### 12.3 Build order by performance per rupee

1. Eval harness and frozen test sets (under ₹5,000 of API calls).
2. Deterministic verifier and tools.
3. Context builder.
4. Cheap worker model, retries, escalation rules.
5. Logging and feedback capture, with customer data rights in the contract.
6. Only then a LoRA SFT experiment (₹1,000 to ₹5,000 of compute).
7. A GRPO pilot only if headroom exists (₹5,000 to ₹50,000).

## 13. Capital allocation under severe constraints

Spend the first rupees on finding a paying customer and on evaluation, not on GPUs: at every budget below ₹1 lakh, the highest-return line items are customer discovery, frontier and open-model evaluation calls, and verifier engineering (your time). GPU training becomes worth funding only after Gate 2 in section 14.

### 13.1 Return on each kind of spend, early stage

| Spend category | Early ROI | Why |
| --- | --- | --- |
| Customer discovery and acquisition | Highest | Without a buyer, every other line is wasted; costs are small (travel, outreach tools) |
| Evaluation systems and baselines | Very high | Decides whether training is worth anything; a 500-item single-turn eval on Sonnet 5.5 costs about ₹4,500 (section 7.4) |
| Tool integration and verifiers | Very high | Mostly founder time; largest measured effect on narrow-task accuracy |
| Retrieval infrastructure | High | Free tiers (Postgres with pgvector, local embeddings) are enough at pilot scale |
| Product engineering and hosting | High once a pilot exists | Free tiers cover the first months |
| Synthetic data from open teachers | Medium | About $100 per 160M output tokens (section 7.3); only useful with a verifier |
| High-quality domain data and licensing | Medium, rising | Valuable after you know the exact task; best source is customer data with rights |
| Human annotation | Medium | Use domain-expert spot checks (for example a CA article clerk at ₹300 to ₹500 an hour) rather than bulk labeling |
| GPU training | Low until Gate 2 | A full LoRA SFT run costs ₹650 to ₹1,700; the risk is not cost but training before knowing it helps |

### 13.2 What each budget buys

| Budget | Allocation | Realistic outcome | Postpone |
| --- | --- | --- | --- |
| ₹5,000 (about $53) | ₹1,500 eval API calls; ₹1,500 concierge-pilot API credits; ₹1,000 domain, landing page, email; ₹1,000 discovery travel. Compute: Kaggle and free OpenRouter models only | 15 discovery calls, a 200-item eval set from real samples, baselines B0, B2, B3 on it, one concierge pilot offer | All training beyond a free Kaggle QLoRA toy |
| ₹25,000 (about $263) | ₹5,000 eval; ₹3,000 open-teacher data; ₹2,000 managed SFT runs (3 to 5 runs); ₹5,000 product hosting and tools for 3 months; ₹5,000 customer acquisition; ₹5,000 reserve | Reduced section 8 experiment (SFT only), first paid pilot, decision on whether weights matter | GRPO, 27B experiments, hiring |
| ₹1,00,000 (about $1,050) | ₹15,000 full eval incl. agentic frontier baselines with caching; ₹10,000 synthetic data; ₹10,000 expert spot-check annotation; ₹25,000 training incl. one small GRPO run; ₹15,000 product infra for 6 months; ₹15,000 acquisition, company registration, contract templates; ₹10,000 reserve | Full section 8 experiment, 3 to 5 paying pilots, a Gate 2 decision backed by data | Own hardware, continued pretraining, full-time hires |
| ₹5,00,000 (about $5,260) | Only after Gate 2: ₹1,00,000 training and eval; ₹1,00,000 domain data and expert annotation; ₹1,50,000 part-time engineer or intern for 4 months; ₹75,000 acquisition; ₹25,000 legal, compliance, trademark; ₹50,000 runway reserve | A repeatable sales motion with 5 to 10 customers, an adapter that beats the cheap hosted model on cost per success, if the data supports it | Pretraining, GPU purchase, multi-node runs |

### 13.3 The first ₹10,000 and the next ₹50,000

**First ₹10,000:** ₹3,000 on evaluation baselines (frontier and open models on a 200-item set built from real buyer samples); ₹2,000 on API credits to run a concierge pilot; ₹2,000 on domain, landing page and email; ₹3,000 on discovery (travel, outreach). Zero on GPUs: Kaggle's free quota covers any toy QLoRA run.

**Next ₹50,000, only after a paid pilot exists:** ₹10,000 to complete frozen test sets and full baselines; ₹8,000 on verifier-filtered data from a permissively licensed open teacher; ₹7,000 on SFT sweeps and a small GRPO pilot (about ₹5,200 on Tinker, section 7.3); ₹10,000 on product infrastructure for 3 to 6 months; ₹10,000 on acquiring the next 3 to 5 customers; ₹5,000 on company registration and a standard contract with data-use and DPA clauses.

**Postpone:** buying GPUs, continued pretraining, any run on more than one GPU, paid ads before you know your buyer, and full-time hires.

### 13.4 Free compute and credits: real versus marketing

| Source | Genuinely accessible to you now? | Notes |
| --- | --- | --- |
| Kaggle notebooks | Yes | About 30 GPU-hours a week on T4 x2 or P100; fine for QLoRA up to 8B |
| Google Colab free tier | Yes, unreliable | Variable T4 availability; not for long runs |
| Free models on OpenRouter (e.g. Gemma 4 31B, Nemotron 3.5 variants listed at $0) | Yes | Rate-limited; good for dev and synthetic data drafts |
| Managed fine-tuning trial credits | Sometimes | Check current offers; earlier Tinker coverage mentioned $150 for new users ([DataCamp tutorial](https://www.datacamp.com/tutorial/tinker-tutorial)) |
| Cloud entry tiers (Google Start, Microsoft entry, AWS Founders) | Yes, small | Hundreds to low thousands of dollars; mostly general cloud, not big GPUs |
| Cloud AI tiers up to $350k | No, until you have institutional investors | Marketing headline for funded startups |
| IndiaAI compute portal | Possibly, after company registration | Subsidized GPUs at a reported ₹65 to ₹150 an hour; application-gated, demand exceeds supply |
| Google TPU Research Cloud | Possibly | Free TPUs for accepted researchers and individuals; expects public output; verify program status |
| Academic partnerships | Yes, if you have a professor sponsor | Your university's GPU cluster plus a co-authored paper from the section 8 experiment is the cheapest serious compute available |
| NVIDIA Inception | Membership, yes; compute credits, generally no | Discounts and visibility rather than free GPUs |

## 14. Go/no-go decision gates

Each gate has a measurable pass condition and a cumulative spend cap; no money moves to the next phase until the gate is passed, and model training is funded only after Gate 2 shows the weights beat the no-training system.

*Roadmap: Gate 0 Real pain (weeks 0 to 2, cap Rs 10k), Gate 1 Paid pilot (weeks 3 to 6, cap Rs 35k), Gate 2 Weights earn it (weeks 7 to 12, cap Rs 1L), Gate 3 Repeatable (months 4 to 6, cap Rs 5L), Gate 4 Fundable (months 7 to 12). Interactive version in the "Go/no-go gates" section of the web page.*

Gate 2 is the Void Models decision: a Pareto win under the section 8 rules keeps model training in the company; anything less turns Void Models into a software-first company that may still use small models as a cost lever.

**Precise definitions for the gates:**

- **Accuracy bar (Gate 1):** agreed with the customer before the pilot, measured on their data, for example 97% field-level accuracy on extracted documents or 70% merged-without-edit CI fixes.
- **Pareto win (Gate 2):** non-inferior to the best frontier system within the margin the eval can resolve (4 points on a 1,000-task eval), at no more than one third of its full cost per accepted result and no more than the cheap tier's (section 8.7). Where the customer requires on-premises data, Gate 2 is instead met by hitting their accuracy bar at acceptable cost.
- **Gross margin (Gate 3):** revenue minus inference, hosting, payment fees and support labour, excluding founder salary.
- **Ported in under a week (Gate 4):** a new open base model fine-tuned and evaluated on frozen test sets within 7 days; customer acceptance and migration may take longer, but the pipeline, not one checkpoint, is the asset.

## 15. Investment committee decision

**Decision: decline Void Models as a model-first company; back a software-first, verifier-centric vertical AI business in small tranches, with small-model training as an option exercised only at Gate 2.** The model-first path has a low chance of revenue inside a year and its main asset, fine-tuned weights, depreciates within months; the software-first path reaches revenue in weeks and accumulates the data and evals that make training worthwhile later.

### 15.1 Scorecard

Probabilities are subjective judgments, informed by this report and by two independent adversarial reviews run on its draft (GPT-6.1 Sol and Gemini 4 Argon via OpenRouter); where they disagreed, the range is shown. They are not empirical forecasts.

| Dimension | Model-first Void Models | Software-first vertical AI (recommended) |
| --- | --- | --- |
| Strict accuracy win over the best current frontier on one narrow, verifiable task | 15% to 25% overall; under 5% in coding or CI repair; about 35% on non-public local formats | Not required |
| Cost-adjusted (Pareto) win under section 8 rules | 30% to 45% | Optional upside |
| Paying customers within 6 months | 15% to 25% | 50% to 70% |
| ₹1 lakh collected revenue within 3 months | Under 10% | 40% to 65% |
| Capital intensity | Low cash, high founder time; rises with RL and eval scope | Low |
| Time to first revenue | 6 to 12+ months | 1 to 3 months |
| Largest downside | Fine-tunes obsoleted by the next open release; no buyer | Low defensibility; trapped in custom services |
| Opportunity cost versus ordinary AI SaaS | High | This is the ordinary AI SaaS path, plus an option on models |

### 15.2 Scenarios over 24 months

| Path | Case | Probability (subjective) | What it looks like |
| --- | --- | --- | --- |
| Software-first | Optimistic | 15% | 30 to 80 customers in one vertical, ₹3 to 6 lakh monthly revenue, an own adapter that cuts cost per accepted result 40%+ or unlocks on-premises deals; seed-fundable |
| Software-first | Base | 50% | Services plus a small product at ₹50,000 to ₹2 lakh monthly revenue; founder modestly paid; training marginal; real eval and data assets |
| Software-first | Pessimistic | 35% | Under ₹30,000 monthly revenue after 12 months, ₹1 to 5 lakh spent; stop or pivot; the founder is highly employable as an applied-AI engineer |
| Model-first | Optimistic | 5% | A well-regarded open specialist model with users but little revenue; acqui-hire interest |
| Model-first | Base | 35% | Published fine-tunes and benchmarks, some attention, negligible revenue |
| Model-first | Pessimistic | 60% | Weights overtaken by the next open base within months; money and 6 to 12 months spent |

Probability-weighted monthly revenue at month 24, using scenario midpoints: about ₹1.35 lakh for software-first versus under ₹5,000 for model-first. The gap is driven by time to first customer, not by model quality.

### 15.3 If this were my own money

| Budget | What I would fund | What I would refuse |
| --- | --- | --- |
| ₹25,000 | Paid concierge pilots in one vertical with checkable outputs (₹10,000), an eval set and baselines on real customer samples (₹5,000), hosting and tools (₹5,000), reserve (₹5,000) | Any GPU spend beyond free Kaggle trials |
| ₹1 lakh | All of the above, plus the minimum-viable section 8 experiment on the customer's own task (₹25,000 to ₹35,000 including a small GRPO pilot), a first product version, company registration; half held as runway until Gate 2 | Large RL runs, 27B-scale training, paid ads |
| ₹5 lakh | Only after Gates 2 and 3: up to ₹1 lakh on model work if Gate 2 passed, the rest on product, distribution, a part-time engineer and compliance; never more than a quarter on training | Pretraining, continued pretraining, GPU purchases, hires before references |

At every budget I would invest in an AI application for one workflow, not in specialized model training. The reasons are in the numbers: compute is cheap but evaluation, data and sales are not (section 7); the cheapest frontier tier fell 10x in a year (section 11); and the business models that work sell outcomes, not tokens (section 10).

## Answers to the five questions

**Q1. Can a 3B to 14B specialized model genuinely outperform today's strongest frontier models on a narrow, valuable task? Under exactly what conditions?**

Yes, but rarely on raw accuracy against the best current system given equal tools; usually on cost-adjusted terms. A strict accuracy win needs all of: (1) a narrow, stable input distribution; (2) machine-checkable outputs; (3) thousands of verified in-distribution examples the frontier never saw, such as non-public document formats, a proprietary label scheme or a customer's own history; (4) a frontier competitor given the same context, tools and attempt budget; (5) a gain that holds on fresh, private and out-of-distribution tests with clustered statistics. Subjective odds: 15% to 25% overall, under 5% in coding, about 35% on non-public local formats. Open-ended software engineering is not such a task: the best open models of 30B or less trail the best closed systems by roughly 10 to 18 points on SWE-bench Verified.

**Q2. What is the highest-leverage affordable method to achieve this without renting massive GPU clusters?**

A deterministic verifier and a frozen eval set first; then rejection-sampled distillation from a permissively licensed open teacher into a LoRA adapter on a 9B to 27B open base; served with best-of-N against the verifier and escalation to a frontier model; GRPO only if pass@16 sits well above pass@1 after SFT. Compute per cycle is ₹1,000 to ₹10,000 on one rented GPU or a managed API. Because a permitted open teacher is weaker than the frontier, the decisive signal must come from the verifier and from customer data, not from the teacher.

**Q3. Is training our own model weights necessary to build a profitable business, or an avoidable expense?**

Avoidable at the start. Train only when at least one holds: inference is 15% to 25%+ of revenue and a small model cuts full cost per accepted result by half at equal verified accuracy; the customer requires on-premises, air-gapped or data-resident processing; a latency budget cheap APIs cannot meet; or a proprietary label space where prompting plateaus below the agreed accuracy bar. Even then it means a LoRA adapter on an open base, never your own pretraining.

**Q4. What is the most realistic way to reach the first ₹1 lakh in revenue, and then a sustainable company?**

Paid concierge pilots in one vertical with checkable outputs: for example four pilots at ₹25,000 with mid-size Indian CA practices, logistics firms or customs brokers drowning in messy local documents, or one or two projects at $1,000 to $2,500 for overseas small businesses with the same workflow. Deliver with cheap and frontier APIs, validators, and yourself in the loop. Then convert pilots to monthly contracts priced on volume or outcome (at least ₹4,000 a month in India, $49 to $500 abroad), productize the repeated workflow, keep contractual data rights, and move through Gates 2 to 4; adapters come only when Gate 2 passes.

**Q5. If the evidence does not support the hypothesis, what company would I build with these skills and this capital?**

The same company without the model-lab ambition: **a verifier-first document and reconciliation automation business for accounting workflows**, starting with Indian CA firms and mid-market finance teams, then, once SOC 2 or ISO 27001 is affordable, India's offshore accounting centres serving US firms. The US accountant workforce was about 1.78 million in 2024, roughly 10% below 2019, and firms such as RSM US, Moss Adams, Sikich and CohnReznick are expanding in India ([Allwork, citing Reuters](https://allwork.space/2025/04/us-accounting-firms-turn-to-india-as-cpa-shortage-worsens/), grade C). Expect competition from Thomson Reuters, Wolters Kluwer, BlackLine and funded AI accounting startups, so win on one painful local workflow first. Second choice: an evaluation-and-reliability service that builds eval suites, regression tests and verifiers for companies deploying LLMs, which uses exactly the skills this plan develops.

## Sources

As of 23 October 2026. Foundational papers are linked inline where cited. They come from the published literature and were not re-opened for this report.

**Opened for this report (current data)**

| Source | Date | Grade | Used for |
| --- | --- | --- | --- |
| [OpenRouter models API](https://openrouter.ai/api/v1/models) | Read 23 Oct 2026 | B | Prices for 468 models, release dates |
| [open.er-api USD rates](https://open.er-api.com/v6/latest/USD); Frankfurter (ECB) | 23 and 22 Oct 2026 | A | ₹94.90 and ₹94.82 per USD |
| [llm-stats SWE-bench Verified](https://llm-stats.com/benchmarks/swe-bench-verified) | Updated 23 Oct 2026 | C | Coding scores and sizes |
| [Liu et al., Coding Agents Have Converged](https://arxiv.org/abs/2609.17394) | 15 Sep 2026 | B | Leaderboard resolution, scaffold effects |
| [Runpod pricing](https://www.runpod.io/pricing) | Updated 8 Oct 2026 | B | GPU prices |
| [Lambda pricing](https://lambda.ai/pricing) | Read 23 Oct 2026 | B | GPU prices |
| [Jarvislabs pricing](https://jarvislabs.ai/pricing) | Read 23 Oct 2026 | B | Indian provider GPU prices |
| [amcompute GPU price aggregator](https://www.amcompute.com/gpu-cloud-prices) | Updated 17 Oct 2026 | C | Lowest and median GPU prices |
| [Tinker pricing](https://thinkingmachines.ai/tinker/pricing/) | 19 Oct 2026 | B | Managed SFT and RL costs |
| [Fireworks pricing](https://fireworks.ai/pricing) | Read 23 Oct 2026 | B | LoRA SFT prices, LoRA serving at base price |
| [Unsloth requirements](https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements) | Read 23 Oct 2026 | B | VRAM minimums |
| [Anthropic help: training on outputs](https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model) | 16 Mar 2026 | A | Distillation restriction |
| [Security document classification preprint](https://arxiv.org/abs/2605.20368) | 19 May 2026 | C | Specialization evidence |
| [Huynh and Pham, anxiety detection](https://arxiv.org/abs/2609.20847) | Aug 2026 | B | Counter-evidence: shortcut learning |
| [Kaliaperumal, BERT to Frontier Agents](https://arxiv.org/abs/2608.13675) | 13 Aug 2026 | C | Context on task-targeted models (abstract only) |
| [Communications Today on IndiaAI](https://www.communicationstoday.co.in/indiaai-mission-faces-gap-between-capacity-and-deployed-gpus/) | 1 Oct 2026 | C | IndiaAI capacity, GPU-hours sanctioned |

**Secondary (from search summaries; verify before relying on them)**

| Topic | Source | Grade |
| --- | --- | --- |
| Qwen3.6-27B scores and license | [NVIDIA NGC listing](https://catalog.ngc.nvidia.com/orgs/nim/teams/qwen/models/qwen3.6-27b) | C |
| Qwen3.8-27B release, license dispute | [Let's Data Science](https://letsdatascience.com/news/alibaba-releases-qwen38-27b-for-local-ai-workloads-ce2a4a89) | C |
| Gemma 4 sizes and Apache-2.0 license | [The Decoder](https://the-decoder.com/googles-gemma-4-is-now-available-with-apache-2-0-licensing-for-the-first-time/) | C |
| Meta Muse Glimmer-30B | [Implicator](https://www.implicator.ai/meta-releases-30b-open-weight-muse-glimmer-and-promises-spark-1-2-weights/) | C |
| IBM Granite 4.2 | [Crypto Briefing](https://cryptobriefing.com/ibm-granite-4-2-models-local-deployment/) | C |
| SWE-bench Pro snapshot | [Morph tracker](https://www.morphllm.com/swe-bench-pro) | C |
| Cursor Composer 2 on Kimi K2.5 | [DataCamp](https://www.datacamp.com/blog/composer-2) | C |
| Fast-apply model claims | [Morph comparisons](https://www.morphllm.com/comparisons) | C |
| BIRD text-to-SQL results | [Technology Magazine](https://technologymagazine.com/news/huawei-datagallery-tops-benchmark-for-enterprise-text2sql-ai) | C |
| METR 2025 RCT and 2026 follow-up | [METR paper](https://arxiv.org/abs/2507.09089); [METR](https://metr.org/) | A / B |
| Enterprise AI spend, open-source share | [Menlo Ventures 2025 report](https://menlovc.com/perspective/2025-the-state-of-generative-ai-in-the-enterprise/) | C |
| OpenPipe acquisition | [CoreWeave](https://coreweave.com/news/coreweave-to-acquire-openpipe-leader-in-reinforcement-learning) | A |
| Support AI pricing | [Gleap on Intercom Fin](https://www.gleap.io/blog/intercom-fin-ai-pricing-2026) | C |
| Code review pricing | [CodeRabbit tracker](https://www.usagepricing.com/blueprint/coderabbit) | C |
| Document extraction pricing | [imagetotable comparison](https://imagetotable.ai/blog/document-extraction-pricing-breakdown-2026) | C |
| Indian CA tool pricing | [ICAI use-case page](https://ai.icai.org/usecases_details.php?id=250) | C |
| E2E Networks H100 price | [E2E](https://www.e2enetworks.com/gpus/nvidia-h100.md) | B |
| DPDP Rules timeline | [Mondaq](https://www.mondaq.com/india/data-protection/1708164/digital-personal-data-protection-rules-2025-notified) | B |
| Sarvam Indic claims | [BuildFastWithAI](https://www.buildfastwithai.com/blogs/sarvam-105b-india-s-open-source-llm-for-22-indian-languages-2026) | C |
| India RCM market | [Ken Research](https://www.kenresearch.com/india-revenue-cycle-management-market) | D |
| AI developer rates | [Ad Snipper](https://adsnipper.com/blog/cost-to-hire-ai-developer/) | C |
| Startup credit programs | [Better Stack comparison](https://betterstack.com/community/comparisons/best-startup-cloud-credit-programs/) | C |
| TPU Research Cloud | [TRC](https://sites.research.google/trc/about/) | B |
| US accountant shortage, India shift | [Allwork, citing Reuters](https://allwork.space/2025/04/us-accounting-firms-turn-to-india-as-cpa-shortage-worsens/) | C |
| RTX 5090 tensor throughput | [NVIDIA developer forum](https://forums.developer.nvidia.com/t/rtx-5090-peak-bf16-tensor-tflops/350543) | C |

**Open questions not resolved here:** OpenAI's exact current terms on competing models (page blocked), DeepSeek V4's license text, IndiaAI portal eligibility for unfunded student companies, and measured (not calculated) training throughput on your own data.
