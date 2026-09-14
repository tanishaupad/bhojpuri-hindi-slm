# Phase 3: Hindi Model H and Nepali Model L

**Author:** Tanisha Upadhyay

**Course:** CL3-410 / Language Modeling and Applications

**Date:** September 2026
**Status:** Main experiments and sample-size ablations complete

## 1. Scope

Phase 3 studies whether the two independently pretrained models learn controlled relational
reasoning after supervised finetuning:

- **Model H:** Hindi, using the 24,000-piece Hindi tokenizer and final Hindi Phase 2 checkpoint.
- **Model L:** Nepali, using the 16,000-piece Nepali tokenizer and step-16,720 Nepali Phase 2
  checkpoint.

The models do not share tokenizers, embeddings, weights, or reasoning examples. Both use the same
four task families, dataset sizes, training objective, generation budgets, and deterministic
verification rules. This makes Hindi–Nepali the primary comparison throughout this report.

## 2. Reasoning datasets

Each language has 25,000 native-language examples. Answers follow only from facts stated in the
prompt; no outside knowledge is needed.

| Split | Examples per language | Purpose |
|---|---:|---|
| Training | 20,000 | Supervised reasoning finetuning |
| Validation | 2,000 | Checkpoint selection and development evaluation |
| Primary test | 2,000 | Familiar names and question wording |
| Challenge names | 500 | Names excluded from training |
| Challenge wording | 500 | Question variant excluded from training |

The four balanced families are direct comparison, three-entity maximum/minimum selection, value
equality, and two-hop transitivity. Dataset construction rejects duplicate worlds, duplicate
normalized prompts, unknown tokens, and sequences exceeding the 256-token context. The Nepali
release contains 25,000 unique worlds, has zero cross-split overlap, and has a maximum serialized
length of 52 tokens.

## 3. Finetuning protocol

Both models use full-model causal finetuning. Prompt and padding positions are masked, so loss is
computed only over the target answer and explanation. Optimization uses AdamW, an effective batch
size of 16, 100 warmup steps, and cosine learning-rate decay from `1e-4` to `1e-5`. The main runs
train for 2,500 optimizer steps.

Nepali checkpoint selection evaluates all 2,000 validation examples across all four task families.
The historical Hindi run originally selected checkpoints with a smaller validation subset; the
reported Hindi full-validation value is a post-hoc evaluation of the retained step-2,500 model.

| Result | Hindi Model H | Nepali Model L |
|---|---:|---:|
| Main training examples | 20,000 | 20,000 |
| Optimizer steps | 2,500 | 2,500 |
| Full reasoning validation loss | 0.0082 | 0.0131 |
| Full reasoning validation perplexity | 1.008 | 1.013 |
| Nepali target tokens processed | — | 1,468,136 |

The selected Nepali checkpoint SHA-256 is
`ded00eaeaa650f6ecbbb8550b41ff95abe2f0f584962b500baf5cf571a9bb745`.

## 4. Evaluation protocol

Evaluation uses greedy decoding under two protocols:

1. The **original protocol** permits 16 new tokens and tests short answer recovery.
2. The **explanation protocol** permits 96 new tokens and asks for an answer followed by a reason.

Explicit-answer accuracy extracts the declared answer field and compares it exactly with the gold
answer. Joint verification requires both the correct explicit answer and a fully supported
explanation. The symbolic checker accepts only complete language-specific value and comparison
claims, rejects unsupported claims, and requires enough premises to prove the result.

## 5. Main held-out results

### 5.1 Primary test

| Metric | Hindi Model H | Nepali Model L |
|---|---:|---:|
| Explicit answer, 16-token protocol | 61.55% | 56.30% |
| Explicit answer, 96-token protocol | **86.20%** | 78.95% |
| Joint verified answer and explanation | **86.15%** | 78.55% |

The longer protocol substantially improves both answer recovery and proof completion. The
16-token protocol truncates explanations and therefore has zero full-proof verification for both
models; this is a generation-budget limitation rather than evidence that no reasoning was learned.

### 5.2 Reasoning-family breakdown under the 96-token protocol

| Family | Hindi joint verification | Nepali joint verification |
|---|---:|---:|
| Direct comparison | **94.80%** | 86.20% |
| Extremum | **83.00%** | 69.00% |
| Equality | **66.80%** | 59.00% |
| Two-hop transitivity | 100.00% | 100.00% |
| Overall | **86.15%** | 78.55% |

Both models verify every familiar-name two-hop transitivity case. Hindi is stronger on the other
three families, especially extremum selection. These results establish performance on the released
controlled tasks; they do not establish general mathematical or linguistic reasoning.

### 5.3 Generalization challenges

| Challenge, 96-token protocol | Hindi explicit answer | Hindi joint | Nepali explicit answer | Nepali joint |
|---|---:|---:|---:|---:|
| Unseen names | 15.80% | 0.00% | 14.40% | 0.00% |
| Unseen wording | 66.20% | **64.60%** | 58.60% | 57.40% |

Both models transfer partially to reserved wording but fail to produce verified reasoning with new
names. Equality questions use fixed yes/no labels, so aggregate unseen-name answer accuracy can be
nonzero even when unfamiliar entity names are not generated reliably. Token fragmentation and name
familiarity are plausible contributors, but this experiment does not isolate their causal effects.

## 6. Generic-language retention

The original Phase 2 test streams were scored before and after the 20k reasoning finetune.

| Language | Pretrained PPL | Finetuned PPL | PPL multiplier | Pretrained BPB | Finetuned BPB |
|---|---:|---:|---:|---:|---:|
| Hindi | 39.57 | 628.12 | 15.87× | 0.5282 | 0.9252 |
| Nepali | 48.85 | 23,992.87 | 491.19× | 0.4377 | 1.1352 |

Reasoning behavior improves while generic next-token prediction degrades substantially. The much
larger Nepali increase indicates stronger task specialization under this schedule. BPB is included
because token-level perplexity is not directly comparable across the 24k Hindi and 16k Nepali
tokenizers.

## 7. Sample-size ablations

Each ablation uses a deterministic, family-balanced subset and two training epochs. Accuracy is a
diagnostic on the same first 100 direct-comparison validation examples; it is narrower than the
complete held-out evaluation.

| Samples | Hindi PPL | Hindi accuracy | Nepali PPL | Nepali accuracy |
|---:|---:|---:|---:|---:|
| 0 | 39.57 | 0% | 48.85 | 0% |
| 1,000 | 55.30 | 49% | 76.18 | 49% |
| 3,000 | 68.89 | 52% | 99.43 | 49% |
| 5,000 | 87.38 | 40% | 175.19 | **56%** |
| 8,000 | 158.50 | 52% | 411.36 | 42% |
| 12,000 | 333.86 | 41% | 1,730.74 | 44% |
| 20,000 | 628.12 | **85%** | 23,992.87 | 52% |

For Nepali, 5k gives the highest direct diagnostic accuracy with far less forgetting than 20k.
That result does not by itself make 5k the preferred final checkpoint: selection would require the
complete four-family test and both challenge evaluations. The non-monotonic diagnostic accuracy
also shows why reasoning validation loss alone is insufficient for choosing a sample budget.

## 8. Attention analysis

Attention was extracted from the pretrained and finetuned models on fixed same-language prompts.
The values summarize probability concentration and backward distance; they do not identify causal
reasoning circuits.

| Layer summary | Hindi Model H | Nepali Model L |
|---|---:|---:|
| Layer 1 entropy, pre → fine | 2.706 → 2.644 | 2.669 → 2.664 |
| Layer 1 distance, pre → fine | 11.144 → 11.318 | 10.078 → 10.599 |
| Layer 8 entropy, pre → fine | 1.990 → 1.970 | **2.108 → 1.770** |
| Layer 8 distance, pre → fine | 10.326 → 10.400 | **9.639 → 12.196** |

Nepali shows a marked late-layer shift toward sharper, longer-range attention on the selected
prompt. Hindi changes are smaller on its selected prompt. These are descriptive examples rather
than population-level or causal findings.

## 9. Conclusions

Supervised finetuning teaches both independently pretrained models the controlled reasoning task.
Hindi Model H achieves higher overall joint verification than Nepali Model L, while both achieve
500/500 on familiar-name two-hop transitivity. Both fail the unseen-name proof challenge and both
lose generic-language modeling quality, with a substantially larger retention cost for Nepali.

The sample-size study suggests that Nepali can acquire useful direct-comparison behavior with much
less than 20k examples. A follow-up should evaluate the 3k, 5k, and 8k Nepali checkpoints across all
four task families and challenges, and should test parameter-efficient or mixed-objective
finetuning to reduce forgetting.

## 10. Reproducibility artifacts

- Nepali release evidence: `report/audits/generated/phase3/reasoning_nepali_v1/`
- Hindi evaluation: `report/results/generated/phase3/finetuned_v1/hindi/`
- Nepali evaluation: `report/results/generated/phase3/finetuned_v1/nepali/`
- Nepali ablations: `report/results/generated/phase3/ablation_nepali/`
- Attention tables and heatmaps: `report/figures/generated/phase3/attention/`
- Nepali training log: `report/logs/generated/phase3/nepali_finetune_training.csv`
- Nepali checkpoint manifest: `report/results/generated/phase3/nepali_checkpoint_manifest.json`
- Reproduction runner: `kaggle/run_phase3_nepali.py`

Large resume-capable checkpoints and generated dataset files remain outside Git. Their hashes and
reproduction commands are recorded in the manifests and companion reports.
