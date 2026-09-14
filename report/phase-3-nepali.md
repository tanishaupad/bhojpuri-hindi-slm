# Phase 3: Nepali reasoning finetuning

## Status

The Nepali Phase 3 implementation, deterministic reasoning release, main 20k finetuning run,
held-out evaluation, retention comparison, attention analysis, and sample-size ablations are
complete. The final main checkpoint was selected at step 2,500 using all 2,000 validation examples.

## Dataset

The release contains 25,000 independently generated Nepali examples over direct comparison,
largest/smallest selection, equality, and two-hop transitivity:

| Split | Examples | Purpose |
|---|---:|---|
| Train | 20,000 | Main supervised finetuning |
| Validation | 2,000 | Full-split checkpoint selection |
| Test | 2,000 | Primary held-out evaluation |
| Challenge names | 500 | Names absent from training |
| Challenge wording | 500 | Question variant absent from training |

All 25,000 answers were derived by the symbolic solver and independently audited during release.
Worlds and normalized prompts are disjoint across splits. Complete prompt-plus-answer sequences are
at most 52 tokens with the production 16k Nepali tokenizer, contain no unknown token, and fit within
the model's 256-token context. The compact release evidence is under
`report/audits/generated/phase3/reasoning_nepali_v1/` and the generated data are under
`tmp/phase3/reasoning_v1/nepali/`.

## Main experiment

The Nepali run matches the established Hindi/Bhojpuri Phase 3 protocol: full-model causal
finetuning from the final Phase 2 checkpoint, target-only loss, effective batch size 16, 2,500
optimizer steps, AdamW, 100-step warmup, and cosine decay from `1e-4` to `1e-5`. Checkpoint
selection evaluates all 2,000 validation examples across all four task families.

```bash
python -m lma_phase3.finetune \
  --config nepali/configs/phase3.toml \
  --pretrained-checkpoint /artifact-root/nepali/phase2/runs/pretrain_v1/checkpoint_step_0016720.pt \
  --train-data tmp/phase3/reasoning_v1/nepali/train.jsonl.gz \
  --val-data tmp/phase3/reasoning_v1/nepali/validation.jsonl.gz \
  --template-config nepali/configs/phase3_templates.json \
  --output-dir tmp/phase3/runs/nepali/finetune_v1 \
  --device cuda
```

Evaluate the pretrained baseline and selected finetuned checkpoint on validation, test, unseen-name,
and unseen-wording splits. The finetuned evaluator runs both the 16-token answer protocol and the
96-token explanation protocol and reports explicit-answer accuracy, explanation verification,
joint success, answer presence, and task-family breakdowns.

```bash
python -m lma_phase3.evaluate_finetuned \
  --config nepali/configs/phase3.toml \
  --checkpoint tmp/phase3/runs/nepali/finetune_v1/best_checkpoint.pt \
  --dataset-root tmp/phase3/reasoning_v1 \
  --output-dir report/results/generated/phase3/finetuned_v1/nepali \
  --device cuda --protocols both
```

Generic-language retention must be evaluated on the unchanged Phase 2 packed Nepali test stream.
Attention comparison should use the Phase 2 and selected Phase 3 checkpoints on one fixed Nepali
reasoning prompt, with the same layers and heads used for Hindi.

## Main results

The main run processed 1,468,136 supervised target tokens. Its final and best validation loss was
0.0131 (perplexity 1.013). The selected checkpoint SHA-256 is
`ded00eaeaa650f6ecbbb8550b41ff95abe2f0f584962b500baf5cf571a9bb745`.

| Evaluation | Explicit answer | Verified explanation and answer |
|---|---:|---:|
| Primary test, 16-token protocol | 56.30% | 0.00% |
| Primary test, 96-token protocol | **78.95%** | **78.55%** |
| Unseen names, 96-token protocol | 14.40% | 0.00% |
| Unseen wording, 96-token protocol | 58.60% | 57.40% |

On the 96-token primary test, task-level joint verification was 86.2% for direct comparison, 69.0%
for extremum selection, 59.0% for equality, and 100.0% for two-hop transitivity. The unseen-name
result shows that the model did not reliably transfer entity generation to names absent from
training. The unseen-wording result shows partial transfer to the reserved question wording.

Generic Nepali test perplexity rose from 48.85 before finetuning to 23,992.87 after the main 20k
run; bits per byte rose from 0.4377 to 1.1352. This is strong evidence of task specialization and
loss of generic language-model retention under full-model finetuning.

Attention changes were descriptive rather than causal. Mean layer-1 entropy was nearly unchanged
(2.669 to 2.664), while mean layer-8 entropy fell from 2.108 to 1.770 and mean layer-8 attention
distance increased from 9.639 to 12.196 tokens. Six paired heatmaps and the complete 54-head table
are under `report/figures/generated/phase3/attention/nepali/`.

## Sample-size ablations

The ablation sweep uses balanced, deterministic subsets of 1k, 3k, 5k, 8k, and 12k examples, two
epochs per subset, plus the pretrained baseline and main 20k model when available. For every budget
it measures generic-test perplexity/BPB, reasoning validation loss, and a fixed 100-example direct
comparison accuracy diagnostic.

```bash
python scripts/ablation_samples_vs_ppl.py \
  --languages nepali \
  --artifact-root /artifact-root \
  --dataset-root tmp/phase3/reasoning_v1 \
  --work-dir tmp/phase3/ablation_nepali \
  --output-dir report/results/generated/phase3/ablation_nepali \
  --device cuda
```

The final comparison should report Hindi and Nepali results under identical splits, generation
budgets, and metrics. Token-level perplexity should be interpreted alongside bits per byte because
the two models use different vocabulary sizes.

### Ablation results

| Samples | Generic PPL | Generic BPB | Reasoning val loss | Direct diagnostic accuracy |
|---:|---:|---:|---:|---:|
| 0 | 48.85 | 0.4377 | 2.4357 | 0% |
| 1,000 | 76.18 | 0.4877 | 0.0994 | 49% |
| 3,000 | 99.43 | 0.5177 | 0.0502 | 49% |
| 5,000 | 175.19 | 0.5815 | 0.0431 | **56%** |
| 8,000 | 411.36 | 0.6776 | 0.0203 | 42% |
| 12,000 | 1,730.74 | 0.8393 | 0.0191 | 44% |
| 20,000 | 23,992.87 | 1.1352 | **0.0131** | 52% |

The diagnostic accuracy uses only the first 100 direct-comparison validation examples, so it is not
a substitute for the full held-out evaluation. Within that diagnostic, 5k provides the highest
accuracy and a much smaller retention penalty than 20k. Selecting a final alternative checkpoint
would require full four-family evaluation of the candidate rather than choosing solely from this
100-example diagnostic.
