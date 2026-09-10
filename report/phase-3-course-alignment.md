# Phase 3 course-alignment evaluation

Updated 11 September 2026. The assigned comparative/ordering question structures, reasoning_v1 datasets, tokenizers, trained weights, and historical prediction files are unchanged. This update adds evaluation and changes the default validation scope for future training runs.

## 1. Generic-text retention after supervised finetuning

The SFT lecture separates task behavior from generic language modeling (Lecture 9, slide 26). Both original Phase 2 test streams were scored with the pretrained and best finetuned checkpoint in float32 on MPS, using the same 256-target windows and final partial window. The streams and tokenizers match the published SHA-256 hashes. Hindi has 3,521,022 scored targets and Bhojpuri 222,584. EOS likelihood is included; BPB divides by the original text UTF-8 byte count.

| Language | Checkpoint | Cross-entropy (nats/token) | Perplexity | Bits per byte |
|---|---|---:|---:|---:|
| Hindi | Pretrained | 3.6781 | 39.57 | 0.5282 |
| Hindi | Finetuned | 6.4427 | 628.12 | 0.9252 |
| Bhojpuri | Pretrained | 4.1420 | 62.93 | 0.6503 |
| Bhojpuri | Finetuned | 8.5049 | 4,939.06 | 1.3353 |

Generic-text perplexity increases **15.87x for Hindi** and **78.49x for Bhojpuri**. These runs improve benchmark response behavior while losing substantial predictive performance on their original language-modeling distribution. This is consistent with forgetting/task specialization; it does not measure how much individual factual knowledge was erased. The very low reasoning-target validation perplexities in the main report measure a different distribution and cannot substitute for this check.

The original raw-text packing is retained deliberately for a paired comparison. Its known whitespace-normalization/unknown-token limitation remains; the update does not silently change test tokens or claim a corrected pretraining experiment.

Evidence: [Hindi retention](results/generated/phase3/retention_v1/hindi.json), [Bhojpuri retention](results/generated/phase3/retention_v1/bhojpuri.json). Each file records checkpoint, tokenizer, token-stream, manifest and evaluator hashes, device, precision, token count and byte denominator.

## 2. Explicit-answer accuracy, explanation verification, and joint success

The new metric extracts the declared answer field instead of searching for the reference anywhere in an explanation. It removes an optional leading `उत्तर:`/`जवाब:` label. With `कारण:`, the entire preceding span is the answer field; otherwise the first sentence or bare response is used. Only terminal punctuation and NFC/whitespace are normalized. Comparison with the gold answer is exact. Missing or noncanonical extracted fields (including multiple or repeated answers within that field) count as incorrect; no answer-choice search or paraphrase matching occurs. This is format-based answer accuracy, not a general semantic judge.

**Explanation verification** reuses the existing grammar-limited semantic checker unchanged. **Explicit joint success** requires a correct extracted answer AND a fully verified explanation. Historical whole-output exact match, answer presence, and the older semantic joint metric remain available separately.

The rule was applied to all 8,000 validation continuations and 16 finetuned success/failure examples were inspected before the held-out rescore; no extractor changes followed that inspection. This is a developer check, not independent human calibration. Earlier work had inspected test outputs, so the results are explicitly **retrospective**, not preregistered. The frozen [protocol](results/generated/phase3/explicit_answer_v1/protocol.json) records the rule and validation evidence hashes.

### Primary test: 2,000 examples per language

| Language | Protocol | Pretrained answer accuracy | Finetuned answer accuracy | Finetuned explanation verification | Explicit joint success | Answer presence (diagnostic) |
|---|---|---:|---:|---:|---:|---:|
| Hindi | original, 16 tokens | 0.00% | 61.55% | 0.00% | 0.00% | 83.85% |
| Hindi | explanation, 96 tokens | 0.00% | 86.20% | 86.15% | 86.15% | 91.70% |
| Bhojpuri | original, 16 tokens | 0.00% | 50.15% | 0.00% | 0.00% | 85.00% |
| Bhojpuri | explanation, 96 tokens | 0.00% | 71.40% | 69.90% | 69.90% | 89.15% |

A correct name copied into a premise cannot rescue an incorrect answer field. The 16-token protocol frequently truncates explanations: its zero verification rate is bounded by that output budget and must not be presented as evidence that the models never produce valid explanations. Prompt wording and generation budgets differ between protocols, so comparisons between 16 and 96 tokens do not isolate a single causal factor.

### Challenges under the 96-token protocol: 500 examples each

| Language | Challenge | Explicit-answer accuracy | Explanation verification | Explicit joint success |
|---|---|---:|---:|---:|
| Hindi | challenge_wording | 66.20% | 79.00% | 64.60% |
| Hindi | challenge_names | 15.80% | 0.00% | 0.00% |
| Bhojpuri | challenge_wording | 69.40% | 68.00% | 68.00% |
| Bhojpuri | challenge_names | 13.60% | 0.00% | 0.00% |

The three evaluation regimes mirror SFT lecture slides 24–25: familiar entities/wording, held-out names, and held-out wording. The wording challenge changes only the reserved question variant. Name-challenge accuracy includes equality questions with yes/no answers, so a nonzero aggregate does not establish reliable generation of unseen names. Zero joint verification on that challenge remains a central limitation.

The [complete summary](results/generated/phase3/explicit_answer_v1/scores/summary.json) includes validation, primary test, both challenges, each task family, both languages, and both model stages under both protocols: **40,000 saved continuations**. Per-example compressed scores and input/output SHA-256 hashes are stored beside it. Historical prediction files are read only.

## 3. Representative validation for new training runs

`lma_phase3.finetune` now defaults to `--validation-scope full`. Every periodic validation pass scores all 2,000 examples, covering direct comparison, extremum, equality and transitivity, and uses token-weighted response loss. Right-padding and prompt labels stay masked. This addresses the earlier first-400-example selection subset, which covered direct comparison only.

The original v1 models were not retrained. Their post-hoc full-validation comparison of retained steps 2,000, 2,250 and 2,500 remains valid and supports step 2,500 among those candidates. New checkpoints record `full_validation_v1`; resume rejects a switch between full and historical subset selection so best-loss values are not mixed. `--validation-scope legacy_subset` retains the original config-limited selection behavior for reproducing/resuming v1. A new full-validation experiment should use a new output directory.

## 4. Interpretation and scope

- Preserve the assignment-specified question structures. Additional wording variety is optional and was not added.
- Describe the 100% transitivity result as 500/500 verified outputs on familiar task structures under the 96-token protocol, not proof of general reasoning.
- Treat subword fragmentation, entity familiarity and corpus breadth as possible explanatory factors. The experiments do not isolate a primary cause.
- Treat attention entropy/distance changes as descriptive observations on selected prompts. They do not prove that particular heads implement necessary reasoning operations.
- Full SFT is a course-supported training choice. This update adds neither preference optimization nor adapters.

## 5. Reproduction

Use the unchanged Phase 2 artifact root and the existing local best SFT checkpoints. The following loop runs both complete retention comparisons:

```bash
artifact_root='/path/to/LMA_Individual_Project'
for language in hindi bhojpuri; do
  if [ "$language" = hindi ]; then step=0020300; else step=0004150; fi
  python -m lma_phase3.evaluate_retention \
    --pretrained-config "$language/configs/phase2.toml" \
    --finetuned-config "$language/configs/phase3.toml" \
    --pretrained-checkpoint "$artifact_root/$language/phase2/runs/pretrain_v1/checkpoint_step_$step.pt" \
    --finetuned-checkpoint "tmp/phase3/runs/$language/finetune_v1/best_checkpoint.pt" \
    --test-manifest "$artifact_root/$language/phase2/tokenized_v1/test.manifest.json" \
    --output "tmp/phase3/retention_reproduced/$language.json" \
    --device mps --batch-size 8
done
```

Use `--device cpu` or `--device cuda` where appropriate; small floating-point differences can occur. Rescore existing predictions into a new, empty output directory:

```bash
python -m lma_phase3.rescore_explicit_answers \
  --dataset-root tmp/phase3/reasoning_v1 \
  --results-root report/results/generated/phase3 \
  --output-dir tmp/phase3/explicit_answer_reproduced
```

The rescore reads the retained pretrained semantic outputs and original finetuned `.jsonl.gz` predictions. If restoring the repository on another machine, restore those finetuned prediction files alongside the saved summaries. The saved explicit-answer evidence contains all rescored continuations and the hashes of the originals.

For new generation, `lma_phase3.evaluate_finetuned` now emits the new answer and explicit-joint fields alongside the existing metrics. No original predictions need to be regenerated for the retrospective comparison.

## 6. Verification

The full existing suite plus the initial new checks passed (166 tests); the final targeted suite passed all 14 checks after adding generation-summary integration coverage. Across these runs, all 167 current tests were exercised. Ruff passes over `src` and `tests`. Regression checks cover wrong answers with the gold name inside a premise, malformed answer fields, duplicate/missing IDs, metric aggregation, full grouped-task validation with token weighting, and equivalence of batched retention scoring to the original Phase 2 evaluator including final partial windows.
