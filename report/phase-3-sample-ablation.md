# Phase 3: Sample Budget and Generic Language Retention

## Experimental scope

Five subset runs per language used 1,000, 3,000, 5,000, 8,000 and 12,000 examples.
The 20,000-example point reuses the historical v1 run; zero is the pretrained checkpoint.
Each subset has equal counts from direct, extremum, equality and transitivity tasks, with
seed 20260823. Subsets are separately sampled, not nested. Two passes at effective batch
size 16 give 125-1,500 steps for the subset runs and 2,500 steps for v1. Peak/final learning
rates are 1e-4/1e-5, with cosine decay and warmup `max(10, min(100, round(steps * 0.1)))`.
Thus sample count, optimizer steps and warmup duration vary together; this experiment does
not isolate the effect of unique examples at fixed compute. There is one seed per condition.

Generic CE/PPL score the complete original Phase 2 **test** streams: 3,521,022 targets for
Hindi and 222,584 for Bhojpuri, including EOS. The raw-text normalization limitation remains.
Reasoning loss/PPL cover all 2,000 validation examples. In contrast, the sweep's accuracy
column and blue plot line use **only the first 100 validation examples, all direct comparisons**,
with original prompts and 16 generated tokens. They are not full-benchmark accuracy.

The choice to inspect Hindi 5k and Bhojpuri 3k followed examination of generic-test PPL.
This is retrospective, exploratory comparison, with no predefined retention threshold or
selection objective. These are illustrative smaller-budget checkpoints, not established
optima. The generic test stream has been consulted during selection and is not an untouched
selection-independent estimate. Future selection should use generic validation data.

## Saved sweep results

| Language | Samples | Steps | Generic CE | Generic PPL | PPL / base | Full validation loss | Full validation PPL | Direct-only accuracy (first 100) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hindi | 0 | 0 | 3.6781 | 39.57 | 1.00x | 2.3798 | 10.80 | 0.0% |
| Hindi | 1,000 | 125 | 4.0128 | 55.30 | 1.40x | 0.0935 | 1.10 | 49.0% |
| Hindi | 3,000 | 375 | 4.2326 | 68.89 | 1.74x | 0.0484 | 1.05 | 52.0% |
| Hindi | 5,000 | 625 | 4.4702 | 87.38 | 2.21x | 0.0294 | 1.03 | 40.0% |
| Hindi | 8,000 | 1,000 | 5.0657 | 158.50 | 4.01x | 0.0239 | 1.02 | 52.0% |
| Hindi | 12,000 | 1,500 | 5.8107 | 333.86 | 8.44x | 0.0196 | 1.02 | 41.0% |
| Hindi | 20,000 | 2,500 | 6.4427 | 628.12 | 15.87x | 0.0082 | 1.01 | 85.0% |
| Bhojpuri | 0 | 0 | 4.1420 | 62.93 | 1.00x | 3.7464 | 42.37 | 0.0% |
| Bhojpuri | 1,000 | 125 | 5.1401 | 170.74 | 2.71x | 0.2119 | 1.24 | 56.0% |
| Bhojpuri | 3,000 | 375 | 5.8646 | 352.35 | 5.60x | 0.0752 | 1.08 | 39.0% |
| Bhojpuri | 5,000 | 625 | 6.1255 | 457.37 | 7.27x | 0.0556 | 1.06 | 32.0% |
| Bhojpuri | 8,000 | 1,000 | 7.0437 | 1,145.64 | 18.21x | 0.0311 | 1.03 | 28.0% |
| Bhojpuri | 12,000 | 1,500 | 7.7134 | 2,238.10 | 35.57x | 0.0232 | 1.02 | 59.0% |
| Bhojpuri | 20,000 | 2,500 | 8.5049 | 4,939.06 | 78.49x | 0.0168 | 1.02 | 49.0% |

## Interpretation

Hindi 5k raises generic PPL from 39.57 to 87.38 (2.21x), versus 628.12 at 20k.
Bhojpuri 3k raises PPL from 62.93 to 352.35 (5.60x), versus 4,939.06 at 20k.
These are substantial losses in generic prediction, although smaller than at 20k. They do
not demonstrate that 3k-5k avoids forgetting or that 5k-8k is a universal safe range.
The results support the TA's suggestion to measure retention rather than accepting a sample
budget without checking it. Bhojpuri already loses predictive performance at 1k (2.71x PPL).

Full-test explicit answer accuracy under the 96-token protocol is 61.90% for Hindi 5k and
45.35% for Bhojpuri 3k, compared with 86.20% and 71.40% at 20k. Explicit joint success is
55.45% and 23.05%, compared with 86.15% and 69.90%. Thus improved retention relative to 20k
comes with worse familiar-task performance. Lower teacher-forced loss alone is not proof
of better generated answers; failures cannot be attributed only to surface formatting.

Hindi 5k improves the names-challenge aggregate to 26.20% explicit accuracy and 17.60% joint
success (88/500), versus 15.80% and 0% at 20k. Its 88 joint successes consist of 49 equality
and 39 transitivity examples; direct and extremum each have zero joint successes. Direct
explicit accuracy is only 1/125 and extremum is 0/125. This is limited, task-dependent
generalization, not reliable unfamiliar-name handling. Bhojpuri remains at zero joint
success on that challenge. Entity familiarity, tokenization, data scale and training exposure
are possible explanations; these runs do not identify a causal mechanism.

## Reproduction and retained evidence

The historical code is `scripts/ablation_samples_vs_ppl.py` at Git commit `912c20a`.
The current script preserves canonical subset RNG order, labels the 100-case diagnostic,
writes new results under a separate directory, and saves complete recovery state for new runs.
Old final checkpoints remain weight-only: optimizer/scheduler/RNG state and original per-step
logs were not saved, and cannot be reconstructed after the fact. Their embedded configuration
records the base 2,500-step config rather than each shortened schedule; use the actual steps
and warmup above, also recorded in `ablation_artifact_manifest.json`.

Published sweep numbers are retained unchanged. The comparison tables are generated from
`explicit_answer_v1/scores/summary.json` and `optimal_eval/*/evaluation_summary.json` by
`scripts/build_ablation_report.py`; the legacy `optimal_eval` folder name is only a storage name.
Compressed predictions for the main and auxiliary models are included under the result folders.
The manifest records current checkpoint, dataset, tokenizer and prediction hashes. It is a
retrospective inventory, not a claim that hashes were recorded during original training.

Restore the Phase 2 and main Phase 3 artifacts from the README Drive links. Generate or
download `reasoning_v1` as described in the dataset-release report. Then, from the repository root:

```bash
python scripts/ablation_samples_vs_ppl.py \
  --artifact-root /path/to/LMA_Individual_Project \
  --dataset-root /path/to/reasoning_v1 \
  --work-dir tmp/phase3/ablation_reproduced \
  --output-dir tmp/phase3/ablation_reproduced/results \
  --device auto
```

Rerun the same command after interruption to resume new runs from `checkpoint_resume.pt`.
The current shuffled epoch is reconstructed before restoring the saved RNG state. New checkpoints
save weights, optimizer, scheduler, scaler, step, actual configuration, input hashes and RNG state.
Device/library differences can affect floating-point results. Published historical predictions
remain the evidence for the reported scores; a reproduction is a separately versioned run.

Evaluate the smaller Hindi checkpoint on all four splits and both protocols:

```bash
python -m lma_phase3.evaluate_finetuned \
  --config hindi/configs/phase3.toml \
  --checkpoint tmp/phase3/ablation_reproduced/runs/hindi/run_5000/checkpoint_final.pt \
  --dataset-root /path/to/reasoning_v1 \
  --output-dir tmp/phase3/ablation_reproduced/evaluation/hindi_5k \
  --protocols both --device auto
```

For Bhojpuri use its config, `runs/bhojpuri/run_3000/checkpoint_final.pt`, and a separate
`evaluation/bhojpuri_3k` output directory. No auxiliary checkpoint Drive link is claimed here;
the committed predictions and hashes allow score inspection without those weights, and the
commands reproduce the auxiliary training from the linked base artifacts.

Redraw the published figure without training, or rebuild its report tables:

```bash
python scripts/ablation_samples_vs_ppl.py --plot-only \
  --output-dir report/figures/generated/phase3
python scripts/build_ablation_report.py
```
