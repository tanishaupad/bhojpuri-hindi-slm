"""Rebuild the report's ablation tables from saved, protocol-specific metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "report/results/generated/phase3"


def build_tables() -> tuple[str, str]:
    """Keep the 100-case diagnostic separate from full-split generation scores."""
    sweep = json.loads((RESULTS / "samples_vs_ppl_ablation.json").read_text())
    old = json.loads((RESULTS / "explicit_answer_v1/scores/summary.json").read_text())["results"]
    small = {
        language: json.loads(
            (RESULTS / f"optimal_eval/{language}_{size}k/evaluation_summary.json").read_text()
        )
        for language, size in (("hindi", 5), ("bhojpuri", 3))
    }
    sweep_lines = [
        "| Language | Samples | Steps | Generic CE | Generic PPL | PPL / base | Full validation loss | Full validation PPL | Direct-only accuracy (first 100) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for language in ("hindi", "bhojpuri"):
        baseline = sweep[language][0]["generic_perplexity"]
        for row in sweep[language]:
            n = row["samples"]
            sweep_lines.append(
                f"| {language.title()} | {n:,} | {n // 8:,} | {row['generic_cross_entropy']:.4f} "
                f"| {row['generic_perplexity']:,.2f} | {row['generic_perplexity'] / baseline:.2f}x "
                f"| {row['reasoning_val_loss']:.4f} | {row['reasoning_val_perplexity']:.2f} "
                f"| {row['reasoning_val_accuracy']:.1f}% |"
            )
    lines = [
        "| Protocol / split | Metric | Hindi base | Hindi 5k | Hindi 20k | Bhojpuri base | Bhojpuri 3k | Bhojpuri 20k |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    records = []
    for protocol, limit in (("original", 16), ("explanation", 96)):
        for split, title in (
            ("test", "Test (2,000)"),
            ("validation", "Validation (2,000)"),
            ("challenge_wording", "Wording (500)"),
            ("challenge_names", "Names (500)"),
        ):
            metrics = [
                ("Explicit answer accuracy", "explicit_answer_correct", "explicit_answer_accuracy")
            ]
            if protocol == "explanation":
                metrics.append(
                    (
                        "Explicit joint success",
                        "explicit_answer_and_explanation_verified",
                        "explicit_answer_and_explanation_verified_rate",
                    )
                )
            if split == "test":
                metrics.append(("Answer presence", "answer_present", "answer_present_rate"))
            for label, old_key, new_key in metrics:
                values = []
                for language in ("hindi", "bhojpuri"):
                    values.extend(
                        [
                            old[f"{language}/{protocol}/pretrained/{split}"]["rates"][old_key],
                            small[language][protocol][split][new_key],
                            old[f"{language}/{protocol}/finetuned/{split}"]["rates"][old_key],
                        ]
                    )
                lines.append(
                    f"| {limit} tokens / {title} | {label} | "
                    + " | ".join(f"{100 * x:.2f}%" for x in values)
                    + " |"
                )
                records.append([protocol, split, label, *values])
    with (RESULTS / "sample_budget_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "protocol",
                "split",
                "metric",
                "hindi_base",
                "hindi_5k",
                "hindi_20k",
                "bhojpuri_base",
                "bhojpuri_3k",
                "bhojpuri_20k",
            ]
        )
        writer.writerows(records)
    return "\n".join(sweep_lines), "\n".join(lines)


def main() -> None:
    """Update existing submission reports; never regenerate predictions or training."""
    sweep, comparison = build_tables()
    protocol = """Five subset runs per language used 1,000, 3,000, 5,000, 8,000 and 12,000 examples.
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
"""
    findings = """Hindi 5k raises generic PPL from 39.57 to 87.38 (2.21x), versus 628.12 at 20k.
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
"""
    section = f"""### 6.5 SFT Sample Budget vs. Generic Retention Ablation

#### Protocol and limitations

{protocol}
{sweep}

![Generic test PPL and direct-only validation accuracy](figures/generated/phase3/samples_vs_ppl_tradeoff.png)

#### Full-split comparison of illustrative smaller-budget checkpoints

The following table uses the complete split for each row, unlike the 100-case sweep diagnostic.
Explicit joint success requires both the correct declared answer and a fully verified explanation.

{comparison}

#### Interpretation

{findings}
The submitted main models remain the resume-capable 20k v1 checkpoints linked in Section 10.
The auxiliary weight-only snapshots do not replace them. Reproduction, hashes and evidence
locations are documented in [the ablation report](phase-3-sample-ablation.md).

"""
    path = ROOT / "report/phase-3.md"
    current = path.read_text()
    start = current.index("### 6.5 SFT Sample Budget")
    end = current.index("## 7. Error Analysis", start)
    path.write_text(current[:start] + section + current[end:])
    path = ROOT / "report/phase-3-sample-ablation.md"
    reproduction = """## Reproduction and retained evidence

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
python scripts/ablation_samples_vs_ppl.py \\
  --artifact-root /path/to/LMA_Individual_Project \\
  --dataset-root /path/to/reasoning_v1 \\
  --work-dir tmp/phase3/ablation_reproduced \\
  --output-dir tmp/phase3/ablation_reproduced/results \\
  --device auto
```

Rerun the same command after interruption to resume new runs from `checkpoint_resume.pt`.
The current shuffled epoch is reconstructed before restoring the saved RNG state. New checkpoints
save weights, optimizer, scheduler, scaler, step, actual configuration, input hashes and RNG state.
Device/library differences can affect floating-point results. Published historical predictions
remain the evidence for the reported scores; a reproduction is a separately versioned run.

Evaluate the smaller Hindi checkpoint on all four splits and both protocols:

```bash
python -m lma_phase3.evaluate_finetuned \\
  --config hindi/configs/phase3.toml \\
  --checkpoint tmp/phase3/ablation_reproduced/runs/hindi/run_5000/checkpoint_final.pt \\
  --dataset-root /path/to/reasoning_v1 \\
  --output-dir tmp/phase3/ablation_reproduced/evaluation/hindi_5k \\
  --protocols both --device auto
```

For Bhojpuri use its config, `runs/bhojpuri/run_3000/checkpoint_final.pt`, and a separate
`evaluation/bhojpuri_3k` output directory. No auxiliary checkpoint Drive link is claimed here;
the committed predictions and hashes allow score inspection without those weights, and the
commands reproduce the auxiliary training from the linked base artifacts.

Redraw the published figure without training, or rebuild its report tables:

```bash
python scripts/ablation_samples_vs_ppl.py --plot-only \\
  --output-dir report/figures/generated/phase3
python scripts/build_ablation_report.py
```
"""
    path.write_text(
        "# Phase 3: Sample Budget and Generic Language Retention\n\n"
        "## Experimental scope\n\n"
        + protocol
        + "\n## Saved sweep results\n\n"
        + sweep
        + "\n\n## Interpretation\n\n"
        + findings
        + "\n"
        + reproduction
    )


if __name__ == "__main__":
    main()
