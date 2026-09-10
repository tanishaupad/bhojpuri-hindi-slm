# Assignment compliance review and self-verification guide

## Status update — 11 September 2026

The 10 September audit below is a historical record, not the current submission status.
Its test/lint counts, uncommitted-work finding, and missing-remote-branch finding are superseded.

- Phase 3 implementation, results, finetuning logs and compressed prediction records are now
  committed. The remote `phase-3` branch was verified at `df0900f` before these follow-up fixes.
- The current suite contains 153 tests and repository-wide lint passes. Six finetuning tests
  previously required ignored local datasets. They now generate deterministic 16-example
  fixtures for each language from committed templates and tokenizers, covering all four task
  families. The suite is verified from a clean export without staged datasets.
- All 20,000 saved finetuned predictions reproduced their recorded scores in the latest review.
  This remains a saved-output consistency check, not a fresh full model-generation run.
- README CLI flags, checkpoint navigation, qualitative quotations, attention-table values,
  loss-objective documentation and the direct-only periodic-validation disclosure have been
  corrected. Full-validation scores for the three retained checkpoints are recorded. Step 2,500
  is described as best among those three candidates, not a global optimum.
- The Bhojpuri failure example now explains that answer presence credits the correct name
  `गुड्डू` in a copied premise despite the incorrect explicit answer `बनवारी`.
- The dataset-release page now links to completed finetuning and evaluation. The user reports
  completing the Drive-access check; this is user confirmation rather than an independently
  reproduced signed-out download in this review.
- Resume code now rejects a missing `latest.json`, checks selected architecture fields and
  reconstructs seeded data order; a controlled continuation test is present. These changes do
  not establish exact recovery for every configuration or retroactively change old checkpoints.

Remaining limitations include Hindi's shortfall from the approximately 500M-token target,
the unrecounted manual fraction in the final raw packed training streams, and the finite,
template-based scope of reasoning and attention conclusions. The original findings below
should be read alongside this update, not treated as an unresolved checklist.

## Historical audit — 10 September 2026

Audit: 10 September 2026. Scope: the current local working tree, including work added after the Antigravity handoff. This is a review, not a revision of the experiments or a claim about awarded marks.

Specification: `M26/LMA/LMA_Individual_Project_v1.pdf`, pages 1–12. Final deadline: 16 September 2026, 11:59 P.M. Phase 3 must be submitted on `phase-3` through the existing GitHub Classroom repository.

**Verdict: the core implementation and experiments substantially match the assignment, but this snapshot is not ready to submit.** The strongest measured finetuning results reproduce. The remaining problems concern report accuracy, validation selection, recovery/reproducibility, and what is actually included in the submission.

## What was checked directly

- All **151 software tests passed** in the existing local environment. Repository-wide lint found **19 issues** (15 in Python files, 4 in the Kaggle notebook), primarily unused names/import formatting. The report's blanket “84 tests, zero lint errors” is not the current result.
- All **50,000 synthetic examples** were rechecked for correct labels, rendering, task quotas, token counts and unique named-world/prompt boundaries. No shared world or prompt was found across these language/split partitions. Current files match the release hashes; current tokenizer/template files match the release manifests.
- Full answer-plus-explanation training sequences fit the context: maximum **111 Hindi / 123 Bhojpuri input tokens**, no unknown-token IDs in the checked examples.
- Both local best finetuned checkpoint hashes match the published manifest. Both contain weights, populated AdamW state, scheduler state at step 2,500, configuration and RNG states. Configurations match the checked-in language configs.
- **20,000 saved finetuned predictions** were checked against dataset IDs, prompts, reference answers, decoded token IDs, token limits, presence checks and semantic scores. Every recomputed stored count agreed. This checks internal consistency, not unrestricted prose understanding.
- **64 outputs were independently regenerated** on CPU from the best checkpoints: one example per task, split, language and protocol. All 64 matched the saved token IDs exactly. This is a sample provenance check, not a fresh full generation of all 20,000 outputs.
- The semantic scorer still matches the hash frozen for the pretrained rescores.
- The live remote lists `main`, `phase-1` and `phase-2`, with **no remote `phase-3`**. Local `phase-3` still points to the Phase 2 commit `d19b552`; Phase 3 work is in uncommitted/untracked files.
- Both finetuning folders, checkpoint files and logs exist in the local Drive filesystem. Signed-out access and completed remote Drive synchronization were **not** verified.

Machine-readable evidence is in [audits/generated/compliance_review](audits/generated/compliance_review/). The reusable prediction check is [../scripts/verify_phase3_results.py](../scripts/verify_phase3_results.py).

## Requirement-by-requirement assessment

| Assignment requirement | Finding | Assessment |
|---|---|---|
| Python/PyTorch; student implementation of attention, positions, causal mask | Primitive embeddings/linear layers, explicit Q/K/V reshaping, causal masking, feed-forward layers, residuals and normalization are implemented. No prohibited model/attention classes found in `src`. Causality tests pass. | Meets inspected implementation requirements |
| Two independent monolingual models | Separate Hindi/Bhojpuri tokenizers, configs and checkpoints. Exact model sizes: 25,285,248 / 24,870,912. | Meets architecture/resource separation requirements inspected |
| Approximately 25M parameters | Both are close to 25M. The PDF specifies an approximate target, not a hard 25,000,000 ceiling. | Meets; correct the report's “ceiling” wording |
| Approximately 500M training tokens per language | Original packed train streams contain approximately 337M Hindi / 23.14M Bhojpuri. Processed pretraining tokens are 332.60M / 67.99M. | Target not reached; report shortfalls accurately |
| At least 20% manually collected final training tokens | Historical normalized corpus statistics report 25.06% Hindi / 22.49% Bhojpuri. Collection/provenance code and stats exist. | Reported as meeting threshold; final raw packed-training manual fraction was not freshly recounted |
| Lower-resource shortfall justification | Bhojpuri is allowed; public-source/cleaning/manual-ratio constraints are discussed. The PDF explicitly permits a justified lower-resource shortfall. | Present; this does not automatically exempt Hindi's shortfall |
| Independent tokenizers trained from scratch and diagnostics | Training code, separate model/vocab files, fertility/UNK/frequency figures and examples exist. | Meets inspected deliverables |
| Independent pretraining, AdamW, schedule, checkpoints and logs | Implemented and Phase 2 artifacts are present. Current audit did not rerun pretraining. | Evidence present |
| CE/PPL/BPB; four decoding settings; BLEU/chrF/ROUGE-L; diversity | Phase 2 saved results contain greedy and temperatures 0.5/1.0/1.5, required metrics, samples, curves and attention summaries. Full-validation recomputations exist. | Deliverables present |
| Own synthetic comparative reasoning data and leakage controls | Four appropriate task families, two target-language corpora, train/validation/test and separate challenges; audited generation. | Meets inspected requirements |
| Independent reasoning finetuning with fixed tokenizers | Both best checkpoints exist; same architecture/tokenizers as their respective pretraining configurations. Training scripts load each supplied pretrained checkpoint. | Implemented, with provenance hardening recommended |
| Pretrained/finetuned reasoning evaluation | Matching 16-token and 96-token protocols, exact match, answer presence, proof checks and raw outputs exist. Main counts reproduce. | Meets evaluation evidence, subject to metric interpretation |
| Checkpoint resume/reproducibility | Required core states are present, but exact shuffled-data continuation and config/data validation are missing. | Partial; see finding 5 |
| Early/late attention and multiple heads, pre/post comparison | Six figures per language, layers 1 and 8, heads 0/1/2; CSVs cover all 54 heads. | Present; table errors and figure layout need correction |
| Final report addressing all four research questions | Sections exist and contain substantial results. Several quotations, numbers and causal claims are unsupported or wrong. | Needs correction |
| README reproduction and Drive links | Commands use nonexistent flags; finetuned links reuse pretrained folder IDs. | Needs correction |
| Figures/logs/results in graded branch; final `phase-3` push | Phase 3 is not committed/pushed; finetuning logs remain only in ignored staging/Drive; full finetuned predictions are ignored. | Not yet submitted |
| Ability to explain and verify AI-assisted work | Must be demonstrated by the student; cannot be established by a passing test suite. | Personal verification required |
| Optional no-position ablation | Not found; explicitly optional in the specification. | Not a required gap |

## Corrections needed before submission

### 1. Replace incorrect report examples and attention values

`report/phase-3.md`, lines 372–375, labels `bho-test-000003` as a successful transitivity example. The actual saved row is a **direct height comparison**:

- Facts: Banwari 106 cm, Guddu 197 cm; asks who is taller.
- Generated answer: Banwari, followed by the false assertion that Banwari's height is greater than Guddu's.
- Actual `joint_fully_verified`: **false**.

The quoted successful transitivity question and response do not belong to that ID. Select a real successful transitivity row and copy it verbatim, or retain this as a failure case.

The Hindi example at lines 367–370 preserves the mathematical meaning but is also not verbatim: the saved output ends `सुमित की उम्र अशोक से ज़्यादा है।`, not the reported `अशोक की उम्र सुमित से कम है।`. A generated-output quotation must match the record, or be labelled as a paraphrase.

Several Section 8 attention rows do not match their saved CSVs:

| Model/head | Reported entropy before → after | Actual saved CSV |
|---|---|---|
| Hindi L8 H1 | 1.8866 → 1.5012 | **2.1614 → 1.9130** |
| Bhojpuri L1 H0 | 2.6841 → 2.6391 | **2.7917 → 2.7737** |
| Bhojpuri L2 H5 | 1.8679 → 1.3402 | **2.6079 → 2.2683** |
| Bhojpuri L8 H5 | 2.4512 → 1.8105 | **2.4157 → 2.1336** |

In particular, Bhojpuri L2 H5 distance **decreases by 2.1773** in the CSV; the report claims an increase of 2.73. This changes the interpretation. Rebuild every attention-table row from the CSV and revise the associated prose.

### 2. State the actual validation selection protocol

Both Phase 3 configs set `validation_batches=50`, microbatch size 8. The validation loader is unshuffled, and the first 400 validation rows are **all direct comparisons**. Consequently the reported 0.00407 / 0.01679 selection losses cover this subset, not the complete four-task validation distribution.

The subsequent generation evaluation does cover all 2,000 validation rows, but that does not retrospectively change checkpoint selection. At minimum disclose the selection subset. Prefer rescoring the retained candidate checkpoints on the complete validation set, selecting from validation alone, then regenerating test/challenge results only if the chosen checkpoint changes. A full pretraining restart is not implied.

### 3. Repair the README commands and checkpoint links

The README's finetuning examples use `--data-dir` and `--tokenizer-path`, which the finetune CLI does not accept; required arguments are `--train-data`, `--val-data`, and `--template-config`. I reproduced the argument-parser failure without starting training.

The evaluation CLI expects `--dataset-root`; it does not accept the README's `--data-dir`, `--tokenizer-path`, or `--language`. Attention expects `--tokenizer` and a required `--prompt`, not `--tokenizer-path`/`--language`.

At README lines 574–577 the finetuned links have exactly the same folder IDs as the corresponding pretrained links. The displayed Phase 3 filesystem suffix does not change the URL target. Replace these with actual Phase 3 links, or a verified accessible root link plus clear navigation. Test downloads signed out.

### 4. Include required evidence in the submission

The live remote has no Phase 3 branch. The PDF grades the branch state at the deadline, not the local folder. Commit and push after review; this audit did not do either.

The full finetuning logs exist under `tmp/phase3/runs/{language}/finetune_v1/training.log` and Drive, but no Phase 3 logs were found under `report/`. Copy the small logs into a versioned report location. `git check-ignore` also confirms finetuned compressed prediction files are ignored. Include these compact evaluation records, or an adequate versioned sample set plus all metrics, so the qualitative examples and results are reviewable. The PDF does not require every raw prediction, but all graded figures/tables and finetuning logs must be in the submission.

### 5. Make recovery claims accurate and test actual continuation

The checkpoint contains model, optimizer, scheduler, step, config and RNG states. However, `finetune.py` does not save the data-loader permutation/epoch/cursor; it recreates a shuffled iterator after restoring RNG. A minimal reproduction produced next IDs `[4, 12, 11, 7]` uninterrupted but `[21, 20, 16, 14]` after the existing restore pattern.

This is recovery of model/training state with a different next batch, not exact continuation. The current checkpoint test only checks save/load and calls RNG restoration; it does not compare resumed training to uninterrupted training.

Also, resume does not validate saved config against requested config or bind data/tokenizer hashes, and requesting `--resume` with no `latest.json` silently begins a new run. Fix these cases and test interruption after a real optimizer step in a disposable directory. Preserve completed run artifacts.

### 6. Correct the loss claim and strengthen experiment provenance

The reported scalar train loss and validation loss are token-weighted. The gradients are different: each microbatch's mean loss is divided by the accumulation count, so two microbatches receive equal weight even when they contain different numbers of active target tokens. This does **not** implement the report's globally token-weighted optimizer-step equation.

Either document the actual mean-of-microbatch-means objective for the completed run or implement globally token-weighted accumulation for a separately versioned run. This discrepancy does not invalidate the saved generation scores.

Finetuned evaluation outputs do not record checkpoint/dataset/tokenizer/scorer hashes or EOS/length stop reason as thoroughly as the original baseline. The sample regeneration and rescoring support the present artifacts, but add a manifest for robust reproducibility. Avoid overwriting earlier experiment directories.

### 7. Narrow scientific claims and reconcile remaining text

- Attention entropy/distance shifts are descriptive evidence. They do not prove a particular head implements a comparison circuit. Demonstrating causal responsibility requires interventions/ablations.
- 100% verified transitivity on this finite test is a test-set result, not proof of unrestricted reasoning ability. Unseen-name verification is zero in both models.
- Calling token fragmentation the primary cause of unseen-name failure is unsupported without a targeted analysis/control. Present it as a hypothesis.
- Hindi does **not** have the larger transformer backbone: 16.07M remaining parameters versus Bhojpuri's 18.73M. Both have nine layers. Correct Section 9's reversed capacity explanation.
- Distinguish complete explanations from answer-and-explanation joint success. For Hindi wording challenge, saved explanation verification is **395/500 (79.0%)**, whereas joint success is **323/500 (64.6%)**. Section 7.2 conflates them.
- README's Bhojpuri unseen-name presence should be **13.6%**, not 13.8%.
- Report current tests/lint honestly; clarify or remove stale “pending” passages inside the report marked complete.
- Hindi's approximately 337M corpus is below the 500M target. The explicit lower-resource allowance applies to Bhojpuri. Explain Hindi's actual shortfall without asserting full target compliance or a guaranteed grading exception.
- The historical manual fractions refer to normalized pre-decontamination counts. If claiming the precise fraction in the final packed pretraining streams, recompute those training totals; the held-out recount alone does not establish it.

### 8. Clean up figure readability and portable testing

A reviewed Hindi Phase 3 attention heatmap has box-like whitespace-token labels and a colorbar overlaying the right plot and some ticks. Axes/titles/color scale are present, so this is a readability correction, not missing attention analysis. Render every figure after fixing layout and token labels.

Several new finetuning tests depend on ignored `tmp/phase3/reasoning_v1/...` data. They pass in this populated workspace but are not self-contained in a fresh checkout. Generate tiny fixtures in tests or document dataset staging explicitly. Fix the 19 lint findings without describing them as model-quality failures.

## Scores that reproduced

These are full primary-test denominators, 2,000 examples per language. Presence can credit copying or mentioning multiple candidates and is not final-answer accuracy.

| Measure | Hindi | Bhojpuri |
|---|---:|---:|
| Original 16-token answer presence | 1,677/2,000 = **83.85%** | 1,700/2,000 = **85.00%** |
| Explanation 96-token answer presence | 1,834/2,000 = **91.70%** | 1,783/2,000 = **89.15%** |
| Explanation protocol joint fully verified | 1,723/2,000 = **86.15%** | 1,398/2,000 = **69.90%** |
| Explanation protocol correct recognized conclusion | 1,723/2,000 = **86.15%** | 1,407/2,000 = **70.35%** |
| Explanation protocol resolved conclusion coverage | 1,998/2,000 | 1,944/2,000 |

Keep the user's requested presence metric and bounded semantic checks separately. An unresolved statement is not automatically false. Exact match is already reported as zero for whole continuations; the PDF does not impose a minimum score.

## How to verify the work yourself

Run commands from the project root:

```bash
cd /Users/tanishaupadhyay/Documents/ChatGPT/LMA_Assignment1
```

1. **Read the specification beside the deliverables.** Use PDF pages 5–6 for corpora/tokenizers, 6–9 for architecture and LM evaluation, and 10–11 for finetuning/reporting. Tick off the requirement table above. Separate “file exists,” “I reproduced it,” and “submitted on GitHub.”

2. **Check code health.** With the local `.venv` and staged datasets:

   ```bash
   .venv/bin/python -m pytest -q
   .venv/bin/ruff check .
   ```

   At this audit: 151 tests pass; lint reports 19 issues. Passing tests do not establish that every report claim is correct.

3. **Recompute saved evaluation scores without training.**

   ```bash
   .venv/bin/python scripts/verify_phase3_results.py
   ```

   Expect a final `PASS` covering 20,000 saved predictions. If datasets are elsewhere, supply `--dataset-root /path/to/reasoning_v1`. This checks saved evidence, not a fresh model run. The full generations must be present locally.

4. **Manually solve examples from each family.** Open the dataset and corresponding compressed prediction records, find matching IDs, then verify the numeric answer yourself. For transitivity, draw both arrows and follow them. Include successes, failures, and both challenge sets. Check the *entire* output for wrong facts, opposing conclusions or copied names; do not accept an answer solely because its name appears.

5. **Verify checkpoint identity.** Compare hashes with `report/results/generated/phase3/checkpoint_manifest.json`:

   ```bash
   shasum -a 256 tmp/phase3/runs/hindi/finetune_v1/best_checkpoint.pt
   shasum -a 256 tmp/phase3/runs/bhojpuri/finetune_v1/best_checkpoint.pt
   ```

   Expected prefixes: Hindi `8ac1b2c96642`, Bhojpuri `cccd2ed6ce9c`. Full hashes must agree. Check that the chosen checkpoint is selected from validation, and understand the current direct-only subset limitation. The recorded sample regeneration audit provides 64 exact matches; a full fresh evaluation is a separate, longer check.

6. **Check loss and attention claims against raw files.** Compare each table entry to `training.log` or `attention_comparison.csv`. Calculate entropy/distance changes as after minus before. Open every PNG at readable size. A color change cannot establish causation; describe what is visibly measured.

7. **Check reproduction instructions before expensive runs.** Compare README flags to:

   ```bash
   .venv/bin/python -m lma_phase3.finetune --help
   .venv/bin/python -m lma_phase3.evaluate_finetuned --help
   .venv/bin/python -m lma_phase3.attention_compare --help
   ```

   Then test recovery in a new scratch run: compare uninterrupted training with stop/save/resume for the same total steps, including next example IDs, LR, optimizer step and weights. Do not run this experiment in the completed training directories.

8. **Verify what the grader can access.** Open every Drive link in a signed-out/private window and download the expected file. For both finetuned models, confirm it is the Phase 3 checkpoint. On GitHub, check the actual `phase-3` file tree rather than your editor's files.

   ```bash
   git status --short
   git ls-remote --heads origin
   git ls-tree -r --name-only HEAD -- report src/lma_phase3
   git check-ignore report/results/generated/phase3/finetuned_v1/hindi/protocol_explanation_96tokens/test_predictions.jsonl.gz
   ```

   At this audit, the remote lacks `phase-3`, HEAD lacks Phase 3 changes, and that prediction path is ignored. A local commit timestamp is not proof of the time a push reached GitHub; preserve Classroom/GitHub receipt evidence.

9. **Prepare to explain the work orally.** Be able to explain `B × T × d → B × h × T × d_head`, causal masking, the next-token shift, prompt-loss masking, PPL/BPB denominators, why two tokenizers are separate, dataset leakage checks, and the difference between answer presence and verified explanation. Explain the AI-assisted parts and how you verified or changed them, as required by the PDF.

## Scope limits

This review did not repeat full corpus collection, manual-source classification, pretraining, full 20,000-output model generation, or every original Phase 2 metric computation. It checked implementation, existing artifacts, all synthetic labels/current finetuned scores, and sampled model reproduction. Signed-out Drive access and historical deadline receipt remain user-side checks. No training code, existing report, model, dataset, remote branch or sharing permission was changed by this audit.
