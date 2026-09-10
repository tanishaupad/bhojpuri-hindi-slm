# Corrected evaluation without required answer/reason headers

The previous explanation scorer incorrectly required `कारण:` before examining facts. Its zero coverage measured compliance with a delimiter, and did not implement the intended evaluation of prose. That method is superseded by this rescore. Saved model generations are unchanged; no finetuning or new generation was performed.

We preserve the requested answer-presence metric and separately check recognizable statements against the question’s numeric facts or comparison graph. Unknown text remains unresolved. No LLM judges the outputs.

## Metrics and scope

- **Answer present:** the correct name or yes/no label appears anywhere as a whole word. This is the user-requested score; it can credit copying or an incorrect sentence mentioning the answer.
- **Supported fact found:** at least one recognized factual assertion is true of the supplied problem. A copied premise counts here but is not a complete explanation.
- **Unsupported claim found:** at least one recognized assertion contradicts or is not entailed by the supplied facts. Unknown prose does not count as a false claim.
- **Recognizable conclusion:** an explicit answer sentence or comparisons expressing the requested result. Coordinated subjects, questions and ambiguous trailing names are not silently promoted to answers.
- **Sufficient justification:** the recognized, supported statements supply all required task evidence. Numeric tasks require the supplied values and a comparison/equality conclusion; two-hop tasks require both links and the endpoint comparison. A natural supported superlative can express the conclusion.
- **Fully verified explanation:** sufficient justification, no unsupported assertions, and no uninterpreted extra clauses. The separate supported-subset metric does not validate unknown extra text.

Headers, sentence order and one exact reference wording are not required. The parser supports a finite set of Hindi/Bhojpuri factual and comparison constructions, selected affirmative introductions, natural comparative/superlative forms, and conjunctions of recognized statements. It is not an unrestricted semantic judge. These are post-hoc diagnostics; parser coverage and human-labelled validation precision/recall have not been established for arbitrary prose. Do not call the verified rate “actual reasoning accuracy.”

## Primary-test results

Each row contains 2,000 outputs. Counts in different columns can overlap. The two generation protocols differ in both prompt and token budget and must remain separate.

| Generation | Language | Answer present | Supported fact found | Unsupported claim found | Recognizable conclusion | Fully verified explanation |
|---|---|---:|---:|---:|---:|---:|
| Original prompt, 16 tokens | Hindi | 340/2000 (17.00%) | 118 | 328 | 0 | 0 |
| Original prompt, 16 tokens | Bhojpuri | 62/2000 (3.10%) | 0 | 0 | 0 | 0 |
| Explanation prompt, 96 tokens | Hindi | 132/2000 (6.60%) | 25 | 73 | 0 | 0 |
| Explanation prompt, 96 tokens | Bhojpuri | 46/2000 (2.30%) | 0 | 0 | 0 | 0 |

The conclusion is **not that every output is wrong**. The primary-test conclusions remain unresolved by this checker, while factual assertions are now examined despite missing headers. The checker finds no sufficient supported proof in either generation protocol, across all splits. This is a bounded verification result, not evidence that every possible paraphrase is incorrect.

## All splits

| Generation | Language | Split | N | Answer present | Supported fact | Unsupported claim | Correct / incorrect / conflicting / unresolved conclusion |
|---|---|---|---:|---:|---:|---:|---|
| Original prompt, 16 tokens | hindi | validation | 2000 | 375 | 124 | 310 | 0 / 0 / 0 / 2000 |
| Original prompt, 16 tokens | hindi | test | 2000 | 340 | 118 | 328 | 0 / 0 / 0 / 2000 |
| Original prompt, 16 tokens | hindi | challenge_names | 500 | 96 | 23 | 72 | 0 / 0 / 0 / 500 |
| Original prompt, 16 tokens | hindi | challenge_wording | 500 | 91 | 24 | 62 | 0 / 0 / 0 / 500 |
| Original prompt, 16 tokens | bhojpuri | validation | 2000 | 43 | 0 | 0 | 0 / 0 / 0 / 2000 |
| Original prompt, 16 tokens | bhojpuri | test | 2000 | 62 | 0 | 0 | 0 / 0 / 0 / 2000 |
| Original prompt, 16 tokens | bhojpuri | challenge_names | 500 | 38 | 0 | 0 | 0 / 0 / 0 / 500 |
| Original prompt, 16 tokens | bhojpuri | challenge_wording | 500 | 16 | 0 | 0 | 0 / 0 / 0 / 500 |
| Explanation prompt, 96 tokens | hindi | validation | 2000 | 126 | 29 | 66 | 0 / 1 / 0 / 1999 |
| Explanation prompt, 96 tokens | hindi | test | 2000 | 132 | 25 | 73 | 0 / 0 / 0 / 2000 |
| Explanation prompt, 96 tokens | hindi | challenge_names | 500 | 38 | 6 | 9 | 0 / 0 / 0 / 500 |
| Explanation prompt, 96 tokens | hindi | challenge_wording | 500 | 19 | 11 | 22 | 0 / 0 / 0 / 500 |
| Explanation prompt, 96 tokens | bhojpuri | validation | 2000 | 26 | 0 | 0 | 0 / 0 / 0 / 2000 |
| Explanation prompt, 96 tokens | bhojpuri | test | 2000 | 46 | 0 | 0 | 0 / 0 / 0 / 2000 |
| Explanation prompt, 96 tokens | bhojpuri | challenge_names | 500 | 31 | 0 | 0 | 0 / 0 / 0 / 500 |
| Explanation prompt, 96 tokens | bhojpuri | challenge_wording | 500 | 13 | 0 | 0 | 0 / 0 / 0 / 500 |

## Audited examples and scoring safeguards

For `hi-test-000000` in the 96-token run, the question gives Ashok’s age as 13 and Sumit’s as 25 and asks who is younger. The output is “यह बात तो तय है कि अशोक की उम्र 13 साल है।” The new checker credits answer presence and a supported fact without needing `कारण:`. It leaves the comparison conclusion unresolved and the justification incomplete: only one age is stated. The old checker called this a missing explanation solely because the delimiter was absent.

For `hi-test-000020` in the original run, “रोहित और रोहित की उम्र 40 साल है। रोहित की उम्र 4” contains a coordinated subject and a truncated repetition. Splitting on “और” must not turn the first “रोहित” into a final answer. An early development rescore made that error, producing the preliminary 87-correct figure mentioned during review; that count is withdrawn. Regression tests now reject this interpretation.

The first Bhojpuri primary-test continuation in the 96-token run repeats unrelated blog-like text. It contains no recognized problem assertion, so it remains unverified. Zero recognized claims is not a finding that every sentence in it is false.

## Verification and reproducibility

Both saved generation sets were rescored: 10,000 outputs per protocol, 20,000 total. Input dataset and prediction hashes, unique ID coverage, and output score hashes are checked/recorded. All 10,000 solver-derived reference explanations are accepted with the reason marker removed in each run (the same reference set checked twice). All 145 software tests pass, including both languages, reordered proofs, negation, questions, coordinated subjects, trailing fragments, invented numbers, and unknown extra prose. Reference acceptance validates the supported grammar; it does not establish recall on unrestricted text.

Per-example statements, unknown clauses, presence flags and per-task summaries are retained in:

- `report/results/generated/phase3/semantic_original_v1/`
- `report/results/generated/phase3/semantic_explanations_v1/`

```bash
python -m lma_phase3.rescore_semantics \
  --predictions-root report/results/generated/phase3/pretrained_v1 \
  --dataset-root tmp/phase3/reasoning_v1 \
  --output-dir /path/to/new-original-scores

python -m lma_phase3.rescore_semantics \
  --predictions-root report/results/generated/phase3/pretrained_proof_v2 \
  --dataset-root tmp/phase3/reasoning_v1 \
  --output-dir /path/to/new-explanation-scores
```

For the later finetuned comparison, retain answer presence and these separate verification/coverage categories under a frozen scoring version. Any further parser development should use validation examples; report this rescore as exploratory because held-out outputs informed the corrections. A small independent human-labelled validation sample is needed before claiming calibrated free-prose explanation accuracy.
