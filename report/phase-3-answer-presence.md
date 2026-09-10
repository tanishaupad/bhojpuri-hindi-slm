# Supplemental answer-presence scores

The user requested credit when the correct answer appears anywhere in generated text. This
post-hoc metric reuses all 10,000 original greedy generations; prompts, models, token limits and
outputs are unchanged. It does not replace the frozen strict exact-match baseline.

## Overall results

| Split | Hindi answer presence | Bhojpuri answer presence | Strict exact match (both) |
|---|---:|---:|---:|
| validation | 18.75% (375/2000) | 2.15% (43/2000) | 0% |
| test | 17.00% (340/2000) | 3.10% (62/2000) | 0% |
| challenge_names | 19.20% (96/500) | 7.60% (38/500) | 0% |
| challenge_wording | 18.20% (91/500) | 3.20% (16/500) | 0% |

## Task-level answer-presence rates

| Split / task | Hindi | Bhojpuri |
|---|---:|---:|
| validation / direct | 26.80% | 2.80% |
| validation / extremum | 19.60% | 2.20% |
| validation / equality | 0.00% | 0.20% |
| validation / transitivity | 28.60% | 3.40% |
| test / direct | 23.60% | 4.80% |
| test / extremum | 17.20% | 4.40% |
| test / equality | 0.00% | 0.00% |
| test / transitivity | 27.20% | 3.20% |
| challenge_names / direct | 25.60% | 11.20% |
| challenge_names / extremum | 17.60% | 8.00% |
| challenge_names / equality | 1.60% | 0.00% |
| challenge_names / transitivity | 32.00% | 11.20% |
| challenge_wording / direct | 27.20% | 8.00% |
| challenge_wording / extremum | 19.20% | 3.20% |
| challenge_wording / equality | 0.00% | 0.80% |
| challenge_wording / transitivity | 26.40% | 0.80% |

## Matching and interpretation

Both prediction and reference receive Unicode NFC and whitespace normalization. A match must
be a whole name or yes/no label: adjacent letters, combining marks, numbers or connector
punctuation prevent a match. Thus `ना` inside `नाम`, or `राम` inside `रामू`, does not count.
Punctuation around a complete answer is allowed. No explanation correctness or final conclusion
is inferred. Mentioning every candidate would earn presence credit, so competing mentions are
flagged separately. Negated or factually incorrect mentions can also receive presence credit.

On the primary Hindi test, 340 outputs contain the correct answer; 97 also contain another
candidate and 243 contain no alternative candidate. Bhojpuri has 62 correct-answer mentions,
none with another candidate. These are mention counts, not verified reasoning successes.

For transparency, unconstrained substring matching would credit 80 Bhojpuri test outputs
(4.0%) instead of 62 (3.1%); the extra 18 are matches inside larger words. Hindi is unchanged.
The whole-answer rule avoids these substring false positives.

## Reproduction and artifacts

```bash
python -m lma_phase3.score_answer_presence \
  --predictions-root report/results/generated/phase3/pretrained_v1 \
  --dataset-root tmp/phase3/reasoning_v1 \
  --output-dir /path/to/new-presence-results
```

Per-example scores, aggregate and per-task metrics, and input/output hashes are saved under
`report/results/generated/phase3/pretrained_presence_v1/`. The scorer verifies the original
prediction, dataset and template hashes and reproduces every strict score before adding presence.
Use this same supplemental rule for the eventual finetuned models if comparing presence rates.
