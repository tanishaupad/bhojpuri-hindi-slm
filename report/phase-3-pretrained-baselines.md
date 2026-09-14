# Pretrained reasoning baselines

Both models were evaluated before reasoning finetuning on the frozen reasoning_v1 release.
Each model used its own original checkpoint and tokenizer. Input files and checkpoints
were verified against recorded SHA-256 hashes. No training was performed.

## Overall exact-match accuracy

| Split | Hindi | Nepali | Examples per language |
|---|---:|---:|---:|
| validation | 0.00% (0/2000) | 0.00% (0/2000) | 2000 |
| test | 0.00% (0/2000) | 0.00% (0/2000) | 2000 |
| challenge_names | 0.00% (0/500) | 0.00% (0/500) | 500 |
| challenge_wording | 0.00% (0/500) | 0.00% (0/500) | 500 |

## Primary-test explicit-answer baseline

| Metric | Hindi | Nepali |
|---|---:|---:|
| Correct first declared answer | 0/2,000 | 0/2,000 |


## Task-level accuracy

| Split / task | Hindi | Nepali | Examples per language |
|---|---:|---:|---:|
| validation / direct | 0.00% | 0.00% | 500 |
| validation / extremum | 0.00% | 0.00% | 500 |
| validation / equality | 0.00% | 0.00% | 500 |
| validation / transitivity | 0.00% | 0.00% | 500 |
| test / direct | 0.00% | 0.00% | 500 |
| test / extremum | 0.00% | 0.00% | 500 |
| test / equality | 0.00% | 0.00% | 500 |
| test / transitivity | 0.00% | 0.00% | 500 |
| challenge_names / direct | 0.00% | 0.00% | 125 |
| challenge_names / extremum | 0.00% | 0.00% | 125 |
| challenge_names / equality | 0.00% | 0.00% | 125 |
| challenge_names / transitivity | 0.00% | 0.00% | 125 |
| challenge_wording / direct | 0.00% | 0.00% | 125 |
| challenge_wording / extremum | 0.00% | 0.00% | 125 |
| challenge_wording / equality | 0.00% | 0.00% | 125 |
| challenge_wording / transitivity | 0.00% | 0.00% | 125 |

## Protocol and interpretation

Greedy decoding; stop at EOS or 16 new tokens. Exact match compares the entire generated
continuation with the canonical answer after Unicode NFC and whitespace normalization only.
No punctuation removal, substring extraction, or multiple-choice restriction was used.
The saved summaries record the float32 evaluation device for each run. Equal-length prompts are batched without
padding; only final-position vocabulary logits are projected. Tests verify equivalence to
the full forward pass and per-example generation, as well as independent EOS stopping.

These are corpus-pretrained language models, not instruction-finetuned models. Exact-match
failure can reflect failure to emit the required short answer or stop cleanly as well as
incorrect comparison reasoning. Noncanonical output therefore does not by itself establish
that the relation could not be represented internally. Keep the same scoring protocol after
finetuning so that before/after differences remain comparable. Test/challenge results must
not be used for checkpoint or hyperparameter selection.

The name challenge holds names out of synthetic training only. The wording challenge holds
out one reviewed question variant, while task structures and premise forms remain familiar.

## Primary-test error categories

| Language | Correct | Wrong canonical answer | Noncanonical output | Empty output | EOS stops |
|---|---:|---:|---:|---:|---:|
| hindi | 0 | 0 | 2000 | 0 | 498 |
| nepali | 0 | 0 | 2000 | 0 | 51 |

## Qualitative examples

Examples below are the first saved primary-test item in each observed outcome category;
this deterministic selection does not represent the frequency of each category.

### hindi: noncanonical_output (hi-test-000000)

अशोक की उम्र 13 साल है। सुमित की उम्र 25 साल है। अशोक और सुमित में से किसकी उम्र सबसे कम है? उत्तर:

Expected: **अशोक**

Generated: "2014 में सुमित ने 2014 में शादी की"

Stop: token_limit

### nepali: noncanonical_output (ne-test-000000)

अर्जुनको उचाइ 107 सेन्टिमिटर छ। नवीनको उचाइ 173 सेन्टिमिटर छ। अर्जुन र नवीनमध्ये सबैभन्दा अग्लो को छ? उत्तर:

Expected: **नवीन**

Generated: "180 सेन्टिमिटर (100 सेन्टिमिटर) को एक सानो आकारको"

Stop: token_limit

## Artifacts

Full predictions, individual split metrics and run summaries are under
`report/results/generated/phase3/pretrained_v1/{hindi,nepali}/`.
Summaries record checkpoint, tokenizer, dataset, evaluator and prediction hashes.
The full validation and primary-test sets each contain 2,000 examples per language;
each challenge contains 500. All requested examples were scored.
