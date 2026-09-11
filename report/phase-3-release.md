# Phase 3 reasoning_v1 dataset release

Status: generated, audited, and copied to the project Drive filesystem. User approved the
naturalness of both 50-example development pilots. The [pretrained reasoning baselines](phase-3.md#4-pretrained-reasoning-baselines) are complete. Finetuning and held-out evaluation are also complete; see the [final Phase 3 report](phase-3.md).
The dataset design and construction details are documented in [the Phase 3 report](phase-3.md#2-dataset-design-and-construction).

## Contents per language

| Split | Examples | Per task family | Purpose |
|---|---:|---:|---|
| Train | 20,000 | 5,000 | Finetuning |
| Validation | 2,000 | 500 | Checkpoint and hyperparameter selection |
| Test | 2,000 | 500 | Primary before/after comparison |
| challenge_names | 500 | 125 | Names absent from core splits |
| challenge_wording | 500 | 125 | Question variant absent from core splits |

Total: 25,000 per language, 50,000 overall. Four families: direct comparison, extremum, equality,
and two-hop transitivity. Numeric domains are age and height. The original pilot records are not
in these files. Core and challenge name pools are separately specified in each language's
`configs/phase3_release.json`; language name pools are disjoint.

## Wording review and limitations

The user's statement “they all seem natural to me” is recorded in
[samples/generated/phase3/pilot_v1/wording_review.json](samples/generated/phase3/pilot_v1/wording_review.json).
Only language approval was entered in the review worksheets; answer-review cells remain blank.
The labels have computational correctness evidence rather than an invented human review.

Variant 0 is used for core splits and unseen names; variant 1 is reserved for unseen wording.
Both are wording patterns already reviewed in the development pilot. Primary splits use one
question phrasing per direction/domain. This is less varied than the initial six-pattern proposal.
The challenge changes question wording, not premise phrasing or answer cues. All names are
masculine to keep agreement consistent with the pilot. Added names are template substitutions.

## Verification

- Every label checked by numerical sorting or enumerating all possible entity rankings, separately
  from the generator's transitive-closure solver.
- Each transitive example requires both edges; no direct endpoint comparison or ambiguous fork.
- Exact task quotas, and label/direction/position/domain balances to within one per family.
- All 50,000 canonical named worlds and normalized prompts are unique across all release splits
  and both languages. All 100 development worlds are excluded from every release split.
- Names for `challenge_names` do not occur in core pools; question variants for `challenge_wording`
  do not occur in core splits. Task structures overlap intentionally.
- Full prompt + answer + EOS fits within 256 tokens: maximum 47 Hindi, 53 Bhojpuri. No UNK IDs
  in accepted examples. Serialization collapses whitespace consistently.
- Repeated candidates are regenerated, with rejection counts reported. Checks do not establish
  that every possible form of semantic overlap has been excluded.
- Tests exercise corrupted labels/prompts, ambiguous reasoning, independent solver behavior,
  split isolation, challenge separation, reproducibility and deterministic compressed bytes.

Compact manifests and the overall audit are in
[audits/generated/phase3/reasoning_v1](audits/generated/phase3/reasoning_v1).

## Frozen evaluation protocol

Start each model from its own final Phase 2 checkpoint and retain its tokenizer. Serialize the
same normalized prompt used for finetuning; omit the answer and all metadata from model input.
Use greedy decoding, stopping at EOS or 16 generated tokens. Normalize generated and reference
answers using Unicode NFC and whitespace collapse/trim; compare the entire result for exact match.
Do not extract an entity substring from an otherwise incorrect continuation. Report overall and
per-task primary-test accuracy, with both challenge scores separately. Use validation alone for
hyperparameters and checkpoint selection. Tokenization and answer-loss boundaries must be checked
when implementing the finetuning loader. Models may use different numbers of tokens per name.

The name challenge tests names unseen in the synthetic training data, not necessarily names
unseen during language-model pretraining. The wording challenge is unseen during finetuning but
was available for development review. Neither challenge demonstrates arbitrary reasoning transfer.

## Storage and reproduction

Within the [existing project Drive folder](https://drive.google.com/drive/folders/1zjJ5bSM69g431qnSRHXnXOwO_7rJOpce?usp=share_link):

- `hindi/phase3/reasoning_v1/`
- `bhojpuri/phase3/reasoning_v1/`

Each folder contains five compressed JSONL splits and `manifest.json`. The Drive-filesystem copies
were hash-verified; cloud sync and remote shareability have not been rechecked during this release.
Large datasets remain excluded from Git. Local staging is under `tmp/phase3/reasoning_v1/`.

From the repository root with the virtual environment active:

```bash
python -m lma_phase3.release \
  --output-dir /path/to/new-release-directory \
  --evidence-dir /path/to/new-audit-directory
python -m pytest -q tests/test_phase3_problems.py tests/test_phase3_generate.py tests/test_phase3_release.py
```

Use a fresh output directory; the generator refuses to overwrite an existing release. Preserve
this v1 when making later corrections, version changed datasets, and rerun comparable baselines.
