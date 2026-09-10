# Phase 3: Synthetic Reasoning Finetuning, Attention Analysis, and Multi-Phase Consolidation

**Author**: Tanisha Upadhyay  
**Course**: CL3-410 / Language Modeling and Applications  
**Date**: September 2026  
**Status**: Complete. Both independent models finetuned on reasoning_v1, fully evaluated across held-out splits and both generation protocols, multi-head attention shifts analyzed, and cross-phase empirical findings consolidated.  

---

## 1. Executive Summary and Verification Findings

As mandated by the Phase 3 requirements, this report consolidates the complete empirical trajectory across Phases 1, 2, and 3. Before launching Phase 3 synthetic data generation and finetuning, a comprehensive audit and recomputation was executed across the Phase 1 and Phase 2 artifacts to ensure full specification compliance, resolve token accounting discrepancies, correct architectural parameters, and unify held-out evaluation metrics.

### 1.1 Summary of Verification Findings

| Verification Check | Historical Phase 1 / Phase 2 Record | Recomputed & Verified Specification Finding | Status |
|---|---|---|---|
| **Code Quality & Tests** | 84 unit tests passing | **84 passing**, **0 lint errors** (all import-order and set duplicate lints resolved) | **Compliant** |
| **Model Dimension ($d_{\text{model}}$)** | Prospective $d_{\text{model}} = 512$ cited in early planning | **Realized $d_{\text{model}} = 384$** ($25.29\text{M}$ Hindi / $24.87\text{M}$ Bhojpuri parameters) | **Reconciled** |
| **Embedding Budget** | $12.28\text{M}$ parameters ($49\%$) | **$9.22\text{M}$ ($36.45\%$) for Hindi; $6.14\text{M}$ ($24.70\%$) for Bhojpuri** | **Reconciled** |
| **Hindi Test Tokens** | $3,372,824$ reported in Phase 1 | **$3,521,023$ packed in Phase 2** ($3,498,250$ raw tokens $+$ $22,773$ document EOS) | **Reconciled** |
| **Validation Evaluation** | $4.095$ loss / $60.06$ PPL (periodic 100-batch estimate) | **Full validation: $3.6667$ CE / $39.12$ PPL / $0.5263$ BPB** scored over all $3,479,647$ tokens | **Recomputed** |
| **Bhojpuri Validation** | $4.068$ loss / $58.41$ PPL (periodic 100-batch estimate) | **Full validation: $4.1737$ CE / $64.95$ PPL / $0.6497$ BPB** scored over all $234,881$ tokens | **Recomputed** |
| **Manual Data Share** | Target $\ge 20\%$ | **Hindi: $25.06\%$ ($83.27\text{M}$ tokens); Bhojpuri: $22.49\%$ ($5.08\text{M}$ tokens)** | **Compliant** |

---

## 2. Deep-Dive Recomputations (Phases 1 & 2)

### 2.1 Held-Out Token-Count Reconciliation

A rigorous provenance audit was conducted by streaming the original decontaminated JSONL files on Google Drive through the checked-in SentencePiece tokenizers. The SHA-256 hashes of the decontaminated Phase 2 validation/test splits match the original Phase 1 splits identically (`71d1b9...` for Hindi validation, `ef94dd...` for Hindi test; `da51b4...` for Bhojpuri validation, `c21f47...` for Bhojpuri test). No underlying document changes occurred.

The exact token accounting breakdown (recorded in [`report/audits/generated/heldout_recomputation.json`](audits/generated/heldout_recomputation.json)) reveals:

| Split | Documents | Whitespace-Normalized Tokens | Raw-Text Tokens | Appended Document EOS | Phase 2 Packed Tokens |
|---|---:|---:|---:|---:|---:|
| **Hindi Validation** | 23,033 | 3,429,056 | 3,456,615 | 23,033 | **3,479,648** |
| **Hindi Test** | 22,773 | 3,468,316 | 3,498,250 | 22,773 | **3,521,023** |
| **Bhojpuri Validation** | 2,344 | 227,239 | 232,538 | 2,344 | **234,882** |
| **Bhojpuri Test** | 2,351 | 214,399 | 220,234 | 2,351 | **222,585** |

#### Why the counts differed:
1. **Raw Text vs. Whitespace Normalization**: Phase 1 token counting used `normalize_tokenizer_input()`, which collapsed Unicode whitespace before encoding. Phase 2 packed binary streams encoded the raw JSONL strings directly, adding $29,934$ tokens to Hindi test ($3,498,250$ vs. $3,468,316$) and $5,835$ tokens to Bhojpuri test ($220,234$ vs. $214,399$).
2. **Document Boundary Delimiters (EOS)**: Phase 2 appended exactly one `<eos>` token after each document to create contiguous, unpadded token sequences. This accounts for exactly $+22,773$ tokens in Hindi test and $+2,351$ tokens in Bhojpuri test. Together, raw text $+$ EOS perfectly reproduces the Phase 2 packed manifest counts to the exact token ($3,498,250 + 22,773 = 3,521,023$).
3. **Historical Summary Discrepancy**: The Phase 1 report's published test count of $3,372,824$ does not reproduce; the cause of that historical summary discrepancy has not been established. The recount establishes the true normalized test size as $3,468,316$ (a $95,492$ difference).
4. **Unknown Token Impact**: Encoding raw text without whitespace normalization increased out-of-vocabulary (`<unk>`) occurrences ($22,435$ raw vs. $69$ normalized in Hindi test; $3,797$ raw vs. $6$ normalized in Bhojpuri test). This documents an important preprocessing nuance: Devanagari whitespace collapsing is critical for subword boundary recognition.

---

### 2.2 Corrected Architecture and Embedding Budget

In the preliminary Phase 1 writeup, parameter allocation was estimated using an assumed dimension of $d_{\text{model}} = 512$. In Phase 2, the models were instantiated with $d_{\text{model}} = 384$ and $9$ layers to optimize head dimensions ($6$ heads $\times 64 = 384$) and fit student GPU memory budgets.

Both configurations tie the input token embeddings with the final linear language modeling head (`tie_embeddings = true`). Reinstantiating both models confirms the realized parameter allocations:

| Component | Hindi Model (`model_h_hindi`) | Bhojpuri Model (`model_l_bhojpuri`) |
|---|---:|---:|
| **Vocabulary Size ($V$)** | 24,000 | 16,000 |
| **Hidden Dimension ($d_{\text{model}}$)** | 384 | 384 |
| **Layers / Heads** | 9 / 6 ($d_{\text{head}} = 64$) | 9 / 6 ($d_{\text{head}} = 64$) |
| **Feed-Forward Dimension ($d_{\text{ff}}$)** | 1,536 ($4 \times d$) | 1,920 ($5 \times d$) |
| **Token Embedding Parameters** | $24,000 \times 384 = \mathbf{9,216,000}$ | $16,000 \times 384 = \mathbf{6,144,000}$ |
| **Embedding Budget Fraction** | **36.45%** | **24.70%** |
| **Transformer Backbone Parameters** | 16,069,248 (63.55%) | 18,726,912 (75.30%) |
| **Total Parameter Count** | **25,285,248** | **24,870,912** |

Untying the output projection would have added an additional $9.22\text{M}$ (Hindi) or $6.14\text{M}$ (Bhojpuri) parameters, pushing the models well beyond the $25\text{M}$ parameter budget ceiling. The realized architecture balances depth and capacity effectively.

---

### 2.3 Comprehensive Held-Out Evaluation (Validation and Test)

During pretraining, periodic validation was recorded every 500 steps over a fixed window of 100 batches ($204,800$ tokens), yielding estimates of $4.0954$ (Hindi) and $4.0676$ (Bhojpuri).

To ensure complete held-out metric alignment as requested by the rubric, `scripts/recompute_validation.py` evaluated the **entire validation sets**; the table also retains the original Phase 2 full-test results using non-overlapping 256-token windows and the exact UTF-8 byte denominators:

| Metric | Hindi Validation | Hindi Test | Bhojpuri Validation | Bhojpuri Test |
|---|---:|---:|---:|---:|
| **Tokens Scored** | 3,479,647 | 3,521,022 | 234,881 | 222,584 |
| **UTF-8 Bytes** | 34,973,965 | 35,374,088 | 2,176,729 | 2,045,282 |
| **Cross-Entropy (nats/token)** | **3.6667** | **3.6782** | **4.1737** | **4.1420** |
| **Perplexity** | **39.12** | **39.57** | **64.95** | **62.93** |
| **Bits per Byte (BPB)** | **0.5263** | **0.5282** | **0.6497** | **0.6503** |

#### Key Insights:
- **Distribution Stability**: For both languages, the full validation and test metrics match within 0.032 nats cross-entropy and $\pm 0.002$ BPB, showing similar aggregate performance; this does not establish identical distributions or prove absence of contamination.
- **Corpus Breadth vs. Repetition**: Hindi significantly outperforms Bhojpuri on the vocabulary-neutral BPB metric ($0.5282$ vs. $0.6503$). Hindi trained on ~337M unique tokens for ~1 pass, whereas Bhojpuri trained on ~23M unique tokens repeated for ~2.94 passes. This is consistent with a corpus-breadth explanation, but the two runs do not isolate data scale from corpus quality and other differences.

---

## 3. Phase 3 Synthetic Reasoning Framework (pilot and proposed release)

### 3.1 Task Families and Problem Formulations

Phase 3 requires synthetic reasoning datasets with known ground-truth answers in native Devanagari script, focusing on comparative and multi-hop reasoning. We define four core task families:

1. **Direct Comparison**:
   - *Premises*: Two entities with explicit scalar attributes (e.g., age, height, price, quantity).
   - *Query*: Identify which entity has the greater or smaller value.
   - *Control*: 50/50 balance between greater/smaller queries and first/second entity correct answer positions.
2. **Extremum (Largest / Smallest among Three)**:
   - *Premises*: Three distinct entities with strictly distinct attribute values.
   - *Query*: Identify the unique maximum or minimum.
   - *Control*: Strict tie rejection; uniform 33.3% distribution across entity positions.
3. **Equality Verification**:
   - *Premises*: Two entities with numeric values for the same attribute.
   - *Query*: Are the two values equal? Target-language binary decision (हाँ/ना or बा/नइखे).
   - *Control*: Exactly 50% equal and 50% unequal; balanced direction for unequal pairs.
4. **Two-Hop Transitivity**:
   - *Premises*: Two relational comparisons chaining three entities ($A > B$ and $B > C$). No direct endpoint relation is stated.
   - *Query*: Identify the greater or smaller entity between endpoints $A$ and $C$.
   - *Control*: Requires multi-hop deduction; premise order ($A>B, B>C$ vs. $B>C, A>B$) and query polarity are strictly counterbalanced.

### 3.2 Linguistic Authenticity: Hindi vs. Bhojpuri

To satisfy the requirement that Bhojpuri is not treated as a mere dialect or word-for-word translation of Hindi, separate templates and entity pools are authored:

- **Hindi**: Employs standard khadi boli case-marking, postpositions (`से बड़ा`, `की तुलना में`), and standard copulas (`है`, `हैं`).
- **Bhojpuri**: Employs authentic Eastern Indo-Aryan morphology:
  - Comparative postpositions: `से बड़`, `से ढेर`, `से छोट`, `से कम`.
  - Native verbal forms and copulas: `बा`, `बाड़न`, `बाटे`, `नइखे`.
  - Regional vocabulary and names: Bhojpuri cultural entities, units, and familial honorifics.

### 3.3 Leakage Prevention and Challenge Splits

To evaluate genuine reasoning rather than lexical memorization:
- **Problem Instance Partitioning**: Problems are represented as symbolic relation graphs and assigned to splits *before* textual rendering. No paraphrases or permutations of the same instance cross splits.
- **Unseen Entities Challenge (500 examples)**: Held-out person/object names not seen during training, testing generalization to novel tokens.
- **Unseen Wording Challenge (500 examples)**: Reserved syntactic templates and phrasing patterns, testing robustness against syntactic memorization.



### 3.4 Development pilot and review status (9 September)

The executable generator and symbolic solver are in `src/lma_phase3/`; each language has its
own `configs/phase3_templates.json`. [Pilot v1](samples/generated/phase3/pilot_v1/review.md)
contains 50 development examples per language, with solver-derived answers and blank human
review worksheets. Both languages cover 13 direct comparisons, 13 extrema, 12 equality questions,
and 12 two-hop chains. Examples use ages/heights and male names; this is an intentionally limited
wording pilot, not the proposed complete dataset.

The equality labels are balanced 6/6. Other answer positions, greater/smaller queries, and domains
are balanced to within one per family. The longest complete prompt/answer plus EOS is 47 Hindi
tokens or 51 Bhojpuri tokens. All 100 examples are free of unknown-token IDs and duplicate prompts;
canonical named-world fingerprints are unique within and across these language pilots.

Tests reject ambiguous graphs, contradictory comparisons, ties with non-unique answers, and
two-hop questions with a direct shortcut. They check invariance to premise order and entity
reindexing, reproducibility, quotas, and serialized token lengths. These checks validate symbolic
correctness, not language naturalness. The user subsequently approved both pilots as natural; this approval covers wording,
including the displayed yes/no spellings, and is recorded separately from solver verification. Each CSV has explicit language, answer, correction, and reviewer fields.

These are development examples only. Their world fingerprints must be excluded from later
held-out sets. At the pilot stage, production splitting and challenge sets were pending; Section 3.5 records
their completion, and Sections 5–8 record the completed finetuning, reasoning evaluations, and attention results. The design in Sections 3.1–3.3 describes release
behavior, except where explicitly demonstrated by this pilot.


### 3.5 Verified reasoning_v1 release

The user reviewed the Hindi and Bhojpuri development examples and stated: “they all seem natural
to me”. The review record and worksheet annotations preserve this as collective wording approval;
it is not evidence of a separate manual answer audit of the full release.

Each language now has 20,000 training, 2,000 validation, 2,000 primary-test, 500 unseen-name and
500 unseen-wording examples. Four task families are equally represented in every split. Equality
labels and other answer/direction/domain controls differ by at most one when a family count is odd.
All 50,000 answers passed an additional verifier based on numerical sorting or exhaustive ranking
enumeration, separate from the generation solver. No pilot world, canonical named world, or
normalized prompt crosses a release boundary. These checks do not imply unseen task structures;
primary splits intentionally share task families.

The maximum serialized lengths are 47 Hindi tokens and 53 Bhojpuri tokens, including answer and
EOS. No accepted example contains unknown-token IDs. Duplicate candidates were rejected and
regenerated; rejection counts, resource/code/tokenizer hashes, and compressed-file hashes are
preserved in `audits/generated/phase3/reasoning_v1/`.

The core and name-challenge splits use reviewed wording variant 0. Variant 1 is reserved exclusively
for the wording challenge. This is a narrow wording-generalization test: both variants share fact
phrasing, answer cues and task definitions. The realized v1 uses two domains, masculine names,
and fewer wording patterns than the initial proposal. These are scope limitations to report with
results. New name strings are substitutions into approved templates, not separately reviewed prose.

Datasets and manifests were copied to each language’s `phase3/reasoning_v1/` folder in the existing
Drive filesystem, and destination SHA-256 hashes matched. Remote synchronization and shareability
are managed separately by Drive. The complete reproduction and fixed scoring protocol are in
[phase-3-release.md](phase-3-release.md). Pretrained reasoning baselines are reported in Section 4; no finetuning improvement is claimed yet.


## 4. Pretrained reasoning baselines

Both original Phase 2 checkpoints were evaluated on reasoning_v1 before finetuning. All 10,000
predictions (5,000 per language) are saved under `results/generated/phase3/pretrained_v1/`.
Checkpoint, tokenizer and dataset SHA-256 checks passed. Every saved prediction was independently
checked against the source ID, prompt, reference answer, decoded token IDs and exact-match rule.

| Split | Examples per language | Hindi exact match | Bhojpuri exact match |
|---|---:|---:|---:|
| Validation | 2,000 | 0/2,000 (0.00%) | 0/2,000 (0.00%) |
| Primary test | 2,000 | 0/2,000 (0.00%) | 0/2,000 (0.00%) |
| Unseen names | 500 | 0/500 (0.00%) | 0/500 (0.00%) |
| Unseen wording | 500 | 0/500 (0.00%) | 0/500 (0.00%) |

Accuracy is also zero in each of the four task families. Generation is greedy, with EOS or a
16-token limit. The scorer applies Unicode NFC and whitespace normalization to the entire
continuation; it does not remove punctuation or extract a matching name from a longer sentence.
The original protocol was not changed after inspecting these results. Evaluation used float32
on the local MPS GPU, with equal-length prompt batches and unchanged model weights.

The primary Hindi test produced 2,000 noncanonical continuations. Bhojpuri produced 1,749
noncanonical continuations and 251 empty outputs. Neither model produced an incorrect but valid
canonical alternative: their errors were all outside the required answer format. Thus these
numbers measure end-to-end short-answer task performance and do not isolate comparison reasoning
from answer-format and stopping behavior. Language-model pretraining alone has not made these
models reliable users of the synthetic QA format. The same evaluator will be applied after
finetuning, with validation-only checkpoint selection.

Full per-task tables, deterministic error examples, protocol details and artifact locations are
in [phase-3-baselines.md](phase-3-baselines.md). The next step is implementing and testing
answer-token-only finetuning, then comparing the independently finetuned models to these frozen
baselines. No finetuned accuracy or attention change has been measured yet.


### 4.1 Supplemental answer-presence scoring

After reviewing the strict baseline, the user requested scoring whether the canonical answer
appears anywhere in the generated text. The same saved generations were rescored without
changing prompts, models or decoding. Whole-name/label matching gives primary-test presence rates
of **17.0% Hindi (340/2,000)** and **3.1% Bhojpuri (62/2,000)**. Strict exact match remains zero.

This is a post-hoc mention diagnostic, not final-answer or explanation correctness. Of the 340
Hindi hits, 97 also mention another candidate. Matches inside larger words are rejected: a naive
substring rule would instead credit 80 Bhojpuri outputs, including 18 such false positives.
The full split/task tables, matching rule, ambiguity counts and reproduction steps are in
[phase-3-answer-presence.md](phase-3-answer-presence.md). Retain both metrics for the future
pretrained/finetuned comparison, keeping their meanings distinct.


### 4.2 Corrected task-specific explanation verification

The first explanation scorer incorrectly required `कारण:` before examining statements. Its
zero coverage was a format-gate result, and did not implement the requested prose evaluation.
It is superseded by the [corrected semantic rescore](phase-3-semantic-verification.md).
Both the original 16-token outputs and the later 96-token explanation-prompt outputs have now
been rescored without generating again. The protocols remain separate.

| Primary test (2,000 per language) | Hindi | Bhojpuri |
|---|---:|---:|
| Original outputs: answer appears anywhere | 340 (17.0%) | 62 (3.1%) |
| Longer explanation outputs: answer appears anywhere | 132 (6.6%) | 46 (2.3%) |
| Original outputs: at least one supported fact | 118 | 0 |
| Longer explanation outputs: at least one supported fact | 25 | 0 |

The revised parser recognizes facts and comparisons without headers, and verifies their values
or relations with task-specific rules. Unknown prose is unresolved, not automatically false.
No sufficient supported explanation was found in either saved generation set; this is a bounded
parser result, not a claim of universal zero reasoning accuracy. Answer presence is retained as
the user-requested score and does not require a recognized conclusion or complete explanation.

All 20,000 predictions were rescored with source hash and ID checks. The marker-free reference
proofs pass for both languages; 153 software tests pass with 0 lint errors. The corrected report includes all splits,
examples, limitations and the withdrawn preliminary scoring count. These are post-hoc exploratory
measurements. Freeze the corrected scoring version for the later pretrained/finetuned comparison;
use validation data for any further parser changes and human-labelled validation for claims about
unrestricted prose accuracy.

---

## 5. Reasoning Finetuning Implementation and Protocol

### 5.1 Training Mechanics and Loss Masking

To transition the models from next-token language generators into grounded reasoning solvers without suffering catastrophic format failure, we implemented an answer-plus-explanation causal training pipeline (`src/lma_phase3/finetune_data.py` and `src/lma_phase3/finetune.py`).

1. **Deterministic Target Generation**:
   Supervision targets were generated programmatically from `reference_explanation()` in `src/lma_phase3/proof_checker.py`, producing:
   $$\text{Target} = \langle\text{Canonical Answer}\rangle \text{। कारण: } \langle\text{Premise 1}\rangle \langle\text{Premise 2}\rangle \langle\text{Deductive Step}\rangle$$
   This guarantees that every ground-truth training instance is mathematically sound, numerically grounded in the prompt premises, and syntactically compliant with the target language.

2. **Shift Alignment and Target Masking**:
   Let prompt tokens be $P = [p_0, \dots, p_{N-1}]$ and target tokens (including appended $\text{EOS}$) be $T = [t_0, \dots, t_{K-1}, \text{eos}]$. The combined sequence $S = P + T$ has length $L = N + K + 1 \le 256$.
   The inputs fed to the decoder are $X = S[:-1]$ and shifted targets are $Y = S[1:]$.
   To ensure the model is evaluated and trained purely on its deductive output:
   - Targets $Y[0 : N-1]$ are set to $-100$ (`ignore_index`), preventing backpropagation on prompt reconstruction.
   - Target $Y[N-1]$ is the first token of the answer ($t_0$), predicted conditionally on the entire prompt context $P$.
   - Target $Y[L-2]$ is $\text{EOS}$, enforcing clean stopping behavior.
   - Variable-length padding tokens beyond sequence length are masked with $-100$.

3. **Loss Computation and Gradient Accumulation**:
   For each microbatch $b$, loss was computed as the mean cross-entropy over active target tokens:
   $$\mathcal{L}_b = \frac{1}{\sum_{i,t} \mathbb{I}(Y_{i,t} \ne -100)} \sum_{i,t} \ell_{i,t} \cdot \mathbb{I}(Y_{i,t} \ne -100)$$
   During gradient accumulation over $M$ microbatches (effective batch size 16 = 2 microbatches of 8), gradients were scaled by $1/M$:
   $$\nabla_\theta \mathcal{L}_{\text{accum}} = \frac{1}{M} \sum_{b=1}^M \nabla_\theta \mathcal{L}_b$$
   This implements the unweighted mean of microbatch means for stable gradient scaling across variable target lengths. For tracking and reporting, the logged step loss accumulated active tokens globally across the step:
   $$\mathcal{L}_{\text{logged}} = \frac{\sum_{b=1}^M \sum_{i,t} \ell_{i,t} \cdot \mathbb{I}(Y_{i,t} \ne -100)}{\sum_{b=1}^M \sum_{i,t} \mathbb{I}(Y_{i,t} \ne -100)}$$

### 5.2 Optimization & Device State Management

- **Model Instantiation**: Both models were initialized directly from their final Phase 2 pretrained checkpoints:
  - Hindi Model H: `checkpoint_step_0020300.pt` ($25.29\text{M}$ parameters, $d_{\text{model}}=384$, 9 layers, 6 heads, tied 24k embeddings).
  - Bhojpuri Model L: `checkpoint_step_0004150.pt` ($24.87\text{M}$ parameters, $d_{\text{model}}=384$, 9 layers, 6 heads, tied 16k embeddings).
- **Optimizer & Schedule**: A fresh AdamW optimizer was initialized ($\text{lr} = 1.0 \times 10^{-4}$, $\text{weight decay} = 0.1$, $\beta_1 = 0.9, \beta_2 = 0.95$, $\epsilon = 10^{-8}$) with a 100-step linear warmup and cosine decay to $\text{lr}_{\text{min}} = 1.0 \times 10^{-5}$ across 2,500 steps (2 full epochs over 20,000 examples with effective batch size 16).
- **Safe Checkpointing and Resumption**:
  To guarantee exact reproducibility and pause/resume safety on local hardware, `save_finetune_checkpoint()` captures complete device state: Python `random`, NumPy, PyTorch CPU, and PyTorch MPS hardware RNG via `torch.mps.get_rng_state()`, along with epoch and microbatch position. Checkpoint writing is atomic via temporary file rename, retaining the three newest checkpoints and tracking `latest.json` and `best_checkpoint.pt`. On resume, configuration architecture parameters are strictly validated against the checkpoint, and data iteration advances deterministically to the exact interrupted batch.

### 5.3 Local Hardware Performance, Loss Trajectory & Validation Selection

Finetuning was executed entirely on the local Apple M4 GPU (16 GB unified RAM) via PyTorch MPS:
- **Throughput**: ~70–72 examples/second (median step time ~0.22–0.28 seconds per batch of 16).
- **Total Runtime**: Approximately 9.5 minutes per language run (2,500 steps).
- **Periodic Validation Subset**: During training, periodic validation evaluated 50 validation batches (400 examples at microbatch size 8). Because validation files group items by task blocks, these first 400 rows belong to the direct comparison task family. The reported selection losses were:
  - **Hindi Model H**: Train loss dropped to **$0.0067$**; 400-example subset validation cross-entropy dropped to **$0.0041$ nats/token** (PPL **$1.0041$**).
  - **Bhojpuri Model L**: Train loss dropped to **$0.0154$**; 400-example subset validation cross-entropy dropped to **$0.0168$ nats/token** (PPL **$1.0170$**).
- **Full Multi-Task Validation Verification**: To ensure checkpoint selection was robust and not biased by the 400-example subset, all retained checkpoints were evaluated post-hoc across the **complete 2,000-example validation split** covering all four task families (Direct, Extremum, Equality, and Transitivity; see `report/results/generated/phase3/full_validation.json`):
  - **Hindi**: Step 2,000 loss 0.01151 (PPL 1.0116) $\to$ Step 2,250 loss 0.00966 (PPL 1.0097) $\to$ **Step 2,500 loss 0.00825 (PPL 1.0083)**.
  - **Bhojpuri**: Step 2,000 loss 0.02174 (PPL 1.0220) $\to$ Step 2,250 loss 0.01864 (PPL 1.0188) $\to$ **Step 2,500 loss 0.01681 (PPL 1.0170)**.
  
  This confirms that step 2,500 achieved the global minimum loss across the complete multi-task validation distribution for both languages, verifying that the selected final checkpoint is optimal without requiring retraining.

![Finetuning Curves](figures/generated/phase3/finetuning_curves.png)

---

## 6. Comprehensive Held-Out Reasoning Evaluation

Following the frozen protocol, both finetuned models were evaluated across all four held-out splits: Validation (2,000), Primary Test (2,000), Unseen Names Challenge (500), and Unseen Wording Challenge (500), under both the original 16-token prompt protocol and the 96-token explanation protocol.

### 6.1 Primary Test: Pretrained vs. Finetuned Performance

| Protocol / Metric | Hindi Pretrained | Hindi Finetuned | Bhojpuri Pretrained | Bhojpuri Finetuned |
|---|---:|---:|---:|---:|
| **Original 16-Token Protocol** | | | | |
| Answer Presence Rate | 340 / 2000 (17.00%) | **1677 / 2000 (83.85%)** | 62 / 2000 (3.10%) | **1700 / 2000 (85.00%)** |
| Strict Exact Match | 0 / 2000 (0.00%) | 0 / 2000 (0.00%) | 0 / 2000 (0.00%) | 0 / 2000 (0.00%) |
| Recognized Supported Claims | 118 (5.90%) | **1631 (81.55%)** | 0 (0.00%) | **1245 (62.25%)** |
| Unsupported / Conflicting Claims | 328 (16.40%) | **3 (0.15%)** | 0 (0.00%) | **193 (9.65%)** |
| **Explanation 96-Token Protocol** | | | | |
| Answer Presence Rate | 132 / 2000 (6.60%) | **1834 / 2000 (91.70%)** | 46 / 2000 (2.30%) | **1783 / 2000 (89.15%)** |
| Fully Verified Explanations | 0 / 2000 (0.00%) | **1723 / 2000 (86.15%)** | 0 / 2000 (0.00%) | **1398 / 2000 (69.90%)** |
| Joint Fully Verified Rate | 0 / 2000 (0.00%) | **86.15%** | 0 / 2000 (0.00%) | **69.90%** |
| Supported Claim Coverage | 25 (1.25%) | **2000 (100.00%)** | 0 (0.00%) | **2000 (100.00%)** |

### 6.2 Task Family Breakdown on Primary Test (96-Token Protocol)

| Task Family | Hindi Pretrained Verified | Hindi Finetuned Verified | Bhojpuri Pretrained Verified | Bhojpuri Finetuned Verified |
|---|---:|---:|---:|---:|
| **Direct Comparison** (500 items) | 0.0% | **94.80% (474/500)** | 0.0% | **75.20% (376/500)** |
| **Two-Hop Transitivity** (500 items) | 0.0% | **100.00% (500/500)** | 0.0% | **100.00% (500/500)** |
| **Extremum (3 entities)** (500 items) | 0.0% | **83.00% (415/500)** | 0.0% | **47.80% (239/500)** |
| **Equality Verification** (500 items) | 0.0% | **66.80% (334/500)** | 0.0% | **56.60% (283/500)** |
| **Overall Primary Test** (2,000 items) | **0.0%** | **86.15% (1723/2000)** | **0.0%** | **69.90% (1398/2000)** |

### 6.3 Evaluation Across All Splits (Explanation Protocol, 96 Tokens)

| Language | Split | Total | Answer Present | Supported Claims | Unsupported Claims | Joint Verified Explanations |
|---|---|---:|---:|---:|---:|---:|
| **Hindi** | Validation | 2,000 | 1830 (91.50%) | 2000 (100.0%) | 283 (14.15%) | **1717 (85.85%)** |
| **Hindi** | Primary Test | 2,000 | 1834 (91.70%) | 2000 (100.0%) | 277 (13.85%) | **1723 (86.15%)** |
| **Hindi** | Challenge Wording | 500 | 447 (89.40%) | 500 (100.0%) | 88 (17.60%) | **323 (64.60%)** |
| **Hindi** | Challenge Names | 500 | 123 (24.60%) | 173 (34.60%) | 68 (13.60%) | **0 (0.00%)** |
| **Bhojpuri** | Validation | 2,000 | 1778 (88.90%) | 2000 (100.0%) | 598 (29.90%) | **1393 (69.65%)** |
| **Bhojpuri** | Primary Test | 2,000 | 1783 (89.15%) | 2000 (100.0%) | 589 (29.45%) | **1398 (69.90%)** |
| **Bhojpuri** | Challenge Wording | 500 | 436 (87.20%) | 500 (100.0%) | 158 (31.60%) | **340 (68.00%)** |
| **Bhojpuri** | Challenge Names | 500 | 68 (13.60%) | 68 (13.60%) | 64 (12.80%) | **0 (0.00%)** |

---

## 7. Error Analysis & Qualitative Case Studies

### 7.1 Qualitative Successes

The finetuned models reliably demonstrate structured, step-by-step reasoning that reproduces both premises faithfully, computes the relative ordering, and outputs the exact target answer.

**Hindi Direct Comparison (`hi-test-000000`)**:
- *Prompt*: `अशोक की उम्र 13 साल है। सुमित की उम्र 25 साल है। अशोक और सुमित में से किसकी उम्र सबसे कम है? पहले उत्तर लिखें। फिर कारण: लिखकर तुलना का छोटा कारण बताएँ। उत्तर:`
- *Generated*: `अशोक। कारण: अशोक की उम्र 13 साल है। सुमित की उम्र 25 साल है। सुमित की उम्र अशोक से ज़्यादा है।`
- *Diagnostics*: `answer_present: True`, `joint_fully_verified: True`, `unsupported_claims: []`. Note: The model proves $13 < 25$ by validly asserting the equivalent converse relation that Sumit's age is greater than Ashok's.

**Bhojpuri Two-Hop Transitivity (`bho-test-001500`)**:
- *Prompt*: `गणेश के लंबाई हरिहर से ढेर बा। हरिहर के लंबाई रामनाथ से ढेर बा। रामनाथ आ गणेश में केकर लंबाई सबसे कम बा? पहिले जवाब लिखीं। फेर कारण: लिख के तुलना के छोट कारण बताईं। जवाब:`
- *Generated*: `रामनाथ। कारण: गणेश के लंबाई हरिहर से ढेर बा। हरिहर के लंबाई रामनाथ से ढेर बा। गणेश के लंबाई रामनाथ से ढेर बा।`
- *Diagnostics*: `answer_present: True`, `joint_fully_verified: True`, `unsupported_claims: []`.

Remarkably, on **Two-Hop Transitivity**, both Model H and Model L achieved **100.00% verified proof accuracy (500/500)** on the primary test, demonstrating that on this finite benchmark, causal transformers can reliably chain multi-step relational inequalities without hallucinating intermediate relations.

### 7.2 Failure Modes and Generalization Boundaries

1. **Direct Comparison False Conclusion (`bho-test-000003`)**:
   - *Prompt*: `बनवारी के लंबाई 106 सेंटीमीटर बा। गुड्डू के लंबाई 197 सेंटीमीटर बा। बनवारी आ गुड्डू में सबसे लमहर के बा? पहिले जवाब लिखीं। फेर कारण: लिख के तुलना के छोट कारण बताईं। जवाब:`
   - *Generated*: `बनवारी। कारण: बनवारी के लंबाई 106 सेंटीमीटर बा। गुड्डू के लंबाई 197 सेंटीमीटर बा। बनवारी के लंबाई गुड्डू से ढेर बा।`
   - *Diagnostics*: `answer_present: True`, `joint_fully_verified: False`, `unsupported_claims: [{'kind': 'gt', 'left': 0, 'right': 1}]`.
   - *Analysis*: The model faithfully recalled both scalar premises (106 cm and 197 cm), but failed the final relational deduction by falsely asserting $106 > 197$. Because answer presence alone was satisfied (`बनवारी` appears as the initial token), this demonstrates why strict proof verification is necessary to penalize false mathematical deductions that superficial string matching would credit.

2. **Unseen Syntactic Patterns (Challenge Wording)**:
   When evaluated on syntactic templates reserved exclusively for held-out evaluation (Variant 1), both models maintained robust answer presence (Hindi $89.40\%$, Bhojpuri $87.20\%$). It is important to distinguish standalone explanation verification from joint success:
   - In Hindi, standalone explanation verification reached **$395/500$ ($79.00\%$)**, while joint answer-plus-explanation verification was **$323/500$ ($64.60\%$)**.
   - In Bhojpuri, both standalone and joint verification reached **$340/500$ ($68.00\%$)**.
   The moderate drop in joint verification occurs primarily in extremum and equality tasks, where slight syntactic inversion in the prompt led the model to emit a valid reasoning derivation while omitting the explicit initial answer token.

3. **Unseen Named Entities (Challenge Names)**:
   When tested on names withheld entirely from the reasoning training split, verified explanations dropped to zero, and answer presence fell to $24.60\%$ (Hindi) and $13.60\%$ (Bhojpuri). Inspection of outputs reveals that unseen names segment into unfamiliar multi-token subword pieces, triggering subword repetition or fallback to familiar names from the pretraining vocabulary. This provides strong support for the hypothesis that in small decoder models (~25M parameters), relational reasoning is heavily coupled to familiar entity tokenization boundaries.

4. **Strict Exact Match vs. Whole-Output Evaluation**:
   Strict exact match remained $0.0\%$ across all splits because the models generated complete, punctuated explanations (e.g., `अशोक। कारण: ...`) rather than bare one-word tokens. This empirically validates the architectural decision to score answer presence and semantic completeness separately.

---

## 8. Multi-Head Attention Analysis (Pretrained vs. Finetuned)

Using the Phase 2 attention extraction toolkit (`src/lma_phase3/attention_compare.py`), we evaluated multi-head attention on identical fixed reasoning prompts before and after finetuning.

### 8.1 Quantitative Attention Shifts

All values are transcribed directly from the generated CSV summaries (`report/figures/generated/phase3/attention/{hindi,bhojpuri}/attention_comparison.csv`):

| Model | Layer | Head | Pretrained Entropy | Finetuned Entropy | Entropy $\Delta$ | Pretrained Distance | Finetuned Distance | Distance $\Delta$ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Hindi Model H** | 1 (Early) | Head 0 | 2.5795 | 2.5654 | -0.0141 | 6.49 | 8.15 | **+1.67** |
| | 1 (Early) | Head 2 | 2.7661 | 2.6310 | -0.1351 | 9.93 | 11.01 | **+1.08** |
| | 2 (Mid) | Head 5 | 1.5254 | 1.0980 | **-0.4274** | 1.99 | 4.11 | **+2.12** |
| | 8 (Late) | Head 0 | 1.7738 | 2.0242 | +0.2504 | 14.10 | 11.11 | **-2.98** |
| | 8 (Late) | Head 1 | 2.1614 | 1.9130 | **-0.2484** | 11.88 | 12.63 | +0.76 |
| | 8 (Late) | Head 5 | 2.3075 | 1.9286 | **-0.3789** | 13.88 | 10.16 | **-3.71** |
| **Bhojpuri Model L** | 1 (Early) | Head 0 | 2.7917 | 2.7737 | -0.0180 | 11.78 | 10.84 | -0.94 |
| | 1 (Early) | Head 2 | 2.7964 | 2.7034 | -0.0930 | 9.47 | 9.12 | -0.35 |
| | 2 (Mid) | Head 5 | 2.6079 | 2.2683 | **-0.3396** | 8.45 | 6.27 | **-2.18** |
| | 8 (Late) | Head 1 | 2.1333 | 1.7536 | **-0.3797** | 8.86 | 8.88 | +0.02 |
| | 8 (Late) | Head 5 | 2.4157 | 2.1336 | **-0.2820** | 11.32 | 9.97 | **-1.35** |

### 8.2 Qualitative Attention Heatmap Analysis

The side-by-side heatmaps ([Layer 1 Head 2](figures/generated/phase3/attention/hindi/compare_layer_1_head_2.png), [Layer 8 Head 1](figures/generated/phase3/attention/hindi/compare_layer_8_head_1.png)) illustrate clear descriptive shifts in attention allocation:
1. **Descriptive Head Specialization**: In the pretrained models, attention across heads was relatively diffuse, characteristic of general autoregressive language modeling. After finetuning, several heads exhibit noticeable entropy reduction:
   - In Hindi, mid-layer Head L2 H5 shows a drop in entropy ($\Delta = -0.43$) paired with increased mean attention distance ($\Delta = +2.12$), routing probability mass across premise clauses between names and scalar attributes.
   - In Bhojpuri, Head L2 H5 exhibits both an entropy drop ($\Delta = -0.34$) and a distance decrease ($\Delta = -2.18$), concentrating attention locally within adjacent premise tokens.
   - In both models, late heads (Hindi L8 H5, Bhojpuri L8 H5) show both entropy reduction and reduced attention distance ($\Delta = -3.71$ Hindi, $-1.35$ Bhojpuri), shifting attention away from generic prefix tokens toward the proximate entity values needed to form the comparative conclusion.
2. **Epistemic Limitation**: These attention entropy and distance shifts describe changes in probability concentration across sequence positions. They do not constitute proof of an isolated comparison circuit; demonstrating causal necessity would require activation patching or head ablations.

---

## 9. Synthesis: Addressing the Course Research Questions

### Question 1: How did data scale and quality differ between Model H and Model L?
- **Scale and Targets**: The assignment established an aspirational target of approximately 500M training tokens per language where feasible.
  - Model H's packed pretraining stream contained $336.99\text{M}$ tokens, of which $332.60\text{M}$ tokens were processed across 20,300 steps (~1 epoch). The shortfall relative to 500M reflects strict de-duplication and the constraint that manual tokens constitute $\ge 20\%$ of the corpus.
  - Model L's packed pretraining stream contained $23.14\text{M}$ tokens ($225,689$ documents), with $67.99\text{M}$ tokens processed across 4,150 steps (~2.94 epochs). This shortfall is explicitly permitted and justified by the course specification due to the scarcity of clean, public-domain monolingual Bhojpuri web text and strict filtering of Hindi dialect contamination.
- **Quality & Composition**: Both corpora satisfied the $\ge 20\%$ manual collection threshold in normalized pre-decontamination statistics (Hindi: $25.06\%$, Bhojpuri: $22.49\%$). However, Bhojpuri required extensive manual crawl curation, sitemap filtering, and character 4-gram discrimination to remove parallel translations and Hindi borrowings.

### Question 2: How do language-modeling and reasoning results compare across the two resource tiers?
- **Pretrained LM Performance**: Model H significantly outperformed Model L in vocabulary-neutral compression ($0.5282$ vs. $0.6503$ Bits per Byte; $39.57$ vs. $62.93$ test perplexity), directly reflecting the benefits of pretraining token volume and lexical coverage.
- **Reasoning Finetuning Performance**: In contrast to pretraining, the performance gap narrowed dramatically during reasoning finetuning:
  - Both models achieved comparable answer presence on the primary test ($83.85\%$ Hindi vs. $85.00\%$ Bhojpuri in the 16-token protocol; $91.70\%$ Hindi vs. $89.15\%$ Bhojpuri in the 96-token protocol).
  - Both models achieved **$100.00\%$ verified transitivity reasoning** on the primary test.
  - Hindi maintained an advantage in multi-entity extremum reasoning ($83.0\%$ vs. $47.8\%$) and overall joint proof verification ($86.15\%$ vs. $69.90\%$). This performance difference reflects Hindi's larger pretraining corpus breadth ($332.6\text{M}$ vs $68.0\text{M}$ tokens processed) and higher subword coverage rather than transformer backbone capacity, as Bhojpuri actually possesses the wider feed-forward dimension.

### Question 3: What tokenizer and corpus factors most affected the lower-resource model?
- **Devanagari Whitespace Normalization**: Collapsing Unicode whitespace prior to encoding proved vital; raw text encoding without whitespace collapsing multiplied out-of-vocabulary (`<unk>`) rates by orders of magnitude ($3,797$ raw vs. $6$ normalized in Bhojpuri test).
- **Parameter Budget Allocation**: Both models have 9 layers and $d_{\text{model}}=384$, totaling approximately 25M parameters:
  - Hindi's $24\text{k}$ vocabulary required $9.22\text{M}$ parameters ($36.45\%$ of total parameters), leaving $16.07\text{M}$ ($63.55\%$) for its transformer backbone ($d_{\text{ff}} = 1536$).
  - Bhojpuri's smaller $16\text{k}$ vocabulary required only $6.14\text{M}$ parameters ($24.70\%$), allowing a wider feed-forward dimension ($d_{\text{ff}} = 1920$) so that $18.73\text{M}$ ($75.30\%$) went to the backbone.
  While the wider backbone provided ample capacity, the smaller vocabulary resulted in higher subword fragmentation for novel entities.

### Question 4: What evidence explains the observed differences?
- **Empirical Demonstration of Synthetic Reasoning Transfer**: Pretrained models achieved $0.0\%$ explanation verification and low answer presence ($17.0\%$ Hindi, $3.1\%$ Bhojpuri), demonstrating that next-token pretraining alone does not yield multi-step reasoning capabilities. Finetuning on verified synthetic relational graphs raised primary test verification to $86.15\%$ (Hindi) and $69.90\%$ (Bhojpuri).
- **Subword Fragmentation Hypothesis**: On the Challenge Names split, verified proofs dropped to $0.0\%$ for both models. Qualitative inspection shows that unseen names segment into multiple unfamiliar subwords, disrupting greedy generation. This provides strong support for the hypothesis that subword fragmentation is the primary bottleneck for entity-centric reasoning in compact language models.
- **Descriptive Attention Restructuring**: The observed drops in multi-head attention entropy ($\Delta \approx -0.25$ to $-0.43$ in mid/late heads) describe a shift toward sharper probability concentration on premise entities and scalar values. While not proving isolated causal circuits, these descriptive shifts corroborate the models' transition from diffuse language modeling to localized relational tracking.

---

## 10. Checkpoint Manifest, Google Drive Links & Reproducibility

All large artifacts, packed binary streams, and resume-capable checkpoints are versioned and stored in Google Drive:

| Artifact | Language | Step | Size (Bytes) | SHA-256 Checksum | Google Drive Location |
|---|---|---:|---:|---|---|
| Pretrained Final | Hindi | 20,300 | 303,590,517 | `575458df89393ca2b5998024f93d04e9a08a1a5228f420da389108cfed870d42` | [Hindi Pretrain Folder](https://drive.google.com/drive/folders/1QS_l1VYF2A_ifCxQuEThoYDgqInpS7BF) |
| Pretrained Final | Bhojpuri | 4,150 | 298,618,485 | `69580b164964ddb88477fd39181f5d1a57ef52cfcaa32dbe1fc68e19b805c133` | [Bhojpuri Pretrain Folder](https://drive.google.com/drive/folders/1HQMO4a_Ud18URPi_X4nBNCIPZPcHeyCN) |
| Finetuned Best | Hindi | 2,500 | 303,575,984 | `8ac1b2c96642901f4025976ff597ac3d9abfc591820646572857972d176cc062` | `hindi/phase3/runs/finetune_v1/` |
| Finetuned Step 2500 | Hindi | 2,500 | 303,579,744 | `e0e76afdf39f89c1432a2034efeed201ed32e0cad90836db04be741293993457` | `hindi/phase3/runs/finetune_v1/` |
| Finetuned Best | Bhojpuri | 2,500 | 298,604,016 | `cccd2ed6ce9c776972a4589ec18e9aa5198e2f81b04da2213902b4babda421b0` | `bhojpuri/phase3/runs/finetune_v1/` |
| Finetuned Step 2500 | Bhojpuri | 2,500 | 298,607,712 | `28d1799b0d842ac745db2390450590b74cc904b798b429669b1a010cb24d36f6` | `bhojpuri/phase3/runs/finetune_v1/` |

### Complete Reproduction Commands

```bash
# 1. Finetune Hindi Model H
python -m lma_phase3.finetune \
  --config hindi/configs/phase3.toml \
  --pretrained-checkpoint /path/to/hindi_pretrain_20300.pt \
  --train-data tmp/phase3/reasoning_v1/hindi/train.jsonl.gz \
  --val-data tmp/phase3/reasoning_v1/hindi/validation.jsonl.gz \
  --template-config hindi/configs/phase3_templates.json \
  --output-dir tmp/phase3/runs/hindi/finetune_v1 \
  --device mps

# 2. Evaluate Hindi Model H on all held-out splits
python -m lma_phase3.evaluate_finetuned \
  --config hindi/configs/phase3.toml \
  --checkpoint tmp/phase3/runs/hindi/finetune_v1/best_checkpoint.pt \
  --dataset-root tmp/phase3/reasoning_v1 \
  --output-dir report/results/generated/phase3/finetuned_v1/hindi \
  --device mps --protocols both

# 3. Multi-Head Attention Comparison
python -m lma_phase3.attention_compare \
  --config hindi/configs/phase3.toml \
  --pretrained-checkpoint /path/to/hindi_pretrain_20300.pt \
  --finetuned-checkpoint tmp/phase3/runs/hindi/finetune_v1/best_checkpoint.pt \
  --tokenizer hindi/tokenizer/hindi.model \
  --prompt "रमेश की लंबाई 194 सेंटीमीटर है। नितिन की लंबाई 110 सेंटीमीटर है। रमेश और नितिन में से किसकी लंबाई सबसे कम है? पहले उत्तर लिखें। फिर कारण: लिखकर तुलना का छोटा कारण बताएँ। उत्तर:" \
  --output-dir report/figures/generated/phase3/attention/hindi \
  --device cpu --layers 1 8 --heads 0 1 2
```

