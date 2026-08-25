# Phase 2: Decoder-only Transformer Implementation and Training

## 1. Pre-training split decontamination

Validation and test documents are protected before any model sees the training split. The audit
is asymmetric: it searches training against held-out data, records candidates, and never modifies
validation or test. This prevents evaluation leakage while preserving the original held-out sets.

### 1.1 Detection methods

The implementation in `src/lma_phase2/decontaminate.py` performs three checks:

1. **Exact document:** SHA-256 of NFC-normalized, case-folded, whitespace-collapsed text.
2. **Exact paragraph:** the same hash over line-preserving paragraphs with at least 100 characters
   and 15 words.
3. **Near document:** Bottom-k MinHash over word 5-grams, with sketch size 64, index prefix size 16,
   at least two shared prefix hashes, and estimated Jaccard similarity at least 0.95.

Match precedence is exact document, near document, then exact paragraph, so one training document
appears at most once in the review set. SQLite holds protected signatures, completed training IDs,
and candidate pairs, making a stopped audit safe to resume. Parallel workers compute immutable
text features, while matching and database writes remain ordered and single-writer.

### 1.2 Full-corpus audit results

| Language | Protected validation + test documents | Training documents checked | Exact documents | Exact paragraphs | Near documents | Flagged training documents |
|---|---:|---:|---:|---:|---:|---:|
| Hindi | 45,806 | 2,237,951 | 0 | 0 | 75 | 75 |
| Bhojpuri | 4,695 | 225,992 | 0 | 253 | 107 | 360 |

The absence of exact-document matches in both languages is consistent with the corpus-wide exact
deduplication completed before splitting. Hindi also has no exact-paragraph leakage. Its 75 near
candidates are all IndicCorp-to-IndicCorp pairs. Reviewed examples include genuine republications
with punctuation or dateline differences, but also distinct calendar-year pages sharing a fixed
template. Therefore, near candidates require pair-level review before removal.

Bhojpuri's candidates include clear copies across Bhojpuri Wikipedia and the downloaded
BhojpuriCorpus, as well as exact news paragraphs reused across Anjoria, `fish_food`, Wikipedia,
JNU-BHLTR, and blog sources. The exact-paragraph matches are strong evidence of held-out content
inside a larger training document. Near-document candidates still require review because the
MinHash score is an estimate.

### 1.3 Artifacts and decision boundary

- `report/audits/generated/hindi_decontamination_report.json`
- `report/audits/generated/hindi_train_heldout_matches.csv`
- `report/audits/generated/bhojpuri_decontamination_report.json`
- `report/audits/generated/bhojpuri_train_heldout_matches.csv`

The CSV files retain both sources, document IDs, URLs, text samples, protected split, similarity,
review decision, reviewer policy, and rationale. Review was completed on 25 August 2026 using a
user-directed threshold policy. For Hindi, the 9 near-document matches with estimated similarity
at least 0.97 were marked `remove`, while the remaining 66 were marked `keep`. For Bhojpuri, all
253 exact-paragraph matches and the 50 near-document matches with similarity at least 0.98 were
marked `remove`; the remaining 57 near-document matches were marked `keep`. Thus 312 training
documents are approved for removal across both languages. The official Google Drive splits remain
unchanged. Versioned replacements were generated after review, removing only the 312 approved
training documents. Their packed-data counts and checksums are recorded below; validation and test
remain byte-for-byte unchanged.

## 2. Transformer architecture

Both models use the same student-implemented decoder-only Transformer backbone: learned token and
absolute position embeddings, embedding dropout, nine pre-normalized blocks, a final LayerNorm,
and a tied token/output matrix. Each block contains six-head causal self-attention and a GELU
feed-forward network. No `nn.Transformer*`, `nn.MultiheadAttention`, Hugging Face model class, or
pretrained model component is used.

| Setting | Model H: Hindi | Model L: Bhojpuri |
|---|---:|---:|
| Vocabulary | 24,000 | 16,000 |
| Context length | 256 | 256 |
| Model dimension | 384 | 384 |
| Layers | 9 | 9 |
| Heads | 6 | 6 |
| Head dimension | 64 | 64 |
| Feed-forward dimension | 1,536 | 1,920 |
| Dropout | 0.1 | 0.1 |
| Exact trainable parameters | **25,285,248** | **24,870,912** |

The attention depth, model width, head count, and head dimension are held fixed to make the
resource-level comparison interpretable. Bhojpuri's smaller vocabulary saves 3.07M embedding
parameters, so some of that budget is reassigned to its feed-forward width while keeping both
models near 25M parameters. Weight tying avoids a second vocabulary-sized output matrix. Learned
absolute positions are simple and fully inspectable, but constrain inference to at most 256 tokens
without resizing and retraining the position table.

For an input `(B, T, 384)`, one linear layer produces Q/K/V and each is reshaped to
`(B, T, 6, 64)` and transposed to `(B, 6, T, 64)`. Attention scores have shape
`(B, 6, T, T)` and are divided by `sqrt(64)` before an explicit upper-triangular future mask and
softmax. Heads are transposed and concatenated back to `(B, T, 384)` before the output projection.
Pre-norm keeps the residual path direct and is more stable than post-norm for optimization.

## 3. Packed data and pretraining protocol

The Phase 2 tokenizer pass streams canonical Phase 1 JSONL, appends the language tokenizer's EOS
ID after every document, and writes little-endian `uint16` token streams. This avoids repeatedly
tokenizing hundreds of millions of tokens during Colab training. The pass checkpoints at document
boundaries and records input, tokenizer, and binary SHA-256 checksums. Random training windows come
from a dedicated PyTorch generator whose state is checkpointed; validation windows are sequential
and deterministic.

The completed packed artifacts are summarized in
`report/audits/generated/phase2_data_manifest.json`:

| Language | Train documents | Train tokens | Validation tokens | Test tokens | Removed train documents |
|---|---:|---:|---:|---:|---:|
| Hindi | 2,237,942 | 336,995,217 | 3,479,648 | 3,521,023 | 9 |
| Bhojpuri | 225,689 | 23,142,975 | 234,882 | 222,585 | 303 |

Both full-size models completed a one-step CPU training and validation smoke test using these
manifests. Each wrote a complete checkpoint and successfully reloaded it through `--resume latest`.
This validates the final data/model identity checks and recovery path; it is not reported as a
pretraining result.

| Setting | Hindi | Bhojpuri |
|---|---:|---:|
| Micro-batch sequences | 8 | 8 |
| Gradient accumulation | 8 | 8 |
| Effective tokens/step | 16,384 | 16,384 |
| Optimizer steps | 20,300 | 4,150 |
| Approx. sampled training tokens | 332.6M | 68.0M |
| Intended corpus passes | about 1 | about 3 |
| Peak learning rate | 3e-4 | 3e-4 |
| LR floor | 3e-5 | 3e-5 |
| Warmup | 500 steps | 200 steps |
| Validation/checkpoint interval | 500 | 250 |

Training uses AdamW (`betas=(0.9, 0.95)`, weight decay 0.1), linear warmup, cosine decay, mixed
precision on CUDA, gradient clipping at 1.0, and validation throughout training. Each atomic
checkpoint contains model weights, optimizer state, scheduler state, gradient-scaler state,
completed step, tokens seen, best validation loss, full resolved configuration, Python/NumPy/
PyTorch/CUDA random states, and the packed-data sampling generator state.

## 4. Evaluation implementation

The evaluation path computes full held-out cross-entropy, perplexity, and bits per UTF-8 byte. It
generates continuations using greedy decoding and seeded sampling at temperatures 0.5, 1.0, and
1.5, then calculates corpus BLEU-4, chrF++, mean ROUGE-L F1, Distinct-1/2, and repeated 4-gram
rate. Attention probabilities can be returned from every custom attention block. The report tools
produce early/late-layer heatmaps for multiple heads and per-head mean entropy and mean causal
attention distance. Numerical results, curves, samples, and plots will be added after independent
GPU pretraining completes.
