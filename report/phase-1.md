# Phase 1: Data Collection and Tokenizer Construction

> Status: work in progress. Replace every `TBD` with generated results before submission.

## 1. Language selection

- **Model H:** Hindi (`hi`, Devanagari), selected because abundant public text is available.
- **Model L:** Bhojpuri (`bho`, Devanagari), selected from the assignment's allowed lower-resource list.

The two corpora, tokenizers, vocabularies, and later model weights are completely independent.

## 2. Corpus collection

### 2.1 Source inventory

| Language | Source | Collection type | Method | Licence | Clean training tokens |
|---|---|---|---|---|---:|
| Hindi | Hindi Wikipedia | Manual | MediaWiki Action API | CC BY-SA 4.0 | TBD |
| Hindi | Hindi Wikisource | Manual | MediaWiki Action API | CC BY-SA 4.0 | TBD |
| Hindi | IndicCorp v2 (`hin_Deva`) | Downloaded | HF streaming | CC0 | TBD |
| Bhojpuri | Bhojpuri Wikipedia | Manual | MediaWiki Action API | CC BY-SA 4.0 | TBD |
| Bhojpuri | Approved web sources | Manual | Rate-limited HTML crawl | Per-source | TBD |
| Bhojpuri | BhojpuriCorpus | Downloaded | HF streaming | Mixed per record | TBD |

### 2.2 Collection process

TBD: explain checkpoints, rate limits, source auditing, and provenance schema.

### 2.3 Cleaning and quality filtering

TBD: report Unicode NFC normalization, boilerplate removal, script ratios, Hindi/Bhojpuri
contamination filtering, and manually audited samples.

### 2.4 Deduplication

TBD: report exact SHA-256 and near-duplicate SimHash settings, removals by source, and the policy
of retaining the manually collected copy when manual and downloaded sources overlap.

### 2.5 Dataset statistics

| Language | Split | Documents | Tokens | Manual tokens | Downloaded tokens | Manual fraction |
|---|---|---:|---:|---:|---:|---:|
| Hindi | Train | TBD | TBD | TBD | TBD | TBD |
| Hindi | Validation | TBD | TBD | TBD | TBD | TBD |
| Hindi | Test | TBD | TBD | TBD | TBD | TBD |
| Bhojpuri | Train | TBD | TBD | TBD | TBD | TBD |
| Bhojpuri | Validation | TBD | TBD | TBD | TBD | TBD |
| Bhojpuri | Test | TBD | TBD | TBD | TBD | TBD |

TBD: justify any Bhojpuri shortfall from the approximately 500M-token target.

## 3. Tokenizer construction

Separate SentencePiece tokenizers were trained from scratch using only the corresponding language's
training split. Candidate vocabulary sizes were evaluated on held-out validation text.

| Language | Vocabulary | Fertility | Characters/token | UNK rate | Rare-type rate |
|---|---:|---:|---:|---:|---:|
| Hindi | 16,000 | TBD | TBD | TBD | TBD |
| Hindi | 24,000 | TBD | TBD | TBD | TBD |
| Hindi | 32,000 | TBD | TBD | TBD | TBD |
| Bhojpuri | 16,000 | TBD | TBD | TBD | TBD |
| Bhojpuri | 24,000 | TBD | TBD | TBD | TBD |
| Bhojpuri | 32,000 | TBD | TBD | TBD | TBD |

### 3.1 Selected tokenizers

- Hindi: TBD
- Bhojpuri: TBD

TBD: justify each choice using held-out fertility, unknown rate, vocabulary utilization, and the
approximately 25M-parameter model budget.

### 3.2 Tokenization examples

TBD: include representative normal, rare-word, numeric, and code-mixed examples for both languages.

## 4. Reproducibility

TBD: record environment, exact commands, seeds, source snapshot dates, Drive links, and checksums.

## 5. Limitations

TBD: discuss Bhojpuri spelling variation, Hindi/Awadhi contamination, source/domain imbalance,
licensing constraints, OCR errors, and limitations of automatic near-duplicate detection.
