[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Hindi and Nepali Monolingual Small Language Models (SLMs)

This repository builds fully independent decoder-only causal language models (~25M parameters) from scratch for two Indic languages in Devanagari script:

- **Model H (Language #1):** Hindi (`hi`, Devanagari), the higher-resource Indian language.
- **Model L (Language #2):** Nepali (`ne`, Devanagari), the primary Language #2 target (~300M tokens, $\ge 20\%$ manual tokens; 273.17M train tokens at 22.74% manual achieved).

The corpora, tokenizers, vocabulary artifacts, training splits, and model weights are completely independent. Token strings can naturally coincide because both languages use Devanagari; there is no shared embedding table, tokenizer, or weight parameter.

---

## 1. Project Requirements & Status

| Criterion | Requirement | Hindi (Model H) | Nepali (Model L) |
|---|---|---:|---:|
| **Target Scale** | ~300M – 500M training tokens | 332,246,742 tokens | 273,165,831 tokens |
| **Manual Data Share** | $\ge 20\%$ of training tokens | **25.06%** (83.27M tokens) | **22.74%** (62.11M tokens) |
| **Data Partitioning** | 98% Train / 1% Val / 1% Test | 2,237,951 / 23,033 / 22,773 docs | 704,658 / 7,377 / 7,333 docs |
| **Selected Tokenizer** | SentencePiece Unigram from scratch | 24,000 vocab (fertility 1.2928) | 16,000 vocab (fertility 1.3522) |
| **Model Size** | ~25M parameters ($d_{\text{model}}=384$) | 25,285,248 parameters | 24,870,912 parameters |
| **Embedding Budget** | Balance depth vs. embeddings | 36.45% embedding / 63.55% transformer | 24.70% embedding / 75.30% transformer |
| **Pipeline Reproducibility**| Deterministic seeds & manifests | Seed `lma-phase-1-2026` | Seed `lma-nepali-phase-1-2026` |

---

## Phase 2 Results

Both independent decoder-only Transformers completed approximately one corpus pass using the same
384-dimensional, nine-layer, six-head backbone and a 256-token context window.

| Result | Hindi | Nepali |
|---|---:|---:|
| Parameters | 25,285,248 | 24,870,912 |
| Training steps | 20,300 | 16,720 |
| Tokens processed | 332,595,200 | 273,940,480 |
| Validation loss / perplexity | 4.0954 / 60.06 | 4.1663 / 64.48 |
| Test loss / perplexity | 3.6782 / 39.57 | 3.8888 / 48.85 |
| Test bits per UTF-8 byte | 0.5282 | 0.4377 |

The complete architecture, training protocol, generation metrics, attention analysis, and artifact
inventory are documented in [`report/phase-2.md`](report/phase-2.md).

---

## 2. Dataset & Split Summary

All token statistics reflect SentencePiece Unigram encoding with the selected production tokenizers (`hi_unigram_24000` for Hindi; `nepali_unigram_16000` for Nepali):

### Split Breakdown

| Language | Split | Documents | Characters | Words | Tokens | Manual Tokens | Downloaded Tokens | Manual Share |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Hindi** | Train | 2,237,951 | 1,327,105,658 | 256,929,141 | 332,246,742 | 83,274,136 | 248,972,606 | **25.06%** |
| Hindi | Validation | 23,033 | 13,678,052 | 2,652,001 | 3,428,944 | 859,881 | 2,569,063 | 25.08% |
| Hindi | Test | 22,773 | 13,446,801 | 2,608,981 | 3,372,824 | 839,451 | 2,533,373 | 24.89% |
| **Hindi Total** | | **2,283,757** | **1,354,230,511** | **262,190,123** | **339,048,510** | **84,973,468** | **254,075,042** | **25.06%** |
| **Nepali** | Train | 704,658 | 1,315,750,818 | 202,155,376 | 273,165,831 | 62,106,565 | 211,059,266 | **22.74%** |
| Nepali | Validation | 7,377 | 14,143,440 | 2,176,567 | 2,943,222 | 652,869 | 2,290,353 | 22.18% |
| Nepali | Test | 7,333 | 13,868,241 | 2,129,750 | 2,885,028 | 726,768 | 2,158,260 | 25.19% |
| **Nepali Total** | | **719,368** | **1,343,762,499** | **206,461,693** | **278,994,081** | **63,486,202** | **215,507,879** | **22.76%** |

### Clean Training Tokens by Source

| Language | Source Name | Collection Type | Documents | Clean Training Tokens | Licence |
|---|---|---|---:|---:|---|
| Hindi | PIB Hindi | Manual | 1,489 | 446,674 | Reproduction with attribution |
| Hindi | Hindi Wikipedia | Manual | 149,420 | 65,666,897 | CC BY-SA 4.0 |
| Hindi | Hindi Wikisource | Manual | 47,816 | 12,382,392 | CC BY-SA 4.0 |
| Hindi | Hindi Wikibooks | Manual | 4,212 | 4,151,331 | CC BY-SA 4.0 |
| Hindi | IndicCorp v2 | Downloaded | 2,035,014 | 248,972,606 | CC0-1.0 |
| Nepali | Nepali Wikipedia | Manual | 22,091 | 10,835,528 | CC BY-SA 4.0 |
| Nepali | Onlinekhabar | Manual | 135,504 | 51,271,037 | Source rights retained; research |
| Nepali | Sangraha Verified Web | Downloaded | 149,249 | 61,294,242 | CC BY 4.0 |
| Nepali | Sangraha Unverified Web | Downloaded | 397,814 | 149,765,024 | CC BY 4.0 |

---

## 3. Tokenizer Construction & Diagnostics

Independent **SentencePiece Unigram** models were trained exclusively on each language's training split (`character_coverage=1.0`, `split_digits=True`). Candidate models were evaluated on the held-out validation split:

| Language | Vocabulary | Fertility (tokens/word) | Chars / Token | UNK Rate | Val Utilization | Singleton Pieces | Rare Pieces ($\le 5$) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Hindi** | 16,000 | 1.3562 | 3.800 | 0.001418% | 76.04% | 0.81% | 2.14% | Candidate |
| **Hindi** | **24,000** | **1.2928** | **3.986** | **0.001487%** | **83.93%** | **0.86%** | **3.02%** | **Selected** |
| Hindi | 32,000 | 1.2630 | 4.080 | 0.001522% | 87.64% | 1.84% | 7.91% | Candidate |
| **Nepali** | **16,000** | **1.3522** | **4.805** | **0.001359%** | **92.89%** | **0.50%** | **1.47%** | **Selected** |
| Nepali | 24,000 | 1.2821 | 5.068 | 0.001433% | 95.17% | 0.49% | 4.16% | Candidate |
| Nepali | 32,000 | 1.2439 | 5.224 | 0.001477% | 95.95% | 1.14% | 11.27% | Candidate |

### Selection Rationale

- **Hindi (Selected: 24,000 Unigram)**: Captures frequent bureaucratic and administrative compound words with an optimal fertility drop (1.356 $\rightarrow$ 1.293) and 83.93% validation utilization. Embedding parameters consume 9.22M weights (36.45% of total budget), leaving 63.55% for attention and feed-forward depth.
- **Nepali (Selected: 16,000 Unigram)**: Achieves 92.89% validation utilization with only 1.47% rare pieces ($\le 5$) and 0.50% singletons. At $d_{\text{model}}=384$, the 16k vocabulary uses 6.14M parameters (24.70% of total budget), allocating **75.30%** of model capacity to transformer depth. Larger vocabularies (32k) suffer extreme tail sparsity (11.27% rare pieces) and starve transformer layers.

---

## 4. Repository Layout

```text
.
├── README.md                              # Main repository overview and documentation
├── pyproject.toml                         # Project metadata and CLI entrypoints
├── src/
│   ├── lma_phase1/                        # Phase 1: collection, cleaning, dedup, tokenizer CLI
│   │   ├── collectors/                    # MediaWiki, WordPress REST, PIB importers
│   │   ├── preprocess.py                  # NFC normalization, wikitext & script heuristics
│   │   ├── deduplicate.py                 # Exact document SHA-256 deduplication
│   │   ├── paragraph_deduplicate.py       # Corpus-wide exact paragraph deduplication
│   │   ├── minhash_deduplicate.py         # Bottom-k MinHash near-duplicate deduplication
│   │   ├── downsample.py                  # Deterministic manual/downloaded balancing
│   │   ├── split.py                       # 98/1/1 deterministic split generator
│   │   ├── stats.py                       # Corpus token statistics and audit verification
│   │   └── tokenizer/                     # SentencePiece training and evaluation routines
│   └── lma_phase2/                        # Phase 2: PyTorch Transformer pretraining & evaluation
├── tests/                                 # Pytest test suite (65 passing tests)
├── hindi/
│   ├── configs/                           # Hindi preprocessing and source TOML configs
│   ├── data/
│   │   ├── figures/                       # Token frequency distribution plots
│   │   └── stats/                         # Split reports, evaluation metrics, source CSVs
│   └── tokenizer/                         # Production hindi.model / hindi.vocab (24k)
├── nepali/
│   ├── configs/                           # Nepali preprocessing and source TOML configs
│   ├── data/
│   │   ├── figures/                       # Token frequency distribution plots
│   │   └── stats/                         # Split reports, evaluation metrics, source CSVs
│   └── tokenizer/                         # Production nepali.model / nepali.vocab (16k)
└── report/
    ├── phase-1.md                         # Detailed Phase 1 technical report
    ├── figures/generated/                 # Generated token distribution charts
    └── tables/generated/                  # Per-source audit and statistics tables
```

---

## 5. Installation & Verification

Requires Python 3.11 or 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Run the unit test suite and linter:

```bash
pytest -v
ruff check src/ tests/
```

All 65 tests verify preprocessing, deduplication, splitting, text metrics, and collectors with zero errors.

---

## 6. End-to-End Pipeline Reproduction

### Step 1: Manual Collection

```bash
# Hindi Wikipedia
lma-collect-mediawiki   --api-url https://hi.wikipedia.org/w/api.php   --language hi --source-name hindi_wikipedia   --license CC-BY-SA-4.0 --output hindi/data/raw/manual/hindi_wikipedia.jsonl.gz   --checkpoint hindi/data/checkpoints/hindi_wikipedia.json

# Nepali Wikipedia
lma-collect-mediawiki   --api-url https://ne.wikipedia.org/w/api.php   --language ne --source-name nepali_wikipedia   --license CC-BY-SA-4.0 --output nepali/data/raw/manual/nepali_wikipedia.jsonl.gz   --checkpoint nepali/data/checkpoints/nepali_wikipedia.json

# Nepali Onlinekhabar (WordPress REST API)
python -m lma_phase1.collectors.wordpress   --base-url https://www.onlinekhabar.com/wp-json/wp/v2/posts   --language ne --source-name onlinekhabar   --output nepali/data/raw/manual/onlinekhabar.jsonl.gz   --checkpoint nepali/data/checkpoints/onlinekhabar.json
```

### Step 2: Downloaded Collection

```bash
# Hindi: IndicCorp v2 shards
lma-collect-hf   --dataset ai4bharat/IndicCorpV2   --data-file data/hi-1.txt --data-file data/hi-2.txt --data-file data/hi-3.txt   --split train --language hi --source-name indiccorp_v2_hindi   --license CC0-1.0 --output hindi/data/raw/downloaded/indiccorp_v2_hindi.jsonl.gz

# Nepali: Sangraha Verified Web
lma-collect-hf   --dataset ai4bharat/sangraha --subset verified --split nep   --language ne --source-name sangraha_verified_nep   --license CC-BY-4.0 --output nepali/data/raw/downloaded/sangraha_verified_nep.jsonl.gz

# Nepali: Sangraha Unverified Web
lma-collect-hf   --dataset ai4bharat/sangraha --subset unverified --split nep   --language ne --source-name sangraha_unverified_nep   --license CC-BY-4.0 --output nepali/data/raw/downloaded/sangraha_unverified_nep.jsonl.gz
```

### Step 3: Cleaning & Quality Filtering

```bash
# Hindi Cleaning
lma-preprocess   --input hindi/data/raw/manual/hindi_wikipedia.jsonl.gz   --input hindi/data/raw/downloaded/indiccorp_v2_hindi.jsonl.gz   --language hi --config hindi/configs/preprocessing.toml   --output hindi/data/interim/cleaned.jsonl.gz   --rejected hindi/data/interim/rejected.jsonl.gz   --report hindi/data/stats/preprocessing.json

# Nepali Cleaning
lma-preprocess   --input nepali/data/raw/manual/nepali_wikipedia.jsonl.gz   --input nepali/data/raw/manual/onlinekhabar.jsonl.gz   --input nepali/data/raw/downloaded/sangraha_verified_nep.jsonl.gz   --input nepali/data/raw/downloaded/sangraha_unverified_nep.jsonl.gz   --language ne --config nepali/configs/preprocessing.toml   --output nepali/data/interim/nepali_cleaned.jsonl.gz   --rejected nepali/data/interim/nepali_rejected.jsonl.gz   --report nepali/data/stats/nepali_preprocessing.json
```

### Step 4: Hierarchical Deduplication & Downsampling

```bash
# Document Deduplication (prioritizing manual text)
lma-deduplicate   --input nepali/data/interim/nepali_cleaned.jsonl.gz   --output nepali/data/interim/nepali_document_deduplicated.jsonl.gz   --database nepali/data/checkpoints/nepali_document_dedup.sqlite   --report nepali/data/stats/nepali_document_dedup_report.json

# Corpus-Wide Paragraph Deduplication (>= 100 chars, >= 15 words)
lma-deduplicate-paragraphs   --input nepali/data/interim/nepali_document_deduplicated.jsonl.gz   --output nepali/data/interim/nepali_paragraph_deduplicated.jsonl.gz   --database nepali/data/checkpoints/nepali_paragraph_dedup.sqlite   --report nepali/data/stats/nepali_paragraph_dedup_report.json

# Deterministic Downsampling (22% manual word buffer)
lma-downsample   --input nepali/data/interim/nepali_paragraph_deduplicated.jsonl.gz   --output nepali/data/interim/nepali_balanced22_v2.jsonl.gz   --target-manual-word-fraction 0.22   --seed lma-nepali-downsample-2026   --report nepali/data/stats/downsampling_22pct_v2.json
```

### Step 5: Deterministic 98/1/1 Splitting

```bash
lma-split   --input nepali/data/interim/nepali_balanced22_v2.jsonl.gz   --output-dir nepali/data/splits   --train-ratio 0.98 --validation-ratio 0.01 --test-ratio 0.01   --seed lma-nepali-phase-1-2026   --report nepali/data/stats/nepali_splits_report.json
```

### Step 6: Tokenizer Training & Evaluation

```bash
# Train Unigram Candidates (16k, 24k, 32k)
lma-train-tokenizers   --input nepali/data/splits/train.jsonl.gz   --language ne --output-dir nepali/tokenizer   --vocabulary-sizes 16000 24000 32000 --character-coverage 1.0   --report nepali/data/stats/nepali_tokenizer_training_report.json

# Evaluate on Held-Out Validation Split
lma-evaluate-tokenizer   --model nepali/tokenizer/nepali_unigram_16000.model   --input nepali/data/splits/validation.jsonl.gz   --language Nepali --output nepali/data/stats/nepali_eval_16000.json   --figure report/figures/generated/nepali_token_frequency.png
```

### Step 7: Final Corpus Statistics

```bash
lma-corpus-stats   --input train=nepali/data/splits/train.jsonl.gz   --input validation=nepali/data/splits/validation.jsonl.gz   --input test=nepali/data/splits/test.jsonl.gz   --tokenizer-model nepali/tokenizer/nepali_unigram_16000.model   --output nepali/data/stats/nepali_final_corpus_stats.json   --source-csv nepali/data/stats/nepali_sources_16k.csv
```

---

## 7. Artifact Storage & Google Drive Reference

Large corpus splits, token streams, and model checkpoints are preserved in Google Drive; code, configurations, tokenizer models, audit CSVs, and reports are tracked in Git:

- **Google Drive Root**: [LMA Individual Project Drive Folder](https://drive.google.com/drive/folders/1zjJ5bSM69g431qnSRHXnXOwO_7rJOpce?usp=share_link)
  - `hindi/splits/` — Finalized Hindi train, validation, and test splits
  - `hindi/tokenizers/` — Hindi SentencePiece Unigram candidate models and vocabs (16k, 24k, 32k)
  - `hindi/processed/` — Deduplicated and balanced intermediate checkpoints
  - `nepali/splits/` — Finalized Nepali train, validation, and test splits
  - `nepali/tokenizers/` — Nepali SentencePiece Unigram candidate models and vocabs (16k, 24k, 32k)
  - `nepali/interim/` — Deduplicated and balanced intermediate checkpoints
  - `manifests/` — Deduplication, audit, and tokenizer evaluation reports
