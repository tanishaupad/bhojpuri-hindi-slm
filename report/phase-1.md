# Phase 1: Data Collection and Tokenizer Construction

## 1. Language selection

- **Language #1 (Model H):** Hindi (`hi`, Devanagari script), selected as the higher-resource Indian language due to abundant availability of public digital text, governmental archives, and encyclopedic resources.
- **Language #2 (Model L):** Nepali (`ne`, Devanagari script), selected as Language #2, targeting ~300M tokens with $\ge 20\%$ manually collected data (achieving 273.17M train tokens at 22.74% manual tokens, and 278.99M total tokens across all splits).

The corpora, tokenizers, vocabulary artifacts, training splits and model weights are independent. Token strings can naturally coincide because both languages use Devanagari; there is no shared embedding table or tokenizer.

---

## 2. Corpus collection

### 2.1 Source inventory

| Language | Source | Collection type | Method | Licence | Clean training tokens |
|---|---|---|---|---|---:|
| Hindi | PIB Hindi | Manual | Polite release crawl / DOM extraction | Reproduction with attribution | 446,674 |
| Hindi | Hindi Wikipedia | Manual | MediaWiki Action API (v6 cleaned) | CC BY-SA 4.0 | 65,666,897 |
| Hindi | Hindi Wikisource | Manual | MediaWiki Action API | CC BY-SA 4.0 | 12,382,392 |
| Hindi | Hindi Wikibooks | Manual | MediaWiki Action API | CC BY-SA 4.0 | 4,151,331 |
| Hindi | IndicCorp v2 (`data/hi-{1,2,3}.txt`) | Downloaded | Hugging Face streaming | CC0-1.0 | 248,972,606 |
| **Hindi Total** | | | | | **332,246,742** |
| Nepali | Nepali Wikipedia | Manual | MediaWiki Action API (v2 cleaned) | CC BY-SA 4.0 | 10,835,528 |
| Nepali | Onlinekhabar | Manual | WordPress REST API | Source rights retained; course research | 51,271,037 |
| Nepali | Sangraha Verified Web | Downloaded | Hugging Face streaming (`ai4bharat/sangraha`) | CC BY 4.0 | 61,294,242 |
| Nepali | Sangraha Unverified Web | Downloaded | Hugging Face streaming (`ai4bharat/sangraha`) | CC BY 4.0 | 149,765,024 |
| **Nepali Total** | | | | | **273,165,831** |

---

### 2.2 Collection process

All raw data collectors write to a unified, provenance-tracking document schema (`src/lma_phase1/schema.py`), recording `document_id`, `text`, `source_name`, `source_url`, `collection_type` (`manual` vs. `downloaded`), `collection_date`, `license`, and structured `metadata`.

- **Hindi Manual Sources**: Namespace-0 page dumps from Hindi Wikipedia, Wikisource, and Wikibooks were gathered using rate-limited MediaWiki Action API collectors with checkpoint continuation. Press Information Bureau (PIB) Hindi articles were collected and imported with body verification, quarantining Akamai access-denied responses.
- **Hindi Downloaded Source**: IndicCorp v2 Hindi was streamed from three raw text shards (`data/hi-1.txt`, `data/hi-2.txt`, `data/hi-3.txt`) directly into compressed JSONL.
- **Nepali Manual Sources**: Namespace-0 articles from Nepali Wikipedia were collected using the MediaWiki Action API with resume checkpoints (22,568 retained articles after strict template, table, category, inline media `\d+px`, thumbnail/caption markers `thumb|` / `अङ्गुठाकार|`, and trailing reference/external-link section cleaning). Onlinekhabar news articles (138,358 clean articles) were collected via categorized REST API endpoints with provenance tracking and boilerplate removal.
- **Nepali Downloaded Sources**: Sangraha (AI4Bharat) Nepali web extractions were streamed from both verified (`sangraha_verified_nep:web`, 149,249 train documents) and unverified (`sangraha_unverified_nep`, 397,814 train documents) subsets, filtered against machine-translation markers and non-Devanagari boilerplate.

---

### 2.3 Cleaning and quality filtering

Preprocessing enforces a strict multi-stage filtering cascade:
1. **Unicode NFC Normalization**: Devanagari text is normalized to NFC, converting irregular combining character sequences into canonical representations, removing zero-width non-joiners (`\u200c`) and zero-width joiners (`\u200d`) that produce invisible encoding fragmentation, and converting HTML entities.
2. **Boilerplate & Line Repetition**: Lines $\ge 20$ characters appearing 3+ times in the same document are stripped.
3. **Source-Specific Cleaning**:
   - *Wikipedia / Wikibooks*: Strips leading and residual `{{template}}` blocks, HTML tags/comments, table markup (`{|`, `|-`, `|}`), category headers (`श्रेणी:`), media captions (`अङ्गुठाकार|`, `thumb|`), and truncates trailing reference/external-link sections (`सन्दर्भ सामग्रीहरू`, `बाह्य कडीहरू`, `सन्दर्भहरू`).
   - *PIB*: Removes entry-date/byline lines (`प्रविष्टि तिथि:`), release IDs (`(रिलीज़ आईडी:`), signature blocks, and embedded English paragraphs.
   - *Onlinekhabar*: Strips author bylines, timestamps, related-article link clusters, and site navigation footers.
4. **Script & Quality Heuristics**:
   - Hindi and Nepali thresholds: $\ge 200$ characters, $\ge 30$ words, $\ge 70\%$ Devanagari letter ratio, $\le 20\%$ Latin letter ratio, $\le 25\%$ digit ratio.

---

### 2.4 Deduplication

Deduplication was conducted in three hierarchical phases, preserving manual document priority:
1. **Exact Document Deduplication**: SHA-256 hashing of NFC-normalized, whitespace-collapsed, case-folded text. Manual documents were indexed first, dropping identical downloaded copies. (Removed 3,028 duplicates in Hindi; 17,813 exact duplicates and 1,293 near duplicates removed in Nepali across 987,583 examined documents, retaining 968,477 documents: 161,265 manual, 807,212 downloaded).
2. **Corpus-Wide Exact Paragraph Deduplication**: Paragraphs with $\ge 100$ characters and $\ge 15$ words were indexed in SQLite. If an identical paragraph appeared in an earlier document, it was removed. If paragraph removal reduced the document below 200 characters / 30 words, the document was dropped. (Removed 29,625 duplicate paragraphs and 16,565 empty/short documents in Hindi; 135,681 duplicate paragraphs and 4,961 below-floor documents removed in Nepali, retaining 963,515 documents: 160,926 manual, 802,589 downloaded). An audit of sampled pairs confirmed valid removal of shared boilerplates and cross-source repeats with zero false positives.
3. **Near-Duplicate Detection & Deterministic Downsampling**: The document stage used 64-bit SimHash over word 5-grams with a Hamming-distance threshold of 3 to eliminate near-duplicates. For Nepali, downloaded records were then deterministically downsampled using seed `lma-nepali-downsample-2026` to cap downloaded words at 161,040,310, retaining 719,368 documents (160,926 manual, 558,442 downloaded) and achieving an exact manual word fraction of 22.00000% (22.76% manual tokens overall; 22.74% in train).
4. **Cross-Lingual Overlap & Leakage Audit**: An automated cross-lingual audit (`scripts/audit_cross_lingual_overlap.py`, recorded in `nepali_cross_lingual_overlap.json`) compared all 719,368 Nepali documents against the complete set of 2,283,757 indexed Hindi documents and 230,687 indexed Bhojpuri documents using exact stable text hashing. Zero exact matches (0 Hindi, 0 Bhojpuri) were found, demonstrating complete cross-lingual isolation and zero train/eval leakage across regional language collections.

---

### 2.5 Historical and Final Phase 1 dataset statistics

After deterministic downsampling of downloaded records to achieve a $\ge 22\%$ manual-word buffer, corpora were partitioned using a **98% / 1% / 1%** document-level split with cryptographic hashing on document and work group IDs.

| Language | Split | Documents | Tokens | Manual tokens | Downloaded tokens | Manual token fraction |
|---|---|---:|---:|---:|---:|---:|
| **Hindi** | Train | 2,237,951 | 332,246,742 | 83,274,136 | 248,972,606 | **25.06%** |
| Hindi | Validation | 23,033 | 3,428,944 | 859,881 | 2,569,063 | 25.08% |
| Hindi | Test | 22,773 | 3,372,824 | 839,451 | 2,533,373 | 24.89% |
| **Hindi Overall** | | **2,283,757** | **339,048,510** | **84,973,468** | **254,075,042** | **25.06%** |
| **Nepali** | Train | 704,658 | 273,165,831 | 62,106,565 | 211,059,266 | **22.74%** |
| Nepali | Validation | 7,377 | 2,943,222 | 652,869 | 2,290,353 | 22.18% |
| Nepali | Test | 7,333 | 2,885,028 | 726,768 | 2,158,260 | 25.19% |
| **Nepali Overall** | | **719,368** | **278,994,081** | **63,486,202** | **215,507,879** | **22.76%** |

> [!NOTE]
> **Token-Count Reconciliation between Phase 1 and Phase 2**:
> The token counts in the table above reflect whitespace-normalized SentencePiece encoding (`normalize_tokenizer_input()`), which was used during Phase 1 corpus analysis. In Phase 2, token streams were packed directly from raw text before appending one `<eos>` token per document to form contiguous training and evaluation streams (see [`heldout_recomputation.json`](audits/generated/heldout_recomputation.json)).
> - **Hindi Test**: Phase 2 packed test stream contains **3,521,023 tokens** = 3,498,250 raw tokens + 22,773 document EOS tokens. Re-encoding the identical decontaminated test split with Phase 1 whitespace normalization yields **3,468,316 tokens** (a 29,934 token difference due to raw vs. normalized whitespace, plus a 95,492 token reconciliation from the preliminary Phase 1 summary count).

#### Justification for Hindi Dataset Size Shortfall
Hindi's historical training summary was **332.25M tokens**, below the approximately 500M training-token target. The final packed stream contains **336.995M tokens**, including EOS. Filtering, the collected manual-text budget and compute constraints limited the completed experiment; no grading exception is claimed.

#### Justification for Nepali Dataset Size Shortfall
Nepali's final training corpus contains **273.17M tokens** across 704,658 documents (and **278.99M total tokens** across all splits), falling short of the unconstrained 500M theoretical target. This shortfall is the result of a deliberate quality, authenticity, and compliance trade-off:
1. **Preservation of the $\ge 20\%$ Manual Requirement**: Under the course constraints, downloaded text must not exceed $4\times$ manual text to maintain $\ge 20\%$ manual representation. With 62.11M clean manual training tokens gathered from Onlinekhabar and Nepali Wikipedia, maintaining a rigorous $\ge 22\%$ manual-word buffer required capping downloaded web records at 161.04M words (211.06M train tokens).
2. **Exclusion of Noisy Web Scrapes & Machine-Translated Data**: While additional uncurated web dumps exist in raw crawls, automated audits and pilot evaluations revealed pervasive English/Latin boilerplates, machine-translated fragments, and corrupted OCR text. These were systematically purged through strict script ratio filtering ($\ge 70\%$ Devanagari), exact document deduplication (17,813 duplicates dropped), and corpus-wide paragraph deduplication (135,681 duplicate paragraphs removed).
3. **Model Representation Depth**: At 273.17M tokens, the dataset provides extensive coverage for our compact Model L architecture, comfortably satisfying Chinchilla-scaling regimes without synthetic noise.

---

## 3. Tokenizer construction

Separate **SentencePiece Unigram** tokenizers were trained from scratch exclusively on each language's training split (`character_coverage=1.0`, `split_digits=True`, `normalization_rule=identity`). Candidate vocabulary sizes were evaluated on held-out validation splits:

| Language | Vocabulary size | Fertility (tokens/word) | Characters / token | UNK rate | Held-out vocab utilization |
|---|---:|---:|---:|---:|---:|
| **Hindi** | 16,000 | 1.3562 | 3.800 | 0.001418% | 76.04% |
| **Hindi** | **24,000** | **1.2928** | **3.986** | **0.001487%** | **83.93%** |
| Hindi | 32,000 | 1.2630 | 4.080 | 0.001522% | 87.64% |
| **Nepali** | **16,000** | **1.3522** | **4.805** | **0.001359%** | **92.89%** |
| Nepali | 24,000 | 1.2821 | 5.068 | 0.001433% | 95.17% |
| Nepali | 32,000 | 1.2439 | 5.224 | 0.001477% | 95.95% |

![Hindi Token Frequency Distribution](figures/generated/hindi_token_frequency.png)
![Nepali Token Frequency Distribution](figures/generated/nepali_token_frequency.png)

---

### 3.1 Selected tokenizers

- **Hindi (Selected: 24,000 Unigram)**:
  - **Fertility & Subwords**: Fertility drops significantly from 1.3562 (16k) to 1.2928 (24k), capturing key compound morphemes with 83.93% held-out vocabulary utilization.
  - **Parameter Budget Allocation**: In our final Phase 2 model architecture ($d_{\text{model}} = 384$, 25,285,248 total parameters; initially projected at $d_{\text{model}} = 512$ in early planning), the 24,000-piece embedding table uses $24,000 \times 384 = 9,216,000$ parameters (36.45% of the total parameter budget), leaving the remaining 16,069,248 parameters (63.55%) for deep multi-head attention and feed-forward layers. A 32k vocabulary would have consumed 12.29M parameters (~49% of the budget) in embeddings alone.
- **Nepali (Selected: 16,000 Unigram)**:
  - **Optimal Subword Balance & Vocabulary Utilization**: The 16,000 Unigram model achieves 92.89% vocabulary utilization on held-out validation text with low fertility (1.3522 tokens/word), 4.805 characters/token, and an unknown rate of 0.001359%. While 24k and 32k marginally decrease fertility to 1.2821 and 1.2439, they suffer sharp increases in rare and singleton subwords: types occurring $\le 5$ times jump from 1.47% (16k) to 4.16% (24k) and 11.27% (32k), with singleton piece rates rising to 1.14% (32k).
  - **Parameter Allocation for Model L**: In our target 12-layer transformer architecture ($d_{\text{model}} = 384$, ~25M total parameters), the 16,000-piece embedding table uses $16,000 \times 384 = 6,144,000$ parameters (24.70% of the total budget), reserving 75.30% of parameters for multi-head attention and feed-forward layers. A 24k vocabulary would consume 36.45% and 32k would consume 48.60% in embeddings alone, severely starving representational capacity.
  - **Effective Token Throughput**: At 16k vocabulary, the training split yields 273.17M tokens (22.74% manual tokens), effectively modeling Nepali agglutinative nominal case markers (*vibhakti*) without excessive subword fragmentation.

---

### 3.2 Tokenization examples

#### Example 1: Standard Hindi Administrative Prose & Inflections
- **Original (Hindi)**: `शिक्षा मंत्रालय ने छात्रवृत्ति योजना के लिए आवेदन आमंत्रित किए हैं।`
- **Tokens (24k)**: `[' शिक्षा', ' मंत्रालय', ' ने', ' छात्रवृत्ति', ' योजना', ' के', ' लिए', ' आवेदन', ' आमंत्रित', ' किए', ' हैं', '।']`
- **Analysis**: Full word-level segmentation for frequent administrative and grammatical particles (`मंत्रालय`, `छात्रवृत्ति`, `के लिए`).

#### Example 2: Hindi Compound Words & Rare Stems
- **Original (Hindi)**: `दूरसंचार अवसंरचना विकास प्राधिकरण`
- **Tokens (24k)**: `[' दूर', 'संचार', ' अव', 'संरचना', ' विकास', ' प्राधिकरण']`
- **Analysis**: Clean prefix and root decomposition (`दूर + संचार`, `अव + संरचना`).

#### Example 3: Hindi Numeric Handling
- **Original**: `15 अगस्त 1947`
- **Tokens (24k)**: `[' 1', '5', ' अगस्त', ' 1', '9', '4', '7']`
- **Analysis**: `split_digits=True` isolates individual digits, preventing combinatorial vocabulary explosion for calendar years.

#### Example 4: Nepali Administrative Prose & Postpositions (Vibhakti)
- **Original (Nepali)**: `शिक्षा मन्त्रालयले छात्रवृत्तिका लागि आवेदन आह्वान गरेको छ।`
- **Tokens (16k)**: `[' शिक्षा', ' मन्त्रालयले', ' छात्रवृत्ति', 'का', ' लागि', ' आवेदन', ' आह्वान', ' गरेको', ' छ', '।']`
- **Analysis**: Whole-word preservation of administrative nouns (`शिक्षा`, `आवेदन`, `आह्वान`) and nominal-agentive postposition attachment (`मन्त्रालयले`), while cleanly isolating the genitive clitic `का` from `छात्रवृत्ति`.

#### Example 5: Nepali Verbal Morphology & Pronoun Inflections
- **Original (Nepali)**: `उनीहरूले घरमा गएर सबै कुरा सम्झाउनुपर्छ।`
- **Tokens (16k)**: `[' उनीहरूले', ' घरमा', ' गएर', ' सबै', ' कुरा', ' सम्झ', 'ाउन', 'ुपर्छ', '।']`
- **Analysis**: Retains third-person plural agentive pronoun (`उनीहरूले`), locative noun (`घरमा`), converb (`गएर`), and regular morphological segmentation of obligation verbs (`सम्झ + ाउन + ुपर्छ`).

#### Example 6: Nepali Compound Proper Nouns & Dates
- **Original (Nepali)**: `सञ्चार तथा सूचना प्रविधि मन्त्रालय 2026 मार्च 13`
- **Tokens (16k)**: `[' सञ्चार', ' तथा', ' सूचना', ' प्रविधि', ' मन्त्रालय', ' 2', '0', '2', '6', ' मार्च', ' 1', '3']`
- **Analysis**: Clean single-token units for all lexical stems in the compound title, while `split_digits=True` decomposes multi-digit years into single-digit units.

---

## 4. Reproducibility

- **Python Version**: Python 3.12 (Apple Silicon / macOS)
- **Key Dependencies**: `sentencepiece>=0.2`, `datasets>=3.0`, `beautifulsoup4>=4.12`, `matplotlib>=3.8`
- **Deterministic Seeds**:
  - Preprocessing & Dedup Hash: `lma-phase-1-2026`
  - Downsampling: `lma-downsample-2026` (Hindi), `lma-nepali-downsample-2026` (Nepali)
  - Split Generator: `lma-phase-1-2026` (Hindi), `lma-nepali-phase-1-2026` (Nepali) (98/1/1 split ratio)
  - SentencePiece training: `character_coverage=1.0`, `split_digits=True`.

### Pipeline Commands

#### Hindi Pipeline

```bash
# 1. Deduplication (Document SimHash & Exact Paragraph)
python -m lma_phase1.paragraph_deduplicate \
  --input LMA_Processing/input/balanced_preparagraph_20pct_v1.jsonl.gz \
  --output LMA_Processing/output/hindi_paragraph_dedup_v1.jsonl.gz \
  --database LMA_Processing/output/hindi_paragraph_dedup_v1.sqlite \
  --report LMA_Processing/output/hindi_paragraph_report.json \
  --minimum-paragraph-characters 100 --minimum-paragraph-words 15

python -m lma_phase1.minhash_deduplicate \
  --input LMA_Processing/output/hindi_paragraph_dedup_v1.jsonl.gz \
  --output LMA_Processing/output/hindi_minhash_v1.jsonl.gz \
  --database LMA_Processing/output/hindi_minhash_v1.sqlite \
  --report LMA_Processing/output/hindi_minhash_report.json \
  --mode apply --similarity-threshold 0.95 --index-collection-type manual

# 2. Downsampling to 22% manual word buffer
python -m lma_phase1.downsample \
  --input LMA_Processing/output/hindi_minhash_v1.jsonl.gz \
  --output LMA_Processing/output/hindi_balanced_22pct_v1.jsonl.gz \
  --report LMA_Processing/output/hindi_downsample_report.json \
  --target-manual-word-fraction 0.22 --seed lma-downsample-2026

# 3. Splitting (98/1/1)
python -m lma_phase1.split \
  --input LMA_Processing/output/hindi_balanced_22pct_v1.jsonl.gz \
  --output-dir LMA_Processing/output/hindi_splits_v1 \
  --report LMA_Processing/output/hindi_split_report.json \
  --train-ratio 0.98 --validation-ratio 0.01 --test-ratio 0.01

# 4. Tokenizer Training & Evaluation
python -m lma_phase1.tokenizer.train \
  --input LMA_Processing/output/hindi_splits_v1/train.jsonl.gz \
  --language hi --output-dir LMA_Processing/output/hindi_tokenizers \
  --report LMA_Processing/output/hindi_tokenizer_training_report.json \
  --vocabulary-sizes 16000 24000 32000 --character-coverage 1.0

python -m lma_phase1.tokenizer.evaluate \
  --model LMA_Processing/output/hindi_tokenizers/hi_unigram_24000.model \
  --input LMA_Processing/output/hindi_splits_v1/validation.jsonl.gz \
  --language Hindi --output LMA_Processing/output/hindi_eval.json \
  --figure report/figures/generated/hindi_token_frequency.png

# 5. Exact Final Corpus Statistics
python -m lma_phase1.stats \
  --input train=LMA_Processing/output/hindi_splits_v1/train.jsonl.gz \
  --input validation=LMA_Processing/output/hindi_splits_v1/validation.jsonl.gz \
  --input test=LMA_Processing/output/hindi_splits_v1/test.jsonl.gz \
  --tokenizer-model LMA_Processing/output/hindi_tokenizers/hi_unigram_24000.model \
  --output LMA_Processing/output/hindi_final_corpus_stats.json \
  --source-csv report/tables/generated/hindi_sources.csv
```

#### Nepali Pipeline

```bash
# 1. Deduplication (Document & Paragraph)
python -m lma_phase1.deduplicate \
  --input nepali/data/interim/nepali_cleaned.jsonl.gz \
  --output nepali/data/interim/nepali_document_deduplicated.jsonl.gz \
  --database nepali/data/checkpoints/nepali_document_dedup.sqlite \
  --report nepali/data/stats/nepali_document_dedup_report.json

python -m lma_phase1.paragraph_deduplicate \
  --input nepali/data/interim/nepali_document_deduplicated.jsonl.gz \
  --output nepali/data/interim/nepali_paragraph_deduplicated.jsonl.gz \
  --database nepali/data/checkpoints/nepali_paragraph_dedup.sqlite \
  --report nepali/data/stats/nepali_paragraph_dedup_report.json \
  --minimum-paragraph-characters 100 --minimum-paragraph-words 15

# 2. Downsampling to 22% manual word buffer
python -m lma_phase1.downsample \
  --input nepali/data/interim/nepali_paragraph_deduplicated.jsonl.gz \
  --output nepali/data/interim/nepali_balanced22_v2.jsonl.gz \
  --report nepali/data/stats/downsampling_22pct_v2.json \
  --target-manual-word-fraction 0.22 --seed lma-nepali-downsample-2026

# 3. Splitting (98/1/1)
python -m lma_phase1.split \
  --input nepali/data/interim/nepali_balanced22_v2.jsonl.gz \
  --output-dir nepali/data/splits \
  --report nepali/data/stats/nepali_splits_report.json \
  --train-ratio 0.98 --validation-ratio 0.01 --test-ratio 0.01 \
  --seed lma-nepali-phase-1-2026

# 4. Tokenizer Training & Evaluation
python -m lma_phase1.tokenizer.train \
  --input nepali/data/splits/train.jsonl.gz \
  --language ne --output-dir nepali/tokenizer \
  --report nepali/data/stats/nepali_tokenizer_training_report.json \
  --vocabulary-sizes 16000 24000 32000 --character-coverage 1.0

python -m lma_phase1.tokenizer.evaluate \
  --model nepali/tokenizer/nepali_unigram_16000.model \
  --input nepali/data/splits/validation.jsonl.gz \
  --language Nepali --output nepali/data/stats/nepali_eval_16000.json \
  --figure report/figures/generated/nepali_token_frequency.png

# 5. Exact Final Corpus Statistics
python -m lma_phase1.stats \
  --input train=nepali/data/splits/train.jsonl.gz \
  --input validation=nepali/data/splits/validation.jsonl.gz \
  --input test=nepali/data/splits/test.jsonl.gz \
  --tokenizer-model nepali/tokenizer/nepali_unigram_16000.model \
  --output nepali/data/stats/nepali_final_corpus_stats.json \
  --source-csv nepali/data/stats/nepali_sources_16k.csv
```

### Artifact Storage & Google Drive Reference

All large corpus splits (`train.jsonl.gz`, `validation.jsonl.gz`, `test.jsonl.gz`), trained tokenizer models (`16k`, `24k`, `32k`), deduplicated checkpoints, and audit manifests are stored in the project's Google Drive:

- **Google Drive Project Root**: [LMA Individual Project Drive Folder](https://drive.google.com/drive/folders/1zjJ5bSM69g431qnSRHXnXOwO_7rJOpce?usp=share_link)
  - `hindi/splits/` — Finalized Hindi train, validation, and test splits
  - `hindi/tokenizers/` — Hindi SentencePiece Unigram candidate models and vocabs
  - `hindi/processed/` — Deduplicated and balanced intermediate checkpoints
  - `nepali/splits/` — Finalized Nepali train, validation, and test splits
  - `nepali/tokenizers/` — Nepali SentencePiece Unigram candidate models and vocabs (16k, 24k, 32k)
  - `nepali/interim/` — Deduplicated and balanced intermediate checkpoints
  - `manifests/` — Deduplication, audit, and tokenizer evaluation reports

---

## 5. Limitations and discussion

1. **Translationese & Synthetic Artifact Avoidance**: A substantial portion of open-source multilingual Indic data originates from translated parallel benchmarks (e.g., FLORES, MultiUN, translated Common Crawl). These were systematically excluded by provenance filtering to prevent models from learning unnatural syntactic patterns ("translationese").
2. **Standardization & Curation in High-Resource Hindi**: While Hindi digital resources are abundant, extensive manual filtering and deduplication were required to remove government portal boilerplate, repeated headers, and uninformative media summaries.
3. **Nepali Scale, Agglutinative Morphology, and Provenance Balance**: Scaling the Nepali corpus as Language #2 to 273.17M training tokens (278.99M total tokens across all splits) required extensive collection of native manual sources (138,358 clean Onlinekhabar articles and 22,568 Wikipedia articles), securing 62.11M manual training tokens (22.74% of the training split, satisfying the $\ge 20\%$ requirement with a verified buffer). The selected 16,000 Unigram tokenizer achieves 1.3522 fertility, 4.805 characters/token, and 92.89% validation utilization, effectively balancing subword granularity for Nepali agglutinative case markers (*vibhakti*) while allocating 75.3% of the model parameter budget to transformer layers rather than sparse embedding weights. The shortfall against 500M tokens was necessary to protect the 20% manual threshold without admitting uncurated or synthetic web-crawl noise.
