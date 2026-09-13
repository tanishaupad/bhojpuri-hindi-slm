# Phase 1: Data Collection and Tokenizer Construction

## 1. Language selection

- **Language #1 (Model H):** Hindi (`hi`, Devanagari script), selected as the higher-resource Indian language due to abundant availability of public digital text, governmental archives, and encyclopedic resources.
- **Language #2 (Model L):** Nepali (`ne`, Devanagari script), selected as Language #2, targeting ~300M tokens with $\ge 20\%$ manually collected data (achieving 275.35M train tokens at 22.78% manual tokens, and 281.22M total tokens across all splits). In addition, Bhojpuri (`bho`, Devanagari script) is preserved as an approved low-resource comparison language and corpus.

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
| Bhojpuri | Bhojpuri Wikisource | Manual | MediaWiki Action API | CC BY-SA 4.0 | 53,205 |
| Bhojpuri | Bhojpuri Wikipedia | Manual | MediaWiki Action API (v4 cleaned) | CC BY-SA 4.0 | 1,689,774 |
| Bhojpuri | Maina Bhojpuri | Manual | Blogger structured JSON feed | Source rights retained; course research | 768,163 |
| Bhojpuri | Bhojpuri News | Manual | Blogger structured JSON feed | Source rights retained; course research | 871,582 |
| Bhojpuri | Anjoria (pre-2023 pilot) | Manual | Sitemap-enumerated public HTML + LLR | Source rights retained; course research | 1,695,362 |
| Bhojpuri | Filtered BhojpuriCorpus | Downloaded | Hugging Face streaming | Mixed per record | 17,499,677 |
| **Bhojpuri Total** | | | | | **22,577,763** |
| Nepali | Nepali Wikipedia | Manual | MediaWiki Action API (v2 cleaned) | CC BY-SA 4.0 | 11,455,382 |
| Nepali | Onlinekhabar | Manual | WordPress REST API | Source rights retained; course research | 51,280,008 |
| Nepali | Sangraha Verified Web | Downloaded | Hugging Face streaming (`ai4bharat/sangraha`) | CC BY 4.0 | 61,743,709 |
| Nepali | Sangraha Unverified Web | Downloaded | Hugging Face streaming (`ai4bharat/sangraha`) | CC BY 4.0 | 150,866,779 |
| **Nepali Total** | | | | | **275,345,878** |

---

### 2.2 Collection process

All raw data collectors write to a unified, provenance-tracking document schema (`src/lma_phase1/schema.py`), recording `document_id`, `text`, `source_name`, `source_url`, `collection_type` (`manual` vs. `downloaded`), `collection_date`, `license`, and structured `metadata`.

- **Hindi Manual Sources**: Namespace-0 page dumps from Hindi Wikipedia, Wikisource, and Wikibooks were gathered using rate-limited MediaWiki Action API collectors with checkpoint continuation. Press Information Bureau (PIB) Hindi articles were collected and imported with body verification, quarantining Akamai access-denied responses.
- **Hindi Downloaded Source**: IndicCorp v2 Hindi was streamed from three raw text shards (`data/hi-1.txt`, `data/hi-2.txt`, `data/hi-3.txt`) directly into compressed JSONL.
- **Bhojpuri Manual Sources**: In addition to Bhojpuri Wikipedia and Wikisource, two dedicated Bhojpuri literary blog archives (Maina and Bhojpuri News) were collected through Blogger feeds, filtering out posts with translation labels (`अनुवाद`). Anjoria articles modified before 2023 were extracted via public XML sitemaps.
- **Bhojpuri Downloaded Source**: BhojpuriCorpus was streamed with strict partition filtering to discard translation/parallel benchmark splits (`eng_bho`, `google_smol`, `fleurs_bhojpuri`).
- **Nepali Manual Sources**: Namespace-0 articles from Nepali Wikipedia were collected using the MediaWiki Action API with resume checkpoints (24,220 retained articles after strict template, table, category, and wikitext cleaning). Onlinekhabar news articles (138,358 clean articles) were collected via categorized REST API endpoints with provenance tracking and boilerplate removal.
- **Nepali Downloaded Sources**: Sangraha (AI4Bharat) Nepali web extractions were streamed from both verified (`sangraha_verified_nep:web`, 150,305 train documents) and unverified (`sangraha_unverified_nep`, 400,686 train documents) subsets, filtered against machine-translation markers and non-Devanagari boilerplate.

---

### 2.3 Cleaning and quality filtering

Preprocessing enforces a strict multi-stage filtering cascade:
1. **Unicode NFC Normalization**: Devanagari text is normalized to NFC, converting irregular combining character sequences into canonical representations, removing zero-width non-joiners (`\u200c`) and zero-width joiners (`\u200d`) that produce invisible encoding fragmentation, and converting HTML entities.
2. **Boilerplate & Line Repetition**: Lines $\ge 20$ characters appearing 3+ times in the same document are stripped.
3. **Source-Specific Cleaning**:
   - *Wikipedia / Wikibooks*: Strips leading and residual `{{template}}` blocks, HTML tags/comments, table markup (`{|`, `|-`, `|}`), category headers (`श्रेणी:`), media captions (`अङ्गुठाकार|`, `thumb|`), and truncates trailing reference/external-link sections (`सन्दर्भ सामग्रीहरू`, `बाह्य कडीहरू`, `सन्दर्भहरू`).
   - *PIB*: Removes entry-date/byline lines (`प्रविष्टि तिथि:`), release IDs (`(रिलीज़ आईडी:`), signature blocks, and embedded English paragraphs.
   - *Onlinekhabar*: Strips author bylines, timestamps, related-article link clusters, and site navigation footers.
   - *Anjoria*: Truncates social-media comments, category navigation lists, and footer boilerplate.
4. **Script & Quality Heuristics**:
   - Hindi and Nepali thresholds: $\ge 200$ characters, $\ge 30$ words, $\ge 70\%$ Devanagari letter ratio, $\le 20\%$ Latin letter ratio, $\le 25\%$ digit ratio.
   - Bhojpuri thresholds (lenient low-resource floor): $\ge 30$ characters, $\ge 5$ words, $\ge 70\%$ Devanagari ratio, $\le 20\%$ Latin ratio, $\le 25\%$ digit ratio.
5. **Statistical Language Identification**: For mixed sources like Anjoria, a character 4-gram log-likelihood-ratio (LLR) classifier trained on held-out Hindi and Bhojpuri seed corpora separated authentic Bhojpuri from Hindi promotional text.

---

### 2.4 Deduplication

Deduplication was conducted in three hierarchical phases, preserving manual document priority:
1. **Exact Document Deduplication**: SHA-256 hashing of NFC-normalized, whitespace-collapsed, case-folded text. Manual documents were indexed first, dropping identical downloaded copies. (Removed 3,028 duplicates in Hindi; 1,312 in Bhojpuri; 17,810 exact duplicates and 1,328 near duplicates removed in Nepali across 989,284 examined documents, retaining 970,146 documents).
2. **Corpus-Wide Exact Paragraph Deduplication**: Paragraphs with $\ge 100$ characters and $\ge 15$ words were indexed in SQLite. If an identical paragraph appeared in an earlier document, it was removed. If paragraph removal reduced the document below 200 characters / 30 words, the document was dropped. (Removed 29,625 duplicate paragraphs and 16,565 empty/short documents in Hindi; 7,488 paragraphs in Bhojpuri; 135,684 duplicate paragraphs and 4,981 below-floor documents removed in Nepali, retaining 965,165 documents). An audit of sampled pairs confirmed valid removal of shared boilerplates and cross-source repeats with zero false positives.
3. **Near-Duplicate Detection & Deterministic Downsampling**: Bottom-k MinHash over word 5-grams was applied to detect near-duplicates. For Nepali, downloaded records were deterministically downsampled using seed `lma-nepali-downsample-2026` to cap downloaded words at 162,208,048, retaining 725,037 documents (162,578 manual, 562,459 downloaded) and achieving an exact manual word fraction of 22.00% (22.80% manual tokens overall; 22.78% in train).

---

### Final training recount (11 September 2026)

Every final decontaminated training document was re-encoded with the released tokenizer.
The reconstructed packed stream matches the published Phase 2 SHA-256 for both languages.
Raw content counts exclude EOS; the packed count includes one EOS per document.

| Language | Final train documents | Raw content tokens | Manual content tokens | Manual fraction | Packed tokens (with EOS) |
|---|---:|---:|---:|---:|---:|
| Hindi | 2,237,942 | 334,757,275 | 85,857,319 | 25.6476% | 336,995,217 |
| Bhojpuri | 225,689 | 22,917,286 | 5,453,089 | 23.7947% | 23,142,975 |

Both final training corpora exceed 20% manual content tokens. Assigning each EOS to its source
document gives packed manual fractions of 25.5058% Hindi and 23.6302% Bhojpuri. Independently
normalizing whitespace gives 332,150,099 Hindi / 22,371,425 Bhojpuri training tokens, with
manual fractions 25.0640% / 21.9352%. These are different denominators, not competing estimates.
The recount verifies recorded collection types and token counts; it is not a fresh manual
review of the linguistic quality or collection classification of every source.

Evidence: [final_training_recount.json](audits/generated/final_training_recount.json).
The historical pre-decontamination tables below are retained as historical records.

### 2.5 Historical Phase 1 dataset statistics

After deterministic downsampling of downloaded records to achieve a $\ge 22\%$ manual-word buffer, corpora were partitioned using a **98% / 1% / 1%** document-level split with cryptographic hashing on document and work group IDs. The following is the historical pre-decontamination summary, not a fresh count of the final packed training streams. Hindi's held-out normalized counts were later corrected to 3,429,056 validation tokens and 3,468,316 test tokens; the historical overall totals therefore should not be reused as current totals.

| Language | Split | Documents | Tokens | Manual tokens | Downloaded tokens | Manual token fraction |
|---|---|---:|---:|---:|---:|---:|
| **Hindi** | Train | 2,237,951 | 332,246,742 | 83,274,136 | 248,972,606 | **25.06%** |
| Hindi | Validation | 23,033 | 3,428,944 | 859,881 | 2,569,063 | 25.08% |
| Hindi | Test | 22,773 | 3,372,824 | 839,451 | 2,533,373 | 24.89% |
| **Hindi Overall** | | **2,283,757** | **339,048,510** | **84,973,468** | **254,075,042** | **25.06%** |
| **Bhojpuri** | Train | 225,992 | 22,577,763 | 5,078,086 | 17,499,677 | **22.49%** |
| Bhojpuri | Validation | 2,344 | 227,239 | 50,949 | 176,290 | 22.42% |
| Bhojpuri | Test | 2,351 | 214,399 | 52,302 | 162,097 | 24.39% |
| **Bhojpuri Overall** | | **230,687** | **23,019,401** | **5,181,337** | **17,838,064** | **22.51%** |
| **Nepali** | Train | 710,200 | 275,345,878 | 62,735,390 | 212,610,488 | **22.78%** |
| Nepali | Validation | 7,434 | 2,962,133 | 659,799 | 2,302,334 | 22.27% |
| Nepali | Test | 7,403 | 2,909,630 | 733,328 | 2,176,302 | 25.20% |
| **Nepali Overall** | | **725,037** | **281,217,641** | **64,128,517** | **217,089,124** | **22.80%** |

> [!NOTE]
> **Token-Count Reconciliation between Phase 1 and Phase 2**:
> The token counts in the table above reflect whitespace-normalized SentencePiece encoding (`normalize_tokenizer_input()`), which was used during Phase 1 corpus analysis. In Phase 2, token streams were packed directly from raw text before appending one `<eos>` token per document to form contiguous training and evaluation streams (see [`heldout_recomputation.json`](audits/generated/heldout_recomputation.json)).
> - **Hindi Test**: Phase 2 packed test stream contains **3,521,023 tokens** = 3,498,250 raw tokens + 22,773 document EOS tokens. Re-encoding the identical decontaminated test split with Phase 1 whitespace normalization yields **3,468,316 tokens** (a 29,934 token difference due to raw vs. normalized whitespace, plus a 95,492 token reconciliation from the preliminary Phase 1 summary count).
> - **Bhojpuri Test**: Phase 2 packed test stream contains **222,585 tokens** = 220,234 raw tokens + 2,351 document EOS tokens. Re-encoding with whitespace normalization yields **214,399 tokens**, matching Phase 1's reported count exactly.

#### Justification for Bhojpuri Dataset Size Shortfall
Hindi's historical training summary was **332.25M tokens**, below the approximately 500M training-token target. The final packed stream contains **336.995M tokens**, including EOS. Filtering, the collected manual-text budget and compute constraints limited the completed experiment; the explicit lower-resource shortfall allowance does not automatically exempt Hindi, and no grading exception is claimed. Bhojpuri's historical training summary was **22.58M tokens**, with the following justification for its permitted lower-resource shortfall:
1. **Scarcity of Native Monolingual Text**: High proportions of online "Bhojpuri" text are machine-translated benchmark subsets (which were deliberately excluded to preserve linguistic authenticity) or Hindi text mistakenly tagged as Bhojpuri.
2. **Exclusion of Non-Devanagari / OCR-Scrambled Data**: Scanned PDF magazines (e.g., Sirijan, Diya Bati) produced corrupted Devanagari combining mark orders and were omitted to protect corpus quality.
3. **Manual Ratio Compliance**: Maintaining a valid 22.49% manual token fraction required capping downloaded records proportionally to the ~5.18M manual tokens collected.

#### Justification for Nepali Dataset Size Shortfall
Nepali's final training corpus contains **275.35M tokens** across 710,200 documents (and **281.22M total tokens** across all splits), falling short of the unconstrained 500M theoretical target. This shortfall is the result of a deliberate quality, authenticity, and compliance trade-off:
1. **Preservation of the $\ge 20\%$ Manual Requirement**: Under the course constraints, downloaded text must not exceed $4\times$ manual text to maintain $\ge 20\%$ manual representation. With 62.74M clean manual training tokens gathered from Onlinekhabar and Nepali Wikipedia, maintaining a rigorous $\ge 22\%$ manual-word buffer required capping downloaded web records at 162.2M words (212.61M train tokens).
2. **Exclusion of Noisy Web Scrapes & Machine-Translated Data**: While additional uncurated web dumps exist in raw crawls, automated audits and pilot evaluations revealed pervasive English/Latin boilerplates, machine-translated fragments, and corrupted OCR text. These were systematically purged through strict script ratio filtering ($\ge 70\%$ Devanagari), exact document deduplication (17,810 duplicates dropped), and corpus-wide paragraph deduplication (135,684 duplicate paragraphs removed).
3. **Model Representation Depth**: At 275.35M tokens, the dataset provides extensive coverage for our compact Model L architecture, comfortably satisfying Chinchilla-scaling regimes without synthetic noise.

---

## 3. Tokenizer construction

Separate **SentencePiece Unigram** tokenizers were trained from scratch exclusively on each language's training split (`character_coverage=1.0`, `split_digits=True`, `normalization_rule=identity`). Candidate vocabulary sizes were evaluated on held-out validation splits:

| Language | Vocabulary size | Fertility (tokens/word) | Characters / token | UNK rate | Held-out vocab utilization |
|---|---:|---:|---:|---:|---:|
| **Hindi** | 16,000 | 1.3562 | 3.800 | 0.001418% | 76.04% |
| **Hindi** | **24,000** | **1.2928** | **3.986** | **0.001487%** | **83.93%** |
| Hindi | 32,000 | 1.2630 | 4.080 | 0.001522% | 87.64% |
| **Bhojpuri** | **16,000** | **1.3341** | **3.764** | **0.001760%** | **78.79%** |
| Bhojpuri | 24,000 | 1.2809 | 3.921 | 0.001833% | 66.40% |
| Bhojpuri | 32,000 | 1.2546 | 4.003 | 0.001872% | 56.29% |
| **Nepali** | **16,000** | **1.3533** | **4.803** | **0.001148%** | **92.79%** |
| Nepali | 24,000 | 1.2828 | 5.067 | 0.001211% | 95.10% |
| Nepali | 32,000 | 1.2448 | 5.222 | 0.001248% | 95.93% |

![Hindi Token Frequency Distribution](figures/generated/hindi_token_frequency.png)
![Bhojpuri Token Frequency Distribution](figures/generated/bhojpuri_token_frequency.png)
![Nepali Token Frequency Distribution](figures/generated/nepali_token_frequency.png)

---

### 3.1 Selected tokenizers

- **Hindi (Selected: 24,000 Unigram)**:
  - **Fertility & Subwords**: Fertility drops significantly from 1.3562 (16k) to 1.2928 (24k), capturing key compound morphemes with 83.93% held-out vocabulary utilization.
  - **Parameter Budget Allocation**: In our final Phase 2 model architecture ($d_{\text{model}} = 384$, 25,285,248 total parameters; initially projected at $d_{\text{model}} = 512$ in early planning), the 24,000-piece embedding table uses $24,000 \times 384 = 9,216,000$ parameters (36.45% of the total parameter budget), leaving the remaining 16,069,248 parameters (63.55%) for deep multi-head attention and feed-forward layers. A 32k vocabulary would have consumed 12.29M parameters (~49% of the budget) in embeddings alone.
- **Bhojpuri (Selected: 16,000 Unigram)**:
  - **High Utilization**: The 16,000 vocabulary achieves 78.79% utilization on held-out text with low fertility (1.3341) and an unknown rate of 0.00176%. Larger vocabularies (24k/32k) exhibit steep drops in utilization (down to 56.29%) and overfit rare morphological variations in the smaller corpus.
  - **Parameter Saving**: With $d_{\text{model}} = 384$, the 16,000-piece vocabulary requires $16,000 \times 384 = 6,144,000$ parameters (24.70% of the 24,870,912 total model parameters), leaving 18,726,912 parameters (75.30%) for Transformer depth.
- **Nepali (Selected: 16,000 Unigram)**:
  - **Optimal Subword Balance & Vocabulary Utilization**: The 16,000 Unigram model achieves 92.79% vocabulary utilization on held-out validation text with low fertility (1.3533 tokens/word), 4.803 characters/token, and an unknown rate of 0.001148%. While 24k and 32k marginally decrease fertility to 1.2828 and 1.2448, they suffer sharp increases in rare and singleton subwords: types occurring $\le 5$ times jump from 1.50% (16k) to 4.14% (24k) and 11.19% (32k), with singleton piece rates rising from 0.51% (16k) to 1.25% (32k).
  - **Parameter Allocation for Model L**: In our target 12-layer transformer architecture ($d_{\text{model}} = 384$, ~25M total parameters), the 16,000-piece embedding table uses $16,000 \times 384 = 6,144,000$ parameters (24.70% of the total budget), reserving 75.30% of parameters for multi-head attention and feed-forward layers. A 24k vocabulary would consume 36.45% and 32k would consume 48.60% in embeddings alone, severely starving representational capacity.
  - **Effective Token Throughput**: At 16k vocabulary, the training split yields 275.35M tokens (22.78% manual tokens), effectively modeling Nepali agglutinative nominal case markers (*vibhakti*) without excessive subword fragmentation.

---

### 3.2 Tokenization examples

#### Example 1: Standard Devanagari Prose & Inflections
- **Original (Hindi)**: `शिक्षा मंत्रालय ने छात्रवृत्ति योजना के लिए आवेदन आमंत्रित किए हैं।`
- **Tokens (24k)**: `[' शिक्षा', ' मंत्रालय', ' ने', ' छात्रवृत्ति', ' योजना', ' के', ' लिए', ' आवेदन', ' आमंत्रित', ' किए', ' हैं', '।']`
- **Analysis**: Full word-level segmentation for frequent administrative and grammatical particles (`मंत्रालय`, `छात्रवृत्ति`, `के लिए`).

#### Example 2: Bhojpuri Verbal Morphology & Postpositions
- **Original (Bhojpuri)**: `ऊ लोगन के घरे जाके सब बात बतावे के चाहीं।`
- **Tokens (16k)**: `[' ऊ', ' लोगन', ' के', ' घरे', ' जाके', ' सब', ' बात', ' बतावे', ' के', ' चाहीं', '।']`
- **Analysis**: Preserves dialect-specific oblique markers and verbal inflections (`लोगन`, `घरे`, `बतावे`, `चाहीं`) as complete subwords.

#### Example 3: Compound Words & Rare Stems
- **Original (Hindi)**: `दूरसंचार अवसंरचना विकास प्राधिकरण`
- **Tokens (24k)**: `[' दूर', 'संचार', ' अव', 'संरचना', ' विकास', ' प्राधिकरण']`
- **Analysis**: Clean prefix and root decomposition (`दूर + संचार`, `अव + संरचना`).

#### Example 4: Numeric Handling
- **Original**: `15 अगस्त 1947`
- **Tokens (24k)**: `[' 1', '5', ' अगस्त', ' 1', '9', '4', '7']`
- **Analysis**: `split_digits=True` isolates individual digits, preventing combinatorial vocabulary explosion for calendar years.

#### Example 5: Nepali Administrative Prose & Postpositions (Vibhakti)
- **Original (Nepali)**: `शिक्षा मन्त्रालयले छात्रवृत्तिका लागि आवेदन आह्वान गरेको छ।`
- **Tokens (16k)**: `[' शिक्षा', ' मन्त्रालयले', ' छात्रवृत्ति', 'का', ' लागि', ' आवेदन', ' आह्वान', ' गरेको', ' छ', '।']`
- **Analysis**: Whole-word preservation of administrative nouns (`शिक्षा`, `आवेदन`, `आह्वान`) and nominal-agentive postposition attachment (`मन्त्रालयले`), while cleanly isolating the genitive clitic `का` from `छात्रवृत्ति`.

#### Example 6: Nepali Verbal Morphology & Pronoun Inflections
- **Original (Nepali)**: `उनीहरूले घरमा गएर सबै कुरा सम्झाउनुपर्छ।`
- **Tokens (16k)**: `[' उनीहरूले', ' घरमा', ' गएर', ' सबै', ' कुरा', ' सम्झ', 'ाउन', 'ुपर्छ', '।']`
- **Analysis**: Retains third-person plural agentive pronoun (`उनीहरूले`), locative noun (`घरमा`), converb (`गएर`), and regular morphological segmentation of obligation verbs (`सम्झ + ाउन + ुपर्छ`).

#### Example 7: Nepali Compound Proper Nouns & Dates
- **Original (Nepali)**: `सञ्चार तथा सूचना प्रविधि मन्त्रालय 2026 मार्च 13`
- **Tokens (16k)**: `[' सञ्चार', ' तथा', ' सूचना', ' प्रविधि', ' मन्त्रालय', ' 2', '0', '2', '6', ' मार्च', ' 1', '3']`
- **Analysis**: Clean single-token units for all lexical stems in the compound title, while `split_digits=True` decomposes multi-digit years into single-digit units.

---

## 4. Reproducibility

- **Python Version**: Python 3.12 (Apple Silicon / macOS)
- **Key Dependencies**: `sentencepiece>=0.2`, `datasets>=3.0`, `beautifulsoup4>=4.12`, `matplotlib>=3.8`
- **Deterministic Seeds**:
  - Preprocessing & Dedup Hash: `lma-phase-1-2026`
  - Downsampling: `lma-downsample-2026`
  - Split Generator: `lma-phase-1-2026` (98/1/1 split ratio)
  - SentencePiece training: `character_coverage=1.0`; no explicit `seed=42` is passed by the training code. Input sampling, threads and library versions may affect retraining; the released tokenizer files identify the tokenizers used in all reported experiments.

### Pipeline Commands

```bash
# 1. Deduplication (Paragraph & MinHash)
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

### Artifact Storage & Google Drive Reference

All large corpus splits (`train.jsonl.gz`, `validation.jsonl.gz`, `test.jsonl.gz`), trained tokenizer models (`16k`, `24k`, `32k`), deduplicated checkpoints, and audit manifests are stored in the project's Google Drive:

- **Google Drive Project Root**: [LMA Individual Project Drive Folder](https://drive.google.com/drive/folders/1zjJ5bSM69g431qnSRHXnXOwO_7rJOpce?usp=share_link)
  - `hindi/splits/` — Finalized Hindi train, validation, and test splits
  - `hindi/tokenizers/` — Hindi SentencePiece Unigram candidate models and vocabs
  - `hindi/processed/` — Deduplicated and balanced intermediate checkpoints
  - `bhojpuri/splits/downsampled22_v1/` — Finalized Bhojpuri splits
  - `bhojpuri/tokenized/` — Bhojpuri candidate models and vocabs
  - `nepali/splits/` — Finalized Nepali train, validation, and test splits
  - `nepali/tokenizers/` — Nepali SentencePiece Unigram candidate models and vocabs (16k, 24k, 32k)
  - `nepali/interim/` — Deduplicated and balanced intermediate checkpoints
  - `manifests/` — Deduplication, audit, and tokenizer evaluation reports

---

## 5. Limitations and discussion

1. **Dialectal & Orthographic Variation in Bhojpuri**: Unlike Hindi, which possesses a standardized grammatical and orthographic convention (e.g., standard Devanagari spellings across Central Hindi Directorate guidelines), Bhojpuri exhibits regional orthographic fluidity (e.g., variation between `बाटे`, `बा`, and `हवे` for existential copulas, and alternate spelling of nasal matras). The Unigram tokenizer mitigates this by decomposing rare variants into base stems and inflectional markers.
2. **Translationese & Synthetic Artifact Avoidance**: A substantial portion of open-source multilingual Indic data originates from translated parallel benchmarks (e.g., FLORES, MultiUN, translated Common Crawl). These was systematically excluded by provenance filtering to prevent models from learning unnatural syntactic patterns ("translationese").
3. **Low-Resource Data Availability vs. Model Scaling**: The ~23M token corpus for Bhojpuri highlights the fundamental data scarcity for non-scheduled Indian languages. To achieve effective pretraining in Phase 2 without overfitting, Model L will be trained with adjusted epoch counts (3–4 passes) and a compact parameter configuration matching empirical recommendations from Lecture 3.
4. **Nepali Scale, Agglutinative Morphology, and Provenance Balance**: Scaling the Nepali corpus as Language #2 to 275.35M training tokens (281.22M total tokens across all splits) required extensive collection of native manual sources (138,358 clean Onlinekhabar articles and 24,220 Wikipedia articles), securing 62.74M manual training tokens (22.78% of the training split, satisfying the $\ge 20\%$ requirement with a verified buffer). The selected 16,000 Unigram tokenizer achieves 1.3533 fertility, 4.803 characters/token, and 92.79% validation utilization, effectively balancing subword granularity for Nepali agglutinative case markers (*vibhakti*) while allocating 75.3% of the model parameter budget to transformer layers rather than sparse embedding weights. The shortfall against 500M tokens was necessary to protect the 20% manual threshold without admitting uncurated or synthetic web-crawl noise.
