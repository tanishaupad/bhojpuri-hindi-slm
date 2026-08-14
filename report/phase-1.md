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
| Hindi | PIB Hindi | Manual | Polite release crawl | Reproduction with attribution | TBD |
| Hindi | IndicCorp v2 (`data/hi-{1,2,3}.txt`) | Downloaded | HF streaming | CC0 | TBD |
| Bhojpuri | Bhojpuri Wikisource | Manual | MediaWiki Action API | CC BY-SA 4.0 | TBD |
| Bhojpuri | Bhojpuri Wikipedia | Manual | MediaWiki Action API | CC BY-SA 4.0 | TBD |
| Bhojpuri | Filtered BhojpuriCorpus | Downloaded | HF streaming | Mixed per record | TBD |

### 2.2 Collection process

All collectors write the common document schema and preserve source URL, source document ID,
collection method, collection type, licence, collection date, and source-specific metadata. The
MediaWiki collector is rate-limited and resumable from a continuation checkpoint. Hugging Face
datasets are streamed directly into compressed JSONL rather than materialized in the Git working
tree. Four deterministic audit worksheets were reviewed manually: 20 PIB releases, 100 IndicCorp
documents, 100 Bhojpuri Wikisource pages, and 20 examples from each of the 13 BhojpuriCorpus
partitions (260 rows).

PIB rejected ordinary scripted requests in the collection environment, so direct article pages were
opened through a normal browser session and then imported into the common schema. The importer
validates the page body rather than trusting navigation success: in the August 2026 attempt, 635 of
1,013 PRIDs returned an Akamai access-denied body and were quarantined. The clean partial raw set
contains 378 genuine articles; their failed PRIDs are retained in the import report for a slower,
rate-limited retry.

### 2.3 Cleaning and quality filtering

Text is normalized to Unicode NFC, control and zero-width characters are removed, whitespace is
normalized, and repeated within-document lines are removed. Documents are then filtered using
minimum length, Devanagari-letter share, Latin-letter share, and digit-share thresholds. The PIB
pilot was accepted subject to paragraph-level removal of standard PIB metadata/footer text and
embedded English social-media blocks.

The current PIB partial preprocessing pass accepted 370 releases and 333,161 whitespace words. It
removed 1,100 standard metadata/separator/signature lines and 204 English-dominant embedded
paragraphs. The accepted output contains 99.13% Devanagari and 0.87% Latin letters; eight unusual
releases were quarantined by the script/digit filters for inspection.

The BhojpuriCorpus partitions `eng_bho`, `google_smol`, and `fleurs_bhojpuri` are excluded by
provenance because they originate in parallel or translation benchmark material. Fluency alone was
not treated as evidence of native authorship. In the current pre-deduplication pass, 30,461 records
were removed by this rule. Conservative low-resource length floors of 30 characters and five words
retained 304,634 documents and 19,174,263 whitespace words after adding the manual Wikipedia crawl;
final SentencePiece token counts will replace these diagnostic word counts. A deterministic
20-document sample of automatically rejected records was exported for manual false-rejection
review.

The first Wikisource crawl supplied only 33,694 retained words, so it cannot meet the 20%
manual-token requirement alone. A complete Bhojpuri Wikipedia crawl supplied 9,149 raw articles, of
which 8,233 passed preprocessing and contributed 1,177,217 words. A manual-first exact-deduplication
measurement removed 824 documents and retained 303,810 documents / 19,134,337 words. The retained
manual portion is 1,210,851 words (6.328%); the downloaded portion is 17,923,486 words. Using words
as a planning proxy, another 3.27 million manual words are needed to keep the whole downloaded
portion. With no additional manual sources, downloaded text would need to be capped near 4.84
million words, giving a roughly 6.05-million-word corpus before near deduplication. The final 20%
decision will use tokenizer tokens rather than whitespace words.

### 2.4 Deduplication

Exact duplicate comparison uses a SHA-256 hash of NFC-normalized, case-folded, whitespace-collapsed
text. Manual records are indexed before downloaded records, so an identical downloaded copy is
removed. The exact-only measurement removed 824 of 304,634 examined documents. Near-duplicate
SimHash deletion has not yet been applied: its threshold will first be validated on manually reviewed
duplicate/non-duplicate pairs to avoid false removal of short low-resource text.

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
