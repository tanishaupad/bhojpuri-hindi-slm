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
| Bhojpuri | Maina Bhojpuri | Manual | Blogger JSON feed | Source rights retained; course research | TBD |
| Bhojpuri | Bhojpuri News | Manual | Blogger JSON feed | Source rights retained; course research | TBD |
| Bhojpuri | Anjoria (pre-2023 pilot) | Manual | Sitemap-enumerated public HTML | Source rights retained; course research | TBD |
| Bhojpuri | Filtered BhojpuriCorpus | Downloaded | HF streaming | Mixed per record | TBD |

### 2.2 Collection process

All collectors write the common document schema and preserve source URL, source document ID,
collection method, collection type, licence, collection date, and source-specific metadata. The
MediaWiki collector is rate-limited and resumable from a continuation checkpoint. Hugging Face
datasets are streamed directly into compressed JSONL rather than materialized in the Git working
tree. Four deterministic audit worksheets were reviewed manually: 20 PIB releases, 100 IndicCorp
documents, 100 Bhojpuri Wikisource pages, and 20 examples from each of the 13 BhojpuriCorpus
partitions (260 rows).

Two additional public Blogger archives were extracted through their structured feeds. Maina
contributed 1,504 raw posts after excluding eight posts explicitly labelled `अनुवाद`; Bhojpuri
News contributed 3,066 raw posts. Embedded social-media blocks and raw URLs were removed during
extraction. The common quality pass accepted 4,566 documents and 1,183,589 whitespace words (Maina:
527,045; Bhojpuri News: 656,544). A new 200-row worksheet, stratified as 100 documents per source,
awaits manual review for Hindi-only or translation-derived text.

Magazine PDF pilots were not added directly. Visually correct Sirijan pages produced scrambled
Devanagari combining-mark order through ordinary PDF text extraction, while an older Sirijan issue
and the tested Diya Bati issue were image-based. Hindi OCR produced readable but imperfect text, so
PDF-scale ingestion is deferred unless OCR output receives its own quality audit.

Anjoria exposes 12,920 sitemap-listed public article URLs last modified before 2023. Its
`robots.txt` disallows the WordPress API, so only the allowed sitemap and public article pages are
used. An initial sample shows substantial native Bhojpuri alongside some Hindi promotional copy;
the source therefore receives stronger Hindi-versus-Bhojpuri filtering before inclusion. A
deterministic 6,000-page target was partitioned into four resumable crawl states. Collection was
paused after repeated `429` responses, leaving a frozen snapshot of 3,663 retained raw articles and
1,712,641 whitespace words. The sitemap URL list was cached after one polite enumeration, and the
unfinished checkpoints were preserved in case further manual data becomes necessary.

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

For Anjoria, a balanced character four-gram log-likelihood-ratio model is trained from independently
held-out Bhojpuri and Hindi seed corpora. Automatic acceptance requires a score above the stricter
of the Bhojpuri holdout fifth percentile and Hindi holdout 99.5th percentile; a lower audit boundary
uses the corresponding first and 95th percentiles. On the initial 38-page source sample, this rule
accepted 32 pages, rejected four visibly Hindi promotional articles, and routed two uncertain pages
to manual review. Applied to the frozen snapshot, the rule accepted 3,089 articles, routed 230 to
the ambiguous audit set, and rejected 344 as likely Hindi. The common quality pass subsequently
retained 2,706 accepted articles and 1,306,145 whitespace words. Audit worksheets contain 100
accepted pages, all 230 ambiguous pages, and 100 rejected pages.

The current PIB partial preprocessing pass accepted 370 releases and 333,161 whitespace words. It
removed 1,100 standard metadata/separator/signature lines and 204 English-dominant embedded
paragraphs. The accepted output contains 99.13% Devanagari and 0.87% Latin letters; eight unusual
releases were quarantined by the script/digit filters for inspection. The metadata rule includes
the standard `DD MON YYYY HH:MMPM by PIB Delhi` entry-date/byline line; a regenerated 100-document
audit contains no remaining instances.

The BhojpuriCorpus partitions `eng_bho`, `google_smol`, and `fleurs_bhojpuri` are excluded by
provenance because they originate in parallel or translation benchmark material. Fluency alone was
not treated as evidence of native authorship. In the current pre-deduplication pass, 30,461 records
were removed by this rule. Bhojpuri Wikipedia cleaning removed 599 English-dominant paragraphs,
truncated 6,721 trailing reference/external-link sections, and identified 82 pages with no usable
Bhojpuri content. Conservative low-resource length floors of 30 characters and five words retained
305,012 documents and 19,201,154 whitespace words after adding the manual Wikipedia crawl; final
SentencePiece token counts will replace these diagnostic word counts. A regenerated 100-document
Wikipedia audit contained no sample above 20% Latin letters. A deterministic 20-document sample of
automatically rejected records was exported for manual false-rejection review.

The first Wikisource crawl supplied only 33,694 retained words, so it cannot meet the 20%
manual-token requirement alone. A complete Bhojpuri Wikipedia crawl supplied 9,149 raw articles, of
which 8,611 passed the v4 preprocessing pass and contributed 1,204,108 words. A manual-first
exact-deduplication measurement removed 1,308 documents and retained 303,704 documents / 19,156,459
words. The retained manual portion is 1,237,718 words (6.461%); the downloaded portion is 17,918,741
words. The two newly cleaned Blogger archives supplied 1,183,589 words before cross-source
deduplication. The updated exact-only, manual-first v3 pass examined 309,578 documents, removed
1,312 exact duplicates, and retained 308,266 documents / 20,339,284 words. After adding the cleaned
Anjoria snapshot, the v4 exact pass examined 312,284 documents and retained 310,972. Its 16,144
manual documents contain 3,726,959 words, while 294,828 downloaded documents contain 17,918,470
words, giving a 17.218% manual word proxy. A stable hash sample then removed only downloaded
records: 80,285 documents / 4,704,707 words. The resulting 230,687-document corpus contains
16,940,722 words and has a 22.000% manual word proxy. The final 20% decision uses tokenizer tokens
rather than whitespace words.

### 2.4 Deduplication

Exact duplicate comparison uses a SHA-256 hash of NFC-normalized, case-folded, whitespace-collapsed
text. Manual records are indexed before downloaded records, so an identical downloaded copy is
removed. The updated v4 exact-only measurement removed 1,312 of 312,284 examined documents.
Near-duplicate
SimHash deletion has not yet been applied: its threshold will first be validated on manually reviewed
duplicate/non-duplicate pairs to avoid false removal of short low-resource text.

### 2.5 Dataset statistics

| Language | Split | Documents | Tokens | Manual tokens | Downloaded tokens | Manual fraction |
|---|---|---:|---:|---:|---:|---:|
| Hindi | Train | TBD | TBD | TBD | TBD | TBD |
| Hindi | Validation | TBD | TBD | TBD | TBD | TBD |
| Hindi | Test | TBD | TBD | TBD | TBD | TBD |
| Bhojpuri | Train | 225,992 | 22,577,763 | 5,078,086 | 17,499,677 | 22.492% |
| Bhojpuri | Validation | 2,344 | 227,239 | 50,949 | 176,290 | 22.421% |
| Bhojpuri | Test | 2,351 | 214,399 | 52,302 | 162,097 | 24.395% |

TBD: justify any Bhojpuri shortfall from the approximately 500M-token target.

## 3. Tokenizer construction

Separate SentencePiece tokenizers were trained from scratch using only the corresponding language's
training split. Candidate vocabulary sizes were evaluated on held-out validation text.

| Language | Vocabulary | Fertility | Characters/token | UNK rate | Rare-type rate |
|---|---:|---:|---:|---:|---:|
| Hindi | 16,000 | TBD | TBD | TBD | TBD |
| Hindi | 24,000 | TBD | TBD | TBD | TBD |
| Hindi | 32,000 | TBD | TBD | TBD | TBD |
| Bhojpuri | 16,000 | 1.3341 | 3.7644 | 0.001760% | 59.250% |
| Bhojpuri | 24,000 | 1.2809 | 3.9206 | 0.001833% | 69.752% |
| Bhojpuri | 32,000 | 1.2546 | 4.0028 | 0.001872% | 74.681% |

### 3.1 Selected tokenizers

- Hindi: TBD
- Bhojpuri: 16,000-piece Unigram candidate. It has the highest held-out vocabulary utilization
  (78.79%) and the lowest rare-type rate while retaining low fertility and a negligible unknown
  rate. The 24K and 32K candidates reduce fertility by only 3.99% and 5.96% relative to 16K, while
  using only 66.40% and 56.29% of their vocabularies. Their additional embedding rows would also
  consume a material share of the approximately 25M-parameter model budget.

Tokenizer input collapses Unicode whitespace before encoding, matching the one-paragraph-per-line
training representation. Without this shared preprocessing, literal held-out newlines appeared as
unknown spans under SentencePiece's identity character normalizer. After applying the consistent
input rule, the 16K candidate emitted four unknown tokens among 227,239 validation tokens.

### 3.2 Tokenization examples

TBD: include representative normal, rare-word, numeric, and code-mixed examples for both languages.

## 4. Reproducibility

TBD: record environment, exact commands, seeds, source snapshot dates, Drive links, and checksums.

## 5. Limitations

TBD: discuss Bhojpuri spelling variation, Hindi/Awadhi contamination, source/domain imbalance,
licensing constraints, OCR errors, and limitations of automatic near-duplicate detection.
