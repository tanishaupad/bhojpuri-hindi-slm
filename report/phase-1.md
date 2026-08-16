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
| Hindi | Hindi Wikipedia | Manual | MediaWiki Action API | CC BY-SA 4.0 | TBD |
| Hindi | Hindi Wikisource | Manual | MediaWiki Action API | CC BY-SA 4.0 | TBD |
| Hindi | Hindi Wikibooks | Manual | MediaWiki Action API | CC BY-SA 4.0 | TBD |
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
527,045; Bhojpuri News: 656,544).

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
the ambiguous audit set, and rejected 344 as likely Hindi. A compact stratified audit then reviewed
60 documents: 30 were kept and 30 removed. The removals comprised 22 Hindi, three English, two
bilingual/mixed, two Romanized Bhojpuri, and one Romanized mixed-language document. Importantly,
four automatic accepts were rejected, so grammatical fluency was not treated as evidence of
Bhojpuri. The final inclusion policy retains native Bhojpuri written in Devanagari and excludes
Hindi, English, mixed-language, translated, and Roman-script documents.

Source-specific cleanup also found a long repeated Anjoria site tail containing comments, recent
posts, and category navigation. It truncated this tail in 3,026 documents and removed 265,778
boilerplate lines / 3,103,775 characters before applying the common quality thresholds. The v2 pass
retained 2,698 of 3,089 accepted-filter documents, containing 756,712 whitespace words; 391 were
quarantined, chiefly for excessive Latin script or insufficient Devanagari.

The current PIB partial preprocessing pass accepted 370 releases and 333,161 whitespace words. It
removed 1,100 standard metadata/separator/signature lines and 204 English-dominant embedded
paragraphs. The accepted output contains 99.13% Devanagari and 0.87% Latin letters; eight unusual
releases were quarantined by the script/digit filters for inspection. The metadata rule includes
the standard `DD MON YYYY HH:MMPM by PIB Delhi` entry-date/byline line; a regenerated 100-document
audit contains no remaining instances.

The Hindi manual collection also includes complete namespace-0 snapshots from Hindi Wikipedia,
Hindi Wikisource, and Hindi Wikibooks. Explicit translation credits are rejected for Wikisource
and Wikibooks. The frozen Wikipedia v6 cleanup accepted 91,543 articles after removing reference
tails, English-dominant blocks, templates, HTML, broken wiki-link markup, and table attributes; an
automated residual-markup scan found zero instances of those patterns. The cleaned manual sources
contributed 58,497,224 whitespace words after exact document deduplication. IndicCorp was sampled
as three independently streamed two-million-record raw shards; the common quality pass accepted
3,139,385 documents / 293,226,648 words before exact deduplication and balancing.

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
which 8,611 passed the v4 preprocessing pass and contributed 1,204,108 words. The two cleaned
Blogger archives and the cleaned Anjoria snapshot substantially increase the manual share. The
current v5 exact pass examined 312,276 documents, removed 1,312 exact duplicates, and retained
310,964 documents: 16,136 manual and 294,828 downloaded.

After all deduplication stages, 299,674 documents and 18,685,024 whitespace words remained. The
manual portion contained 16,135 documents / 3,147,135 words; the downloaded portion contained
283,539 documents / 15,537,889 words. A stable hash sample then removed only downloaded records,
retaining all manual material. The resulting pre-split corpus contains 221,200 documents and
14,305,159 words: 3,147,135 manual and 11,158,024 downloaded, giving a 22.000% manual word proxy.
The final assignment check will use tokens from the newly trained tokenizer rather than whitespace
words.

### 2.4 Deduplication

Exact duplicate comparison uses a SHA-256 hash of NFC-normalized, case-folded,
whitespace-collapsed text. Manual records are indexed before downloaded records, so an identical
downloaded copy is removed. The v5 exact-only pass removed 1,312 of 312,276 examined documents.

Corpus-wide exact paragraph deduplication then considered only lines with at least 100 characters
and 15 words, protecting short headings and poetic refrains. It removed 7,488 repeated long
paragraphs (442,355 words) and 6,826 documents that became empty or too short; one additional exact
document duplicate appeared after paragraph removal. A deterministic 40-pair audit included clear
reuse across manual sites and downloaded BhojpuriCorpus partitions.

Finally, bottom-k MinHash compared word five-gram sketches. A 50,000-document pilot showed that
thresholds from 0.75 through 0.90 merged distinct but similarly templated pages, whereas a reviewed
sample at 0.95 consisted of reposts or copies with minor spelling, metadata, or formatting changes.
The strict 0.95 application removed 4,463 near duplicates and protected 191,400 short documents
with fewer than 20 five-grams. The resulting pair audit is retained with the corpus manifests.

For Hindi, the completed exact-only pass examined 3,237,873 quality-filtered documents, removed
3,028 exact duplicates, and retained 3,234,845 documents: 98,330 manual and 3,136,515 downloaded.
A deterministic preliminary balance then retained all 58,497,224 manual words and selected
233,988,884 downloaded words, giving a 20.0000008% manual word proxy across 2,602,213 documents.
This is the current safe checkpoint, not the final corpus. Paragraph deduplication, a reviewed
strict MinHash pass, final 22% balancing, splitting, and tokenizer-based token accounting remain.

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
| Bhojpuri | 16,000 | 1.3341 | 3.7644 | 0.001760% | 59.250% |
| Bhojpuri | 24,000 | 1.2809 | 3.9206 | 0.001833% | 69.752% |
| Bhojpuri | 32,000 | 1.2546 | 4.0028 | 0.001872% | 74.681% |

### 3.1 Selected tokenizers

- Hindi: TBD
- Bhojpuri: provisional 16,000-piece Unigram candidate from the earlier corpus version. It had the highest held-out vocabulary utilization
  (78.79%) and the lowest rare-type rate while retaining low fertility and a negligible unknown
  rate. The 24K and 32K candidates reduce fertility by only 3.99% and 5.96% relative to 16K, while
  using only 66.40% and 56.29% of their vocabularies. Their additional embedding rows would also
  consume a material share of the approximately 25M-parameter model budget.

The candidates will be retrained and re-evaluated after splitting the finalized v1 corpus; the
figures above are retained only as pilot evidence and must not be reported as final. Tokenizer input
collapses Unicode whitespace before encoding, matching the one-paragraph-per-line
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
