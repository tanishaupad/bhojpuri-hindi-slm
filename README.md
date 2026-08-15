[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Hindi and Bhojpuri monolingual language models

This repository builds two fully independent decoder-only language models from scratch:

- **Model H:** Hindi (`hi`, Devanagari), the higher-resource language.
- **Model L:** Bhojpuri (`bho`, Devanagari), the approved lower-resource language.

The current `phase-1` branch contains reproducible corpus collection, preprocessing,
deduplication, split creation, tokenizer training, and tokenizer evaluation code. The two
languages never share documents, tokenizers, vocabularies, checkpoints, or weights.

## Phase 1 requirements tracked here

- Approximately 500M training tokens per language where feasible.
- At least 20% of **each final training corpus** collected manually.
- Separate, document-level train/validation/test splits.
- Separate tokenizers trained from scratch.
- Dataset provenance, cleaning statistics, token counts, and tokenizer diagnostics.

For Bhojpuri, the maximum feasible clean corpus will be reported with an exact tokenizer-based
token count and a documented justification if it is below 500M tokens.

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the unit tests:

```bash
pytest -q
```

## Data policy and layout

Large data is deliberately excluded from Git. Every pipeline record uses the schema in
`src/lma_phase1/schema.py` and retains its source URL, source identifier, collection method,
licence, date, and `manual`/`downloaded` classification.

Generated data is stored locally as:

```text
<language>/data/
  raw/manual/
  raw/downloaded/
  checkpoints/
  interim/
  processed/
  splits/
  stats/
```

Final large artifacts and checksums will be uploaded to Google Drive; links will be added here.

## 1. Manual collection

### MediaWiki pilot

The collector uses page-level MediaWiki API requests, respects continuation tokens, saves a
checkpoint after every batch, and preserves page/revision metadata. A stopped command can be run
again with the same output and checkpoint paths.

Hindi Wikipedia pilot:

```bash
lma-collect-mediawiki \
  --api-url https://hi.wikipedia.org/w/api.php \
  --language hi \
  --source-name hindi_wikipedia \
  --license CC-BY-SA-4.0 \
  --output hindi/data/raw/manual/hindi_wikipedia.jsonl.gz \
  --checkpoint hindi/data/checkpoints/hindi_wikipedia.json \
  --max-pages 100
```

Bhojpuri Wikipedia pilot:

```bash
lma-collect-mediawiki \
  --api-url https://bh.wikipedia.org/w/api.php \
  --language bho \
  --source-name bhojpuri_wikipedia \
  --license CC-BY-SA-4.0 \
  --output bhojpuri/data/raw/manual/bhojpuri_wikipedia.jsonl.gz \
  --checkpoint bhojpuri/data/checkpoints/bhojpuri_wikipedia.json \
  --max-pages 100
```

Inspect and record pilot results in each language's `data/stats/source_audit.csv`. Remove
`--max-pages` to resume and finish an accepted source.

Create a deterministic worksheet for manual language and quality review:

```bash
lma-create-audit-sample \
  --input bhojpuri/data/raw/manual/bhojpuri_wikipedia.jsonl.gz \
  --output bhojpuri/data/stats/bhojpuri_wikipedia_review.csv \
  --size 100
```

Fill in `language_label`, `translation_label`, `quality_label`, reviewer, and notes. Keep the
completed worksheet with the corpus manifests as evidence of manual auditing; do not paste private
or sensitive text into it.

### Approved-domain HTML pilot

Only run this collector after checking the site's terms, redistribution permission, and
`robots.txt`. It enforces same-domain traversal, consults `robots.txt`, rate-limits requests, and
stores its frontier in SQLite.

```bash
lma-collect-web \
  --start-url https://example.org/approved-start-page \
  --language bho \
  --source-name approved_source \
  --license RECORDED-LICENSE \
  --content-selector "article" \
  --output bhojpuri/data/raw/manual/approved_source.jsonl.gz \
  --state-db bhojpuri/data/checkpoints/approved_source.sqlite \
  --max-pages 100 \
  --delay 1.0
```

### PIB browser import

PIB currently returns access-denied responses to the command-line collector in this environment.
After collecting direct article text through a normal browser session, import the append-only browser
JSONL attempts into the canonical schema with:

```bash
lma-import-pib-browser \
  --input pib_browser_attempts.jsonl \
  --input pib_browser_retries.jsonl \
  --output hindi/data/raw/manual/pib_hindi.jsonl.gz \
  --report hindi/data/stats/pib_import.json
```

The importer keeps the last successful attempt for each PRID and explicitly rejects Akamai
`Access Denied` bodies; a successful browser navigation is not by itself considered a successful
article collection.

## 2. Downloaded collection

BhojpuriCorpus, preserving its per-row source, URL, and licence:

```bash
lma-collect-hf \
  --dataset Satyam810/BhojpuriCorpus \
  --split train \
  --language bho \
  --source-name bhojpuri_corpus \
  --license mixed-see-record \
  --text-field text \
  --id-field id \
  --source-field source \
  --license-field license \
  --url-field url \
  --output bhojpuri/data/raw/downloaded/bhojpuri_corpus.jsonl.gz
```

The Bhojpuri preprocessing configuration excludes `eng_bho`, `google_smol`, and
`fleurs_bhojpuri`: all three originate from parallel/translation material even when the retained
Bhojpuri strings are fluent. Source-level exclusions are counted in the preprocessing report.

IndicCorp v2 Hindi will be streamed from its three pinned Hindi text files. Their exact current
Hub paths are recorded in `hindi/configs/sources.toml`; run a small pilot before the full sample
because Hub dataset layouts may change.

```bash
lma-collect-hf \
  --dataset ai4bharat/IndicCorpV2 \
  --data-file data/hi-1.txt \
  --data-file data/hi-2.txt \
  --data-file data/hi-3.txt \
  --split train \
  --language hi \
  --source-name indiccorp_v2_hindi \
  --license CC0-1.0 \
  --output hindi/data/raw/downloaded/indiccorp_v2_hindi.jsonl.gz \
  --max-documents 1000
```

## 3. Preprocessing

Preprocessing applies NFC normalization, removes repeated boilerplate lines, computes script
ratios, rejects configured source partitions and low-quality records. It does not overwrite its
input.

For Bhojpuri Wikipedia, preprocessing drops English-dominant paragraphs and truncates trailing
reference/external-link sections while retaining the Bhojpuri article body. Fully English pages are
rejected by the document-level script thresholds. For PIB, standard entry-date/byline, signature,
release-ID/footer, and English-dominant embedded blocks are removed.

```bash
lma-preprocess \
  --input hindi/data/raw/manual/hindi_wikipedia.jsonl.gz \
  --input hindi/data/raw/downloaded/indiccorp_v2_hindi.jsonl.gz \
  --language hi \
  --config hindi/configs/preprocessing.toml \
  --output hindi/data/interim/cleaned.jsonl.gz \
  --rejected hindi/data/interim/rejected.jsonl.gz \
  --report hindi/data/stats/preprocessing.json
```

Use the analogous Bhojpuri paths and `bhojpuri/configs/preprocessing.toml` for Model L.

## 4. Exact and near deduplication

Manual records are always indexed before downloaded records. Therefore, if a manually scraped
page overlaps a downloaded corpus, the manual copy is retained and the downloaded copy is
removed. Runs are resumable when the output and SQLite index are both preserved.

```bash
lma-deduplicate \
  --input hindi/data/interim/cleaned.jsonl.gz \
  --output hindi/data/processed/deduplicated.jsonl.gz \
  --database hindi/data/checkpoints/dedup.sqlite \
  --report hindi/data/stats/deduplication.json
```

The default near-duplicate rule uses word 5-gram SimHash with Hamming distance at most 3. This
threshold must be validated on a manually inspected duplicate/non-duplicate sample and reported.
Use `--exact-only` for a safe overlap measurement before that manual threshold validation; do not
reuse its output/database pair as the final near-deduplicated run.

## 5. Train/validation/test splits

The deterministic splitter uses a 98/1/1 ratio. Pages carrying the same `book_id`, `work_id`, or
`group_id` are kept in one split.

```bash
lma-split \
  --input hindi/data/processed/deduplicated.jsonl.gz \
  --output-dir hindi/data/splits \
  --report hindi/data/stats/splits.json
```

## 6. Tokenizer candidates

Train separate SentencePiece Unigram candidates on each language's **training split only**:

```bash
lma-train-tokenizers \
  --input hindi/data/splits/train.jsonl.gz \
  --language hi \
  --output-dir hindi/tokenizer/runs \
  --vocabulary-sizes 16000 24000 32000 \
  --report hindi/data/stats/tokenizer_training.json
```

Evaluate all candidates on held-out validation text:

```bash
lma-evaluate-tokenizer \
  --model hindi/tokenizer/runs/hi_unigram_16000.model \
  --model hindi/tokenizer/runs/hi_unigram_24000.model \
  --model hindi/tokenizer/runs/hi_unigram_32000.model \
  --input hindi/data/splits/validation.jsonl.gz \
  --language Hindi \
  --output hindi/data/stats/tokenizer_evaluation.json \
  --figure report/figures/generated/hindi_token_frequency.png
```

Select the vocabulary using fertility, characters per token, unknown-token rate, vocabulary
utilization, and the later approximately 25M-parameter budget. Copy only the selected `.model`
and `.vocab` files from `tokenizer/runs/` into the language's `tokenizer/` directory for Git.

## 7. Exact dataset statistics

After selecting a tokenizer, calculate final token-based statistics and the required manual
fraction:

```bash
lma-corpus-stats \
  --input train=hindi/data/splits/train.jsonl.gz \
  --input validation=hindi/data/splits/validation.jsonl.gz \
  --input test=hindi/data/splits/test.jsonl.gz \
  --tokenizer-model hindi/tokenizer/hindi.model \
  --output hindi/data/stats/final_corpus.json \
  --source-csv report/tables/generated/hindi_sources.csv
```

The command reports `training_manual_token_fraction` and `manual_requirement_met`. Repeat the
entire workflow independently under `bhojpuri/`.

## Reports and artifact links

- Phase 1 report: [`report/phase-1.md`](report/phase-1.md)
- Hindi dataset: TBD
- Bhojpuri dataset: TBD
- Dataset checksums: TBD
