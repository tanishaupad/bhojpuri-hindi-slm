[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Hindi and Bhojpuri monolingual language models

This repository builds two fully independent decoder-only language models from scratch:

- **Model H:** Hindi (`hi`, Devanagari), the higher-resource language.
- **Model L:** Bhojpuri (`bho`, Devanagari), the approved lower-resource language.

The `phase-1` and `phase-2` branches established reproducible corpus collection, preprocessing,
deduplication, split creation, tokenizer training, pre-training data-integrity audits, and from-scratch
PyTorch Transformer pretraining of two ~25M parameter causal decoder models. The current `phase-3`
branch carries out reasoning dataset generation, baseline zero-shot evaluation, causal finetuning
on multi-step reasoning tasks, held-out evaluation across original and explanation protocols, and
before-and-after multi-head attention analysis. The two languages never share documents, tokenizers,
vocabularies, checkpoints, or weights.

## Phase 2 status

Before model training, validation and test are treated as protected data. The training split is
audited asymmetrically for canonical exact-document matches, exact paragraph reuse, and
word-5-gram Bottom-k MinHash candidates at the same conservative 0.95 threshold used in Phase 1.
The audit is resumable, preserves provenance in a review CSV, and never edits the official splits.

| Language | Training documents checked | Exact documents | Exact paragraphs | Near documents | Flagged training documents |
|---|---:|---:|---:|---:|---:|
| Hindi | 2,237,951 | 0 | 0 | 75 | 75 |
| Bhojpuri | 225,992 | 0 | 253 | 107 | 360 |

These candidates were reviewed on 25 August 2026 using a recorded, user-directed policy. Hindi
near-document matches with estimated similarity at least 0.97 were marked `remove` (9), and the
remaining 66 were marked `keep`. For Bhojpuri, all 253 exact-paragraph matches and the 50
near-document matches with similarity at least 0.98 were marked `remove`; the remaining 57 were
marked `keep`. Every CSV row records its decision, reviewer policy, and rationale. Full settings,
counts, and sampled pairs are in
[`report/audits/generated`](report/audits/generated), with interpretation in
[`report/phase-2.md`](report/phase-2.md). Versioned decontaminated training splits have now been
generated in Google Drive: 2,237,942 Hindi documents after 9 removals and 225,689 Bhojpuri
documents after 303 removals. Validation and test remain byte-for-byte unchanged.

The from-scratch Phase 2 model, training, evaluation, and attention code is under
`src/lma_phase2/`, with independent configurations at `hindi/configs/phase2.toml` and
`bhojpuri/configs/phase2.toml`. The exact model sizes are 25,285,248 Hindi parameters and
24,870,912 Bhojpuri parameters. Packed Phase 2 streams contain 336,995,217 Hindi and 23,142,975
Bhojpuri training tokens. Both independent GPU pretraining runs are complete: Hindi reached
20,300 steps / 332,595,200 processed tokens, and Bhojpuri reached 4,150 steps / 67,993,600 tokens.
The final checkpoints, optimizer state, logs, and resume metadata are saved in Google Drive and
their SHA-256 hashes are recorded below.

| Full-test result | Hindi | Bhojpuri |
|---|---:|---:|
| Cross-entropy (nats/token) | 3.6782 | 4.1420 |
| Perplexity | 39.57 | 62.93 |
| Bits per UTF-8 byte | **0.5282** | **0.6503** |

The complete comparison—including all four decoding modes, qualitative sample review, training
curves, and early/late multi-head attention analysis—is in
[`report/phase-2.md`](report/phase-2.md).

```bash
python -m lma_phase2.decontaminate \
  --train /path/to/train.jsonl.gz \
  --validation /path/to/validation.jsonl.gz \
  --test /path/to/test.jsonl.gz \
  --database tmp/decontamination.sqlite \
  --audit-output report/audits/generated/matches.csv \
  --report report/audits/generated/report.json \
  --workers 4
```

### Phase 2 reproduction

The final versioned splits and compact token streams are stored under each language's
`phase2/tokenized_v1` Drive directory. Input, tokenizer, and packed-stream checksums are recorded
in `report/audits/generated/phase2_data_manifest.json`; rerunning an interrupted packing job
resumes at its last durable document boundary.

```bash
lma-tokenize-corpus \
  --input train=/path/to/final/train.jsonl.gz \
  --input validation=/path/to/final/validation.jsonl.gz \
  --input test=/path/to/final/test.jsonl.gz \
  --tokenizer-model hindi/tokenizer/hindi.model \
  --output-dir /path/to/drive/hindi/phase2/tokenized \
  --language hi
```

The equivalent Bhojpuri command uses `bhojpuri/tokenizer/bhojpuri.model`, language code `bho`,
and its own output directory. The two streams and runs must never be mixed.

Start Hindi pretraining, or resume it from the last complete checkpoint:

```bash
lma-train-lm \
  --config hindi/configs/phase2.toml \
  --train-manifest /path/to/hindi/phase2/tokenized/train.manifest.json \
  --validation-manifest /path/to/hindi/phase2/tokenized/validation.manifest.json \
  --output-dir /path/to/drive/hindi/phase2/runs/pretrain_v1

lma-train-lm \
  --config hindi/configs/phase2.toml \
  --train-manifest /path/to/hindi/phase2/tokenized/train.manifest.json \
  --validation-manifest /path/to/hindi/phase2/tokenized/validation.manifest.json \
  --output-dir /path/to/drive/hindi/phase2/runs/pretrain_v1 \
  --resume latest
```

Use `bhojpuri/configs/phase2.toml` and separate Bhojpuri manifests/output for Model L. Checkpoints
contain model, optimizer, scheduler, mixed-precision scaler, step/configuration, and all RNG states.
To stay within hosted-notebook disk limits, each run retains its three newest complete checkpoints.
Evaluation writes its small JSON/CSV/PNG results into `report/`; large checkpoints and packed token
streams remain in Google Drive.

Evaluate the final Hindi checkpoint on its complete held-out test stream:

```bash
lma-evaluate-lm \
  --config hindi/configs/phase2.toml \
  --checkpoint /path/to/hindi/phase2/runs/pretrain_v1/checkpoint_step_0020300.pt \
  --test-manifest /path/to/hindi/phase2/tokenized_v1/test.manifest.json \
  --test-jsonl /path/to/hindi/splits/phase2_decontaminated_v1/test.jsonl.gz \
  --tokenizer-model hindi/tokenizer/hindi.model \
  --output report/results/generated/phase2/hindi_evaluation.json \
  --samples-output report/samples/generated/phase2/hindi_samples.jsonl \
  --attention-csv report/tables/generated/phase2/hindi_attention_summary.csv \
  --figure-dir report/figures/generated/phase2/attention/hindi \
  --generation-examples 50 \
  --prefix-tokens 32 \
  --continuation-tokens 64 \
  --seed 20260823 \
  --device cuda
```

For Bhojpuri, use its config, checkpoint, test data, tokenizer, and language-specific output paths.
The Kaggle bundle can reproduce both evaluations in one command:

```bash
python /path/to/bundle/kaggle/run_phase2_kaggle.py --language both --mode evaluate
```

The final run folders are [Hindi Model H](https://drive.google.com/drive/folders/1QS_l1VYF2A_ifCxQuEThoYDgqInpS7BF)
and [Bhojpuri Model L](https://drive.google.com/drive/folders/1HQMO4a_Ud18URPi_X4nBNCIPZPcHeyCN).
To reproduce attention labels on a system whose default font lacks Devanagari glyphs, pass a font
file to evaluation with `--attention-font /path/to/Devanagari-font.ttf`.

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

Generated data is stored outside Git using this logical layout (currently under the course Google
Drive folder, with temporary local staging only while a large streaming job is active):

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

Final large corpus artifacts, packed streams, and model checkpoints are stored in Google Drive;
small audit and evaluation artifacts are versioned in this repository.

### Final Phase 1 Datasets and Splits

- **Hindi (Model H)**:
  - Total Corpus: 2,283,757 documents / 339,048,510 tokens (with 24k Unigram tokenizer)
  - Train split: 2,237,951 documents / 332,246,742 tokens (**25.06% manual tokens**)
  - Validation split: 23,033 documents / 3,428,944 tokens
  - Test split: 22,773 documents / 3,372,824 tokens
  - Tokenizer: `hindi/tokenizer/hindi.model` (24,000 Unigram, character coverage 1.0)

- **Bhojpuri (Model L)**:
  - Total Corpus: 230,687 documents / 23,019,401 tokens (with 16k Unigram tokenizer)
  - Train split: 225,992 documents / 22,577,763 tokens (**22.49% manual tokens**)
  - Validation split: 2,344 documents / 227,239 tokens
  - Test split: 2,351 documents / 214,399 tokens
  - Tokenizer: `bhojpuri/tokenizer/bhojpuri.model` (16,000 Unigram, character coverage 1.0)

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

### Blogger archives

For approved Blogger sources, use the structured feed rather than crawling navigation pages. The
collector is resumable, stores post labels and dates, and can exclude an explicit translation
label before writing the raw corpus.

```bash
lma-collect-blogger \
  --feed-url https://www.blogger.com/feeds/5766880049298620909/posts/default \
  --language bho \
  --source-name maina_bhojpuri \
  --license source-rights-retained-course-research \
  --exclude-label अनुवाद \
  --output bhojpuri/data/raw/manual/maina_bhojpuri.jsonl.gz \
  --checkpoint bhojpuri/data/checkpoints/maina_bhojpuri.json
```

The current full snapshots contain 1,504 Maina posts and 3,066 Bhojpuri News posts before the
common preprocessing pass. After script/length filtering, 4,566 documents and 1,183,589
whitespace words remain. A deterministic 200-row worksheet samples 100 documents from each source
for native-language and translation review.

### Sitemap-enumerated archives

When a site disallows its API but publishes an allowed XML sitemap, the sitemap collector fetches
only public article pages, checks `robots.txt`, rate-limits requests, and stores crawl state in
SQLite. Anjoria is restricted to pages last modified before 2023 to avoid recent AI-assisted
material; Hindi-only and translated items still require filtering and audit.

```bash
lma-collect-sitemap \
  --sitemap-index https://anjoria.com/sitemap_index.xml \
  --before-date 2023-01-01 \
  --language bho \
  --source-name anjoria_pre2023 \
  --license source-rights-retained-course-research \
  --content-selector '.entry-content, article' \
  --output bhojpuri/data/raw/manual/anjoria_pre2023.jsonl.gz \
  --state-db bhojpuri/data/checkpoints/anjoria_pre2023.sqlite \
  --delay 1.0
```

For the current collection, use the checked-in wrapper below. It draws one reproducible 6,000-page
sample from the 12,920 eligible URLs, caches the complete eligible URL list, and processes four
durable partitions sequentially at a three-second interval. Sequential execution avoids the `429`
responses observed when four workers contacted the same host at once.

```bash
scripts/collect_anjoria_sample6000.sh "/path/to/LMA_Individual_Project"
```

The wrapper accepts an optional second argument for resumed runs. After Anjoria began returning
`429` responses during a sustained crawl, the conservative resume setting became ten seconds:

```bash
scripts/collect_anjoria_sample6000.sh "/path/to/LMA_Individual_Project" 10.0
```

After collection, a character four-gram discriminator trained on separately held-out Bhojpuri and
Hindi seed text separates high-confidence Bhojpuri from obvious Hindi. The default acceptance
cutoff is the stricter of the Bhojpuri holdout's fifth percentile and the Hindi holdout's 99.5th
percentile. Ambiguous documents are written to a separate audit set; they are not automatically
included.

```bash
lma-filter-language \
  --bhojpuri-seed bhojpuri/data/interim/cleaned.jsonl.gz \
  --hindi-seed hindi/data/raw/downloaded/indiccorp_v2_hindi.jsonl.gz \
  --hindi-seed hindi/data/interim/pib_hindi_cleaned.jsonl.gz \
  --input bhojpuri/data/raw/manual/anjoria_pre2023.jsonl.gz \
  --accepted bhojpuri/data/interim/anjoria_accepted.jsonl.gz \
  --borderline bhojpuri/data/interim/anjoria_borderline.jsonl.gz \
  --rejected bhojpuri/data/interim/anjoria_rejected.jsonl.gz \
  --report bhojpuri/data/stats/anjoria_language_filter.json
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
rejected by the document-level script thresholds. For Anjoria, source-specific cleanup truncates
the repeated comment/recent-post/category tail before the quality metrics are calculated. For PIB,
standard entry-date/byline, signature, release-ID/footer, and English-dominant embedded blocks are
removed.

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

## 4. Exact, paragraph, and near deduplication

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

For the current Bhojpuri corpus, the exact-only pass is followed by corpus-wide exact paragraph
deduplication. Only lines of at least 100 characters and 15 words are eligible, protecting short
headings, quotations, and poetic refrains. If removing repeated material makes a document shorter
than the quality floor, that document is removed.

```bash
lma-deduplicate-paragraphs \
  --input bhojpuri/data/processed/deduplicated_exact_v5.jsonl.gz \
  --output bhojpuri/data/processed/deduplicated_paragraph_v1.jsonl.gz \
  --database bhojpuri/data/checkpoints/deduplicated_paragraph_v1.sqlite \
  --report bhojpuri/data/stats/deduplication_paragraph_v1.json
```

The final near-duplicate stage uses bottom-k MinHash over word five-grams. A pilot compared
thresholds from 0.75 upward: values below 0.95 incorrectly grouped distinct articles sharing a
template, while a manually inspected sample at 0.95 contained reposts and copies with only small
formatting or metadata changes. Documents with fewer than 20 word five-grams are protected.

```bash
lma-deduplicate-minhash \
  --input bhojpuri/data/processed/deduplicated_paragraph_v1.jsonl.gz \
  --output bhojpuri/data/processed/deduplicated_minhash95_v1.jsonl.gz \
  --database bhojpuri/data/checkpoints/deduplicated_minhash95_v1.sqlite \
  --audit-output bhojpuri/data/stats/minhash95_pairs_v1.csv \
  --report bhojpuri/data/stats/deduplication_minhash95_v1.json \
  --mode apply \
  --similarity-threshold 0.95
```

If the retained manual portion is below the assignment minimum, preserve every manual document and
deterministically sample only downloaded records. A target above 20% provides a margin before the
final SentencePiece-token measurement:

```bash
lma-downsample \
  --input bhojpuri/data/processed/deduplicated_minhash95_v1.jsonl.gz \
  --output bhojpuri/data/processed/deduplicated_minhash95_downsampled22_v1.jsonl.gz \
  --target-manual-word-fraction 0.22 \
  --seed bhojpuri-minhash95-downsample-22pct-v1 \
  --report bhojpuri/data/stats/downsampling_minhash95_22pct_v1.json
```

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

Training, evaluation, corpus-statistics, and later model-input code must all call the shared
`normalize_tokenizer_input` helper. It collapses Unicode whitespace to match SentencePiece's
one-paragraph-per-line training representation; character normalization itself remains identity
because corpus text has already been normalized to NFC.

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

- **Phase 1 Report**: [`report/phase-1.md`](report/phase-1.md)
- **Phase 2 Final Report**: [`report/phase-2.md`](report/phase-2.md)
- **Phase 3 Final Report**: [`report/phase-3.md`](report/phase-3.md)
- **Phase 2 Training Curves**: [`report/figures/generated/phase2/pretraining_curves.png`](report/figures/generated/phase2/pretraining_curves.png)
- **Phase 3 Finetuning Curves**: [`report/figures/generated/phase3/finetuning_curves.png`](report/figures/generated/phase3/finetuning_curves.png)
- **Phase 3 Attention Heatmaps**: [`report/figures/generated/phase3/attention/hindi/`](report/figures/generated/phase3/attention/hindi/), [`report/figures/generated/phase3/attention/bhojpuri/`](report/figures/generated/phase3/attention/bhojpuri/)
- **Phase 2 Evaluation and Samples**: [`report/results/generated/phase2`](report/results/generated/phase2), [`report/samples/generated/phase2`](report/samples/generated/phase2)
- **Phase 3 Evaluation Results**: [`report/results/generated/phase3/finetuned_v1`](report/results/generated/phase3/finetuned_v1)
- **Hindi Pretrained Checkpoint Folder**: [Google Drive](https://drive.google.com/drive/folders/1QS_l1VYF2A_ifCxQuEThoYDgqInpS7BF)
- **Bhojpuri Pretrained Checkpoint Folder**: [Google Drive](https://drive.google.com/drive/folders/1HQMO4a_Ud18URPi_X4nBNCIPZPcHeyCN)
- **Course Google Drive Root Folder**: [LMA Individual Project Drive Folder](https://drive.google.com/drive/folders/1zjJ5bSM69g431qnSRHXnXOwO_7rJOpce?usp=share_link) (contains `hindi/phase3/runs/finetune_v1/` and `bhojpuri/phase3/runs/finetune_v1/` for Phase 3 finetuned checkpoints)
- **Token Frequency Plot (Hindi)**: [`report/figures/generated/hindi_token_frequency.png`](report/figures/generated/hindi_token_frequency.png)
- **Token Frequency Plot (Bhojpuri)**: [`report/figures/generated/bhojpuri_token_frequency.png`](report/figures/generated/bhojpuri_token_frequency.png)
- **Hindi Source Statistics Table**: [`report/tables/generated/hindi_sources.csv`](report/tables/generated/hindi_sources.csv)
- **Bhojpuri Source Statistics Table**: [`report/tables/generated/bhojpuri_sources.csv`](report/tables/generated/bhojpuri_sources.csv)

### Dataset and Model Checksums (SHA-256)

| Artifact | Language | SHA-256 Checksum |
|---|---|---|
| `hindi/splits/train.jsonl.gz` | Hindi | `35e4a4689e40b70868ec9065bbeb036eb677e219867f60c20b0fb2341d700a8a` |
| `hindi/splits/validation.jsonl.gz` | Hindi | `71d1b9940011ac7998e444380dd82f0037bb65eb3a0a12818684c7066e9594b5` |
| `hindi/splits/test.jsonl.gz` | Hindi | `ef94dde8713b0e7a5d8ff7cbad54ff869c51b6336ec7f23cf24ec6915ebfad12` |
| `hindi/processed/balanced_22pct_v1.jsonl.gz` | Hindi | `8da68dd00742d094a55e1bc987e66264590c14867e2411f298983d7f65834c14` |
| `hindi/tokenizer/hindi.model` (24k) | Hindi | `7cf0b4bf5d70b741ef03c6218d6a858ff43675038ec7493a7e3d740eb76b6680` |
| `bhojpuri/splits/downsampled22_v1/train.jsonl.gz` | Bhojpuri | `d690a2cf757b1cb7a60216ae9cb0f19c968f9b9f939eecdf590ad5ec1dbf6311` |
| `bhojpuri/tokenizer/bhojpuri.model` (16k) | Bhojpuri | `5a703d1592ca07b667825d1945a6c38a2e1d743a6d7aa19c8fcb5cfdccefa9ba` |
| `hindi/phase2/runs/pretrain_v1/checkpoint_step_0020300.pt` | Hindi | `575458df89393ca2b5998024f93d04e9a08a1a5228f420da389108cfed870d42` |
| `bhojpuri/phase2/runs/pretrain_v1/checkpoint_step_0004150.pt` | Bhojpuri | `69580b164964ddb88477fd39181f5d1a57ef52cfcaa32dbe1fc68e19b805c133` |
| `hindi/phase3/runs/finetune_v1/best_checkpoint.pt` | Hindi | `8ac1b2c96642901f4025976ff597ac3d9abfc591820646572857972d176cc062` |
| `hindi/phase3/runs/finetune_v1/checkpoint_step_0002500.pt` | Hindi | `e0e76afdf39f89c1432a2034efeed201ed32e0cad90836db04be741293993457` |
| `bhojpuri/phase3/runs/finetune_v1/best_checkpoint.pt` | Bhojpuri | `cccd2ed6ce9c776972a4589ec18e9aa5198e2f81b04da2213902b4babda421b0` |
| `bhojpuri/phase3/runs/finetune_v1/checkpoint_step_0002500.pt` | Bhojpuri | `28d1799b0d842ac745db2390450590b74cc904b798b429669b1a010cb24d36f6` |

## Phase 3: Reasoning Dataset, Finetuning & Attention Analysis

Phase 3 implements synthetic relational reasoning, deterministic symbolic verification, causal finetuning of the pretrained ~25M parameter decoder models, held-out evaluation across two distinct protocols, and before-and-after multi-head attention entropy analysis.

### Dataset Design & Release (`reasoning_v1`)

The reasoning dataset contains **25,000 examples per language** across four task families: Direct Comparison, Largest/Smallest selection, Value Equality, and Two-Hop Transitivity:
- **Training**: 20,000 examples
- **Validation**: 2,000 examples
- **Primary Test**: 2,000 examples
- **Challenge Unseen Names**: 500 examples (novel person names absent from training)
- **Challenge Unseen Wording**: 500 examples (held-out premise templates and comparative phrasings)

Datasets are independently generated with native Hindi and Bhojpuri templates, names, and grammar. They are stored under `hindi/phase3/reasoning_v1/` and `bhojpuri/phase3/reasoning_v1/` in Google Drive.

```bash
python -m lma_phase3.release --output-dir /path/to/release-root --evidence-dir report/audits/generated/phase3/reasoning_v1
```

### Finetuning Reproduction

Both models are finetuned using causal language modeling with prompt masking ($Y[:N-1] = -100$), loss computed strictly over target tokens, AdamW with cosine decay ($10^{-4} \to 10^{-5}$), batch size 16 (micro 8, grad accum 2), and full RNG state capture:

```bash
# Hindi finetuning (Model H)
python -m lma_phase3.finetune \
  --config hindi/configs/phase3.toml \
  --pretrained-checkpoint /path/to/hindi/phase2/runs/pretrain_v1/checkpoint_step_0020300.pt \
  --train-data /path/to/hindi/phase3/reasoning_v1/train.jsonl.gz \
  --val-data /path/to/hindi/phase3/reasoning_v1/validation.jsonl.gz \
  --template-config hindi/configs/phase3_templates.json \
  --output-dir /path/to/hindi/phase3/runs/finetune_v1 \
  --validation-scope legacy_subset \
  --device mps

# Bhojpuri finetuning (Model L)
python -m lma_phase3.finetune \
  --config bhojpuri/configs/phase3.toml \
  --pretrained-checkpoint /path/to/bhojpuri/phase2/runs/pretrain_v1/checkpoint_step_0004150.pt \
  --train-data /path/to/bhojpuri/phase3/reasoning_v1/train.jsonl.gz \
  --val-data /path/to/bhojpuri/phase3/reasoning_v1/validation.jsonl.gz \
  --template-config bhojpuri/configs/phase3_templates.json \
  --output-dir /path/to/bhojpuri/phase3/runs/finetune_v1 \
  --validation-scope legacy_subset \
  --device mps
```

New training runs default to **full validation across all four task families**. The commands above explicitly reproduce historical v1 subset validation. Omit `--validation-scope legacy_subset` and choose a new output directory for full-validation selection; resume requires the same selection protocol as the saved checkpoint. Existing models were not retrained.

### Held-Out Evaluation Reproduction

Evaluation executes both generation protocols without format gates:
1. **Original Prompt Protocol (16 max tokens)**: Tests direct answer recovery, scoring strict exact match, answer presence (whole-name boundary match), and premise claim support.
2. **Explanation Prompt Protocol (96 max tokens)**: Tests full step-by-step reasoning derivation, verified against the ground-truth world by a deterministic symbolic proof checker.

```bash
# Hindi evaluation
python -m lma_phase3.evaluate_finetuned \
  --config hindi/configs/phase3.toml \
  --checkpoint /path/to/hindi/phase3/runs/finetune_v1/best_checkpoint.pt \
  --dataset-root /path/to/reasoning_v1 \
  --output-dir report/results/generated/phase3/finetuned_v1/hindi \
  --device mps --protocols both

# Bhojpuri evaluation
python -m lma_phase3.evaluate_finetuned \
  --config bhojpuri/configs/phase3.toml \
  --checkpoint /path/to/bhojpuri/phase3/runs/finetune_v1/best_checkpoint.pt \
  --dataset-root /path/to/reasoning_v1 \
  --output-dir report/results/generated/phase3/finetuned_v1/bhojpuri \
  --device mps --protocols both
```

### Course-alignment additions

See [the course-alignment report](report/phase-3-course-alignment.md) for generic-text retention before/after SFT, explicit-answer accuracy, limitations, and reproduction commands. The assigned synthetic question structures and datasets are unchanged. Original metrics below remain historical diagnostics; answer presence is not answer accuracy.

| Added metric | Hindi | Bhojpuri |
|---|---:|---:|
| Explicit-answer accuracy after SFT, primary test, 96 tokens | 86.20% | 71.40% |
| Explicit answer + verified explanation, same protocol | 86.15% | 69.90% |
| Generic-text perplexity before SFT | 39.57 | 62.93 |
| Generic-text perplexity after SFT | 628.12 | 4,939.06 |

The paired generic-text evaluation uses identical Phase 2 test streams and shows substantial loss of predictive performance after SFT. Low reasoning-target loss therefore does not establish retained language ability. The answer rescore is retrospective, uses a frozen leading-answer extraction rule, and preserves historical metrics.


### Key Empirical Results

#### Primary Test Split (2,000 examples per language)

| Metric | Hindi Pretrained | Hindi Finetuned | Bhojpuri Pretrained | Bhojpuri Finetuned |
|---|---|---|---|---|
| **Original (16-tok) Answer Presence** | 17.00% | **83.85%** | 3.10% | **85.00%** |
| **Original (16-tok) Supported Claims** | 5.90% | **81.55%** | 0.00% | **62.25%** |
| **Explanation (96-tok) Answer Presence** | 6.60% | **91.70%** | 2.30% | **89.15%** |
| **Explanation (96-tok) Joint Verified Proofs** | 0.00% | **86.15%** | 0.00% | **69.90%** |
| **Two-Hop Transitivity Verified Proofs** | 0.00% | **100.00%** | 0.00% | **100.00%** |

#### Generalization Challenges (500 examples each)

| Split | Hindi Presence (96-tok) | Hindi Verified Proofs | Bhojpuri Presence (96-tok) | Bhojpuri Verified Proofs |
|---|---|---|---|---|
| **Challenge Wording** | 89.40% | **64.60%** | 87.20% | **68.00%** |
| **Challenge Unseen Names** | 24.60% | 0.00% | 13.60% | 0.00% |

- **Wording challenge**: Under the single reserved question variant, joint verification is 64.60% for Hindi and 68.00% for Bhojpuri. This measures a narrow wording change; premise phrasing and task structures remain familiar.
- **Unseen names**: Joint verification falls to zero. Segmentation and entity familiarity are possible explanations; no controlled experiment establishes a primary cause.

### Multi-Head Attention Analysis

Attention maps on fixed prompts show descriptive changes in attention concentration and distance:
- Selected mid/late heads have lower entropy after finetuning; the exact per-head values are in Section 8 of the Phase 3 report.
- These summaries do not identify causally necessary reasoning heads. Ablations or activation interventions would be needed for that claim.
- Attention comparison tool:

```bash
# Hindi attention comparison
python -m lma_phase3.attention_compare \
  --config hindi/configs/phase3.toml \
  --pretrained-checkpoint /path/to/hindi/phase2/runs/pretrain_v1/checkpoint_step_0020300.pt \
  --finetuned-checkpoint /path/to/hindi/phase3/runs/finetune_v1/best_checkpoint.pt \
  --tokenizer hindi/tokenizer/hindi.model \
  --prompt "रमेश की लंबाई 194 सेंटीमीटर है। नितिन की लंबाई 110 सेंटीमीटर है। रमेश और नितिन में से किसकी लंबाई सबसे कम है? पहले उत्तर लिखें। फिर कारण: लिखकर तुलना का छोटा कारण बताएँ। उत्तर:" \
  --output-dir report/figures/generated/phase3/attention/hindi \
  --device cpu --layers 1 8 --heads 0 1 2

# Bhojpuri attention comparison
python -m lma_phase3.attention_compare \
  --config bhojpuri/configs/phase3.toml \
  --pretrained-checkpoint /path/to/bhojpuri/phase2/runs/pretrain_v1/checkpoint_step_0004150.pt \
  --finetuned-checkpoint /path/to/bhojpuri/phase3/runs/finetune_v1/best_checkpoint.pt \
  --tokenizer bhojpuri/tokenizer/bhojpuri.model \
  --prompt "रामनाथ के उमिर 24 बरिस बा। बनवारी के उमिर 65 बरिस बा। रामनाथ आ बनवारी में केकर उमिर सबसे ढेर बा? पहिले जवाब लिखीं। फेर कारण: लिख के तुलना के छोट कारण बताईं। जवाब:" \
  --output-dir report/figures/generated/phase3/attention/bhojpuri \
  --device cpu --layers 1 8 --heads 0 1 2
```

Full details, error analysis, task breakdowns, and rubric questions are documented in [`report/phase-3.md`](report/phase-3.md).
