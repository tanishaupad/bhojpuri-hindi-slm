#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 ARTIFACT_ROOT [REQUEST_DELAY_SECONDS]" >&2
  exit 2
fi

artifact_root=$1
request_delay=${2:-10.0}
raw_dir="$artifact_root/bhojpuri/raw/manual_anjoria"
checkpoint_dir="$artifact_root/bhojpuri/checkpoints"
url_cache="$checkpoint_dir/anjoria_pre2023_urls_before2023_v1.json"

mkdir -p "$raw_dir" "$checkpoint_dir"

# Part 2 fills the only frontier that was not initialized during the pilot.
# The remaining order then resumes the three existing frontiers, one at a time.
for partition in 2 0 1 3; do
  .venv/bin/python -m lma_phase1.collectors.sitemap \
    --sitemap-index https://anjoria.com/sitemap_index.xml \
    --before-date 2023-01-01 \
    --language bho \
    --source-name anjoria_pre2023 \
    --license source-rights-retained-course-research \
    --content-selector '.entry-content, article' \
    --output "$raw_dir/anjoria_pre2023_sample6000_part${partition}_v1.jsonl.gz" \
    --state-db "$checkpoint_dir/anjoria_pre2023_sample6000_part${partition}_v1.sqlite" \
    --url-cache "$url_cache" \
    --discovery-delay 3.0 \
    --sample-size 6000 \
    --sample-seed anjoria-pre2023-sample6000-v1 \
    --partition-count 4 \
    --partition-index "$partition" \
    --retry-errors \
    --delay "$request_delay"
done
