#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUCKET_NAME="${1:-${BUCKET_NAME:-ml-llm-digests-9981ee3e6e}}"
AWS_REGION="${2:-${AWS_REGION:-us-east-1}}"
SYNC_VIEW_MARKDOWN="${SYNC_VIEW_MARKDOWN:-1}"
MD_SYNC_PARALLEL="${MD_SYNC_PARALLEL:-1}"
DELETE_RECONCILE="${DELETE_RECONCILE:-0}"

SRC_2023="${DIGESTS_2023_DIR:-$ROOT_DIR/../ml_research_analysis_2023}"
SRC_2024="${DIGESTS_2024_DIR:-$ROOT_DIR/../ml_research_analysis_2024}"
SRC_2025="${DIGESTS_2025_DIR:-$ROOT_DIR/../ml_research_analysis_2025}"

for required in aws mktemp; do
  if ! command -v "$required" >/dev/null 2>&1; then
    echo "Error: required command not found: $required" >&2
    exit 1
  fi
done

STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/website-deploy.XXXXXX")"
cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

echo "Staging site assets into: $STAGE_DIR"

# Preserve mtimes so aws s3 sync can skip unchanged files.
mkdir -p "$STAGE_DIR/view" "$STAGE_DIR/assets" "$STAGE_DIR/search"
cp -p "$ROOT_DIR/index.html" "$STAGE_DIR/index.html"
if [[ -f "$ROOT_DIR/mldigest.ico" ]]; then
  cp -p "$ROOT_DIR/mldigest.ico" "$STAGE_DIR/mldigest.ico"
fi
cp -a "$ROOT_DIR/view/." "$STAGE_DIR/view/"
cp -a "$ROOT_DIR/assets/." "$STAGE_DIR/assets/"
cp -a "$ROOT_DIR/search/." "$STAGE_DIR/search/"

if [[ ! -f "$STAGE_DIR/search/manifest.json" ]]; then
  echo "Error: missing search/manifest.json in staged assets." >&2
  echo "Run ./scripts/build_db.sh before deploy." >&2
  exit 1
fi

echo "Syncing staged assets to s3://$BUCKET_NAME/ (no delete reconciliation by default) ..."
aws s3 sync "$STAGE_DIR/" "s3://$BUCKET_NAME/"

if [[ "$DELETE_RECONCILE" == "1" ]]; then
  echo "Running delete reconciliation (DELETE_RECONCILE=1), preserving /view/*.md and /search/*.sqlite ..."
  aws s3 sync "$STAGE_DIR/" "s3://$BUCKET_NAME/" \
    --delete \
    --exclude "view/*.md" \
    --exclude "search/*.sqlite"
fi

# Force cache headers for app shell files.
# NOTE: `aws s3 sync` does not update metadata for unchanged files,
# so we use `cp --recursive` to guarantee header updates.
# These files are small, so re-uploading them each deploy is acceptable.
aws s3 cp "$STAGE_DIR/" "s3://$BUCKET_NAME/" \
  --recursive \
  --exclude "*" \
  --include "*.html" \
  --include "search/manifest.json" \
  --include "search/search-manifest.json" \
  --cache-control "no-store, max-age=0, must-revalidate"

aws s3 cp "$STAGE_DIR/" "s3://$BUCKET_NAME/" \
  --recursive \
  --exclude "*" \
  --include "*.js" \
  --cache-control "no-cache, max-age=0, must-revalidate"

sync_markdown_dir() {
  local src="$1"
  aws s3 sync "$src/" "s3://$BUCKET_NAME/view/" \
    --exclude "*" \
    --include "*.md" \
    --cache-control "public, max-age=31536000, immutable" \
    --no-progress
}

if [[ "$SYNC_VIEW_MARKDOWN" == "1" ]]; then
  echo "Syncing digest markdown files to /view/ (SYNC_VIEW_MARKDOWN=1, MD_SYNC_PARALLEL=$MD_SYNC_PARALLEL) ..."

  md_sources=()
  for src in "$SRC_2023" "$SRC_2024" "$SRC_2025"; do
    if [[ -d "$src" ]]; then
      md_sources+=("$src")
    else
      echo "Warning: source dir not found, skipping: $src"
    fi
  done

  if [[ "${#md_sources[@]}" -gt 0 ]]; then
    if [[ "$MD_SYNC_PARALLEL" == "1" || "${#md_sources[@]}" -eq 1 ]]; then
      for src in "${md_sources[@]}"; do
        sync_markdown_dir "$src"
      done
    else
      pids=()
      for src in "${md_sources[@]}"; do
        sync_markdown_dir "$src" &
        pids+=("$!")
      done

      for pid in "${pids[@]}"; do
        wait "$pid"
      done
    fi
  fi
fi

echo "Done."
echo "S3 object URL: https://$BUCKET_NAME.s3.$AWS_REGION.amazonaws.com/"
echo "Website URL (if static website hosting enabled): http://$BUCKET_NAME.s3-website-$AWS_REGION.amazonaws.com"
