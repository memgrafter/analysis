#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC_2023="${DIGESTS_2023_DIR:-$ROOT_DIR/../ml_research_analysis_2023}"
SRC_2024="${DIGESTS_2024_DIR:-$ROOT_DIR/../ml_research_analysis_2024}"
SRC_2025="${DIGESTS_2025_DIR:-$ROOT_DIR/../ml_research_analysis_2025}"

# Canonical deploy output is "$ROOT_DIR/search".
# Tests may override via OUTPUT_DIR to avoid mutating deploy artifacts.
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/search}"

mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/search-*.sqlite \
      "$OUTPUT_DIR"/manifest.json \
      "$OUTPUT_DIR"/search-manifest.json

CMD=(
  python3 "$ROOT_DIR/scripts/build_search_db.py"
  --source-dir "$SRC_2023"
  --source-dir "$SRC_2024"
  --source-dir "$SRC_2025"
  --output-dir "$OUTPUT_DIR"
  --preview-words 500
)

# Optional test/dev override only. Canonical production path indexes first 500 body words.
if [[ -n "${MAX_FILES:-}" ]]; then
  CMD+=(--max-files "$MAX_FILES")
fi

echo "Building search DB (preview=500 words, FTS=full text detail=column)..."
echo "Output: $OUTPUT_DIR"
echo "Sources:"
echo "  - $SRC_2023"
echo "  - $SRC_2024"
echo "  - $SRC_2025"

"${CMD[@]}"
