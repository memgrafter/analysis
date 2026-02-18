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
)

# Optional test/dev override only.
if [[ -n "${MAX_FILES:-}" ]]; then
  CMD+=(--max-files "$MAX_FILES")
fi

echo "Building search DB (FTS=full text contentless detail=column, no body preview storage)..."
echo "Output: $OUTPUT_DIR"
echo "Sources:"
echo "  - $SRC_2023"
echo "  - $SRC_2024"
echo "  - $SRC_2025"

"${CMD[@]}"

MANIFEST_PATH="$OUTPUT_DIR/manifest.json"
if [[ -f "$MANIFEST_PATH" ]]; then
  DB_FILE="$(python3 - <<'PY' "$MANIFEST_PATH"
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
print(manifest.get('db_file', ''))
PY
)"

  if [[ -n "$DB_FILE" && -f "$OUTPUT_DIR/$DB_FILE" ]]; then
    echo "Building cloud term scores..."
    python3 "$ROOT_DIR/scripts/build_cloud_terms.py" \
      --db-path "$OUTPUT_DIR/$DB_FILE" \
      --output "$OUTPUT_DIR/cloud-terms.json"
  else
    echo "Warning: could not find DB file from manifest for cloud term build" >&2
  fi
fi
