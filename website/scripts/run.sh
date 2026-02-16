#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC_2023="${DIGESTS_2023_DIR:-$ROOT_DIR/../ml_research_analysis_2023}"
SRC_2024="${DIGESTS_2024_DIR:-$ROOT_DIR/../ml_research_analysis_2024}"
SRC_2025="${DIGESTS_2025_DIR:-$ROOT_DIR/../ml_research_analysis_2025}"

cd "$ROOT_DIR"

echo "Starting local server"
echo "URL: http://localhost:${PORT}/"
echo "Viewer routes:"
echo "  /view?id=<digest-id>"
echo "  /view/?id=<digest-id>"
echo "Raw route:"
echo "  /view/<digest-id>.md"
echo "Search routes (after build):"
echo "  /search/"
echo "  /search/?q=<query>"
echo "  /search/manifest.json"

python3 scripts/local_server.py \
  --host 127.0.0.1 \
  --port "$PORT" \
  --root "$ROOT_DIR" \
  --source-dir "$SRC_2023" \
  --source-dir "$SRC_2024" \
  --source-dir "$SRC_2025"
