#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "Serving static site from: $ROOT_DIR"
echo "URL: http://localhost:${PORT}/"

python3 -m http.server "$PORT" --bind 127.0.0.1
