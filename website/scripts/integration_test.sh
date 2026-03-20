#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8010}"
MAX_FILES="${MAX_FILES:-}"

SRC_2023="${DIGESTS_2023_DIR:-$ROOT_DIR/../ml_research_analysis_2023}"
SRC_2024="${DIGESTS_2024_DIR:-$ROOT_DIR/../ml_research_analysis_2024}"
SRC_2025="${DIGESTS_2025_DIR:-$ROOT_DIR/../ml_research_analysis_2025}"
SRC_2026="${DIGESTS_2026_DIR:-$ROOT_DIR/../ml_research_analysis_2026}"

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/website-integration.XXXXXX")"
SERVER_LOG="$LOG_DIR/server.log"
TEST_ROOT="$LOG_DIR/site"
TEST_SEARCH_DIR="$TEST_ROOT/search"

SERVER_PID=""

if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Error: test port $PORT is already in use. Set PORT=<free-port>." >&2
  exit 1
fi

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$LOG_DIR"
}
trap cleanup EXIT

fail() {
  echo "\n[FAIL] $1" >&2
  if [[ -f "$SERVER_LOG" ]]; then
    echo "\n--- server log tail ---" >&2
    tail -n 80 "$SERVER_LOG" >&2 || true
  fi
  exit 1
}

expect_status() {
  local url="$1"
  local expected="$2"
  local code
  code="$(curl -sS -o /dev/null -w "%{http_code}" "$url")"
  if [[ "$code" != "$expected" ]]; then
    fail "Expected HTTP $expected for $url, got $code"
  fi
}

echo "[1/6] Preparing isolated test root"
mkdir -p "$TEST_ROOT" "$TEST_SEARCH_DIR"
ln -s "$ROOT_DIR/index.html" "$TEST_ROOT/index.html"
ln -s "$ROOT_DIR/view" "$TEST_ROOT/view"
ln -s "$ROOT_DIR/cloud" "$TEST_ROOT/cloud"
ln -s "$ROOT_DIR/about" "$TEST_ROOT/about"
ln -s "$ROOT_DIR/assets" "$TEST_ROOT/assets"
cp "$ROOT_DIR/search/index.html" "$TEST_SEARCH_DIR/index.html"

echo "[2/6] Building search DB (isolated, does not touch $ROOT_DIR/search)"
BUILD_ENV=(
  "DIGESTS_2023_DIR=$SRC_2023"
  "DIGESTS_2024_DIR=$SRC_2024"
  "DIGESTS_2025_DIR=$SRC_2025"
  "DIGESTS_2026_DIR=$SRC_2026"
  "OUTPUT_DIR=$TEST_SEARCH_DIR"
)
if [[ -n "$MAX_FILES" ]]; then
  BUILD_ENV+=("MAX_FILES=$MAX_FILES")
fi
env "${BUILD_ENV[@]}" "$ROOT_DIR/scripts/build_db.sh"

MANIFEST_PATH="$TEST_SEARCH_DIR/manifest.json"
[[ -f "$MANIFEST_PATH" ]] || fail "Manifest not found: $MANIFEST_PATH"

DB_FILE="$(python3 - <<'PY' "$MANIFEST_PATH"
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
print(manifest.get('db_file', ''))
PY
)"
[[ -n "$DB_FILE" ]] || fail "manifest.json is missing db_file"
[[ -f "$TEST_SEARCH_DIR/$DB_FILE" ]] || fail "DB file missing: $TEST_SEARCH_DIR/$DB_FILE"

echo "[3/6] Validating built DB contents"
SAMPLE_ID="$(python3 - <<'PY' "$TEST_SEARCH_DIR/$DB_FILE"
import sqlite3
import sys
conn = sqlite3.connect(sys.argv[1])
row = conn.execute("SELECT digest_id FROM digests ORDER BY digest_id LIMIT 1").fetchone()
if not row:
    raise SystemExit(1)
print(row[0])
PY
)"
[[ -n "$SAMPLE_ID" ]] || fail "Could not fetch sample digest_id from DB"

python3 - <<'PY' "$TEST_SEARCH_DIR/$DB_FILE"
import sqlite3
import sys
conn = sqlite3.connect(sys.argv[1])
term_count = conn.execute("SELECT COUNT(*) FROM cloud_term").fetchone()[0]
posting_chunk_count = conn.execute("SELECT COUNT(*) FROM cloud_term_postings").fetchone()[0]
if term_count <= 0 or posting_chunk_count <= 0:
    raise SystemExit(
        f"cloud cache tables look empty (terms={term_count}, postings_chunks={posting_chunk_count})"
    )
print(f"cloud cache rows: terms={term_count}, postings_chunks={posting_chunk_count}")
PY

echo "[4/6] Starting local server on :$PORT (isolated root)"
(
  cd "$ROOT_DIR"
  python3 "$ROOT_DIR/scripts/local_server.py" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --root "$TEST_ROOT" \
    --source-dir "$SRC_2023" \
    --source-dir "$SRC_2024" \
    --source-dir "$SRC_2025" \
    --source-dir "$SRC_2026" \
    >"$SERVER_LOG" 2>&1
) &
SERVER_PID=$!

READY=0
for _ in {1..120}; do
  if curl -sS "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done
[[ "$READY" == "1" ]] || fail "Server did not become ready on port $PORT"

echo "[5/6] Checking routes/assets"
expect_status "http://127.0.0.1:$PORT/" 200
expect_status "http://127.0.0.1:$PORT/view/" 200

VIEW_CODE="$(curl -sS -L -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/view?id=$SAMPLE_ID")"
[[ "$VIEW_CODE" == "200" ]] || fail "Expected /view?id=<sample> to resolve to 200, got $VIEW_CODE"

expect_status "http://127.0.0.1:$PORT/view/$SAMPLE_ID.md" 200
expect_status "http://127.0.0.1:$PORT/search/" 200
expect_status "http://127.0.0.1:$PORT/cloud/" 200
expect_status "http://127.0.0.1:$PORT/about/" 200
expect_status "http://127.0.0.1:$PORT/AGENTS.md" 200
expect_status "http://127.0.0.1:$PORT/search/?q=transformers" 200
expect_status "http://127.0.0.1:$PORT/search/manifest.json" 200
expect_status "http://127.0.0.1:$PORT/search/cloud-terms.json" 200
expect_status "http://127.0.0.1:$PORT/search/$DB_FILE" 200
expect_status "http://127.0.0.1:$PORT/assets/cloud.js" 200
expect_status "http://127.0.0.1:$PORT/assets/shared-header.js" 200
expect_status "http://127.0.0.1:$PORT/assets/sqljs-httpvfs/index.js" 200
expect_status "http://127.0.0.1:$PORT/assets/sqljs-httpvfs/sqlite.worker.js" 200
expect_status "http://127.0.0.1:$PORT/assets/sqljs-httpvfs/sql-wasm.wasm" 200

echo "[6/6] Checking HTTP Range support for SQLite file"
RANGE_HEADERS="$(curl -sS -D - -o /dev/null -H "Range: bytes=0-1023" "http://127.0.0.1:$PORT/search/$DB_FILE")"
echo "$RANGE_HEADERS" | grep -q " 206 " || fail "Expected 206 Partial Content for range request"
echo "$RANGE_HEADERS" | grep -qi "^Content-Range: bytes 0-" || fail "Missing Content-Range header"

echo "\nPASS: integration test runner completed successfully"
