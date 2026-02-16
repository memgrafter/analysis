#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLONE_DIR="${SQLJS_CLONE_DIR:-$HOME/clones/sql.js}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command not found: $cmd" >&2
    exit 1
  fi
}

require_cmd git
require_cmd npm
require_cmd node
require_cmd emcc
require_cmd sha3sum

mkdir -p "$(dirname "$CLONE_DIR")"
if [[ ! -d "$CLONE_DIR/.git" ]]; then
  echo "Cloning sql.js into $CLONE_DIR"
  git clone https://github.com/sql-js/sql.js.git "$CLONE_DIR"
fi

cd "$CLONE_DIR"

echo "Ensuring FTS5 compile flag in Makefile"
python3 - <<'PY'
from pathlib import Path
path = Path('Makefile')
text = path.read_text()
if '-DSQLITE_ENABLE_FTS5' in text:
    raise SystemExit(0)
needle = "\t-DSQLITE_ENABLE_FTS3_PARENTHESIS \\\n"
if needle not in text:
    raise SystemExit('Could not find expected Makefile flags block')
text = text.replace(needle, needle + "\t-DSQLITE_ENABLE_FTS5 \\\n", 1)
path.write_text(text)
PY

echo "Installing npm dependencies"
npm install

echo "Rebuilding sql.js"
npm run rebuild

echo "Verifying FTS5 in built runtime"
node - <<'NODE'
const initSqlJs = require('./dist/sql-wasm.js');
(async () => {
  const SQL = await initSqlJs({ locateFile: f => `./dist/${f}` });
  const db = new SQL.Database();
  db.run('CREATE VIRTUAL TABLE t USING fts5(x);');
  db.run("INSERT INTO t VALUES ('potato rocket')");
  const out = db.exec("SELECT count(*) FROM t WHERE t MATCH 'potato*'");
  const n = out?.[0]?.values?.[0]?.[0] ?? 0;
  if (n < 1) {
    throw new Error('FTS5 test query failed');
  }
  console.log('FTS5 verification passed');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
NODE

mkdir -p "$ROOT_DIR/assets/sqljs"
cp "$CLONE_DIR/dist/sql-wasm.js" "$ROOT_DIR/assets/sqljs/sql-wasm.js"
cp "$CLONE_DIR/dist/sql-wasm.wasm" "$ROOT_DIR/assets/sqljs/sql-wasm.wasm"

echo "Copied runtime assets to:"
echo "  $ROOT_DIR/assets/sqljs/sql-wasm.js"
echo "  $ROOT_DIR/assets/sqljs/sql-wasm.wasm"

echo "Done."
