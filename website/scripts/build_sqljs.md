# build_sqljs.sh

Builds a pinned local `sql.js` runtime with **FTS5 enabled**, then copies runtime assets into this repo.

## Usage

```bash
./scripts/build_sqljs.sh
```

Optional clone location override:

```bash
SQLJS_CLONE_DIR=~/clones/sql.js ./scripts/build_sqljs.sh
```

## What it does

1. Ensures required tools are installed:
   - `git`, `npm`, `node`, `emcc`, `sha3sum`
2. Clones `sql.js` into `~/clones/sql.js` (default) if missing.
3. Ensures `-DSQLITE_ENABLE_FTS5` is present in `Makefile`.
4. Runs `npm install` and `npm run rebuild` in the clone.
5. Verifies FTS5 support with a runtime SQL smoke test.
6. Copies output to:
   - `assets/sqljs/sql-wasm.js`
   - `assets/sqljs/sql-wasm.wasm`

## Result

The search UI (`/search/`) can load the local runtime assets instead of CDN runtime binaries, with FTS5 support guaranteed by this build path.
