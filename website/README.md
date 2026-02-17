### How to run

Build DB:

```bash
  ./scripts/build_db.sh
```

> This is the canonical build path: stores 500-word body preview in `digests`, and builds full-text contentless FTS5 (`detail=column`).
> Optional quick dev/test run: `MAX_FILES=1500 ./scripts/build_db.sh`

Provision infrastructure (bucket only):

```bash
  ./scripts/provision.sh
```

Provision + deploy in one run:

```bash
  ./scripts/provision.sh --with-deploy
```

Run local server:

```bash
  ./scripts/run.sh
```

Run the integration suite (single entrypoint):

```bash
  ./scripts/integration_test.sh
```

> Integration builds are isolated in a temp directory and do not overwrite `search/` deploy artifacts.

Optional faster integration run while iterating:

```bash
  MAX_FILES=1500 ./scripts/integration_test.sh
```
