### How to run

Build DB:

```bash
  ./scripts/build_db.sh
```

> This is the canonical build path: builds full-text contentless FTS5 (`detail=column`) and app metadata in `digests`.
> Optional quick dev/test run: `MAX_FILES=1500 ./scripts/build_db.sh`
> Build also emits `search/cloud-terms.json` and precomputes cloud search cache tables (`cloud_term`, `cloud_term_postings`) in the SQLite DB from vendored seed terms in `data/word_clouds/`.

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
