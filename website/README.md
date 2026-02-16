### How to run

Build DB:

```bash
  python3 scripts/build_search_db.py \
    --source-dir ../ml_research_analysis_2023 \
    --source-dir ../ml_research_analysis_2024 \
    --source-dir ../ml_research_analysis_2025 \
    --output-dir search \
    --preview-words 500
```

Run local server:

```bash
  ./scripts/run.sh
```

Run the integration suite (single entrypoint):

```bash
  ./scripts/integration_test.sh
```

Optional faster integration run while iterating:

```bash
  MAX_FILES=1500 ./scripts/integration_test.sh
```
