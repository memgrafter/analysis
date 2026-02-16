# SQLite-over-HTTP Proposal (Selected)

We will implement search using a single SQLite database (FTS5) hosted on S3 and queried in-browser via sql.js + HTTP range requests.

## Scope

- Build `papers.sqlite` from the markdown corpus.
- Include FTS5 fields: `title`, `core_contribution`, `abstract_text`, `tags`.
- Host `papers.sqlite` and static website assets on S3.
- Query in-browser with sql.js HTTP VFS so only needed DB pages are fetched.

## Why this plan

- Strong search quality (FTS5 ranking + full query syntax).
- Minimal change from current pipeline.
- Static hosting friendly (S3 + range requests).

## Deployment target for today

Generated hash (date-seeded, URL-safe hex, 10 chars): `9981ee3e6e`

- Bucket name: `ml-llm-digests-9981ee3e6e`
- Bucket URL: `s3://ml-llm-digests-9981ee3e6e`

Hash command used:

```bash
DATE_SEED=$(date +%F)
HASH=$(printf "%s" "$DATE_SEED" | shasum -a 256 | awk '{print $1}' | cut -c1-10)
echo "$HASH"
```
