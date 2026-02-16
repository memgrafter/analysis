# Coda Plan: arXiv ID → Viewer Resolution (Post-Search)

## Status
Deferred until after search implementation.

## Goal
Allow `/view` to accept either:
- full digest id (`?id=<digest-id>`), or
- arXiv id (`?id=<arxiv-id>`, e.g. `2306.17290`)

and resolve to a canonical digest view.

## Constraints
- Static hosting only (S3)
- No rewrite layer
- No server-side routing logic

## Proposed Behavior
1. If `id` looks like a full digest id, render directly.
2. If `id` looks like an arXiv id, resolve client-side via manifest lookup.
3. Canonicalize URL to `?id=<full-digest-id>` using client-side history/redirect.
4. Keep raw route unchanged: `/view/<digest-id>.md`.

## Data Contract
Generate a small static lookup artifact during build:
- `arxiv_id -> canonical_digest_id`

Canonical selection rule (initial):
- Choose latest timestamped digest for that arXiv id.

## Implementation Order (after search)
1. Add build step to emit arXiv lookup manifest.
2. Load manifest in `/view/` client code (or shard if needed).
3. Add resolver + canonicalization logic.
4. Add fallback UX when arXiv id is not found.
5. Add basic checks for duplicate/version handling.

## Non-Goals (for now)
- HTTP 301/302 redirects
- Version picker UI
- Rewrite-based pretty routes

## Done Criteria
- `/view/?id=<digest-id>` works (existing)
- `/view/?id=<arxiv-id>` resolves and opens canonical digest
- URL canonicalizes to `?id=<digest-id>`
- No change required to raw markdown storage layout
