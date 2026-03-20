#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERY_ROOT = PROJECT_ROOT / "data" / "word_clouds"
LEGACY_QUERY_ROOT = PROJECT_ROOT.parent.parent / "research_crawler" / "research_paper_analysis_v2" / "queries"

YEAR_OPTIONS = [2023, 2024, 2025, 2026]

YEAR_DIR_MAP: dict[int, str] = {
    2023: "2023",
    2024: "2024",
    2025: "2025",
    2026: "2026",
}

LEGACY_YEAR_DIR_MAP: dict[int, str] = {
    2023: "word_clouds_2023_organic_semantic_cleaned",
    2024: "word_clouds_2024",
    2025: "word_clouds",
}

SORT_NAME_TO_ID = {
    "relevance": 0,
    "newest": 1,
    "title_asc": 2,
}

TOKEN_SANITIZE_RE = re.compile(r"[^a-z0-9\s]")
SEPARATOR_RE = re.compile(r"[-_/]+")


@dataclass(slots=True)
class TermStats:
    display_term: str
    total_count: int
    year_scores: dict[int, int]


def normalize_tokens(value: str) -> list[str]:
    normalized = value.lower()
    normalized = SEPARATOR_RE.sub(" ", normalized)
    normalized = TOKEN_SANITIZE_RE.sub(" ", normalized)
    return [token for token in normalized.split() if token]


def normalize_term_key(value: str) -> str:
    return " ".join(normalize_tokens(value))


def build_fts_query_full_scope(term_key: str) -> str:
    tokens = normalize_tokens(term_key)
    if not tokens:
        return ""

    # Full-scope FTS query across title/core/tags/body_text.
    return " AND ".join(tokens)


def parse_terms(file_path: Path) -> list[str]:
    terms: list[str] = []
    for raw_line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        line = " ".join(line.split())
        if line:
            terms.append(line)

    return terms


def resolve_query_layout(query_root: Path) -> dict[int, str]:
    if (query_root / YEAR_DIR_MAP[2026]).exists():
        return YEAR_DIR_MAP

    if (query_root / YEAR_DIR_MAP[2025]).exists():
        return YEAR_DIR_MAP

    if (query_root / LEGACY_YEAR_DIR_MAP[2025]).exists():
        return LEGACY_YEAR_DIR_MAP

    return YEAR_DIR_MAP


def discover_canonical_files(query_root: Path, year_dir_map: dict[int, str]) -> list[str]:
    # Use the newest available year as canonical seed source.
    canonical_year = 2026 if (query_root / year_dir_map[2026]).exists() else 2025
    canonical_dir = query_root / year_dir_map[canonical_year]
    if not canonical_dir.exists():
        return []

    return sorted(file_path.name for file_path in canonical_dir.glob("*.txt"))


def collect_seed_terms(
    query_root: Path,
    year_dir_map: dict[int, str],
    canonical_files: list[str],
) -> tuple[dict[int, dict[str, str]], dict[int, list[str]]]:
    seeds_by_year: dict[int, dict[str, str]] = {}
    files_used_by_year: dict[int, list[str]] = {}

    for year in YEAR_OPTIONS:
        year_dir = query_root / year_dir_map[year]
        year_terms: dict[str, str] = {}
        files_used: list[str] = []

        if not year_dir.exists():
            seeds_by_year[year] = year_terms
            files_used_by_year[year] = files_used
            continue

        file_names = canonical_files if canonical_files else sorted(path.name for path in year_dir.glob("*.txt"))

        for file_name in file_names:
            file_path = year_dir / file_name
            if not file_path.exists():
                continue

            files_used.append(file_name)
            for raw_term in parse_terms(file_path):
                term_key = normalize_term_key(raw_term)
                if not term_key:
                    continue
                year_terms.setdefault(term_key, raw_term)

        seeds_by_year[year] = year_terms
        files_used_by_year[year] = files_used

    return seeds_by_year, files_used_by_year


def choose_union_terms(seeds_by_year: dict[int, dict[str, str]]) -> dict[str, str]:
    # Prefer newer display forms when a normalized key appears in multiple yearly lists.
    union: dict[str, str] = {}
    for year in [2026, 2025, 2024, 2023]:
        for term_key, display_term in seeds_by_year.get(year, {}).items():
            union.setdefault(term_key, display_term)
    return union


def parse_sort_names(raw: str) -> list[str]:
    names = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    if not names:
        return ["relevance", "newest", "title_asc"]

    out: list[str] = []
    for name in names:
        if name not in SORT_NAME_TO_ID:
            raise SystemExit(f"Invalid sort name: {name}. Allowed: {', '.join(sorted(SORT_NAME_TO_ID))}")
        if name not in out:
            out.append(name)
    return out


def drop_cache_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS cloud_term_hit;
        DROP TABLE IF EXISTS cloud_term_postings;
        DROP TABLE IF EXISTS cloud_term;
        """
    )


def create_cache_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE cloud_term (
            term_id INTEGER PRIMARY KEY,
            term_key TEXT UNIQUE NOT NULL,
            display_term TEXT NOT NULL,
            total_count INTEGER NOT NULL
        );

        CREATE TABLE cloud_term_postings (
            term_id INTEGER NOT NULL,
            sort_id INTEGER NOT NULL,
            chunk_idx INTEGER NOT NULL,
            ids_blob BLOB NOT NULL,
            PRIMARY KEY (term_id, sort_id, chunk_idx)
        ) WITHOUT ROWID;
        """
    )


def encode_ids_blob(ids: list[int]) -> bytes:
    if not ids:
        return b""
    return struct.pack(f"<{len(ids)}I", *ids)


def append_posting_chunks(
    posting_batch: list[tuple[int, int, int, sqlite3.Binary]],
    term_id: int,
    sort_id: int,
    digest_rowids: list[int],
    chunk_size: int,
) -> None:
    chunk_idx = 0
    for i in range(0, len(digest_rowids), chunk_size):
        chunk = digest_rowids[i : i + chunk_size]
        posting_batch.append((term_id, sort_id, chunk_idx, sqlite3.Binary(encode_ids_blob(chunk))))
        chunk_idx += 1


def build_term_cache(
    conn: sqlite3.Connection,
    union_terms: dict[str, str],
    sort_names: list[str],
    include_zero_scores: bool,
    chunk_size: int,
) -> dict[str, TermStats]:
    sort_ids = [SORT_NAME_TO_ID[name] for name in sort_names]

    term_stats: dict[str, TermStats] = {}
    term_batch: list[tuple[int, str, str, int]] = []
    posting_batch: list[tuple[int, int, int, sqlite3.Binary]] = []
    term_id = 0

    started = time.time()
    total_relevance_hits = 0

    cur = conn.cursor()

    terms_items = sorted(union_terms.items(), key=lambda item: item[0])
    for idx, (term_key, display_term) in enumerate(terms_items, start=1):
        fts_query = build_fts_query_full_scope(term_key)
        if not fts_query:
            if include_zero_scores:
                term_stats[term_key] = TermStats(
                    display_term=display_term,
                    total_count=0,
                    year_scores={year: 0 for year in YEAR_OPTIONS},
                )
            continue

        rows = cur.execute(
            """
            SELECT
              d.id AS digest_rowid,
              d.digest_id AS digest_id,
              COALESCE(d.arxiv_id, '') AS arxiv_key,
              lower(COALESCE(d.title, '')) AS title_key,
              COALESCE(d.year, 0) AS year
            FROM digests_fts
            JOIN digests d ON d.id = digests_fts.rowid
            WHERE digests_fts MATCH ?
            ORDER BY bm25(digests_fts) ASC, d.digest_id ASC
            """,
            (fts_query,),
        ).fetchall()

        total_count = len(rows)
        year_scores = {year: 0 for year in YEAR_OPTIONS}
        for _, _, _, _, row_year in rows:
            row_year_int = int(row_year)
            if row_year_int in year_scores:
                year_scores[row_year_int] += 1

        term_stats[term_key] = TermStats(
            display_term=display_term,
            total_count=total_count,
            year_scores=year_scores,
        )

        if total_count == 0 and not include_zero_scores:
            continue

        term_id += 1
        total_relevance_hits += total_count
        term_batch.append((term_id, term_key, display_term, total_count))

        relevance_rowids = [int(row[0]) for row in rows]
        if 0 in sort_ids:
            append_posting_chunks(posting_batch, term_id, 0, relevance_rowids, chunk_size)

        if 1 in sort_ids and total_count > 0:
            newest_rows = sorted(rows, key=lambda row: (row[2], row[1]), reverse=True)
            newest_rowids = [int(row[0]) for row in newest_rows]
            append_posting_chunks(posting_batch, term_id, 1, newest_rowids, chunk_size)

        if 2 in sort_ids and total_count > 0:
            title_rows = sorted(rows, key=lambda row: (row[3], row[1]))
            title_rowids = [int(row[0]) for row in title_rows]
            append_posting_chunks(posting_batch, term_id, 2, title_rowids, chunk_size)

        if len(term_batch) >= 256:
            conn.executemany(
                "INSERT INTO cloud_term (term_id, term_key, display_term, total_count) VALUES (?, ?, ?, ?)",
                term_batch,
            )
            term_batch.clear()

        if len(posting_batch) >= 4000:
            conn.executemany(
                """
                INSERT INTO cloud_term_postings (term_id, sort_id, chunk_idx, ids_blob)
                VALUES (?, ?, ?, ?)
                """,
                posting_batch,
            )
            posting_batch.clear()

        if idx % 100 == 0:
            elapsed = time.time() - started
            print(f"  Cached terms {idx:,} / {len(terms_items):,} (elapsed {elapsed:.1f}s)")

    if term_batch:
        conn.executemany(
            "INSERT INTO cloud_term (term_id, term_key, display_term, total_count) VALUES (?, ?, ?, ?)",
            term_batch,
        )

    if posting_batch:
        conn.executemany(
            """
            INSERT INTO cloud_term_postings (term_id, sort_id, chunk_idx, ids_blob)
            VALUES (?, ?, ?, ?)
            """,
            posting_batch,
        )

    conn.commit()

    term_count = conn.execute("SELECT COUNT(*) FROM cloud_term").fetchone()[0]
    posting_chunk_count = conn.execute("SELECT COUNT(*) FROM cloud_term_postings").fetchone()[0]
    print(
        "Cloud cache rows: "
        f"terms={term_count:,}, postings_chunks={posting_chunk_count:,}, relevance_hits={total_relevance_hits:,}"
    )

    return term_stats


def build_json_payload(
    query_root: Path,
    year_dir_map: dict[int, str],
    canonical_files: list[str],
    seeds_by_year: dict[int, dict[str, str]],
    files_used_by_year: dict[int, list[str]],
    union_terms: dict[str, str],
    term_stats: dict[str, TermStats],
    sort_names: list[str],
    include_zero_scores: bool,
    max_terms_per_year: int,
    chunk_size: int,
) -> dict:
    years_payload: dict[str, dict] = {}

    for year in YEAR_OPTIONS:
        rows: list[dict] = []
        year_seed_terms = seeds_by_year.get(year, {})

        for term_key, display_term in year_seed_terms.items():
            stats = term_stats.get(term_key)
            if stats is None:
                if include_zero_scores:
                    rows.append({"term": display_term, "score": 0, "all_years_score": 0})
                continue

            score = int(stats.year_scores.get(year, 0))
            all_years_score = int(stats.total_count)
            if score == 0 and not include_zero_scores:
                continue

            rows.append(
                {
                    "term": stats.display_term,
                    "score": score,
                    "all_years_score": all_years_score,
                }
            )

        rows.sort(key=lambda item: (-item["score"], -item["all_years_score"], item["term"].casefold()))
        if max_terms_per_year > 0:
            rows = rows[:max_terms_per_year]

        years_payload[str(year)] = {
            "term_count": len(year_seed_terms),
            "scored_term_count": len(rows),
            "files": files_used_by_year.get(year, []),
            "terms": rows,
        }

    all_rows: list[dict] = []
    for term_key, display_term in union_terms.items():
        stats = term_stats.get(term_key)
        if stats is None:
            if include_zero_scores:
                all_rows.append({"term": display_term, "score": 0})
            continue

        score = int(stats.total_count)
        if score == 0 and not include_zero_scores:
            continue

        all_rows.append({"term": stats.display_term, "score": score})

    all_rows.sort(key=lambda item: (-item["score"], item["term"].casefold()))
    if max_terms_per_year > 0:
        all_rows = all_rows[:max_terms_per_year]

    return {
        "schema_version": 3,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "query_root": str(query_root),
        "canonical_files": canonical_files,
        "year_directories": {str(year): year_dir_map[year] for year in YEAR_OPTIONS},
        "scoring": {
            "type": "fts_match_count",
            "search_scope": ["title", "core_contribution", "tags", "body_text"],
            "include_zero_scores": include_zero_scores,
        },
        "cache": {
            "tables": ["cloud_term", "cloud_term_postings"],
            "sorts": sort_names,
            "posting_chunk_size": chunk_size,
            "years_default": YEAR_OPTIONS,
            "scopes_default": ["title", "core", "body"],
            "pagination": "rank_range",
        },
        "years": years_payload,
        "all_years": {
            "term_count": len(union_terms),
            "scored_term_count": len(all_rows),
            "terms": all_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build /cloud term JSON and precomputed cloud search cache tables inside SQLite"
    )
    parser.add_argument("--db-path", type=Path, required=True, help="Path to built SQLite search DB")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "search" / "cloud-terms.json")
    parser.add_argument("--query-root", type=Path, default=DEFAULT_QUERY_ROOT)
    parser.add_argument("--include-zero-scores", action="store_true")
    parser.add_argument(
        "--max-terms-per-year",
        type=int,
        default=0,
        help="Max terms in each year list (0 = unlimited)",
    )
    parser.add_argument(
        "--sorts",
        default="relevance,newest,title_asc",
        help="Comma-separated cached sort set (relevance,newest,title_asc)",
    )
    parser.add_argument(
        "--posting-chunk-size",
        type=int,
        default=512,
        help="Number of digest rowids per posting chunk",
    )
    parser.add_argument(
        "--no-write-db-cache",
        action="store_true",
        help="Do not keep cloud cache tables in SQLite (JSON only)",
    )
    args = parser.parse_args()

    db_path = args.db_path.resolve()
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    if args.posting_chunk_size <= 0:
        raise SystemExit("--posting-chunk-size must be > 0")

    query_root = args.query_root.resolve()
    if not query_root.exists() and args.query_root == DEFAULT_QUERY_ROOT and LEGACY_QUERY_ROOT.exists():
        query_root = LEGACY_QUERY_ROOT.resolve()
        print(f"Using legacy query root fallback: {query_root}")

    if not query_root.exists():
        print(f"Warning: query root not found: {query_root}")

    sort_names = parse_sort_names(args.sorts)
    year_dir_map = resolve_query_layout(query_root)
    canonical_files = discover_canonical_files(query_root, year_dir_map)
    seeds_by_year, files_used_by_year = collect_seed_terms(query_root, year_dir_map, canonical_files)
    union_terms = choose_union_terms(seeds_by_year)

    print(f"Cloud seed terms: {len(union_terms):,}")
    print(f"Cached sorts: {', '.join(sort_names)}")
    print(f"Posting chunk size: {args.posting_chunk_size}")

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")

        existing_tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        had_old_hit_layout = "cloud_term_hit" in existing_tables

        drop_cache_tables(conn)

        if args.no_write_db_cache:
            create_cache_tables(conn)
            term_stats = build_term_cache(
                conn=conn,
                union_terms=union_terms,
                sort_names=["relevance"],
                include_zero_scores=args.include_zero_scores,
                chunk_size=args.posting_chunk_size,
            )
            drop_cache_tables(conn)
            conn.commit()
        else:
            create_cache_tables(conn)
            term_stats = build_term_cache(
                conn=conn,
                union_terms=union_terms,
                sort_names=sort_names,
                include_zero_scores=args.include_zero_scores,
                chunk_size=args.posting_chunk_size,
            )

        if had_old_hit_layout:
            print("Detected old cloud_term_hit layout; running VACUUM to reclaim space...")
            conn.execute("VACUUM")
            conn.commit()
    finally:
        conn.close()

    payload = build_json_payload(
        query_root=query_root,
        year_dir_map=year_dir_map,
        canonical_files=canonical_files,
        seeds_by_year=seeds_by_year,
        files_used_by_year=files_used_by_year,
        union_terms=union_terms,
        term_stats=term_stats,
        sort_names=sort_names,
        include_zero_scores=args.include_zero_scores,
        max_terms_per_year=max(0, args.max_terms_per_year),
        chunk_size=args.posting_chunk_size,
    )
    payload["db_file"] = db_path.name

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    years_summary = ", ".join(
        f"{year}: {payload['years'][str(year)]['scored_term_count']} scored" for year in YEAR_OPTIONS
    )
    print(f"Wrote cloud term data: {output_path}")
    print(f"  {years_summary}")


if __name__ == "__main__":
    main()
