#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERY_ROOT = PROJECT_ROOT / "data" / "word_clouds"
LEGACY_QUERY_ROOT = PROJECT_ROOT.parent.parent / "research_crawler" / "research_paper_analysis_v2" / "queries"

YEAR_DIR_MAP: dict[int, str] = {
    2023: "2023",
    2024: "2024",
    2025: "2025",
}

LEGACY_YEAR_DIR_MAP: dict[int, str] = {
    2023: "word_clouds_2023_organic_semantic_cleaned",
    2024: "word_clouds_2024",
    2025: "word_clouds",
}

TOKEN_SANITIZE_RE = re.compile(r"[^a-z0-9\s]")
SEPARATOR_RE = re.compile(r"[-_/]+")


def normalize_tokens(value: str) -> list[str]:
    normalized = value.lower()
    normalized = SEPARATOR_RE.sub(" ", normalized)
    normalized = TOKEN_SANITIZE_RE.sub(" ", normalized)
    return [token for token in normalized.split() if token]


def build_scoped_fts_query(term: str) -> str:
    tokens = normalize_tokens(term)
    if not tokens:
        return ""

    return " AND ".join(f"(title:{token} OR core_contribution:{token})" for token in tokens)


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
    if (query_root / YEAR_DIR_MAP[2025]).exists():
        return YEAR_DIR_MAP

    if (query_root / LEGACY_YEAR_DIR_MAP[2025]).exists():
        return LEGACY_YEAR_DIR_MAP

    return YEAR_DIR_MAP


def discover_canonical_files(query_root: Path, year_dir_map: dict[int, str]) -> list[str]:
    canonical_dir = query_root / year_dir_map[2025]
    if not canonical_dir.exists():
        return []

    return sorted(file_path.name for file_path in canonical_dir.glob("*.txt"))


def collect_terms_by_year(
    query_root: Path,
    canonical_files: list[str],
    year_dir_map: dict[int, str],
) -> tuple[dict[int, dict[str, str]], dict[int, list[str]]]:
    terms_by_year: dict[int, dict[str, str]] = {}
    files_used_by_year: dict[int, list[str]] = {}

    for year, dir_name in year_dir_map.items():
        year_dir = query_root / dir_name
        seen: dict[str, str] = {}
        files_used: list[str] = []

        if not year_dir.exists():
            terms_by_year[year] = seen
            files_used_by_year[year] = files_used
            continue

        file_names = canonical_files if canonical_files else sorted(path.name for path in year_dir.glob("*.txt"))

        for file_name in file_names:
            file_path = year_dir / file_name
            if not file_path.exists():
                continue

            files_used.append(file_name)
            for term in parse_terms(file_path):
                key = term.casefold()
                if key not in seen:
                    seen[key] = term

        terms_by_year[year] = seen
        files_used_by_year[year] = files_used

    return terms_by_year, files_used_by_year


def fetch_match_count(conn: sqlite3.Connection, fts_query: str, year: int | None) -> int:
    if year is None:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM digests_fts
            JOIN digests d ON d.id = digests_fts.rowid
            WHERE digests_fts MATCH ?
            """,
            (fts_query,),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM digests_fts
            JOIN digests d ON d.id = digests_fts.rowid
            WHERE digests_fts MATCH ?
              AND d.year = ?
            """,
            (fts_query, year),
        ).fetchone()

    return int(row[0] if row else 0)


def build_payload(
    conn: sqlite3.Connection,
    query_root: Path,
    include_zero_scores: bool,
    max_terms_per_year: int,
) -> dict:
    year_dir_map = resolve_query_layout(query_root)
    canonical_files = discover_canonical_files(query_root, year_dir_map)
    terms_by_year, files_used_by_year = collect_terms_by_year(query_root, canonical_files, year_dir_map)

    years_payload: dict[str, dict] = {}
    all_years_terms: dict[str, str] = {}
    count_cache: dict[tuple[str, int | None], int] = {}

    for year in sorted(year_dir_map):
        rows: list[dict] = []
        terms_map = terms_by_year.get(year, {})

        for term_key, term in terms_map.items():
            all_years_terms.setdefault(term_key, term)

            fts_query = build_scoped_fts_query(term)
            if not fts_query:
                if include_zero_scores:
                    rows.append({"term": term, "score": 0, "all_years_score": 0})
                continue

            year_cache_key = (fts_query, year)
            all_cache_key = (fts_query, None)

            if year_cache_key not in count_cache:
                count_cache[year_cache_key] = fetch_match_count(conn, fts_query, year)
            if all_cache_key not in count_cache:
                count_cache[all_cache_key] = fetch_match_count(conn, fts_query, None)

            year_score = count_cache[year_cache_key]
            all_score = count_cache[all_cache_key]

            if year_score == 0 and not include_zero_scores:
                continue

            rows.append(
                {
                    "term": term,
                    "score": year_score,
                    "all_years_score": all_score,
                }
            )

        rows.sort(key=lambda item: (-item["score"], -item["all_years_score"], item["term"].casefold()))

        if max_terms_per_year > 0:
            rows = rows[:max_terms_per_year]

        years_payload[str(year)] = {
            "term_count": len(terms_map),
            "scored_term_count": len(rows),
            "files": files_used_by_year.get(year, []),
            "terms": rows,
        }

    all_rows: list[dict] = []
    for term in all_years_terms.values():
        fts_query = build_scoped_fts_query(term)
        if not fts_query:
            continue

        all_cache_key = (fts_query, None)
        if all_cache_key not in count_cache:
            count_cache[all_cache_key] = fetch_match_count(conn, fts_query, None)

        score = count_cache[all_cache_key]
        if score == 0 and not include_zero_scores:
            continue

        all_rows.append({"term": term, "score": score})

    all_rows.sort(key=lambda item: (-item["score"], item["term"].casefold()))
    if max_terms_per_year > 0:
        all_rows = all_rows[:max_terms_per_year]

    return {
        "schema_version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "query_root": str(query_root),
        "scoring": {
            "type": "fts_match_count",
            "search_scope": ["title", "core_contribution"],
            "include_zero_scores": include_zero_scores,
        },
        "canonical_files": canonical_files,
        "year_directories": {str(year): dir_name for year, dir_name in sorted(year_dir_map.items())},
        "years": years_payload,
        "all_years": {
            "term_count": len(all_years_terms),
            "scored_term_count": len(all_rows),
            "terms": all_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build /cloud word-score data from yearly term lists and SQLite FTS counts")
    parser.add_argument("--db-path", type=Path, required=True, help="Path to built SQLite search DB")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "search" / "cloud-terms.json")
    parser.add_argument("--query-root", type=Path, default=DEFAULT_QUERY_ROOT)
    parser.add_argument("--include-zero-scores", action="store_true")
    parser.add_argument(
        "--max-terms-per-year",
        type=int,
        default=450,
        help="Max terms to emit per year (0 = unlimited)",
    )
    args = parser.parse_args()

    db_path = args.db_path.resolve()
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    query_root = args.query_root.resolve()
    if not query_root.exists() and args.query_root == DEFAULT_QUERY_ROOT and LEGACY_QUERY_ROOT.exists():
        query_root = LEGACY_QUERY_ROOT.resolve()
        print(f"Using legacy query root fallback: {query_root}")

    if not query_root.exists():
        print(f"Warning: query root not found: {query_root}")

    conn = sqlite3.connect(db_path)
    try:
        payload = build_payload(
            conn=conn,
            query_root=query_root,
            include_zero_scores=args.include_zero_scores,
            max_terms_per_year=max(0, args.max_terms_per_year),
        )
    finally:
        conn.close()

    payload["db_file"] = db_path.name

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    years_summary = ", ".join(
        f"{year}: {payload['years'][year]['scored_term_count']} scored"
        for year in sorted(payload["years"].keys())
    )
    print(f"Wrote cloud term data: {output_path}")
    print(f"  {years_summary}")


if __name__ == "__main__":
    main()
