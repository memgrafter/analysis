#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIRS = [
    PROJECT_ROOT.parent / "ml_research_analysis_2023",
    PROJECT_ROOT.parent / "ml_research_analysis_2024",
    PROJECT_ROOT.parent / "ml_research_analysis_2025",
    PROJECT_ROOT.parent / "ml_research_analysis_2026",
]

TIMESTAMP_SUFFIX_RE = re.compile(r"_(\d{8}_\d{6})$")
ARXIV_ID_RE = re.compile(r"^(\d{2})\d{2}\.\d{4,5}(?:v\d+)?$")


@dataclass
class DigestRow:
    digest_id: str
    arxiv_id: str | None
    title: str | None
    core_contribution: str | None
    tags: str
    source_path: str
    year: int | None
    timestamp_suffix: str | None


def strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        return "", text

    boundary = text.find("\n---\n", 4)
    if boundary == -1:
        return "", text

    frontmatter = text[4:boundary]
    body = text[boundary + 5 :]
    return frontmatter, body


def parse_frontmatter(frontmatter: str) -> dict[str, str | list[str]]:
    data: dict[str, str | list[str]] = {}
    lines = frontmatter.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if line.startswith((" ", "\t")):
            i += 1
            continue

        if ":" not in line:
            i += 1
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        if value == "":
            # List form (e.g., tags)
            if i + 1 < len(lines) and lines[i + 1].lstrip().startswith("-"):
                items: list[str] = []
                i += 1
                while i < len(lines) and lines[i].lstrip().startswith("-"):
                    item = lines[i].lstrip()[1:].strip()
                    items.append(strip_quotes(item))
                    i += 1
                data[key] = items
                continue

            # Multiline indented scalar
            chunks: list[str] = []
            i += 1
            while i < len(lines) and lines[i].startswith((" ", "\t")):
                chunks.append(lines[i].strip())
                i += 1
            data[key] = " ".join(chunks).strip()
            continue

        # Folded/literal block scalars.
        if value in {"|", ">", "|-", ">-", "|+", ">+"}:
            block_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].startswith((" ", "\t")):
                block_lines.append(lines[i].strip())
                i += 1

            if value.startswith("|"):
                data[key] = "\n".join(block_lines).strip()
            else:
                data[key] = " ".join(block_lines).strip()
            continue

        # Quoted multiline scalar continuation
        if value.startswith("'") and not value.endswith("'"):
            chunks = [value[1:]]
            i += 1
            while i < len(lines):
                segment = lines[i].strip()
                if segment.endswith("'"):
                    chunks.append(segment[:-1])
                    i += 1
                    break
                chunks.append(segment)
                i += 1
            data[key] = " ".join(chunks).strip()
            continue

        if value.startswith('"') and not value.endswith('"'):
            chunks = [value[1:]]
            i += 1
            while i < len(lines):
                segment = lines[i].strip()
                if segment.endswith('"'):
                    chunks.append(segment[:-1])
                    i += 1
                    break
                chunks.append(segment)
                i += 1
            data[key] = " ".join(chunks).strip()
            continue

        # Plain scalar continuation lines.
        chunks = [strip_quotes(value)]
        i += 1
        while i < len(lines) and lines[i].startswith((" ", "\t")):
            chunks.append(lines[i].strip())
            i += 1
        data[key] = " ".join(chunk for chunk in chunks if chunk).strip()

    return data


def infer_arxiv_id(digest_id: str, frontmatter: dict[str, str | list[str]]) -> str | None:
    fm_arxiv = frontmatter.get("arxiv_id")
    if isinstance(fm_arxiv, str) and fm_arxiv.strip():
        return fm_arxiv.strip().strip("'\"")

    prefix = digest_id.split("_", 1)[0]
    return prefix if prefix else None


def infer_year(arxiv_id: str | None, source_path: Path) -> int | None:
    if arxiv_id:
        m = ARXIV_ID_RE.match(arxiv_id)
        if m:
            return 2000 + int(m.group(1))

    dir_match = re.search(r"(20\d{2})", str(source_path))
    if dir_match:
        return int(dir_match.group(1))

    return None


def timestamp_suffix(digest_id: str) -> str | None:
    m = TIMESTAMP_SUFFIX_RE.search(digest_id)
    return m.group(1) if m else None


def discover_files(source_dirs: list[Path], max_files: int | None) -> tuple[list[Path], list[str], list[str]]:
    files: list[Path] = []
    missing_dirs: list[str] = []
    collisions: list[str] = []
    by_digest_id: dict[str, Path] = {}

    for source_dir in source_dirs:
        if not source_dir.exists():
            missing_dirs.append(str(source_dir))
            continue

        for md_file in source_dir.rglob("*.md"):
            resolved = md_file.resolve()
            digest_id = resolved.stem
            if digest_id in by_digest_id:
                collisions.append(
                    f"{digest_id}\n  - {by_digest_id[digest_id]}\n  - {resolved}"
                )
                continue
            by_digest_id[digest_id] = resolved
            files.append(resolved)

    files.sort(key=lambda p: str(p))
    if max_files is not None:
        files = files[:max_files]

    return files, missing_dirs, collisions


def compute_build_hash(files: list[Path]) -> str:
    h = hashlib.sha256()
    h.update(b"fts_mode=contentless_full_text_detail_column\n")
    h.update(b"digests_schema=v3_no_body_preview\n")
    h.update(b"frontmatter_parser=v2_multiline_scalars\n")

    for file_path in files:
        stat = file_path.stat()
        h.update(str(file_path).encode("utf-8", errors="replace"))
        h.update(b"\0")
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(b"\0")
        h.update(str(stat.st_mtime_ns).encode("utf-8"))
        h.update(b"\n")

    return h.hexdigest()[:12]


def parse_digest_file(file_path: Path) -> DigestRow:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    frontmatter_raw, _body = split_frontmatter(text)
    frontmatter = parse_frontmatter(frontmatter_raw)

    digest_id = file_path.stem
    arxiv_id = infer_arxiv_id(digest_id, frontmatter)

    title = frontmatter.get("title") if isinstance(frontmatter.get("title"), str) else None
    core = (
        frontmatter.get("core_contribution")
        if isinstance(frontmatter.get("core_contribution"), str)
        else None
    )
    tags_val = frontmatter.get("tags")
    if isinstance(tags_val, list):
        tags = ", ".join(item.strip() for item in tags_val if item.strip())
    elif isinstance(tags_val, str):
        tags = tags_val.strip()
    else:
        tags = ""

    return DigestRow(
        digest_id=digest_id,
        arxiv_id=arxiv_id,
        title=title.strip() if title else None,
        core_contribution=core.strip() if core else None,
        tags=tags,
        source_path=str(file_path),
        year=infer_year(arxiv_id, file_path),
        timestamp_suffix=timestamp_suffix(digest_id),
    )


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE digests (
            id INTEGER PRIMARY KEY,
            digest_id TEXT UNIQUE NOT NULL,
            arxiv_id TEXT,
            title TEXT,
            core_contribution TEXT,
            tags TEXT,
            source_path TEXT NOT NULL,
            year INTEGER,
            timestamp_suffix TEXT
        );

        CREATE INDEX idx_digests_arxiv_id ON digests(arxiv_id);

        CREATE VIRTUAL TABLE digests_fts USING fts5(
            title,
            core_contribution,
            tags,
            body_text,
            tokenize='porter unicode61',
            content='',
            detail='column'
        );

        CREATE TABLE arxiv_latest (
            arxiv_id TEXT PRIMARY KEY,
            digest_id TEXT NOT NULL
        );
        """
    )


def insert_rows(conn: sqlite3.Connection, rows: list[DigestRow]) -> None:
    conn.executemany(
        """
        INSERT INTO digests (
            digest_id, arxiv_id, title, core_contribution, tags,
            source_path, year, timestamp_suffix
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                row.digest_id,
                row.arxiv_id,
                row.title,
                row.core_contribution,
                row.tags,
                row.source_path,
                row.year,
                row.timestamp_suffix,
            )
            for row in rows
        ],
    )

    # Contentless FTS5 index built from full body text with detail='column'.
    id_by_digest_id = {
        digest_id: row_id
        for row_id, digest_id in conn.execute("SELECT id, digest_id FROM digests")
    }

    fts_batch: list[tuple[int, str, str, str, str]] = []
    for idx, row in enumerate(rows, start=1):
        file_text = Path(row.source_path).read_text(encoding="utf-8", errors="replace")
        _, body = split_frontmatter(file_text)

        fts_batch.append(
            (
                id_by_digest_id[row.digest_id],
                row.title or "",
                row.core_contribution or "",
                row.tags or "",
                body,
            )
        )

        if len(fts_batch) >= 250:
            conn.executemany(
                """
                INSERT INTO digests_fts (rowid, title, core_contribution, tags, body_text)
                VALUES (?, ?, ?, ?, ?)
                """,
                fts_batch,
            )
            fts_batch.clear()

        if idx % 5000 == 0:
            print(f"  Indexed FTS text {idx:,} / {len(rows):,}")

    if fts_batch:
        conn.executemany(
            """
            INSERT INTO digests_fts (rowid, title, core_contribution, tags, body_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            fts_batch,
        )

    latest_by_arxiv: dict[str, DigestRow] = {}
    for row in rows:
        if not row.arxiv_id:
            continue

        existing = latest_by_arxiv.get(row.arxiv_id)
        if existing is None:
            latest_by_arxiv[row.arxiv_id] = row
            continue

        existing_key = (existing.timestamp_suffix or "", existing.digest_id)
        row_key = (row.timestamp_suffix or "", row.digest_id)
        if row_key > existing_key:
            latest_by_arxiv[row.arxiv_id] = row

    conn.executemany(
        "INSERT INTO arxiv_latest (arxiv_id, digest_id) VALUES (?, ?)",
        [(arxiv_id, row.digest_id) for arxiv_id, row in latest_by_arxiv.items()],
    )


def finalize_db(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO digests_fts(digests_fts) VALUES('optimize')")
    conn.commit()

    # Ensure page size is applied for serving behavior (requires VACUUM).
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA page_size=4096")
    conn.execute("VACUUM")
    conn.commit()


def write_manifest(
    output_dir: Path,
    db_file: str,
    build_hash: str,
    rows: list[DigestRow],
    source_dirs: list[Path],
) -> None:
    arxiv_ids = {row.arxiv_id for row in rows if row.arxiv_id}
    manifest = {
        "build_hash": build_hash,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "db_file": db_file,
        "digest_count": len(rows),
        "arxiv_count": len(arxiv_ids),
        "source_dirs": [str(p) for p in source_dirs],
        "fts_detail": "column",
        "fts_content_mode": "contentless",
        "fts_body_mode": "full_text",
        "body_preview_stored": False,
    }

    manifest_json = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output_dir / "manifest.json").write_text(manifest_json, encoding="utf-8")
    (output_dir / "search-manifest.json").write_text(manifest_json, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLite search DB from digest markdown")
    parser.add_argument("--source-dir", action="append", dest="source_dirs")
    parser.add_argument("--output-dir", default="search")
    parser.add_argument("--db-name", default="search.sqlite")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--fail-on-parse-error", action="store_true", default=True)
    parser.add_argument("--no-fail-on-parse-error", action="store_false", dest="fail_on_parse_error")
    parser.add_argument("--max-parse-errors", type=int, default=0)
    args = parser.parse_args()

    started = time.time()

    source_dirs = [Path(p).resolve() for p in (args.source_dirs or DEFAULT_SOURCE_DIRS)]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files, missing_dirs, collisions = discover_files(source_dirs, args.max_files)

    print("Source directories:")
    for src in source_dirs:
        print(f"  - {src}")

    if missing_dirs:
        print("\nWarning: missing source directories:")
        for missing in missing_dirs:
            print(f"  - {missing}")

    if collisions:
        print("\nError: digest filename collisions detected:")
        for collision in collisions[:20]:
            print(f"\n{collision}")
        if len(collisions) > 20:
            print(f"\n... and {len(collisions) - 20} more")
        raise SystemExit(1)

    if not files:
        raise SystemExit("No markdown files found in source directories")

    print(f"\nDiscovered markdown files: {len(files):,}")
    print("FTS mode: contentless full-text indexing with detail=column")
    print("Storage mode: no body preview text stored in digests table")

    build_hash = compute_build_hash(files)

    db_stem = Path(args.db_name).stem or "search"
    db_file = f"{db_stem}-{build_hash}.sqlite"
    db_path = output_dir / db_file
    temp_db_path = output_dir / f".{db_file}.tmp"

    parse_errors = 0
    rows: list[DigestRow] = []

    for idx, file_path in enumerate(files, start=1):
        try:
            rows.append(parse_digest_file(file_path))
        except Exception as exc:  # pragma: no cover (hard to predict malformed files)
            parse_errors += 1
            print(f"Parse error [{parse_errors}] {file_path}: {exc}")

        if idx % 5000 == 0:
            print(f"  Parsed {idx:,} / {len(files):,}")

    if args.fail_on_parse_error and parse_errors > 0:
        raise SystemExit(f"Failed: encountered {parse_errors} parse errors")

    if parse_errors > args.max_parse_errors:
        raise SystemExit(
            f"Failed: parse errors ({parse_errors}) exceed max allowed ({args.max_parse_errors})"
        )

    if temp_db_path.exists():
        temp_db_path.unlink()

    conn = sqlite3.connect(temp_db_path)
    try:
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")

        init_db(conn)
        insert_rows(conn, rows)
        finalize_db(conn)
    finally:
        conn.close()

    if db_path.exists():
        db_path.unlink()
    temp_db_path.rename(db_path)

    write_manifest(output_dir, db_file, build_hash, rows, source_dirs)

    elapsed = time.time() - started
    arxiv_count = len({row.arxiv_id for row in rows if row.arxiv_id})

    print("\nBuild complete")
    print(f"  output_dir: {output_dir}")
    print(f"  db_file: {db_file}")
    print(f"  digest_count: {len(rows):,}")
    print(f"  arxiv_count: {arxiv_count:,}")
    print(f"  db_size_bytes: {db_path.stat().st_size:,}")
    print(f"  parse_errors: {parse_errors}")
    print(f"  elapsed_seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
