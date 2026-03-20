#!/usr/bin/env python3
"""
Index YAML frontmatter from research analysis markdown files into SQLite.

Optimized for large and frequent reindexing:
- Incremental updates via file metadata (mtime_ns + size)
- Upsert behavior (updates changed files, inserts new files)
- Optional git-scoped indexing (working tree or diff range)
- Optional prune of deleted files
- Batched writes with SQLite tuning pragmas

Usage:
  python scripts/index_frontmatter.py
  python scripts/index_frontmatter.py ml_research_analysis --full
  python scripts/index_frontmatter.py ml_research_analysis --git-changed
  python scripts/index_frontmatter.py ml_research_analysis --git-changed origin/main...HEAD
  python scripts/index_frontmatter.py ml_research_analysis --git-changed --prune

Defaults:
  directory: ml_research_analysis
  db: analysis_outputs/research_index.sqlite
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DB_DEFAULT = "analysis_outputs/research_index.sqlite"
DIR_DEFAULT = "ml_research_analysis"


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id              INTEGER PRIMARY KEY,
            filename        TEXT NOT NULL UNIQUE,
            title           TEXT,
            arxiv_id        TEXT,
            tags            TEXT,
            core_contribution TEXT,
            indexed_at      TEXT DEFAULT (datetime('now')),
            file_mtime_ns   INTEGER,
            file_size       INTEGER,
            source_dir      TEXT
        )
    """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_arxiv_id ON papers(arxiv_id)")

    # Lightweight migration for older DBs that predate source_dir column.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()}
    if "source_dir" not in cols:
        conn.execute("ALTER TABLE papers ADD COLUMN source_dir TEXT")

    conn.commit()


def load_existing_meta(
    conn: sqlite3.Connection,
    source_dir: Optional[str] = None,
) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    if source_dir is not None:
        rows = conn.execute(
            "SELECT filename, file_mtime_ns, file_size FROM papers WHERE source_dir = ?",
            (source_dir,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT filename, file_mtime_ns, file_size FROM papers").fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def scan_markdown_files(dirpath: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    with os.scandir(dirpath) as it:
        for entry in it:
            if not entry.is_file():
                continue
            if not entry.name.endswith(".md"):
                continue
            files[entry.name] = Path(entry.path)
    return files


def parse_simple_yaml(lines: List[str]) -> Optional[dict]:
    """Fast parser for the small frontmatter subset we use."""
    if not lines:
        return None

    result: dict = {}
    current_key: Optional[str] = None
    current_val_lines: List[str] = []
    in_list = False
    list_items: List[str] = []

    def clean_scalar(v: str) -> str:
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            return v[1:-1]
        return v

    def flush() -> None:
        nonlocal current_key, current_val_lines, in_list, list_items
        if current_key is None:
            return
        if in_list:
            result[current_key] = [clean_scalar(x) for x in list_items]
        else:
            val = " ".join(current_val_lines).strip()
            result[current_key] = clean_scalar(val)

        current_key = None
        current_val_lines = []
        in_list = False
        list_items = []

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # New key line: no leading indentation and has key:
        if line and not line.startswith((" ", "\t")) and ":" in line and not stripped.startswith("- "):
            flush()
            key, val = line.split(":", 1)
            current_key = key.strip()
            val = val.strip()
            if val:
                current_val_lines.append(val)
            continue

        # List item for current key.
        if current_key and stripped.startswith("- "):
            in_list = True
            list_items.append(stripped[2:].strip())
            continue

        # Continuation line for current key.
        if current_key is not None:
            if stripped:
                current_val_lines.append(stripped)

    flush()
    return result if result else None


def extract_frontmatter(filepath: Path) -> Optional[dict]:
    """Extract frontmatter from markdown file.

    Strategy:
    1) Read between leading --- delimiters
    2) Fast local parser first
    3) Fallback to PyYAML only if fast parse is missing critical keys
    """
    try:
        with filepath.open("r", encoding="utf-8") as f:
            first = f.readline().rstrip("\n")
            if first != "---":
                return None

            lines: List[str] = []
            for line in f:
                if line.rstrip("\n") == "---":
                    break
                lines.append(line)
            else:
                return None
    except Exception:
        return None

    if not lines:
        return None

    parsed = parse_simple_yaml(lines)
    if parsed and isinstance(parsed, dict):
        # If we got the fields we care about, keep the fast result.
        if any(k in parsed for k in ("title", "arxiv_id", "tags", "core_contribution")):
            return parsed

    # Fallback to PyYAML only when needed.
    try:
        import yaml  # type: ignore

        data = yaml.safe_load("".join(lines))
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return parsed if isinstance(parsed, dict) else None


def normalize_tags(tags_obj: object) -> str:
    if isinstance(tags_obj, list):
        return json.dumps([str(x) for x in tags_obj])
    if tags_obj in (None, ""):
        return "[]"
    return json.dumps([str(tags_obj)])


def run_git(args: List[str], cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git command failed")
    return proc.stdout


def get_repo_root() -> Optional[Path]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if out.returncode != 0:
            return None
        return Path(out.stdout.strip()).resolve()
    except Exception:
        return None


def git_changed_and_deleted_basenames(target_dir: Path, spec: str) -> Tuple[set[str], set[str]]:
    """Return changed and deleted basenames from git.

    spec:
      - WORKTREE: staged/unstaged/untracked changes from `git status --porcelain`
      - any other value: passed directly as git diff range/spec
    """
    repo_root = get_repo_root()
    if repo_root is None:
        raise RuntimeError("Not inside a git repository")

    target_dir = target_dir.resolve()
    try:
        rel_dir = target_dir.relative_to(repo_root)
    except ValueError as e:
        raise RuntimeError(f"Directory not under git repo root: {target_dir}") from e

    changed: set[str] = set()
    deleted: set[str] = set()

    if spec == "WORKTREE":
        out = run_git(
            ["status", "--porcelain=v1", "--untracked-files=all", "--", str(rel_dir)],
            cwd=repo_root,
        )
        for line in out.splitlines():
            if not line:
                continue
            status = line[:2]
            payload = line[3:].strip()
            # rename format: old -> new
            if " -> " in payload:
                payload = payload.split(" -> ", 1)[1].strip()
            base = Path(payload).name
            if not base.endswith(".md"):
                continue
            if "D" in status:
                deleted.add(base)
            else:
                changed.add(base)
        return changed, deleted

    # Diff mode against a range/spec, e.g. origin/main...HEAD or HEAD~1..HEAD
    changed_out = run_git(
        ["diff", "--name-only", "--diff-filter=ACMR", spec, "--", str(rel_dir)],
        cwd=repo_root,
    )
    deleted_out = run_git(
        ["diff", "--name-only", "--diff-filter=D", spec, "--", str(rel_dir)],
        cwd=repo_root,
    )

    for line in changed_out.splitlines():
        base = Path(line.strip()).name
        if base.endswith(".md"):
            changed.add(base)

    for line in deleted_out.splitlines():
        base = Path(line.strip()).name
        if base.endswith(".md"):
            deleted.add(base)

    return changed, deleted


def index_directory(
    dirpath: Path,
    db_path: Path,
    full: bool,
    git_changed_spec: Optional[str],
    prune: bool,
    batch_size: int,
    dry_run: bool,
) -> None:
    os.makedirs(db_path.parent or Path("."), exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    init_db(conn)

    # Resolve to a stable directory name for source_dir tracking.
    source_dir_name = dirpath.resolve().name

    # Scope existing_meta to this source_dir so incremental/prune only
    # see rows that belong to the directory being indexed.
    existing_meta = load_existing_meta(conn, source_dir=source_dir_name)
    all_files = scan_markdown_files(dirpath)
    all_filenames = sorted(all_files.keys())

    git_deleted: set[str] = set()

    if full and git_changed_spec is not None:
        conn.close()
        raise SystemExit("Use either --full or --git-changed, not both")

    if git_changed_spec is not None:
        changed, git_deleted = git_changed_and_deleted_basenames(dirpath, git_changed_spec)
        targets = sorted(name for name in changed if name in all_files)
        missing_from_disk = sorted(name for name in changed if name not in all_files)

        print(
            f"Git-scoped mode ({git_changed_spec}): changed={len(changed)}, "
            f"existing-md={len(targets)}, missing-md={len(missing_from_disk)}, git-deleted={len(git_deleted)}"
        )
    elif full:
        targets = all_filenames
        print(f"Full mode: {len(targets)} files")
    else:
        targets = []
        for name in all_filenames:
            path = all_files[name]
            st = path.stat()
            old = existing_meta.get(name)
            if old is None or old[0] != st.st_mtime_ns or old[1] != st.st_size:
                targets.append(name)
        print(
            f"Incremental mode: total={len(all_filenames)}, indexed={len(existing_meta)}, "
            f"changed_or_new={len(targets)}"
        )

    if not targets and not (prune and (git_deleted or (git_changed_spec is None))):
        print("Nothing to index.")
        conn.close()
        return

    upsert_sql = """
        INSERT INTO papers
            (filename, title, arxiv_id, tags, core_contribution, indexed_at,
             file_mtime_ns, file_size, source_dir)
        VALUES
            (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?)
        ON CONFLICT(filename) DO UPDATE SET
            title=excluded.title,
            arxiv_id=excluded.arxiv_id,
            tags=excluded.tags,
            core_contribution=excluded.core_contribution,
            indexed_at=datetime('now'),
            file_mtime_ns=excluded.file_mtime_ns,
            file_size=excluded.file_size,
            source_dir=excluded.source_dir
    """

    inserted = 0
    updated = 0
    errors = 0
    batch: list[tuple] = []

    for fname in targets:
        fpath = all_files[fname]
        fm = extract_frontmatter(fpath)
        if fm is None:
            errors += 1
            continue

        st = fpath.stat()
        rec = (
            fname,
            str(fm.get("title", "")),
            str(fm.get("arxiv_id", "")),
            normalize_tags(fm.get("tags", [])),
            str(fm.get("core_contribution", "")),
            int(st.st_mtime_ns),
            int(st.st_size),
            source_dir_name,
        )
        batch.append(rec)

        if fname in existing_meta:
            updated += 1
        else:
            inserted += 1

        if len(batch) >= max(batch_size, 1):
            if not dry_run:
                conn.executemany(upsert_sql, batch)
                conn.commit()
            batch = []

    if batch and not dry_run:
        conn.executemany(upsert_sql, batch)
        conn.commit()

    pruned = 0
    if prune:
        if git_changed_spec is not None:
            # Git-scoped prune only removes files git reports as deleted in scope.
            to_delete = sorted(name for name in git_deleted if name in existing_meta)
        else:
            # Prune rows from THIS source_dir that are no longer on disk.
            on_disk = set(all_filenames)
            to_delete = sorted(name for name in existing_meta if name not in on_disk)

        if to_delete:
            pruned = len(to_delete)
            if not dry_run:
                conn.executemany("DELETE FROM papers WHERE filename = ?", [(n,) for n in to_delete])
                conn.commit()

    total_in_db = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    conn.close()

    mode = "git" if git_changed_spec is not None else ("full" if full else "incremental")
    dry = " [dry-run]" if dry_run else ""
    print(
        f"Done ({mode}{dry}). source_dir={source_dir_name}, "
        f"upserted={inserted + updated} (inserted={inserted}, updated={updated}), "
        f"errors={errors}, pruned={pruned}, total_in_db={total_in_db}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index markdown frontmatter into SQLite")
    parser.add_argument(
        "directory",
        nargs="?",
        default=DIR_DEFAULT,
        help=f"Directory to scan (default: {DIR_DEFAULT})",
    )
    parser.add_argument("--db", default=DB_DEFAULT, help=f"SQLite database path (default: {DB_DEFAULT})")
    parser.add_argument("--full", action="store_true", help="Re-parse all .md files and upsert")
    parser.add_argument(
        "--git-changed",
        nargs="?",
        const="WORKTREE",
        metavar="RANGE",
        help=(
            "Index only files changed in git. "
            "No value => working tree/staged/untracked changes. "
            "With value => git diff range/spec (e.g. origin/main...HEAD)."
        ),
    )
    parser.add_argument("--prune", action="store_true", help="Delete DB rows for deleted files")
    parser.add_argument("--batch-size", type=int, default=500, help="DB batch size (default: 500)")
    parser.add_argument("--dry-run", action="store_true", help="Plan and parse, but do not write DB")
    args = parser.parse_args()

    dirpath = Path(args.directory)
    db_path = Path(args.db)

    if not dirpath.is_dir():
        print(f"Error: {dirpath} is not a directory", file=sys.stderr)
        sys.exit(1)

    try:
        index_directory(
            dirpath=dirpath,
            db_path=db_path,
            full=args.full,
            git_changed_spec=args.git_changed,
            prune=args.prune,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
