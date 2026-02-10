#!/usr/bin/env python3
"""
Search topic-relevant digests from analysis_outputs/research_index.sqlite.

Why this exists:
- The `tags` field contains a fixed tail of generic tags in every record.
- So filtering by raw tags alone can produce unusable results.

This script uses a multi-field strategy over:
1) title
2) core_contribution
3) filename slug
4) dynamic tags only (first N tags, default N=5)

Usage examples:
  python scripts/search_topic.py --topic "mixture of experts" --alias moe
  python scripts/search_topic.py --topic quantization --mode strict
  python scripts/search_topic.py --topic "reinforcement learning" --alias rl --limit 25 --json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DB = "analysis_outputs/research_index.sqlite"


@dataclass
class MatchResult:
    filename: str
    title: str
    score: int
    matched_in: list[str]


def parse_aliases(raw_aliases: list[str]) -> list[str]:
    aliases: list[str] = []
    for item in raw_aliases:
        for part in item.split(","):
            alias = part.strip().lower()
            if alias and alias not in aliases:
                aliases.append(alias)
    return aliases


def parse_tags(tags_text: str | None) -> list[str]:
    if not tags_text:
        return []
    try:
        parsed = json.loads(tags_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            out.append(item.lower())
        else:
            out.append(str(item).lower())
    return out


def contains_topic(text: str, phrase: str, hyphen_phrase: str) -> bool:
    t = text.lower()
    return phrase in t or (hyphen_phrase != phrase and hyphen_phrase in t)


def alias_regexes(aliases: Iterable[str]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for alias in aliases:
        escaped = re.escape(alias)
        patterns.append(re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"))
    return patterns


def has_alias(text: str, patterns: list[re.Pattern[str]]) -> bool:
    t = text.lower()
    return any(p.search(t) for p in patterns)


def evaluate_row(
    row: sqlite3.Row,
    phrase: str,
    hyphen_phrase: str,
    alias_patterns: list[re.Pattern[str]],
    dynamic_tag_count: int,
    mode: str,
) -> MatchResult | None:
    filename = (row["filename"] or "").strip()
    title = (row["title"] or "").strip()
    core = (row["core_contribution"] or "").strip()
    tags = parse_tags(row["tags"])
    dynamic_tags = tags[:dynamic_tag_count]

    title_phrase = contains_topic(title, phrase, hyphen_phrase)
    core_phrase = contains_topic(core, phrase, hyphen_phrase)
    filename_phrase = contains_topic(filename, phrase, hyphen_phrase)
    dyn_tags_phrase = any(contains_topic(tag, phrase, hyphen_phrase) for tag in dynamic_tags)

    title_alias = has_alias(title, alias_patterns) if alias_patterns else False
    core_alias = has_alias(core, alias_patterns) if alias_patterns else False
    filename_alias = has_alias(filename, alias_patterns) if alias_patterns else False
    dyn_tags_alias = any(has_alias(tag, alias_patterns) for tag in dynamic_tags) if alias_patterns else False

    alias_any = title_alias or core_alias or filename_alias or dyn_tags_alias

    matched_in: list[str] = []
    if title_phrase:
        matched_in.append("title")
    if core_phrase:
        matched_in.append("core")
    if filename_phrase:
        matched_in.append("filename")
    if dyn_tags_phrase:
        matched_in.append("tags[0:5]")
    if alias_any:
        matched_in.append("alias")

    if mode == "strict":
        # High precision: title/filename signals only.
        hit = title_phrase or filename_phrase or title_alias or filename_alias
        if not hit:
            return None
        score = int(title_phrase) + int(filename_phrase) + int(title_alias) + int(filename_alias)
        return MatchResult(filename=filename, title=title, score=score, matched_in=matched_in)

    # Broad mode
    hit = title_phrase or core_phrase or filename_phrase or dyn_tags_phrase or alias_any
    if not hit:
        return None

    score = (
        3 * int(title_phrase)
        + 2 * int(core_phrase)
        + 2 * int(filename_phrase)
        + 1 * int(dyn_tags_phrase)
        + 1 * int(alias_any)
    )
    return MatchResult(filename=filename, title=title, score=score, matched_in=matched_in)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search topic-specific papers in research index")
    parser.add_argument("--topic", required=True, help="Topic phrase, e.g. 'mixture of experts'")
    parser.add_argument("--alias", action="append", default=[], help="Optional alias/acronym (repeat or comma-separate), e.g. --alias moe")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite path (default: {DEFAULT_DB})")
    parser.add_argument("--mode", choices=["broad", "strict"], default="broad", help="broad=recall, strict=precision")
    parser.add_argument("--limit", type=int, default=50, help="Max rows to print")
    parser.add_argument("--dynamic-tag-count", type=int, default=5, help="How many leading tags to treat as dynamic (default 5)")
    parser.add_argument("--min-score", type=int, default=1, help="Minimum score (broad mode only)")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    phrase = args.topic.strip().lower()
    if not phrase:
        raise SystemExit("--topic cannot be empty")

    hyphen_phrase = re.sub(r"\s+", "-", phrase)
    aliases = parse_aliases(args.alias)
    alias_patterns = alias_regexes(aliases)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT filename, title, core_contribution, tags FROM papers"
    ).fetchall()
    conn.close()

    matches: list[MatchResult] = []
    for row in rows:
        result = evaluate_row(
            row=row,
            phrase=phrase,
            hyphen_phrase=hyphen_phrase,
            alias_patterns=alias_patterns,
            dynamic_tag_count=max(0, args.dynamic_tag_count),
            mode=args.mode,
        )
        if result is None:
            continue
        if args.mode == "broad" and result.score < args.min_score:
            continue
        matches.append(result)

    if args.mode == "broad":
        matches.sort(key=lambda m: (-m.score, m.title.lower(), m.filename.lower()))
    else:
        matches.sort(key=lambda m: (m.title.lower(), m.filename.lower()))

    limited = matches[: max(args.limit, 0)]

    summary = {
        "db": str(db_path),
        "topic": args.topic,
        "aliases": aliases,
        "mode": args.mode,
        "scanned": len(rows),
        "matches": len(matches),
        "returned": len(limited),
    }

    if args.json:
        payload = {
            "summary": summary,
            "results": [
                {
                    "score": m.score,
                    "title": m.title,
                    "filename": m.filename,
                    "matched_in": m.matched_in,
                }
                for m in limited
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    aliases_text = ", ".join(aliases) if aliases else "(none)"
    print(
        f"topic='{args.topic}' | aliases={aliases_text} | mode={args.mode} | "
        f"scanned={len(rows)} | matches={len(matches)} | returned={len(limited)}"
    )
    print()

    if not limited:
        print("No matches.")
        return

    for idx, m in enumerate(limited, start=1):
        hit_fields = ",".join(m.matched_in)
        print(f"{idx:>3}. [score={m.score}] {m.title}")
        print(f"     {m.filename}")
        print(f"     hits: {hit_fields}")


if __name__ == "__main__":
    main()
