#!/usr/bin/env python3
from __future__ import annotations

import argparse
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIRS = [
    PROJECT_ROOT.parent / "ml_research_analysis_2023",
    PROJECT_ROOT.parent / "ml_research_analysis_2024",
    PROJECT_ROOT.parent / "ml_research_analysis_2025",
]


def build_digest_index(source_dirs: list[Path]) -> tuple[dict[str, Path], list[str], list[str]]:
    digest_map: dict[str, Path] = {}
    missing_dirs: list[str] = []
    collisions: list[str] = []

    for source_dir in source_dirs:
        if not source_dir.exists():
            missing_dirs.append(str(source_dir))
            continue

        for md_file in source_dir.rglob("*.md"):
            digest_id = md_file.stem
            if digest_id in digest_map:
                collisions.append(
                    f"{digest_id}\n  - {digest_map[digest_id]}\n  - {md_file.resolve()}"
                )
                continue
            digest_map[digest_id] = md_file.resolve()

    return digest_map, missing_dirs, collisions


class ViewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, digest_map: dict[str, Path], **kwargs):
        self.digest_map = digest_map
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)

        if path == "/view":
            target = "/view/"
            if parsed.query:
                target += f"?{parsed.query}"
            self.send_response(301)
            self.send_header("Location", target)
            self.end_headers()
            return

        if path.startswith("/view/") and path.endswith(".md"):
            tail = path[len("/view/") :]
            if "/" in tail:
                self.send_error(404, "Nested markdown path not supported")
                return
            digest_id = tail[:-3]
            self._serve_raw_markdown(digest_id)
            return

        super().do_GET()

    def _serve_raw_markdown(self, digest_id: str) -> None:
        file_path = self.digest_map.get(digest_id)
        if not file_path:
            self.send_error(404, f"Digest not found: {digest_id}")
            return

        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local static server with /view markdown mapping")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-dir", action="append", dest="source_dirs")
    args = parser.parse_args()

    source_dirs = [Path(p).resolve() for p in (args.source_dirs or DEFAULT_SOURCE_DIRS)]
    digest_map, missing_dirs, collisions = build_digest_index(source_dirs)

    print("Source directories:")
    for src in source_dirs:
        print(f"  - {src}")

    if missing_dirs:
        print("\nWarning: missing source directories:")
        for missing in missing_dirs:
            print(f"  - {missing}")

    if collisions:
        print("\nError: filename collisions detected across source directories:")
        for collision in collisions[:20]:
            print(f"\n{collision}")
        if len(collisions) > 20:
            print(f"\n... and {len(collisions) - 20} more")
        raise SystemExit(1)

    print(f"\nIndexed digests: {len(digest_map):,}")

    handler = lambda *h_args, **h_kwargs: ViewHandler(  # noqa: E731
        *h_args,
        digest_map=digest_map,
        directory=str(args.root),
        **h_kwargs,
    )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving at http://{args.host}:{args.port}/")
    print("Viewer:")
    print("  /view/?id=<digest-id>")
    print("Raw:")
    print("  /view/<digest-id>.md")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
