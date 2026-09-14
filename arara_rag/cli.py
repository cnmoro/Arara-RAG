"""Command line interface.

Index a handful of files and search them in one shot -- useful for demos and
for checking retrieval quality before wiring the stack into a service.

    python -m arara_rag search "qual a alíquota do imposto de renda?" docs/*.txt
    python -m arara_rag search "licenciamento ambiental" docs/ --json -k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import Arara

TEXT_SUFFIXES = {".txt", ".md", ".json", ".csv", ".html", ".xml", ".rst"}


def _iter_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(
                f for f in sorted(path.rglob("*")) if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES
            )
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: {p} is not a file or directory", file=sys.stderr)
    return files


def _cmd_search(args: argparse.Namespace) -> int:
    files = _iter_files(args.paths)
    if not files:
        print("no documents found", file=sys.stderr)
        return 2

    docs = {}
    for f in files:
        try:
            docs[str(f)] = f.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover
            print(f"warning: could not read {f}: {exc}", file=sys.stderr)

    arara = Arara(chunk_mode=args.chunk, max_chunk_chars=args.max_chunk_chars)
    arara.add_documents(docs, show_progress=not args.quiet)
    arara.finalize(build_cxm25=args.mode == "hybrid_cxm25", n_jobs=args.jobs)

    hits = arara.search(args.query, top_k=args.k, mode=args.mode)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "doc_id": h.doc_id,
                        "score": h.score,
                        "chunk_id": h.chunk_id,
                        "start": h.start,
                        "end": h.end,
                        "text": h.text,
                    }
                    for h in hits
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    stats = arara.stats()
    print(
        f"{stats['documents']} documents -> {stats['chunks']} chunks "
        f"({stats['chunk_mode']} chunking, mode={args.mode})\n",
        file=sys.stderr,
    )
    for i, h in enumerate(hits, start=1):
        preview = (h.text or "").replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "..."
        print(f"{i}. [{h.score:.4f}] {h.doc_id}  ({h.start}-{h.end})")
        print(f"   {preview}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="arara", description="CPU-only Portuguese RAG retrieval.")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("search", help="index documents and run one query")
    s.add_argument("query")
    s.add_argument("paths", nargs="+", help="files or directories to index")
    s.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    s.add_argument(
        "--mode",
        default="hybrid",
        choices=["dense", "lexical", "hybrid", "hybrid_cxm25"],
        help="retrieval mode (default hybrid)",
    )
    s.add_argument(
        "--chunk",
        default="tinyzchunk",
        choices=["tinyzchunk", "paragraph", "none"],
        help="chunking strategy (default tinyzchunk)",
    )
    s.add_argument("--max-chunk-chars", type=int, default=2500)
    s.add_argument("--json", action="store_true", help="emit results as JSON with offsets")
    s.add_argument("--jobs", type=int, default=1, help="workers for CXM25 statistics")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=_cmd_search)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
