"""Render bench/results/*.json into markdown tables for the README.

    python -m bench.report            # all suites
    python -m bench.report --suite core
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

COLUMNS = [
    ("experiment", "experiment"),
    ("chunks", "chunks"),
    ("ndcg_at_10", "nDCG@10"),
    ("recall_at_100", "R@100"),
    ("mrr_at_10", "MRR@10"),
    ("ms_per_query", "ms/query"),
]


def load(suite: str, base: Path | None = None) -> list[dict]:
    path = (base or RESULTS_DIR) / f"summary_{suite}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def render(records: list[dict], title: str) -> str:
    if not records:
        return f"### {title}\n\n_no results_\n"
    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [f"### {title}", "", header, sep]
    for r in records:
        cells = []
        for key, _ in COLUMNS:
            v = r.get(key, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if key.startswith(("ndcg", "recall", "mrr")) else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def readme_tables(base: Path | None = None) -> str:
    """Emit the two tables the README embeds, straight from the result JSONs."""
    core = load("core", base)
    order = ["brtaxqa_capped", "faquadir", "faq_bacen", "juristcu", "quati"]
    names = {"brtaxqa_capped": "BRTaxQAR (capped)", "faquadir": "FaQuADIR",
             "faq_bacen": "FaqBacenRetrieval", "juristcu": "JurisTCU", "quati": "Quati"}
    by_task: dict[str, dict[str, dict]] = {}
    for r in core:
        by_task.setdefault(r["task"], {})[r["mode"]] = r
    lines = ["| Task | docs | rel./query | MRR@10 | dense | lexical | hybrid | best |", "|---|---|---|---|---|---|---|---|"]
    for t in order:
        modes = by_task.get(t)
        if not modes:
            continue
        cells = {m: modes.get(m, {}).get("ndcg_at_10") for m in ("dense", "lexical", "hybrid")}
        best = max((m for m in cells if cells[m] is not None), key=lambda m: cells[m])
        fmt = lambda v, d=4: f"{v:.{d}f}" if v is not None else "-"  # noqa: E731
        lex = modes.get("lexical", {})
        lines.append(
            f"| {names[t]} | {modes['dense']['corpus_docs']} | "
            f"{lex.get('rel_per_query', '-')} | {fmt(lex.get('mrr_at_10'), 3)} | "
            f"{fmt(cells['dense'])} | {fmt(cells['lexical'])} | {fmt(cells['hybrid'])} | "
            f"**{best}** |"
        )
    out = ["**Cross-task, nDCG@10** (fixed-window chunking; these corpora are mostly "
           "single-chunk documents, so this isolates dense vs lexical vs fusion)\n"]
    out += lines

    nano_base = (base or RESULTS_DIR) / "nanoe5"
    nano_core = load("core", nano_base)
    if nano_core:
        nano_by_task = {}
        for r in nano_core:
            nano_by_task.setdefault(r["task"], {})[r["mode"]] = r
        out.append("\n**Backend comparison, nDCG@10** — same corpus, same chunking\n")
        out.append("| Task | static dense | nanoE5 dense | static hybrid | nanoE5 hybrid | lexical |")
        out.append("|---|---|---|---|---|---|")
        for t in order:
            modes = by_task.get(t)
            nm = nano_by_task.get(t)
            if not modes or not nm:
                continue
            f = lambda v: f"{v:.4f}" if v is not None else "-"  # noqa: E731
            out.append(
                f"| {names[t]} | {f(modes.get('dense', {}).get('ndcg_at_10'))} | "
                f"{f(nm.get('dense', {}).get('ndcg_at_10'))} | "
                f"{f(modes.get('hybrid', {}).get('ndcg_at_10'))} | "
                f"{f(nm.get('hybrid', {}).get('ndcg_at_10'))} | "
                f"{f(modes.get('lexical', {}).get('ndcg_at_10'))} |"
            )
        out.append("")

    ab = load("brtaxqa", base)
    if ab:
        out.append("\n**BRTaxQAR ablation, nDCG@10** — each row changes one thing\n")
        out.append("| Configuration | chunks | nDCG@10 | R@100 |")
        out.append("|---|---|---|---|")
        labels = {
            "brtaxqa_capped_document_dense": "capped at 32k, one vector per doc *(leaderboard setting)*",
            "brtaxqa_capped_window_dense": "capped at 32k, fixed 2500-char windows",
            "brtaxqa_capped_tinyzchunk_dense": "capped at 32k, **tinyzchunk** boundaries",
            "brtaxqa_capped_tinyzchunk_hybrid": "capped at 32k, tinyzchunk + BM25 (hybrid)",
            "brtaxqa_full_document_dense": "**full documents**, one vector per doc",
            "brtaxqa_full_window_dense": "**full documents**, fixed windows",
            "brtaxqa_full_paragraph_dense": "**full documents**, paragraph splits",
            "brtaxqa_full_tinyzchunk_dense": "**full documents**, tinyzchunk",
            "brtaxqa_full_tinyzchunk_hybrid": "**full documents**, tinyzchunk + BM25",
            "brtaxqa_full_tinyzchunk_hybrid_cxm25": "**full documents**, + CXM25 rerank",
        }
        index = {r["experiment"]: r for r in ab}
        for key in labels:
            r = index.get(key)
            if not r:
                continue
            out.append(
                f"| {labels[key]} | {r['chunks']} | **{r['ndcg_at_10']:.4f}** | {r['recall_at_100']:.4f} |"
            )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=None)
    ap.add_argument("--dir", default=None, help="results dir (default bench/results)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--readme", action="store_true", help="emit the README result tables")
    args = ap.parse_args()
    base = Path(args.dir) if args.dir else None

    if args.readme:
        text = readme_tables(base)
        if args.out:
            Path(args.out).write_text(text)
            print(f"wrote {args.out}")
        else:
            print(text)
        return

    suites = [args.suite] if args.suite else ["core", "brtaxqa", "cxm25", "sweep"]
    chunks = []
    for s in suites:
        recs = load(s, base)
        if recs:
            chunks.append(render(recs, f"{s} suite ({len(recs)} experiments)"))
    text = "\n".join(chunks)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
