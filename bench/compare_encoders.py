"""Side-by-side comparison of the dense encoder backends.

    python -m bench.compare_encoders
    python -m bench.compare_encoders --a static --b use

Reads ``bench/results/<encoder>/summary_*.json`` and prints the deltas that
decide which encoder becomes the default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

RETRIEVAL_ORDER = [
    ("brtaxqa_capped", "BRTaxQAR (capped)"),
    ("faquadir", "FaQuADIR"),
    ("faq_bacen", "FaqBacenRetrieval"),
    ("juristcu", "JurisTCU"),
    ("quati", "Quati"),
]


def load(encoder: str, suite: str) -> list[dict]:
    path = RESULTS_DIR / encoder / f"summary_{suite}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def index(records: list[dict]) -> dict[str, dict]:
    return {r["experiment"]: r for r in records if "experiment" in r}


def fmt(v, digits: int = 4) -> str:
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "-"


def delta(a, b, digits: int = 4) -> str:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "-"
    d = b - a
    mark = "+" if d > 0 else ""
    return f"{mark}{d:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="static", help="baseline encoder (default: static)")
    ap.add_argument("--b", default="use", help="challenger encoder (default: use)")
    args = ap.parse_args()

    a_core, b_core = index(load(args.a, "core")), index(load(args.b, "core"))

    print(f"# Dense encoder comparison: {args.a} (baseline) vs {args.b} (challenger)\n")
    print("## Retrieval, nDCG@10\n")
    print(f"| Task | mode | {args.a} | {args.b} | Δ |")
    print("|---|---|---|---|---|")
    for task, label in RETRIEVAL_ORDER:
        for mode in ("dense", "lexical", "hybrid"):
            key = f"{task}_{mode}"
            ra, rb = a_core.get(key), b_core.get(key)
            if not ra or not rb:
                continue
            va, vb = ra.get("ndcg_at_10"), rb.get("ndcg_at_10")
            print(f"| {label} | {mode} | {fmt(va)} | {fmt(vb)} | {delta(va, vb)} |")

    a_ab, b_ab = index(load(args.a, "brtaxqa")), index(load(args.b, "brtaxqa"))
    rows = [k for k in a_ab if k in b_ab and a_ab[k].get("mode") in ("dense", "hybrid")]
    if rows:
        print("\n## BRTaxQAR ablation, nDCG@10\n")
        print(f"| Configuration | chunks | {args.a} | {args.b} | Δ |")
        print("|---|---|---|---|---|")
        for key in sorted(rows, key=lambda k: (a_ab[k]["chunk"], a_ab[k]["mode"])):
            ra, rb = a_ab[key], b_ab[key]
            va, vb = ra.get("ndcg_at_10"), rb.get("ndcg_at_10")
            print(
                f"| {ra['chunk']} / {ra['mode']} | {ra['chunks']} | "
                f"{fmt(va)} | {fmt(vb)} | {delta(va, vb)} |"
            )

    a_rr, b_rr = index(load(args.a, "rerank")), index(load(args.b, "rerank"))
    rr_rows = [k for k in a_rr if k in b_rr]
    if rr_rows:
        print("\n## Reranking, MAP@1000\n")
        print(f"| Task | mode | {args.a} | {args.b} | Δ |")
        print("|---|---|---|---|---|")
        for key in sorted(rr_rows):
            ra, rb = a_rr[key], b_rr[key]
            va, vb = ra.get("map_at_1000"), rb.get("map_at_1000")
            print(f"| {ra.get('task_name', ra['task'])} | {ra['mode']} | {fmt(va)} | {fmt(vb)} | {delta(va, vb)} |")

    # Build cost, which is where the two backends differ most.
    print("\n## Index build cost\n")
    print(f"| Task | chunks | {args.a} build (s) | {args.b} build (s) |")
    print("|---|---|---|---|")
    seen = set()
    for key, ra in a_core.items():
        rb = b_core.get(key)
        if not rb:
            continue
        sig = (ra["task"], ra["chunk"])
        if sig in seen:
            continue
        seen.add(sig)
        print(
            f"| {ra['task']} / {ra['chunk']} | {ra['chunks']} | "
            f"{ra['index_build_s']} | {rb['index_build_s']} |"
        )

    # Summary verdict.
    deltas = []
    for task, _ in RETRIEVAL_ORDER:
        ra, rb = a_core.get(f"{task}_dense"), b_core.get(f"{task}_dense")
        if ra and rb:
            deltas.append(rb["ndcg_at_10"] - ra["ndcg_at_10"])
    if deltas:
        wins = sum(1 for d in deltas if d > 0)
        print(
            f"\n**Dense retrieval: {args.b} wins {wins}/{len(deltas)} tasks, "
            f"mean Δ nDCG@10 = {sum(deltas)/len(deltas):+.4f}**"
        )


if __name__ == "__main__":
    main()
