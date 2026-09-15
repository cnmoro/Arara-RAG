"""Reranking evaluation on MTEB-BR.

MTEB-BR's reranking tasks hand you a fixed candidate list per query (BM25 hard
negatives) and score only the resulting order, with MAP@1000 as the main score.
That makes the ``identity`` row the BM25 baseline the benchmark ships with, and
every other row a pure reranking gain.

    python -m bench.rerank
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from arara_rag import Arara

from .metrics import evaluate_reranking
from .tasks import RERANK_LOADERS

RESULTS_DIR = Path(__file__).parent / "results"
MODES = ["identity", "lexical", "dense", "hybrid", "cxm25"]
TASKS = ["quati_reranking", "juristcu_reranking"]


def run_task(task_key: str, modes: list[str], chunk: str = "window") -> list[dict]:
    print(f"  loading {task_key} ...", flush=True)
    task = RERANK_LOADERS[task_key]()
    print(
        f"  {len(task.corpus)} docs, {len(task.queries)} queries, "
        f"mean candidates/query "
        f"{sum(len(v) for v in task.candidates.values()) / max(len(task.candidates), 1):.1f}",
        flush=True,
    )

    t0 = time.perf_counter()
    arara = Arara(chunk_mode=chunk)
    arara.add_documents({d: v["text"] for d, v in task.corpus.items()})
    arara.finalize(build_cxm25="cxm25" in modes, n_jobs=8)
    build_s = time.perf_counter() - t0
    print(f"    index: {arara.n_chunks} chunks in {build_s:.1f}s", flush=True)

    records = []
    for mode in modes:
        t0 = time.perf_counter()
        rankings = {}
        for qid, query in task.queries.items():
            cands = task.candidates[qid]
            if mode == "identity":
                rankings[qid] = cands
            else:
                rankings[qid] = arara.rerank(query, cands, mode=mode)
        elapsed = time.perf_counter() - t0
        metrics = evaluate_reranking(rankings, task.qrels, k=1000)
        rec = {
            "experiment": f"{task_key}_{mode}",
            "task": task_key,
            "task_name": task.name,
            "mode": mode,
            "chunk": chunk,
            "corpus_docs": len(task.corpus),
            "queries": len(task.queries),
            "chunks": arara.n_chunks,
            "index_build_s": round(build_s, 2),
            "ms_per_query": round(elapsed / max(len(task.queries), 1) * 1000, 2),
            **{k: round(v, 4) for k, v in metrics.items()},
        }
        records.append(rec)
        print(
            f"    {mode:9s} map@1000={rec.get('map_at_1000')} "
            f"({rec['ms_per_query']} ms/query)",
            flush=True,
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    for task_key in TASKS:
        print(f"[{task_key}]", flush=True)
        recs = run_task(task_key, MODES)
        all_records.extend(recs)
        for rec in recs:
            (out_dir / f"{rec['experiment']}.json").write_text(json.dumps(rec, indent=2))
    out = out_dir / "summary_rerank.json"
    out.write_text(json.dumps(all_records, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
