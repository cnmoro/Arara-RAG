"""Run arara-rag against MTEB-BR retrieval tasks.

    python -m bench.run --suite core
    python -m bench.run --suite brtaxqa
    python -m bench.run --suite cxm25
    python -m bench.run --suite all

Results are written to ``bench/results/`` as per-experiment JSON plus a combined
``summary.json``. Indexing is cached per (task, chunk config) so several
retrieval modes reuse one index build.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from arara_rag import Arara

from .metrics import evaluate
from .tasks import TASK_LOADERS, RetrievalTask

RESULTS_DIR = Path(__file__).parent / "results"
DEPTH = 100


@dataclass
class Experiment:
    name: str
    task: str
    chunk: str = "none"
    mode: str = "hybrid"
    max_chunk_chars: int = 2500
    cxm25: bool = False
    dense_weight: float = 1.0
    lexical_weight: float = 1.0


CORE_TASKS = ["brtaxqa_capped", "faquadir", "faq_bacen", "juristcu", "quati"]

SUITES: dict[str, list[Experiment]] = {
    # Cross-task comparison with the cheapest sane chunking (fixed windows).
    # On these corpora most documents are shorter than the window, so chunking
    # is close to a no-op and the table isolates dense vs lexical vs hybrid.
    "core": [
        Experiment(f"{t}_{m}", task=t, chunk="window", mode=m)
        for t in CORE_TASKS
        for m in ("dense", "lexical", "hybrid")
    ],
    # BRTaxQAR isolates each decision in turn: truncate or not, and chunk or not.
    #   document = one vector per document (the leaderboard's operating point)
    #   window   = fixed 2500-char word-snapped windows
    #   tinyzchunk = learned boundaries
    "brtaxqa": [
        Experiment("brtaxqa_capped_document_dense", "brtaxqa_capped", "document", "dense"),
        Experiment("brtaxqa_capped_window_dense", "brtaxqa_capped", "window", "dense"),
        Experiment("brtaxqa_capped_tinyzchunk_dense", "brtaxqa_capped", "tinyzchunk", "dense"),
        Experiment("brtaxqa_capped_tinyzchunk_hybrid", "brtaxqa_capped", "tinyzchunk", "hybrid"),
        Experiment("brtaxqa_full_document_dense", "brtaxqa_full", "document", "dense"),
        Experiment("brtaxqa_full_window_dense", "brtaxqa_full", "window", "dense"),
        Experiment("brtaxqa_full_paragraph_dense", "brtaxqa_full", "paragraph", "dense"),
        Experiment("brtaxqa_full_tinyzchunk_dense", "brtaxqa_full", "tinyzchunk", "dense"),
        Experiment("brtaxqa_full_tinyzchunk_hybrid", "brtaxqa_full", "tinyzchunk", "hybrid"),
        Experiment("brtaxqa_full_tinyzchunk_hybrid_cxm25", "brtaxqa_full", "tinyzchunk",
                   "hybrid_cxm25", cxm25=True),
    ],
    # Does the CXM25 lexical reranker add anything on top of RRF?
    "cxm25": [
        Experiment("cxm25_brtaxqa_hybrid", "brtaxqa_full", "tinyzchunk", "hybrid"),
        Experiment("cxm25_brtaxqa_hybrid_cxm25", "brtaxqa_full", "tinyzchunk",
                   "hybrid_cxm25", cxm25=True),
        Experiment("cxm25_faquadir_hybrid", "faquadir", "window", "hybrid"),
        Experiment("cxm25_faquadir_hybrid_cxm25", "faquadir", "window", "hybrid_cxm25", cxm25=True),
        Experiment("cxm25_faq_bacen_hybrid", "faq_bacen", "window", "hybrid"),
        Experiment("cxm25_faq_bacen_hybrid_cxm25", "faq_bacen", "window", "hybrid_cxm25", cxm25=True),
        Experiment("cxm25_quati_hybrid", "quati", "window", "hybrid"),
        Experiment("cxm25_quati_hybrid_cxm25", "quati", "window", "hybrid_cxm25", cxm25=True),
    ],
    # Equal-weight RRF loses to lexical-only on every task, so the weights are
    # swept here rather than left at the default.
    "sweep": [
        Experiment(f"sweep_{t}_w{dw:g}x{lw:g}", task=t, chunk="window", mode="hybrid",
                   dense_weight=dw, lexical_weight=lw)
        for t in ("faquadir", "faq_bacen", "brtaxqa_capped", "juristcu")
        for dw, lw in ((1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (1.0, 5.0))
    ],
}

# Chunk configurations keyed by the experiment's ``chunk`` field.
def _build_index(task: RetrievalTask, chunk: str, max_chunk_chars: int) -> tuple[Arara, float, dict]:
    t0 = time.perf_counter()
    arara = Arara(chunk_mode=chunk, max_chunk_chars=max_chunk_chars, candidate_k=DEPTH)
    arara.add_documents({doc_id: d["text"] for doc_id, d in task.corpus.items()})
    arara.finalize()
    build_s = time.perf_counter() - t0
    return arara, build_s, arara.stats()


def run_experiment(
    exp: Experiment,
    cache: dict[tuple[str, str, int, str], tuple[Arara, float, dict]],
    task_cache: dict[str, RetrievalTask],
) -> dict:
    task_key = exp.task
    if task_key not in task_cache:
        print(f"  loading task {task_key} ...", flush=True)
        task_cache[task_key] = TASK_LOADERS[task_key]()
    task = task_cache[task_key]

    key = (task_key, exp.chunk, exp.max_chunk_chars)
    if key not in cache:
        print(
            f"  building index  task={task_key} chunk={exp.chunk} "
            f"docs={len(task.corpus)} ...",
            flush=True,
        )
        cache[key] = _build_index(task, exp.chunk, exp.max_chunk_chars)
        _, build_s, stats = cache[key]
        print(
            f"    -> {stats['chunks']} chunks in {build_s:.1f}s "
            f"(dense {stats['vector_bytes']/1e6:.1f} MB, lex {stats['lexical_bytes']/1e6:.1f} MB)",
            flush=True,
        )
    arara, build_s, stats = cache[key]

    if exp.cxm25 and arara._cxm25 is None:
        print("    building CXM25 scorer ...", flush=True)
        arara.finalize(build_cxm25=True, n_jobs=8)

    if exp.cxm25 and arara._cxm25 is None:
        raise RuntimeError(f"CXM25 unavailable for {exp.name}")

    # Fusion weights only affect search, so a cached index is reused and the
    # weights are set on the shared instance before each run.
    arara.fusion_weights = (exp.dense_weight, exp.lexical_weight)

    t0 = time.perf_counter()
    rankings = {
        qid: arara.retrieve_ranking(text, depth=DEPTH, mode=exp.mode)  # type: ignore[arg-type]
        for qid, text in task.queries.items()
    }
    query_s = time.perf_counter() - t0

    metrics = evaluate(rankings, task.qrels, k_values=(10, 100))
    record = {
        "experiment": exp.name,
        "task": task_key,
        "chunk": exp.chunk,
        "mode": exp.mode,
        "dense_weight": exp.dense_weight,
        "lexical_weight": exp.lexical_weight,
        "max_chunk_chars": exp.max_chunk_chars,
        "corpus_docs": len(task.corpus),
        "queries": len(task.queries),
        "chunks": stats["chunks"],
        "index_build_s": round(build_s, 2),
        "query_total_s": round(query_s, 2),
        "ms_per_query": round(query_s / max(len(task.queries), 1) * 1000, 2),
        "vector_bytes": stats["vector_bytes"],
        "lexical_bytes": stats["lexical_bytes"],
        **{k: round(v, 4) for k, v in metrics.items()},
    }
    if task.meta:
        record["task_meta"] = task.meta
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="core", choices=sorted(SUITES) + ["all"])
    ap.add_argument("--out", default=str(RESULTS_DIR))
    ap.add_argument("--max-queries", type=int, default=None,
                    help="subsample queries (for smoke runs); reported in output")
    args = ap.parse_args()

    exps = SUITES[args.suite] if args.suite != "all" else [
        e for s in ("core", "brtaxqa", "cxm25", "sweep") for e in SUITES[s]
    ]
    if args.max_queries is None:
        exps = list({e.name: e for e in exps}.values())
    else:
        exps = list({e.name: e for e in exps}.values())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_cache: dict[str, RetrievalTask] = {}
    cache: dict[tuple[str, str, int], tuple[Arara, float, dict]] = {}
    records = []
    for exp in exps:
        print(f"[{exp.name}]", flush=True)
        if args.max_queries and exp.task not in task_cache:
            task_cache[exp.task] = TASK_LOADERS[exp.task]()
            t = task_cache[exp.task]
            t.queries = dict(list(t.queries.items())[: args.max_queries])
            t.qrels = {q: v for q, v in t.qrels.items() if q in t.queries}
        rec = run_experiment(exp, cache, task_cache)
        if args.max_queries:
            rec["max_queries"] = args.max_queries
        records.append(rec)
        print(
            f"  ndcg@10={rec.get('ndcg_at_10')} recall@100={rec.get('recall_at_100')} "
            f"mrr@10={rec.get('mrr_at_10')} ({rec['ms_per_query']} ms/query)",
            flush=True,
        )
        (out_dir / f"{exp.name}.json").write_text(json.dumps(rec, indent=2))

    summary_path = out_dir / f"summary_{args.suite}.json"
    summary_path.write_text(json.dumps(records, indent=2))

    # Always also write one summary per suite, so that `--suite all` leaves the
    # same artifacts as running the suites separately. Report tooling reads
    # summary_core.json / summary_brtaxqa.json / summary_cxm25.json /
    # summary_sweep.json, not summary_all.json.
    suite_of = {e.name: name for name, exps in SUITES.items() for e in exps}
    by_suite: dict[str, list[dict]] = {}
    for rec in records:
        suite_name = suite_of.get(rec["experiment"])
        if suite_name:
            by_suite.setdefault(suite_name, []).append(rec)
    for suite_name, recs in by_suite.items():
        (out_dir / f"summary_{suite_name}.json").write_text(json.dumps(recs, indent=2))

    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
