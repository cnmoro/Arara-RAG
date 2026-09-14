"""Cross-check bench.metrics against pytrec_eval on random judgements.

Run with an interpreter that has pytrec_eval installed:
    python bench/validate_metrics.py
"""
from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")
from bench.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k

try:
    import pytrec_eval
except ImportError:  # pragma: no cover
    print("pytrec_eval not installed; skipping")
    raise SystemExit(0)

random.seed(7)
n_docs, n_queries = 200, 60
doc_ids = [f"d{i}" for i in range(n_docs)]

run, qrels = {}, {}
for qi in range(n_queries):
    qid = f"q{qi}"
    ranked = doc_ids[:]
    random.shuffle(ranked)
    run[qid] = {d: float(len(ranked) - r) for r, d in enumerate(ranked)}
    rel = random.sample(doc_ids, k=random.randint(1, 6))
    qrels[qid] = {d: random.choice([1, 2, 3]) for d in rel}

evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100", "recip_rank"})
ref = evaluator.evaluate(run)

rankings = {
    qid: [d for d, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
    for qid, scores in run.items()
}
mine = evaluate(rankings, qrels, k_values=(10, 100))

def ref_mean(key):
    return sum(v[key] for v in ref.values()) / len(ref)

checks = [
    ("ndcg_at_10", mine["ndcg_at_10"], ref_mean("ndcg_cut_10")),
    ("recall_at_100", mine["recall_at_100"], ref_mean("recall_100")),
]
ok = True
for name, got, want in checks:
    delta = abs(got - want)
    flag = "OK " if delta < 1e-9 else "FAIL"
    if delta >= 1e-9:
        ok = False
    print(f"{flag} {name:15s} arara={got:.12f}  pytrec_eval={want:.12f}  delta={delta:.2e}")

# spot-check per-query ndcg
worst = max(abs(ndcg_at_k(rankings[q], qrels[q], 10) - ref[q]["ndcg_cut_10"]) for q in qrels)
print(f"{'OK ' if worst < 1e-9 else 'FAIL'} per-query max ndcg_at_10 delta = {worst:.2e}")
ok = ok and worst < 1e-9
print("\nRESULT:", "metrics match pytrec_eval" if ok else "MISMATCH")
raise SystemExit(0 if ok else 1)
