"""Compare arara-rag against the public MTEB-BR leaderboard.

Downloads the per-task result JSONs published at
https://huggingface.co/datasets/MTEB-BR/mteb-pt-results and places arara's own
numbers among them.

Comparability caveat, enforced by ``TASK_MAP``: the leaderboard evaluates
BRTaxQAR with documents truncated to 32,000 characters, so only arara's
``brtaxqa_capped`` runs are placed against it. arara's full-document runs are a
different (and strictly harder to compare) operating point and are reported
separately as a within-arara ablation.

Also note that the leaderboard contains *embedding models only* -- there is no
BM25 baseline on it. arara's lexical mode is therefore not an apples-to-apples
comparison against a dense encoder, and the report says so rather than
implying parity.

    python -m bench.leaderboard
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
CACHE = Path(__file__).parent.parent / ".cache" / "mteb-br-results"

RETRIEVAL_TASKS = ["Quati", "JurisTCU", "BRTaxQAR", "FaQuADIR", "FaqBacenRetrieval", "MedPTRetrieval"]

# Task -> the metric MTEB-BR uses as that task's main score.
MAIN_SCORE = {
    "Quati": "ndcg_at_10",
    "JurisTCU": "ndcg_at_10",
    "BRTaxQAR": "ndcg_at_10",
    "FaQuADIR": "ndcg_at_10",
    "FaqBacenRetrieval": "ndcg_at_10",
    "MedPTRetrieval": "ndcg_at_10",
    "QuatiReranking": "map_at_1000",
    "JurisTCUReranking": "map_at_1000",
}

# arara experiment prefix -> leaderboard task name
TASK_MAP = {
    "quati": "Quati",
    "juristcu": "JurisTCU",
    "brtaxqa_capped": "BRTaxQAR",
    "faquadir": "FaQuADIR",
    "faq_bacen": "FaqBacenRetrieval",
    "quati_reranking": "QuatiReranking",
    "juristcu_reranking": "JurisTCUReranking",
}

RETRIEVAL_ORDER = RETRIEVAL_TASKS
RERANK_ORDER = ["QuatiReranking", "JurisTCUReranking"]


def fetch_leaderboard() -> dict[str, dict[str, float]]:
    """Return ``{task_name: {model_name: main_score}}``."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        "MTEB-BR/mteb-pt-results",
        repo_type="dataset",
        allow_patterns=["results/*/*/*.json"],
        local_dir=str(CACHE),
    )
    out: dict[str, dict[str, float]] = {t: {} for t in MAIN_SCORE}
    for path in Path(root).glob("results/*/*/*.json"):
        model = path.parent.parent.name.replace("__", "/")
        task = path.stem
        if task not in out:
            continue
        try:
            data = json.loads(path.read_text())
            score = data["scores"]["test"][0][MAIN_SCORE[task]]
        except (KeyError, IndexError, json.JSONDecodeError):
            continue
        out[task][model] = float(score)
    return out


def arara_scores() -> dict[str, dict[str, float]]:
    """Return ``{task_name: {experiment: main_score}}`` from local results."""
    out: dict[str, dict[str, float]] = {}
    if not RESULTS_DIR.exists():
        return out
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        task = TASK_MAP.get(rec.get("task", ""))
        if task is None:
            continue
        metric = MAIN_SCORE[task]
        if metric not in rec:
            continue
        out.setdefault(task, {})[rec["experiment"]] = float(rec[metric])
    return out


def main() -> None:
    lb = fetch_leaderboard()
    ours = arara_scores()

    print("# MTEB-BR comparison\n")
    print("## Retrieval (nDCG@10)\n")
    header = (
        "| Task | arara best | arara mode | leaderboard best | score | "
        "leaderboard median | models | arara percentile |"
    )
    print(header)
    print("|" + "|".join(["---"] * 8) + "|")
    for task in RETRIEVAL_ORDER:
        models = lb.get(task, {})
        mine = ours.get(task, {})
        if not mine:
            continue
        best_exp = max(mine, key=lambda k: mine[k])
        best_val = mine[best_exp]
        mode = best_exp.split("_")[-1]
        if models:
            best_model = max(models, key=lambda k: models[k])
            best_lb = models[best_model]
            sorted_scores = sorted(models.values())
            median = sorted_scores[len(sorted_scores) // 2]
            pct = 100.0 * sum(1 for s in sorted_scores if s < best_val) / len(sorted_scores)
            print(
                f"| {task} | {best_val:.4f} | {mode} | {best_model} | {best_lb:.4f} | "
                f"{median:.4f} | {len(models)} | {pct:.0f}th pct |"
            )
        else:
            print(f"| {task} | {best_val:.4f} | {mode} | - | - | - | 0 | - |")

    print("\n## Reranking (MAP@1000)\n")
    print("| Task | arara best | mode | leaderboard best | score | models | percentile |")
    print("|---|---|---|---|---|---|---|")
    for task in RERANK_ORDER:
        models = lb.get(task, {})
        mine = ours.get(task, {})
        if not mine:
            continue
        best_exp = max(mine, key=lambda k: mine[k])
        best_val = mine[best_exp]
        mode = best_exp.split("_")[-1]
        vals = sorted(models.values())
        if vals:
            best_model = max(models, key=lambda k: models[k])
            pct = 100.0 * sum(1 for s in vals if s < best_val) / len(vals)
            print(
                f"| {task} | {best_val:.4f} | {mode} | {best_model} | {models[best_model]:.4f} | "
                f"{len(vals)} | {pct:.0f}th pct |"
            )
        else:
            print(f"| {task} | {best_val:.4f} | {mode} | - | - | 0 | - |")

    print("\n## Where each arara mode lands\n")
    print("| Task | dense (pct) | lexical (pct) | hybrid (pct) |")
    print("|---|---|---|---|")
    for task in RETRIEVAL_ORDER:
        models = lb.get(task, {})
        mine = ours.get(task, {})
        if not mine or not models:
            continue
        vals = sorted(models.values())

        def pct_of(v: float, _vals=vals) -> str:
            pct = 100.0 * sum(1 for s in _vals if s < v) / len(_vals)
            return f"{v:.4f} ({pct:.0f}th pct)"

        cells = []
        for mode in ("dense", "lexical", "hybrid"):
            exp = next((e for e in mine if e.endswith(f"_{mode}")), None)
            cells.append(pct_of(mine[exp]) if exp else "-")
        print(f"| {task} | " + " | ".join(cells) + " |")

    cache_path = RESULTS_DIR / "leaderboard_comparison.json"
    if RESULTS_DIR.exists():
        cache_path.write_text(json.dumps({"leaderboard": lb, "arara": ours}, indent=2))
        print(f"\nwrote {cache_path}")


if __name__ == "__main__":
    main()
