"""MTEB-BR task loaders.

These replicate the data-loading logic of the corresponding task definitions in
``tardellirs/mteb-br`` (same pinned dataset revisions) so that scores computed
here are comparable to the public leaderboard. The loaders are reimplemented
rather than imported so that arara's benchmark harness does not require the
``mteb``/``torch`` stack.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# Pinned revisions copied from mteb-br task definitions.
QUATI_REPO, QUATI_REV = "MTEB-BR/quati-50k", "5cb87d9561d8ace807305f50e59a4b0af352da2e"
BACEN_REPO, BACEN_REV = "MTEB-BR/faq-bacen", "076d89a68a8b8d2f14e3161631c416ffe29b8463"
FAQUAD_REPO, FAQUAD_REV = "MTEB-BR/faquad-ir", "51fd9e7707bb4971229a0189560379992c3adce2"
JURIS_REPO, JURIS_REV = "LeandroRibeiro/JurisTCU", "ac7bea9e580626a586ffea69b245d26f5a73d44e"
TAXQA_REPO, TAXQA_REV = "unicamp-dl/BR-TaxQA-R", "9f0f4263928c6152a40f888da3e63375a24bb0f3"

# MTEB-BR's BRTaxQAR loader truncates documents here because transformer
# embedders cannot fit them. arara overrides this deliberately.
MTEB_BR_DOC_CAP = 32_000

_TAG = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", _TAG.sub(" ", str(text))).strip()


@dataclass
class RetrievalTask:
    name: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    main_score: str = "ndcg_at_10"
    meta: dict = field(default_factory=dict)


def _load_mteb_format(repo: str, revision: str, name: str) -> RetrievalTask:
    from datasets import load_dataset

    corpus = {
        str(r["_id"]): {"text": r["text"], "title": r.get("title") or ""}
        for r in load_dataset(repo, "corpus", split="test", revision=revision)
    }
    queries = {
        str(r["_id"]): r["text"]
        for r in load_dataset(repo, "queries", split="test", revision=revision)
    }
    qrels: dict[str, dict[str, int]] = {}
    for r in load_dataset(repo, "qrels", split="test", revision=revision):
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    queries = {q: t for q, t in queries.items() if q in qrels}
    return RetrievalTask(name=name, corpus=corpus, queries=queries, qrels=qrels)


def load_quati() -> RetrievalTask:
    return _load_mteb_format(QUATI_REPO, QUATI_REV, "Quati")


def load_faq_bacen() -> RetrievalTask:
    return _load_mteb_format(BACEN_REPO, BACEN_REV, "FaqBacenRetrieval")


def load_faquad_ir() -> RetrievalTask:
    return _load_mteb_format(FAQUAD_REPO, FAQUAD_REV, "FaQuADIR")


def load_juris_tcu() -> RetrievalTask:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def _dl(fname: str) -> str:
        return hf_hub_download(JURIS_REPO, fname, repo_type="dataset", revision=JURIS_REV)

    docs_df = pd.read_csv(_dl("doc.csv"))
    corpus = {}
    for _, row in docs_df.iterrows():
        text = _strip_html(row.get("ENUNCIADO", ""))
        if text:
            corpus[str(row["KEY"])] = {"text": text, "title": ""}

    queries_df = pd.read_csv(_dl("query.csv"))
    queries = {str(r["ID"]): str(r["TEXT"]).strip() for _, r in queries_df.iterrows()}

    qrels_df = pd.read_csv(_dl("qrel.csv"))
    qrels: dict[str, dict[str, int]] = {}
    for _, row in qrels_df.iterrows():
        qrels.setdefault(str(row["QUERY_ID"]), {})[str(row["DOC_ID"])] = int(row["SCORE"])

    queries = {q: t for q, t in queries.items() if q in qrels}
    return RetrievalTask(name="JurisTCU", corpus=corpus, queries=queries, qrels=qrels)


def load_br_taxqa(doc_char_cap: int | None = None) -> RetrievalTask:
    """BR-TaxQA-R, optionally with MTEB-BR's document truncation applied.

    ``doc_char_cap=None`` uses the complete statute text, which is the setting
    arara is designed for and which the leaderboard cannot represent.
    """
    from huggingface_hub import hf_hub_download

    docs_path = hf_hub_download(
        TAXQA_REPO, "referred_legal_documents_QA_2024_v1.1.json",
        repo_type="dataset", revision=TAXQA_REV,
    )
    questions_path = hf_hub_download(
        TAXQA_REPO, "questions_QA_2024_v1.1.json",
        repo_type="dataset", revision=TAXQA_REV,
    )

    with open(docs_path, encoding="utf-8") as fh:
        docs_raw = json.load(fh)
    corpus: dict[str, dict[str, str]] = {}
    n_truncated = 0
    for d in docs_raw:
        fname = (d.get("filename") or "").strip()
        text = (d.get("filedata") or "").strip()
        if doc_char_cap is not None:
            if len(text) > doc_char_cap:
                n_truncated += 1
            text = text[:doc_char_cap]
        if not fname or not text:
            continue
        docid = fname[:-4] if fname.endswith(".txt") else fname
        corpus[docid] = {"text": text, "title": docid}

    with open(questions_path, encoding="utf-8") as fh:
        questions_raw = json.load(fh)
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    for q in questions_raw:
        qid = str(q.get("question_number", "")).strip()
        text = (q.get("question_text") or "").strip()
        if not qid or not text:
            continue
        rel: dict[str, int] = {}
        for key, score in (("formatted_references", 2), ("formatted_embedded_references", 1)):
            for ref in q.get(key) or []:
                f = (ref.get("file") or "").strip()
                if not f:
                    continue
                docid = f[:-4] if f.endswith(".txt") else f
                if docid in corpus and docid not in rel:
                    rel[docid] = score
        if rel:
            queries[qid] = text
            qrels[qid] = rel

    return RetrievalTask(
        name="BRTaxQAR",
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        meta={"doc_char_cap": doc_char_cap, "n_docs_truncated": n_truncated},
    )


TASK_LOADERS = {
    "brtaxqa_full": lambda: load_br_taxqa(doc_char_cap=None),
    "brtaxqa_capped": lambda: load_br_taxqa(doc_char_cap=MTEB_BR_DOC_CAP),
    "faquadir": load_faquad_ir,
    "faq_bacen": load_faq_bacen,
    "juristcu": load_juris_tcu,
    "quati": load_quati,
}


# --------------------------------------------------------------------------
# reranking tasks
# --------------------------------------------------------------------------
RERANK_QUATI_REPO, RERANK_QUATI_REV = (
    "MTEB-BR/quati-reranking",
    "68d40ca9a44e8ea0704fb628f31ace070c16bdbc",
)
RERANK_JURIS_REPO, RERANK_JURIS_REV = (
    "MTEB-BR/juristcu-reranking",
    "83d1eec1aac2ba4e639d72c32a34b4efe70aef82",
)


@dataclass
class RerankingTask:
    """A reranking task: candidate lists are given, only order is decided."""

    name: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    candidates: dict[str, list[str]]
    main_score: str = "map_at_1000"


def _load_reranking(repo: str, revision: str, name: str) -> RerankingTask:
    from datasets import load_dataset

    corpus = {
        str(r["_id"]): {"text": r["text"], "title": r.get("title") or ""}
        for r in load_dataset(repo, "corpus", split="test", revision=revision)
    }
    queries = {
        str(r["_id"]): r["text"]
        for r in load_dataset(repo, "queries", split="test", revision=revision)
    }
    qrels: dict[str, dict[str, int]] = {}
    for r in load_dataset(repo, "qrels", split="test", revision=revision):
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    candidates: dict[str, list[str]] = {}
    for r in load_dataset(repo, "top_ranked", split="test", revision=revision):
        candidates[str(r["query-id"])] = [str(d) for d in r["corpus-ids"]]

    qids = [q for q in queries if q in qrels and candidates.get(q)]
    return RerankingTask(
        name=name,
        corpus=corpus,
        queries={q: queries[q] for q in qids},
        qrels={q: qrels[q] for q in qids},
        candidates={q: candidates[q] for q in qids},
    )


def load_quati_reranking() -> RerankingTask:
    return _load_reranking(RERANK_QUATI_REPO, RERANK_QUATI_REV, "QuatiReranking")


def load_juristcu_reranking() -> RerankingTask:
    return _load_reranking(RERANK_JURIS_REPO, RERANK_JURIS_REV, "JurisTCUReranking")


RERANK_LOADERS = {
    "quati_reranking": load_quati_reranking,
    "juristcu_reranking": load_juristcu_reranking,
}
