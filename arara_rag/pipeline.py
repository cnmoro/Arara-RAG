"""The arara-rag pipeline: chunk, embed, index, retrieve, fuse, rerank."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Literal

import numpy as np

from .chunk import Chunker, canonicalize
from .dense import DEFAULT_DENSE_MODEL, DenseEncoder, DenseIndex
from .fuse import rank_from_scores, reciprocal_rank_fusion
from .lexical import BM25Index, CXM25Scorer
from .text import Tokenizer
from .types import Chunk, Hit

SearchMode = Literal["dense", "lexical", "hybrid", "hybrid_cxm25"]


def _normalize_input(docs) -> list[tuple[str, str]]:
    if isinstance(docs, Mapping):
        items = list(docs.items())
    else:
        items = []
        for i, item in enumerate(docs):
            if isinstance(item, str):
                items.append((str(i), item))
            else:
                doc_id, text = item
                items.append((str(doc_id), text))
    out: list[tuple[str, str]] = []
    for doc_id, text in items:
        if text is None:
            continue
        out.append((str(doc_id), str(text)))
    return out


class Arara:
    """A CPU-first hybrid retrieval stack for Portuguese and English.

    Example:
        >>> from arara_rag import Arara
        >>> a = Arara()
        >>> a.add_documents({"lei": "Art. 1o Esta lei ..." * 50})
        >>> a.search("o que diz o artigo primeiro?", top_k=3)

    Args:
        dense_model: HuggingFace id of a Model2Vec static embedding model.
        chunk_mode: ``"tinyzchunk"`` (default), ``"paragraph"`` or ``"none"``.
        max_chunk_chars: hard ceiling passed to the chunker.
        min_chunk_chars: chunks below this are merged away by tinyzchunk.
        candidate_k: how many chunks each retriever contributes to fusion.
        rrf_k: RRF damping constant.
        lexical: ``"bm25"`` (numpy, full-corpus) -- the fast first stage.
        cxm25: build the optional CXM25 scorer for the ``hybrid_cxm25`` mode.
    """

    def __init__(
        self,
        dense_model: str = DEFAULT_DENSE_MODEL,
        chunk_mode: str = "tinyzchunk",
        max_chunk_chars: int = 2500,
        min_chunk_chars: int = 100,
        candidate_k: int = 100,
        rrf_k: int = 60,
        fusion_weights: tuple[float, float] = (1.0, 1.0),
        lang: str = "pt",
        cache_dir: str | None = None,
        encoder: DenseEncoder | None = None,
    ) -> None:
        self.dense_model = dense_model
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        # (dense, lexical) weights for reciprocal rank fusion.
        self.fusion_weights = fusion_weights
        self.lang = lang
        self.chunker = Chunker(
            mode=chunk_mode,  # type: ignore[arg-type]
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars,
        )
        self.tokenizer = Tokenizer(lang)
        self._encoder = encoder
        self._cache_dir = cache_dir

        self._doc_ids: list[str] = []
        self._doc_text: dict[str, str] = {}
        self._doc_to_chunks: dict[str, list[int]] = {}
        self._chunks: list[Chunk] = []
        self._chunk_doc_idx = np.empty(0, dtype=np.int64)
        self._dense: DenseIndex | None = None
        self._lex: BM25Index = BM25Index()
        self._cxm25: CXM25Scorer | None = None
        self.timings: dict[str, float] = {}

    # -- indexing -----------------------------------------------------------
    @property
    def encoder(self) -> DenseEncoder:
        if self._encoder is None:
            self._encoder = DenseEncoder(self.dense_model, cache_dir=self._cache_dir)
        return self._encoder

    def __len__(self) -> int:
        return len(self._doc_ids)

    def document_text(self, doc_id: str) -> str:
        """The canonical text that chunk offsets index into.

        Line endings are normalised to ``\\n``; the text is otherwise identical
        to what was passed to :meth:`add_documents`.
        """
        return self._doc_text[doc_id]

    def resolve(self, hit: Hit) -> str:
        """Return the exact source text a hit points at."""
        if hit.start is None or hit.end is None:
            raise ValueError("hit has no offsets")
        return self._doc_text[hit.doc_id][hit.start : hit.end]

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    def add_documents(self, docs, show_progress: bool = False) -> int:
        """Chunk, embed and index documents. Returns the number of chunks added."""
        items = _normalize_input(docs)
        if not items:
            return 0

        t0 = time.perf_counter()
        new_chunks: list[Chunk] = []
        new_doc_idx: list[int] = []
        for doc_id, text in items:
            if doc_id in self._doc_ids:
                raise ValueError(f"duplicate doc_id: {doc_id!r}")
            base = len(self._doc_ids)
            self._doc_ids.append(doc_id)
            canonical = canonicalize(text)
            self._doc_text[doc_id] = canonical
            idxs: list[int] = []
            for chunk in self.chunker.split(doc_id, canonical):
                idxs.append(len(new_chunks))
                new_chunks.append(chunk)
                new_doc_idx.append(base)
            self._doc_to_chunks[doc_id] = idxs
        self.timings["chunk_s"] = time.perf_counter() - t0
        if not new_chunks:
            return 0

        t0 = time.perf_counter()
        vectors = self.encoder.encode(
            [c.text for c in new_chunks], show_progress=show_progress
        )
        self.timings["encode_s"] = time.perf_counter() - t0

        if self._dense is None:
            self._dense = DenseIndex(dim=vectors.shape[1])
        self._dense.add(vectors)

        t0 = time.perf_counter()
        terms = [self.tokenizer.terms(c.text) for c in new_chunks]
        self._lex.add(terms)
        self.timings["tokenize_s"] = time.perf_counter() - t0

        self._chunks.extend(new_chunks)
        self._chunk_doc_idx = np.concatenate(
            [self._chunk_doc_idx, np.asarray(new_doc_idx, dtype=np.int64)]
        )
        return len(new_chunks)

    def finalize(self, build_cxm25: bool = False, n_jobs: int = 1) -> "Arara":
        """Build derived structures. Call once after all documents are added."""
        self._lex.finalize()
        if self._dense is not None:
            self._dense.finalize()
        if build_cxm25 and self._cxm25 is None and self._chunks:
            t0 = time.perf_counter()
            self._cxm25 = CXM25Scorer(
                [c.text for c in self._chunks], tokenizer=self.tokenizer, n_jobs=n_jobs
            )
            self.timings["cxm25_build_s"] = time.perf_counter() - t0
        return self

    # -- retrieval ----------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: SearchMode = "hybrid",
        aggregate: str = "max",
    ) -> list[Hit]:
        """Retrieve documents for ``query``.

        Chunk scores are aggregated to document level with ``max``, the standard
        choice when a document is indexed as several spans.
        """
        if not self._chunks:
            return []
        if mode == "hybrid_cxm25" and self._cxm25 is None:
            self.finalize(build_cxm25=True)

        rankings: list[list[int]] = []
        weights: list[float] = []
        single_scores: dict[int, float] | None = None
        chunk_scores: dict[int, float]

        if mode in ("dense", "hybrid", "hybrid_cxm25"):
            qv = self.encoder.encode([query])[0]
            ds, didx = self._dense.search(qv, self.candidate_k)  # type: ignore[union-attr]
            rankings.append([int(i) for i in didx])
            weights.append(self.fusion_weights[0])
            if mode == "dense":
                single_scores = {int(i): float(s) for s, i in zip(ds, didx)}

        if mode in ("lexical", "hybrid", "hybrid_cxm25"):
            terms = self.tokenizer.terms(query)
            ls, lidx = self._lex.search(terms, self.candidate_k)
            rankings.append([int(i) for i in lidx])
            weights.append(self.fusion_weights[1])
            if mode == "lexical":
                single_scores = {int(i): float(s) for s, i in zip(ls, lidx)}

        if mode == "hybrid_cxm25" and self._cxm25 is not None:
            fused = reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=weights)
            cand = [c for c, _ in rank_from_scores(fused, self.candidate_k)]
            scores = self._cxm25.score_candidates(query, cand)
            chunk_scores = {c: float(s) for c, s in zip(cand, scores)}
        elif single_scores is not None:
            chunk_scores = single_scores
        elif len(rankings) == 1:
            chunk_scores = {
                c: 1.0 / (self.rrf_k + r) for r, c in enumerate(rankings[0], start=1)
            }
        else:
            chunk_scores = reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=weights)

        doc_scores: dict[int, float] = {}
        doc_best_chunk: dict[int, int] = {}
        for chunk_idx, score in chunk_scores.items():
            doc_idx = int(self._chunk_doc_idx[chunk_idx])
            if score > doc_scores.get(doc_idx, -np.inf):
                doc_scores[doc_idx] = score
                doc_best_chunk[doc_idx] = chunk_idx

        hits: list[Hit] = []
        for doc_idx, score in rank_from_scores(doc_scores, top_k):
            chunk = self._chunks[doc_best_chunk[doc_idx]]
            hits.append(
                Hit(
                    doc_id=self._doc_ids[doc_idx],
                    score=float(score),
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    start=chunk.start,
                    end=chunk.end,
                )
            )
        return hits

    def search_chunks(self, query: str, top_k: int = 10, mode: SearchMode = "hybrid") -> list[Hit]:
        """Like :meth:`search` but returns chunk-level hits without doc pooling."""
        if not self._chunks:
            return []
        rankings: list[list[int]] = []
        weights: list[float] = []
        if mode in ("dense", "hybrid", "hybrid_cxm25"):
            qv = self.encoder.encode([query])[0]
            _, idx = self._dense.search(qv, top_k)  # type: ignore[union-attr]
            rankings.append([int(i) for i in idx])
            weights.append(self.fusion_weights[0])
        if mode in ("lexical", "hybrid", "hybrid_cxm25"):
            _, idx = self._lex.search(self.tokenizer.terms(query), top_k)
            rankings.append([int(i) for i in idx])
            weights.append(self.fusion_weights[1])
        fused = (
            reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=weights)
            if len(rankings) > 1
            else {c: 1.0 / (self.rrf_k + r) for r, c in enumerate(rankings[0], start=1)}
        )
        out = []
        for chunk_idx, score in rank_from_scores(fused, top_k):
            c = self._chunks[chunk_idx]
            out.append(
                Hit(
                    doc_id=c.doc_id,
                    score=float(score),
                    chunk_id=c.chunk_id,
                    text=c.text,
                    start=c.start,
                    end=c.end,
                )
            )
        return out

    def retrieve_ranking(self, query: str, depth: int = 100, mode: SearchMode = "hybrid") -> list[str]:
        """Ranked doc ids, for benchmark harnesses."""
        return [h.doc_id for h in self.search(query, top_k=depth, mode=mode)]

    def score_documents(
        self, query: str, doc_ids, mode: str = "lexical"
    ) -> dict[str, float]:
        """Score a fixed candidate set of documents for ``query``.

        This is the reranking path: the candidate list is given, and only the
        order within it is decided. A document's score is the best score among
        its chunks. Documents with no indexed chunks score ``-inf``.

        ``mode="hybrid"`` fuses the dense and lexical *document* rankings with
        RRF, using :attr:`fusion_weights`.
        """
        ids = list(dict.fromkeys(str(d) for d in doc_ids))
        chunk_map = {d: self._doc_to_chunks.get(d, []) for d in ids}
        out = {d: float("-inf") for d in ids}
        present = {d: cs for d, cs in chunk_map.items() if cs}
        if not present:
            return out

        dense_scores = lex_scores = cxm25_scores = None
        if mode in ("dense", "hybrid"):
            dense_scores = self._dense.score_all(self.encoder.encode([query])[0])  # type: ignore[union-attr]
        if mode in ("lexical", "hybrid"):
            lex_scores = self._lex.score_all(self.tokenizer.terms(query))
        if mode == "cxm25":
            if self._cxm25 is None:
                self.finalize(build_cxm25=True)
            flat = [c for cs in present.values() for c in cs]
            vals = self._cxm25.score_candidates(query, flat)  # type: ignore[union-attr]
            cxm25_scores = dict(zip(flat, (float(v) for v in vals)))

        if mode == "cxm25":
            for d, cs in present.items():
                out[d] = max(cxm25_scores[c] for c in cs)  # type: ignore[index]
        elif mode == "hybrid":
            d_doc = {d: max(float(dense_scores[c]) for c in cs) for d, cs in present.items()}  # type: ignore[index]
            l_doc = {d: max(float(lex_scores[c]) for c in cs) for d, cs in present.items()}  # type: ignore[index]
            d_rank = {d: r for r, d in enumerate(sorted(d_doc, key=lambda x: -d_doc[x]), 1)}
            l_rank = {d: r for r, d in enumerate(sorted(l_doc, key=lambda x: -l_doc[x]), 1)}
            dw, lw = self.fusion_weights
            for d in present:
                out[d] = dw / (self.rrf_k + d_rank[d]) + lw / (self.rrf_k + l_rank[d])
        elif mode == "dense":
            for d, cs in present.items():
                out[d] = max(float(dense_scores[c]) for c in cs)  # type: ignore[index]
        elif mode == "lexical":
            for d, cs in present.items():
                out[d] = max(float(lex_scores[c]) for c in cs)  # type: ignore[index]
        else:
            raise ValueError(f"unknown scoring mode: {mode!r}")
        return out

    def rerank(
        self, query: str, doc_ids, mode: str = "cxm25", top_k: int | None = None
    ) -> list[str]:
        """Return ``doc_ids`` reordered best-first. Unscorable docs go last."""
        scores = self.score_documents(query, doc_ids, mode=mode)
        order = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
        ranked = [d for d, _ in order]
        return ranked[:top_k] if top_k else ranked

    # -- introspection ------------------------------------------------------
    def stats(self) -> dict:
        dense_bytes = self._dense.nbytes if self._dense is not None else 0
        return {
            "documents": len(self._doc_ids),
            "chunks": len(self._chunks),
            "dense_backend": "model2vec/numpy",
            "dense_model": self.dense_model,
            "dense_bytes": dense_bytes,
            "lexical_bytes": self._lex.nbytes,
            "lexical_backend": self._lex.__class__.__name__,
            "tokenizer": self.tokenizer.backend,
            "chunk_mode": self.chunker.mode,
            "timings": dict(self.timings),
        }
