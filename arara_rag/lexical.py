"""Lexical retrieval.

Two scorers, used at different stages for a reason:

* :class:`BM25Index` -- a numpy inverted index. Fast enough to score a whole
  corpus per query, which is what a full hybrid retrieval stage requires.
* :class:`CXM25Scorer` -- CXM25, which is markedly more accurate than BM25 on
  hard PT-BR pairs (0.705 vs 0.673 pairwise) but scores roughly 71 us/doc, so
  it is only economical over a candidate set. It is therefore a *reranker*.
"""

from __future__ import annotations

import numpy as np

from .text import Tokenizer


class BM25Index:
    """Standard Robertson/Lucene BM25 over an inverted index built with numpy."""

    def __init__(self, k1: float = 0.8, b: float = 0.8) -> None:
        self.k1 = k1
        self.b = b
        self._vocab: dict[str, int] = {}
        self._term_ids: list[np.ndarray] = []
        self._doc_lens: list[int] = []
        self._indptr: np.ndarray | None = None
        self._indices: np.ndarray | None = None
        self._data: np.ndarray | None = None
        self._idf: np.ndarray | None = None
        self._avgdl: float = 1.0
        self._n_docs: int = 0

    def __len__(self) -> int:
        return self._n_docs

    @property
    def nbytes(self) -> int:
        total = 0
        for arr in (self._indptr, self._indices, self._data, self._idf):
            if arr is not None:
                total += int(arr.nbytes)
        return total

    def add(self, token_lists) -> None:
        if not isinstance(self._doc_lens, list):
            # Already finalized; fall back to list form so incremental adds work
            # and force a rebuild of the inverted index.
            self._doc_lens = list(self._doc_lens.tolist())
            self._indptr = None
        for terms in token_lists:
            ids = []
            for t in terms:
                tid = self._vocab.get(t)
                if tid is None:
                    tid = len(self._vocab)
                    self._vocab[t] = tid
                ids.append(tid)
            self._term_ids.append(np.asarray(sorted(ids), dtype=np.int32))
            self._doc_lens.append(len(ids))

    def finalize(self) -> "BM25Index":
        n = len(self._term_ids)
        self._n_docs = n
        if n == 0:
            self._indptr = np.zeros(1, dtype=np.int64)
            self._indices = np.empty(0, dtype=np.int32)
            self._data = np.empty(0, dtype=np.float32)
            self._idf = np.empty(0, dtype=np.float32)
            return self
        lens = np.asarray(self._doc_lens, dtype=np.float32)
        self._avgdl = float(lens.mean()) if n else 1.0

        # Document frequency counts each term at most once per document, so the
        # offsets must be derived from per-document *unique* term ids. Using raw
        # term occurrences here silently leaves uninitialised slots in the
        # postings arrays.
        uniq = [np.unique(ids) for ids in self._term_ids if ids.size]
        flat = (
            np.concatenate(uniq).astype(np.int64, copy=False)
            if uniq
            else np.empty(0, dtype=np.int64)
        )
        indptr = np.zeros(len(self._vocab) + 1, dtype=np.int64)
        np.cumsum(np.bincount(flat, minlength=len(self._vocab)), out=indptr[1:])

        total = int(indptr[-1])
        indices = np.empty(total, dtype=np.int32)
        data = np.empty(total, dtype=np.float32)
        cursor = indptr[:-1].copy()
        for i, ids in enumerate(self._term_ids):
            if ids.size == 0:
                continue
            terms, tf = np.unique(ids, return_counts=True)
            for t, f in zip(terms, tf):
                pos = cursor[t]
                indices[pos] = i
                data[pos] = f
                cursor[t] = pos + 1
        assert np.array_equal(cursor, indptr[1:]), "inverted index build lost postings"

        self._indptr = indptr
        self._indices = indices
        self._data = data
        df = np.diff(indptr).astype(np.float32)
        self._idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self._doc_lens = lens  # type: ignore[assignment]
        return self

    def search(self, terms, top_k: int = 100):
        """Return ``(scores, doc_indices)`` for the top ``top_k`` documents."""
        if self._indptr is None:
            self.finalize()
        n = int(self._doc_lens.shape[0]) if isinstance(self._doc_lens, np.ndarray) else 0
        if n == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        scores = np.zeros(n, dtype=np.float32)
        k1, b, avgdl = self.k1, self.b, self._avgdl
        for term in set(terms):
            tid = self._vocab.get(term)
            if tid is None:
                continue
            lo, hi = self._indptr[tid], self._indptr[tid + 1]
            if hi <= lo:
                continue
            docs = self._indices[lo:hi]
            tf = self._data[lo:hi]
            dl = self._doc_lens[docs]
            denom = tf + k1 * (1.0 - b + b * dl / avgdl)
            scores[docs] += self._idf[tid] * (tf * (k1 + 1.0)) / denom
        k = min(top_k, n)
        part = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
        cand_s = scores[part]
        order = np.argsort(-cand_s, kind="stable")
        return cand_s[order], part[order].astype(np.int64)


class CXM25Scorer:
    """CXM25 lexical scorer, used to rerank a candidate set.

    Construction is the expensive part (corpus statistics plus one tokenization
    pass), so it is built once and reused across every query.
    """

    def __init__(self, texts, tokenizer: Tokenizer | None = None, gram_n: int = 3, n_jobs: int = 1):
        import cxm25
        from cxm25.scoring import CXM25
        from cxm25.stats import build_corpus_stats, tokenize_doc

        self._tokenizer = tokenizer or Tokenizer("pt")
        norm = self._tokenizer.normalizer
        if norm is None:  # pragma: no cover
            raise RuntimeError("CXM25 reranking requires the cxm25 package")
        texts = list(texts)
        self._docobjs = [tokenize_doc(norm, t) for t in texts]
        stats = build_corpus_stats(texts, gram_n=gram_n, n_jobs=n_jobs)
        avg_len = sum(len(o[0]) for o in self._docobjs) / max(len(self._docobjs), 1)
        self._scorer = CXM25(stats.df, stats.df2, stats.N, avg_len, gdf=stats.gdf, gram_n=gram_n)
        self._norm = norm
        self._cxm25 = cxm25

    def score_candidates(self, query: str, candidate_idx) -> np.ndarray:
        """Score the given document indices against ``query``."""
        qt = self._norm(query)
        ctx = self._scorer.prepare(qt)
        return np.asarray(
            [self._scorer.score_prepared(ctx, self._docobjs[i]) for i in candidate_idx],
            dtype=np.float32,
        )
