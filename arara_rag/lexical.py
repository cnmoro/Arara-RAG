"""Lexical retrieval.

Two scorers, used at different stages for a reason:

* :class:`BM25Index` -- a numpy inverted index. Fast enough to score a whole
  corpus per query, which is what a full hybrid retrieval stage requires.
* :class:`CXM25Scorer` -- CXM25, which is markedly more accurate than BM25 on
  hard PT-BR pairs (0.705 vs 0.673 pairwise) but scores roughly 71 us/doc, so
  it is only economical over a candidate set. It is therefore a *reranker*.
"""

from __future__ import annotations

import json
import mmap as _mmap

import numpy as np

from .text import Tokenizer


class SortedVocab:
    """Term -> id map kept on disk, read with a binary search.

    A Python ``dict`` of terms costs on the order of 140 bytes per distinct
    term -- 140 MB for a million-term corpus, held resident for the life of the
    process. The same map as a sorted blob of UTF-8 terms plus offsets and ids
    costs about 17 bytes per term, and the pages a lookup touches are clean and
    evictable, so what stays resident is only what queries actually use.
    """

    __slots__ = ("_blob", "_offsets", "_ids", "_n")

    def __init__(self, blob, offsets, ids) -> None:
        self._blob = blob
        self._offsets = offsets
        self._ids = ids
        self._n = int(offsets.shape[0]) - 1

    def __len__(self) -> int:
        return self._n

    @property
    def nbytes(self) -> int:
        return int(self._blob.nbytes + self._offsets.nbytes + self._ids.nbytes)

    def _term_at(self, i: int) -> bytes:
        return bytes(self._blob[int(self._offsets[i]) : int(self._offsets[i + 1])])

    def get(self, term: str | bytes) -> int | None:
        b = term.encode("utf-8") if isinstance(term, str) else term
        lo, hi = 0, self._n
        offsets, ids = self._offsets, self._ids
        while lo < hi:
            mid = (lo + hi) // 2
            cur = bytes(
                self._blob[int(offsets[mid]) : int(offsets[mid + 1])]
            )
            if cur == b:
                return int(ids[mid])
            if cur < b:
                lo = mid + 1
            else:
                hi = mid
        return None

    def items(self):
        for i in range(self._n):
            yield self._term_at(i).decode("utf-8"), int(self._ids[i])

    def to_dict(self) -> dict[str, int]:
        return dict(self.items())

    @classmethod
    def build(cls, vocab: dict[str, int], directory) -> "SortedVocab":
        from pathlib import Path

        d = Path(directory)
        terms = sorted(vocab)
        encoded = [t.encode("utf-8") for t in terms]
        offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        np.cumsum([len(e) for e in encoded], out=offsets[1:])
        blob = np.frombuffer(b"".join(encoded), dtype=np.uint8)
        ids = np.asarray([vocab[t] for t in terms], dtype=np.int32)
        np.save(d / "bm25_vocab_offsets.npy", offsets)
        np.save(d / "bm25_vocab_blob.npy", blob)
        np.save(d / "bm25_vocab_ids.npy", ids)
        return cls.from_directory(d)

    @classmethod
    def from_directory(cls, directory) -> "SortedVocab | None":
        from pathlib import Path

        d = Path(directory)
        if not (d / "bm25_vocab_offsets.npy").exists():
            return None
        return cls(
            np.load(d / "bm25_vocab_blob.npy", mmap_mode="r"),
            np.load(d / "bm25_vocab_offsets.npy", mmap_mode="r"),
            np.load(d / "bm25_vocab_ids.npy", mmap_mode="r"),
        )


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
        # Set by :meth:`load` when the postings are memory-mapped; a scan then
        # drops the pages it touched so resident memory stays flat.
        self.release_pages: bool = False

    def __len__(self) -> int:
        return self._n_docs

    @property
    def nbytes(self) -> int:
        total = 0
        for arr in (self._indptr, self._indices, self._data, self._idf):
            if arr is not None:
                total += int(arr.nbytes)
        vocab = getattr(self._vocab, "nbytes", None)
        return total + (int(vocab) if vocab else 0)

    def add(self, token_lists) -> None:
        if not isinstance(self._doc_lens, list):
            # Already finalized; fall back to list form so incremental adds work
            # and force a rebuild of the inverted index.
            self._doc_lens = list(self._doc_lens.tolist())
            self._indptr = None
        if not isinstance(self._vocab, dict):
            # A vocabulary loaded from disk is read-only; adding to it means
            # rebuilding anyway, so materialise the dict and carry on.
            self._vocab = self._vocab.to_dict()
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

    def _drop_postings_pages(self, lo: int, hi: int) -> None:
        """madvise one term's posting slice back out of the resident set."""
        if not self.release_pages or hi <= lo:
            return
        for arr in (self._indices, self._data):
            handle = getattr(arr, "_mmap", None)
            if handle is None or not hasattr(handle, "madvise"):
                continue
            page = _mmap.PAGESIZE
            itemsize = np.dtype(arr.dtype).itemsize
            start = (lo * itemsize) // page * page
            stop = min(((hi * itemsize) + page - 1) // page * page, arr.nbytes)
            if stop > start:
                try:
                    handle.madvise(_mmap.MADV_DONTNEED, start, stop - start)
                except (OSError, ValueError):  # pragma: no cover - platform
                    pass

    def score_all(self, terms, mask: np.ndarray | None = None) -> np.ndarray:
        """BM25 score of every document, in index order.

        ``mask`` is a boolean array over slots; masked-out entries score 0.
        """
        if self._indptr is None:
            self.finalize()
        n = self._n_docs
        scores = np.zeros(n, dtype=np.float32)
        if n == 0:
            return scores
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
            self._drop_postings_pages(lo, hi)
        if mask is not None:
            scores = np.where(mask, scores, 0.0)
        return scores

    def search(self, terms, top_k: int = 100, mask: np.ndarray | None = None):
        """Return ``(scores, doc_indices)`` for the top ``top_k`` documents."""
        scores = self.score_all(terms, mask=mask)
        n = scores.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        k = min(top_k, n)
        part = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
        cand_s = scores[part]
        order = np.argsort(-cand_s, kind="stable")
        idx = part[order].astype(np.int64)
        vals = cand_s[order]
        if mask is not None:
            keep = vals > 0
            idx, vals = idx[keep], vals[keep]
        return vals, idx

    # -- persistence --------------------------------------------------------
    def save(self, directory) -> None:
        """Write the inverted index to ``directory`` as ``.npy`` arrays."""
        from pathlib import Path

        self.finalize()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "bm25_indptr.npy", self._indptr)
        np.save(d / "bm25_indices.npy", self._indices)
        np.save(d / "bm25_data.npy", self._data)
        np.save(d / "bm25_idf.npy", self._idf)
        np.save(d / "bm25_doclen.npy", np.asarray(self._doc_lens, dtype=np.float32))
        SortedVocab.build(dict(self._vocab), d)
        (d / "bm25_meta.json").write_text(
            json.dumps(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "avgdl": self._avgdl,
                    "n_docs": self._n_docs,
                    "n_terms": len(self._vocab),
                }
            )
        )

    @classmethod
    def load(cls, directory, mmap: bool = True) -> "BM25Index":
        """Load an inverted index written by :meth:`save`.

        With ``mmap=True`` the postings are paged from disk instead of being
        read into the process, which is what keeps a large index out of RAM.
        """
        from pathlib import Path

        d = Path(directory)
        meta = json.loads((d / "bm25_meta.json").read_text())
        obj = cls(k1=meta["k1"], b=meta["b"])
        mode = "r" if mmap else None
        obj.release_pages = bool(mmap)
        obj._indptr = np.load(d / "bm25_indptr.npy", mmap_mode=mode)
        obj._indices = np.load(d / "bm25_indices.npy", mmap_mode=mode)
        obj._data = np.load(d / "bm25_data.npy", mmap_mode=mode)
        obj._idf = np.load(d / "bm25_idf.npy", mmap_mode=mode)
        obj._doc_lens = np.load(d / "bm25_doclen.npy", mmap_mode=mode)
        obj._avgdl = float(meta["avgdl"])
        obj._n_docs = int(meta["n_docs"])
        vocab = SortedVocab.from_directory(d)
        if vocab is None:  # index written before the vocabulary moved to disk
            vocab = {k: int(v) for k, v in meta["vocab"].items()}
        obj._vocab = vocab
        return obj


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
