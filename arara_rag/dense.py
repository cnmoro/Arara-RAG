"""Dense retrieval: static embeddings plus an exact numpy index.

No PyTorch, no ONNX, no FAISS. A static embedding model is a lookup table, so
encoding is a tokenize-plus-lookup pass and search is a single matrix multiply.
That is the whole trick behind fitting a usable RAG stack on a CPU.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DENSE_MODEL = "cnmoro/static-nomic-384-pten-v2"


class DenseEncoder:
    """Wraps a Model2Vec static embedding model (numpy only)."""

    def __init__(self, model_id: str = DEFAULT_DENSE_MODEL, cache_dir: str | None = None) -> None:
        from model2vec import StaticModel  # imported lazily to keep module import cheap

        kwargs = {"cache_folder": cache_dir} if cache_dir else {}
        self.model_id = model_id
        self.model = StaticModel.from_pretrained(model_id, **kwargs)
        self.dim = int(self.model.dim)

    def encode(self, texts, batch_size: int = 512, show_progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            list(texts), batch_size=batch_size, show_progress_bar=show_progress
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        # Defensive: some static models ship un-normalized; cosine needs unit rows.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        np.divide(vecs, np.maximum(norms, 1e-12), out=vecs)
        return vecs


class DenseIndex:
    """Exact inner-product index over L2-normalized vectors.

    Vectors are retained in float16 to halve memory; scoring accumulates in
    float32 in blocks so peak memory stays bounded regardless of corpus size.
    """

    def __init__(self, dim: int, dtype=np.float16) -> None:
        self.dim = dim
        self.dtype = dtype
        self._vectors: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None
        self._matrix32: np.ndarray | None = None

    def __len__(self) -> int:
        if self._matrix is not None:
            return int(self._matrix.shape[0])
        return int(sum(v.shape[0] for v in self._vectors))

    @property
    def nbytes(self) -> int:
        if self._matrix is not None:
            return int(self._matrix.nbytes)
        return int(sum(v.nbytes for v in self._vectors))

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) vectors, got {vectors.shape}")
        self._vectors.append(vectors.astype(self.dtype, copy=False))
        self._matrix = None  # invalidate
        self._matrix32 = None

    def finalize(self) -> "DenseIndex":
        if self._matrix is None and self._vectors:
            self._matrix = np.concatenate(self._vectors, axis=0)
            self._vectors = []
        return self

    def _searchable(self) -> np.ndarray | None:
        """float16 halves storage, but converting it per query dominates latency.

        The conversion is done once and cached, so query cost scales with the
        corpus rather than with the dtype conversion.
        """
        self.finalize()
        if self._matrix is None or self._matrix.shape[0] == 0:
            return None
        if self._matrix32 is None:
            self._matrix32 = (
                self._matrix.astype(np.float32)
                if self._matrix.dtype != np.float32
                else self._matrix
            )
        return self._matrix32

    def search(self, query: np.ndarray, top_k: int = 100, block: int = 65536):
        """Return ``(scores, indices)`` for the top ``top_k`` vectors."""
        matrix = self._searchable()
        if matrix is None:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        n = matrix.shape[0]
        top_k = min(top_k, n)
        best_s = np.empty(0, dtype=np.float32)
        best_i = np.empty(0, dtype=np.int64)
        for start in range(0, n, block):
            stop = min(start + block, n)
            sims = matrix[start:stop] @ q
            k = min(top_k, sims.shape[0])
            if k < sims.shape[0]:
                part = np.argpartition(-sims, k - 1)[:k]
            else:
                part = np.arange(sims.shape[0])
            cand_s = sims[part]
            cand_i = part.astype(np.int64) + start
            if best_s.size:
                cand_s = np.concatenate([best_s, cand_s])
                cand_i = np.concatenate([best_i, cand_i])
            order = np.argsort(-cand_s, kind="stable")[:top_k]
            best_s = cand_s[order]
            best_i = cand_i[order]
        return best_s, best_i

    def save(self, path: str) -> None:
        self.finalize()
        np.save(path, self._matrix if self._matrix is not None else np.zeros((0, self.dim), self.dtype))

    @classmethod
    def load(cls, path: str) -> "DenseIndex":
        matrix = np.load(path, mmap_mode="r")
        idx = cls(dim=int(matrix.shape[1]), dtype=matrix.dtype)
        idx._matrix = matrix
        return idx
