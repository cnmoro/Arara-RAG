"""The arara-rag pipeline: chunk, embed, index, retrieve, fuse, rerank.

Two storage modes behind one API:

* ``Arara()`` keeps everything in memory -- fastest, best for small corpora;
* ``Arara(path="./index")`` is out-of-core and persistent: vectors live in a
  memory-mapped file, documents, metadata and chunk offsets live in SQLite.

Both support metadata filtering and CRUD. Deletes are tombstones with slot
reuse, because a RAG index is read far more often than it is written.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np

from .chunk import Chunker, canonicalize
from .dense import (DEFAULT_DENSE_MODEL, DEFAULT_NANOE5_VARIANT, DenseEncoder,
                    build_encoder)
from .fuse import rank_from_scores, reciprocal_rank_fusion
from .lexical import BM25Index, CXM25Scorer
from .store import (DEFAULT_SCAN_BLOCK, Catalog, ChunkRecord, MemCatalog,
                    MemVectorStore, VectorStore)
from .text import Tokenizer
from .types import Chunk, Hit

SearchMode = Literal["dense", "lexical", "hybrid", "hybrid_cxm25"]

# Chunking and embedding dominate index build, and both are per-document, so
# they parallelise across processes almost linearly. Below this many documents
# the pool costs more than it saves.
PARALLEL_MIN_DOCS = 64
DEFAULT_MAX_WORKERS = 8

# A scan smaller than this cannot amortise its own syscalls, so the resident
# budget is clamped here rather than down to a single row.
MIN_SCAN_BLOCK = 1024
# BLAS scratch and the per-block index arrays that appear *during* a scan, and
# are therefore not visible in the RSS reading taken just before it.
SCAN_RSS_SLACK = 24 * 1024 * 1024


def rss_bytes() -> int:
    """Resident set size of this process, or 0 where /proc is unavailable."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:  # pragma: no cover - non-Linux
        pass
    return 0


# Forked children inherit these; nothing is pickled except the work items and
# the results. The catalog and vector store are never touched by a child.
_WORKER_STATE: dict = {}


def _limit_threads():
    """Cap BLAS to one thread, if threadpoolctl is available.

    OpenBLAS sizes its thread team from the core count, but a per-document
    matmul (a few hundred rows) is far too small to amortise that: 2000
    documents cost 7.8 s on 16 threads against 2.1 s on one. Inside a process
    pool the oversubscription compounds.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - optional
        return None
    return threadpool_limits(limits=1)


def _worker_prepare(batch):
    """Chunk and embed a batch of documents. Module level, so it can be mapped.

    A batch rather than a single document, because encoders with a per-call cost
    (nanoE5 dequantises its 4-bit weights every call) are far more efficient on
    a large batch: 286 ms per passage at batch 1 against 99 ms at 256.
    """
    import numpy as _np

    chunker = _WORKER_STATE["chunker"]
    encoder = _WORKER_STATE["encoder"]
    per_doc = []
    limit = _limit_threads()

    def run():
        chunks_per_doc = [chunker.split(doc_id, canonical) for doc_id, canonical in batch]
        texts = [c.text for chunks in chunks_per_doc for c in chunks]
        if not texts:
            return [(chunks, _empty(encoder.dim)) for chunks in chunks_per_doc]
        size = max(1, int(getattr(encoder, "batch_size", 256)))
        parts = [encoder.encode(texts[i : i + size]) for i in range(0, len(texts), size)]
        vectors = _np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        out, pos = [], 0
        for chunks in chunks_per_doc:
            n = len(chunks)
            out.append((chunks, vectors[pos : pos + n] if n else _empty(encoder.dim)))
            pos += n
        return out

    if limit is None:
        return run()
    with limit:
        return run()


def _worker_tokenize(batch):
    """Tokenise a batch of ``(slot, text)``. Module level so it can be mapped."""
    tokenizer = _WORKER_STATE["tokenizer"]
    limit = _limit_threads()
    if limit is None:
        return [(slot, tokenizer.terms(text)) for slot, text in batch]
    with limit:
        return [(slot, tokenizer.terms(text)) for slot, text in batch]


def _empty(dim):
    import numpy as _np

    return _np.empty((0, dim), dtype=_np.float32)


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
    return [(str(d), str(t)) for d, t in items if t is not None]


class Arara:
    """A CPU-first hybrid retrieval stack for Portuguese and English.

    Example:
        >>> from arara_rag import Arara
        >>> a = Arara(path="./meu_indice")
        >>> a.add_documents({"lei": texto}, metadata={"ano": 2024})
        >>> a.search("aliquota", top_k=5, where={"ano": {"$gte": 2020}})

    Args:
        path: directory for a persistent, out-of-core index. ``None`` keeps
            everything in memory.
        dense_model: Model2Vec static embedding model.
        chunk_mode: ``"tinyzchunk"`` (default), ``"window"``, ``"paragraph"``
            or ``"document"``.
        candidate_k: chunks each retriever contributes to fusion.
        fusion_weights: ``(dense, lexical)`` weights for RRF.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        dense_model: str = DEFAULT_DENSE_MODEL,
        dense_backend: str = "static",
        dense_variant: str | None = None,
        chunk_mode: str = "tinyzchunk",
        max_chunk_chars: int = 2500,
        min_chunk_chars: int = 100,
        candidate_k: int = 100,
        rrf_k: int = 60,
        fusion_weights: tuple[float, ...] = (1.0, 1.0),
        lang: str = "pt",
        cache_dir: str | None = None,
        encoder: DenseEncoder | None = None,
        workers: int | None = None,
        max_ram_mb: float | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.dense_model = dense_model
        self.dense_backend = dense_backend
        self.dense_variant = dense_variant
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
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
        # None = pick automatically; 1 forces the sequential path.
        self.workers = workers
        # Ceiling on the per-query vector working set. The corpus is scanned a
        # block at a time and the pages are released afterwards, so resident
        # memory does not grow with the index; this bounds the block.
        self.max_ram_mb = max_ram_mb

        self._catalog: Catalog | MemCatalog = Catalog(self.path) if self.path else MemCatalog()
        self._vectors: VectorStore | MemVectorStore | None = None
        self._lex: BM25Index | None = None
        self._lex_dirty = True
        self._cxm25: CXM25Scorer | None = None

        # Fixed-size per-slot bookkeeping: 16 bytes per chunk, so it stays
        # resident even when vectors, text and metadata do not.
        self._slot_doc = np.empty(0, dtype=np.int32)
        self._slot_ord = np.empty(0, dtype=np.int32)
        self._slot_start = np.empty(0, dtype=np.int32)
        self._slot_end = np.empty(0, dtype=np.int32)
        self._doc_ids: list[str] = []
        # doc_id -> position. Built on first use rather than at open: a
        # read-only server answers queries without ever needing it.
        self._doc_index: dict[str, int] | None = None
        self._doc_alive: list[bool] = []
        self.timings: dict[str, float] = {}

        if self.path is not None:
            self._reopen()

    # -- persistence --------------------------------------------------------
    @property
    def persistent(self) -> bool:
        return self.path is not None

    def _scan_block(self, dim: int) -> int:
        """Rows a single scan step may hold resident.

        Without a cap the default block is used. With ``max_ram_mb`` the block
        is derived from what is left of the process ceiling *right now*, so the
        budget shrinks as the process grows and the ceiling is never crossed by
        the scan itself -- index data is dropped again page by page after each
        step.
        """
        if self.max_ram_mb is None or dim <= 0:
            return DEFAULT_SCAN_BLOCK
        row_bytes = max(1, dim * np.dtype(np.float32).itemsize)
        room = float(self.max_ram_mb) * 1024 * 1024 - rss_bytes() - SCAN_RSS_SLACK
        return int(min(DEFAULT_SCAN_BLOCK, max(MIN_SCAN_BLOCK, room // row_bytes)))

    def _retune_scan(self) -> None:
        """Re-derive the scan block from the current footprint before a query."""
        if self.max_ram_mb is None or not isinstance(self._vectors, VectorStore):
            return
        self._vectors.block = self._scan_block(self._vectors.dim)

    def _doc_map(self) -> dict[str, int]:
        """doc_id -> position in :attr:`_doc_ids`, materialised on demand."""
        if self._doc_index is None:
            self._doc_index = {d: i for i, d in enumerate(self._doc_ids)}
        return self._doc_index

    def _reopen(self) -> None:
        """Rebuild slot bookkeeping from a catalog written by a previous run."""
        self._doc_ids = []
        # rowid -> position, so the chunk table can be walked in slot order
        # without a doc_id -> position dict for the whole corpus.
        ord_of = np.full(self._catalog.max_doc_rowid() + 1, -1, dtype=np.int32)
        for doc_ord, (rowid, doc_id) in enumerate(self._catalog.document_rowids()):
            ord_of[rowid] = doc_ord
            self._doc_ids.append(doc_id)
        if not self._doc_ids:
            return
        self._doc_alive = [True] * len(self._doc_ids)
        size = self._catalog.max_slot() + 1
        self._slot_doc = np.full(size, -1, dtype=np.int32)
        self._slot_ord = np.zeros(size, dtype=np.int32)
        self._slot_start = np.zeros(size, dtype=np.int32)
        self._slot_end = np.zeros(size, dtype=np.int32)
        for row, doc_rowid in self._catalog.slot_rows_indexed():
            self._slot_doc[row.slot] = ord_of[doc_rowid]
            self._slot_ord[row.slot] = int(row.chunk_id.rsplit("#", 1)[-1])
            self._slot_start[row.slot] = row.start
            self._slot_end[row.slot] = row.end
        del ord_of
        if size:
            self._vectors = VectorStore(
                self.path, block=self._scan_block(self._catalog.vector_dim())
            )
        if (self.path / "bm25_meta.json").exists():
            self._lex = BM25Index.load(self.path)
            self._lex_dirty = False

    def flush(self) -> None:
        """Persist catalog and vector metadata. Cheap; call after writes."""
        self._catalog.commit()
        if self._vectors is not None:
            self._vectors.flush()

    def close(self) -> None:
        self.flush()
        if self._vectors is not None:
            self._vectors.close()
        self._catalog.close()

    def __enter__(self) -> "Arara":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- indexing -----------------------------------------------------------
    @property
    def encoder(self) -> DenseEncoder:
        if self._encoder is None:
            self._encoder = build_encoder(
                self.dense_backend, self.dense_model,
                cache_dir=self._cache_dir, variant=self.dense_variant,
            )
        return self._encoder

    def _encode_query(self, query: str) -> np.ndarray:
        """Embed one query.

        Asymmetric models (E5 and friends) encode queries and documents
        differently, so this must not go through ``encode``.
        """
        enc = self.encoder
        fn = getattr(enc, "encode_query", enc.encode)
        return fn([query])[0]

    def __len__(self) -> int:
        """Number of live documents."""
        return int(sum(self._doc_alive))

    @property
    def n_chunks(self) -> int:
        return int((self._slot_doc >= 0).sum())

    def _metadata_for(self, doc_id: str, metadata) -> dict:
        if metadata is None:
            return {}
        if isinstance(metadata, Mapping) and isinstance(metadata.get(doc_id), Mapping):
            return dict(metadata[doc_id])
        return dict(metadata)

    def add_documents(self, docs, metadata=None, show_progress: bool = False) -> int:
        """Chunk, embed and index documents. Returns the number of chunks added.

        Re-adding an existing ``doc_id`` replaces that document: its old chunks
        are tombstoned, their slots recycled, and the new version indexed. That
        keeps a frequently edited index from growing forever.

        Args:
            docs: mapping or iterable of ``(doc_id, text)``.
            metadata: dict applied to every document, or a mapping of
                ``doc_id`` to dict.
        """
        items = _normalize_input(docs)
        if not items:
            return 0

        t0 = time.perf_counter()
        # Documents are registered in the parent: the catalog is not written
        # from forked children.
        doc_ids: list[str] = []
        canonicals: list[str] = []
        doc_indices: list[int] = []
        for doc_id, text in items:
            if self._catalog.has_document(doc_id):
                self.delete_document(doc_id)
            canonical = canonicalize(text)
            self._catalog.put_document(doc_id, canonical, self._metadata_for(doc_id, metadata))
            index = self._doc_map()
            if doc_id in index:
                di = index[doc_id]
                self._doc_alive[di] = True
            else:
                di = len(self._doc_ids)
                index[doc_id] = di
                self._doc_ids.append(doc_id)
                self._doc_alive.append(True)
            doc_ids.append(doc_id)
            canonicals.append(canonical)
            doc_indices.append(di)

        prepared = self._prepare_documents(doc_ids, canonicals)
        self.timings["prepare_s"] = time.perf_counter() - t0

        new_chunks: list[Chunk] = []
        new_docs: list[int] = []
        blocks: list[np.ndarray] = []
        for (chunks, vectors), di in zip(prepared, doc_indices):
            if not len(chunks):
                continue
            new_chunks.extend(chunks)
            new_docs.extend([di] * len(chunks))
            blocks.append(vectors)
        if not new_chunks:
            self.flush()
            return 0

        t0 = time.perf_counter()
        vectors = np.concatenate(blocks, axis=0)
        if self._vectors is None:
            self._vectors = (
                VectorStore(self.path, dim=vectors.shape[1],
                            block=self._scan_block(vectors.shape[1]))
                if self.path is not None
                else MemVectorStore(dim=vectors.shape[1])
            )
        slots = self._vectors.append(vectors)

        t0 = time.perf_counter()
        self._place_slots(new_chunks, new_docs, slots)
        self._catalog.add_chunks(
            [
                ChunkRecord(c.chunk_id, c.doc_id, int(s), c.start, c.end)
                for c, s in zip(new_chunks, slots)
            ]
        )
        self._lex_dirty = True
        self._cxm25 = None
        self.timings["bookkeeping_s"] = time.perf_counter() - t0
        self.flush()
        return len(new_chunks)

    def _effective_workers(self, n_docs: int) -> int:
        """How many processes to use for this batch."""
        if self.workers is not None:
            return max(1, int(self.workers))
        if n_docs < PARALLEL_MIN_DOCS:
            return 1
        if "fork" not in mp.get_all_start_methods():
            # A spawned child would have to re-import numpy and reload the
            # weights in every worker; not worth it implicitly.
            return 1
        return max(1, min(os.cpu_count() or 1, DEFAULT_MAX_WORKERS))

    def _prepare_documents(self, doc_ids, canonicals):
        """Chunk and embed documents, in order, optionally across processes."""
        jobs = list(zip(doc_ids, canonicals))
        workers = self._effective_workers(len(jobs))
        _WORKER_STATE["chunker"] = self.chunker
        _WORKER_STATE["encoder"] = self.encoder
        self.timings["workers"] = float(workers)
        # Documents are grouped so the encoder sees a full batch. One document is
        # usually one chunk, which is the worst possible batch for nanoE5.
        group = max(1, int(getattr(self.encoder, "batch_size", 256)))
        batches = [jobs[i : i + group] for i in range(0, len(jobs), group)]
        if workers <= 1:
            return [r for batch in batches for r in _worker_prepare(batch)]
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
                return [r for part in pool.map(_worker_prepare, batches) for r in part]
        except (OSError, RuntimeError):  # pragma: no cover - sandboxed hosts
            # A pool that cannot start must not fail the ingest.
            return [r for batch in batches for r in _worker_prepare(batch)]

    def _place_slots(self, chunks, doc_idx, slots: np.ndarray) -> None:
        need = int(slots.max(initial=-1)) + 1
        if need > self._slot_doc.size:
            for name, fill in (
                ("_slot_doc", -1),
                ("_slot_ord", 0),
                ("_slot_start", 0),
                ("_slot_end", 0),
            ):
                old = getattr(self, name)
                grown = np.full(need, fill, dtype=np.int32)
                grown[: old.size] = old
                setattr(self, name, grown)
        self._slot_doc[slots] = np.asarray(doc_idx, dtype=np.int32)
        self._slot_ord[slots] = np.asarray(
            [int(c.chunk_id.rsplit("#", 1)[-1]) for c in chunks], dtype=np.int32
        )
        self._slot_start[slots] = np.asarray([c.start for c in chunks], dtype=np.int32)
        self._slot_end[slots] = np.asarray([c.end for c in chunks], dtype=np.int32)

    def finalize(self, build_cxm25: bool = False, n_jobs: int = 1, force_lexical: bool = False) -> "Arara":
        """Build derived structures. Call once after all documents are added."""
        if force_lexical or self._lex_dirty or self._lex is None:
            self._lex = self._build_lexical()
            self._lex_dirty = False
        if build_cxm25 and self._cxm25 is None and self._slot_doc.size:
            # Indexed by *slot*, with empty text for dead slots, so that
            # candidate ids stay valid after deletions.
            texts = [
                self._chunk_text(s) if int(self._slot_doc[s]) >= 0 else ""
                for s in range(self._slot_doc.size)
            ]
            t0 = time.perf_counter()
            self._cxm25 = CXM25Scorer(texts, tokenizer=self.tokenizer, n_jobs=n_jobs)
            self.timings["cxm25_build_s"] = time.perf_counter() - t0
        return self

    def _build_lexical(self) -> BM25Index:
        """Build BM25 indexed by *slot*.

        Dead slots get empty term lists so that the BM25 document space matches
        the slot space exactly. Compacting to live chunks only would silently
        shift every id after the first deletion.
        """
        t0 = time.perf_counter()
        n_slots = int(self._slot_doc.size)

        # Read the chunk text in the parent (the catalog is not shared with
        # children), then tokenise; stemming is per-chunk, so it parallelises.
        work: list[tuple[int, str]] = []
        for di, doc_id in enumerate(self._doc_ids):
            if not self._doc_alive[di]:
                continue
            if not self._catalog.has_document(doc_id):
                continue
            text = self._catalog.text_of(doc_id)
            for r in sorted(self._catalog.chunks_of(doc_id), key=lambda r: r.start):
                if 0 <= r.slot < n_slots:
                    work.append((r.slot, text[r.start : r.end]))

        terms: list[list[str]] = [[] for _ in range(n_slots)]
        workers = self._effective_workers(len(work))
        _WORKER_STATE["tokenizer"] = self.tokenizer
        if workers <= 1 or len(work) < PARALLEL_MIN_DOCS:
            for slot, tok in _worker_tokenize(work):
                terms[slot] = tok
        else:
            batches = [work[i : i + 512] for i in range(0, len(work), 512)]
            try:
                with ProcessPoolExecutor(
                    max_workers=min(workers, len(batches)), mp_context=mp.get_context("fork")
                ) as pool:
                    for part in pool.map(_worker_tokenize, batches):
                        for slot, tok in part:
                            terms[slot] = tok
            except (OSError, RuntimeError):  # pragma: no cover - sandboxed hosts
                for slot, tok in _worker_tokenize(work):
                    terms[slot] = tok

        idx = BM25Index()
        idx.add(terms)
        idx.finalize()
        self.timings["lexical_s"] = time.perf_counter() - t0
        if self.path is not None:
            idx.save(self.path)
        return idx

    # -- CRUD ---------------------------------------------------------------
    def delete_document(self, doc_id: str) -> bool:
        """Remove a document. Its chunks are tombstoned and its slots recycled."""
        di = self._doc_map().get(doc_id)
        if di is None or not self._doc_alive[di]:
            return False
        slots = self._catalog.delete_document(doc_id)
        if slots and self._vectors is not None:
            self._vectors.release(slots)
            self._slot_doc[np.asarray(slots, dtype=np.int64)] = -1
        self._doc_alive[di] = False
        self._lex_dirty = True
        self._cxm25 = None
        self.flush()
        return True

    def update_metadata(self, doc_id: str, metadata: dict, merge: bool = True) -> bool:
        """Replace or merge a document's metadata without re-embedding it."""
        current = self._catalog.get_document(doc_id)
        if current is None:
            return False
        text, old = current
        new = {**old, **metadata} if merge else dict(metadata)
        self._catalog.put_document(doc_id, text, new)
        self.flush()
        return True

    def get_document(self, doc_id: str) -> tuple[str, dict] | None:
        """Return ``(text, metadata)``, or ``None`` if absent."""
        return self._catalog.get_document(doc_id)

    def document_text(self, doc_id: str) -> str:
        """The canonical text that chunk offsets index into."""
        return self._catalog.text_of(doc_id)

    def resolve(self, hit: Hit) -> str:
        """Return the exact source text a hit points at."""
        if hit.start is None or hit.end is None:
            raise ValueError("hit has no offsets")
        return self._catalog.text_of(hit.doc_id)[hit.start : hit.end]

    def compact(self) -> int:
        """Rewrite the vector file without tombstoned slots.

        Slots are recycled on write, so this reclaims space rather than fixing
        correctness. Returns the live chunk count.
        """
        if self.path is None or self._vectors is None:
            return self.n_chunks
        live = self._live_slots()
        if live.size == 0:
            return 0

        staging = Path(tempfile.mkdtemp(dir=str(self.path)))
        store = VectorStore(staging, dim=self._vectors.dim, capacity=int(live.size))
        store.append(self._vectors.gather(live))
        store.flush()
        store.close()

        self._slot_doc = self._slot_doc[live]
        self._slot_ord = self._slot_ord[live]
        self._slot_start = self._slot_start[live]
        self._slot_end = self._slot_end[live]

        self._vectors.close()
        for name in ("vectors.npy", "vectors.json"):
            src, dst = staging / name, self.path / name
            if src.exists():
                dst.unlink(missing_ok=True)
                src.replace(dst)
        shutil.rmtree(staging, ignore_errors=True)
        self._vectors = VectorStore(
            self.path, block=self._scan_block(self._catalog.vector_dim())
        )

        records = []
        for s in range(self._slot_doc.size):
            doc_id = self._doc_ids[int(self._slot_doc[s])]
            records.append(
                ChunkRecord(
                    chunk_id=f"{doc_id}#{int(self._slot_ord[s])}",
                    doc_id=doc_id,
                    slot=s,
                    start=int(self._slot_start[s]),
                    end=int(self._slot_end[s]),
                )
            )
        self._catalog.replace_chunks(records)
        self._lex_dirty = True
        self.flush()
        return self.n_chunks

    # -- retrieval ----------------------------------------------------------
    def _live_slots(self) -> np.ndarray:
        if self._slot_doc.size == 0:
            return np.empty(0, dtype=np.int64)
        return np.nonzero(self._slot_doc >= 0)[0].astype(np.int64)

    def _allowed(self, where: dict | None) -> np.ndarray | None:
        """Live slots restricted by ``where``; ``None`` means 'all of them'."""
        live = self._live_slots()
        filtered = self._catalog.filter_slots(where)
        if filtered is None:
            return None if live.size == self._slot_doc.size else live
        return np.intersect1d(live, filtered, assume_unique=True)

    def _chunk_text(self, slot: int) -> str:
        di = int(self._slot_doc[slot])
        text = self._catalog.text_of(self._doc_ids[di])
        return text[int(self._slot_start[slot]) : int(self._slot_end[slot])]

    def _hit(self, slot: int, score: float) -> Hit:
        di = int(self._slot_doc[slot])
        doc_id = self._doc_ids[di]
        return Hit(
            doc_id=doc_id,
            score=float(score),
            chunk_id=f"{doc_id}#{int(self._slot_ord[slot])}",
            text=self._chunk_text(slot),
            start=int(self._slot_start[slot]),
            end=int(self._slot_end[slot]),
        )

    def _prepare(self, where: dict | None):
        if self._slot_doc.size == 0:
            return None, None, False
        allowed = self._allowed(where)
        if allowed is not None and allowed.size == 0:
            return None, None, False
        if self._lex is None or self._lex_dirty:
            self.finalize()
        mask = None
        if allowed is not None:
            mask = np.zeros(self._slot_doc.size, dtype=bool)
            mask[allowed] = True
        return allowed, mask, True

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: SearchMode = "hybrid",
        where: dict | None = None,
    ) -> list[Hit]:
        """Retrieve documents, optionally filtered by metadata.

        ``where`` uses a MongoDB-like syntax:
        ``{"ano": {"$gte": 2020}, "tipo": {"$in": ["lei", "decreto"]}}``.
        """
        self._retune_scan()
        allowed, mask, ok = self._prepare(where)
        if not ok:
            return []

        rankings: list[list[int]] = []
        weights: list[float] = []
        single: dict[int, float] | None = None

        if mode in ("dense", "hybrid", "hybrid_cxm25") and self._vectors is not None:
            ds, didx = self._vectors.search(
                self._encode_query(query), self.candidate_k, allowed
            )
            rankings.append([int(i) for i in didx])
            weights.append(self.fusion_weights[0])
            if mode == "dense":
                single = {int(i): float(s) for s, i in zip(ds, didx)}

        if mode in ("lexical", "hybrid", "hybrid_cxm25"):
            ls, lidx = self._lex.search(self.tokenizer.terms(query), self.candidate_k, mask=mask)
            rankings.append([int(i) for i in lidx])
            weights.append(self.fusion_weights[1])
            if mode == "lexical":
                single = {int(i): float(s) for s, i in zip(ls, lidx)}

        if not rankings:
            return []

        if mode == "hybrid_cxm25" and self._cxm25 is not None:
            fused = reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=weights)
            cand = [c for c, _ in rank_from_scores(fused, self.candidate_k)]
            scores = self._cxm25.score_candidates(query, cand)
            chunk_scores = {c: float(s) for c, s in zip(cand, scores)}
        elif single is not None:
            chunk_scores = single
        else:
            chunk_scores = reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=weights)

        doc_scores: dict[int, float] = {}
        doc_chunk: dict[int, int] = {}
        for slot, score in chunk_scores.items():
            di = int(self._slot_doc[slot])
            if di < 0:
                continue
            if score > doc_scores.get(di, -np.inf):
                doc_scores[di] = score
                doc_chunk[di] = slot
        return [self._hit(doc_chunk[d], s) for d, s in rank_from_scores(doc_scores, top_k)]

    def search_chunks(
        self, query: str, top_k: int = 10, mode: SearchMode = "hybrid", where: dict | None = None
    ) -> list[Hit]:
        """Like :meth:`search` but returns chunk-level hits without doc pooling."""
        self._retune_scan()
        allowed, mask, ok = self._prepare(where)
        if not ok:
            return []
        rankings: list[list[int]] = []
        if mode in ("dense", "hybrid", "hybrid_cxm25") and self._vectors is not None:
            _, idx = self._vectors.search(self._encode_query(query), top_k, allowed)
            rankings.append([int(i) for i in idx])
        if mode in ("lexical", "hybrid", "hybrid_cxm25"):
            _, idx = self._lex.search(self.tokenizer.terms(query), top_k, mask=mask)
            rankings.append([int(i) for i in idx])
        if not rankings:
            return []
        fused = (
            reciprocal_rank_fusion(rankings, k=self.rrf_k)
            if len(rankings) > 1
            else {c: 1.0 / (self.rrf_k + r) for r, c in enumerate(rankings[0], start=1)}
        )
        return [self._hit(slot, score) for slot, score in rank_from_scores(fused, top_k)]

    def retrieve_ranking(
        self, query: str, depth: int = 100, mode: SearchMode = "hybrid", where: dict | None = None
    ) -> list[str]:
        """Ranked doc ids, for benchmark harnesses."""
        return [h.doc_id for h in self.search(query, top_k=depth, mode=mode, where=where)]

    def score_documents(self, query: str, doc_ids, mode: str = "lexical") -> dict[str, float]:
        """Score a fixed candidate set of documents (the reranking path)."""
        ids = list(dict.fromkeys(str(d) for d in doc_ids))
        out = {d: float("-inf") for d in ids}
        slots_of: dict[str, list[int]] = {}
        for d in ids:
            di = self._doc_map().get(d)
            if di is None or not self._doc_alive[di]:
                slots_of[d] = []
                continue
            slots_of[d] = [r.slot for r in self._catalog.chunks_of(d)]
        present = {d: s for d, s in slots_of.items() if s}
        if not present:
            return out
        if self._lex is None or self._lex_dirty:
            self.finalize()

        flat = [c for s in present.values() for c in s]
        if mode in ("lexical", "hybrid"):
            lex = self._lex.score_all(self.tokenizer.terms(query))
        if mode in ("dense", "hybrid") and self._vectors is not None:
            dense = self._vectors.gather(flat) @ self._encode_query(query)
            dense_map = dict(zip(flat, (float(v) for v in dense)))
        if mode == "cxm25":
            if self._cxm25 is None:
                self.finalize(build_cxm25=True)
            vals = self._cxm25.score_candidates(query, flat)
            cxm25_map = dict(zip(flat, (float(v) for v in vals)))

        if mode == "cxm25":
            for d, cs in present.items():
                out[d] = max(cxm25_map[c] for c in cs)
        elif mode == "hybrid":
            d_doc = {d: max(dense_map[c] for c in cs) for d, cs in present.items()}
            l_doc = {d: max(float(lex[c]) for c in cs) for d, cs in present.items()}
            d_rank = {d: r for r, d in enumerate(sorted(d_doc, key=lambda x: -d_doc[x]), 1)}
            l_rank = {d: r for r, d in enumerate(sorted(l_doc, key=lambda x: -l_doc[x]), 1)}
            dw, lw = self.fusion_weights[0], self.fusion_weights[1]
            for d in present:
                out[d] = dw / (self.rrf_k + d_rank[d]) + lw / (self.rrf_k + l_rank[d])
        elif mode == "dense":
            for d, cs in present.items():
                out[d] = max(dense_map[c] for c in cs)
        elif mode == "lexical":
            for d, cs in present.items():
                out[d] = max(float(lex[c]) for c in cs)
        else:
            raise ValueError(f"unknown scoring mode: {mode!r}")
        return out

    def rerank(self, query: str, doc_ids, mode: str = "cxm25", top_k: int | None = None) -> list[str]:
        """Return ``doc_ids`` reordered best-first. Unscorable docs go last."""
        scores = self.score_documents(query, doc_ids, mode=mode)
        ranked = [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))]
        return ranked[:top_k] if top_k else ranked

    # -- introspection ------------------------------------------------------
    def stats(self) -> dict:
        vector_bytes = self._vectors.nbytes if self._vectors is not None else 0
        file_bytes = self._vectors.file_bytes if self._vectors is not None else 0
        return {
            "documents": len(self),
            "deleted_documents": len(self._doc_ids) - len(self),
            "chunks": self.n_chunks,
            "slots_allocated": int(self._slot_doc.size),
            "slot_bookkeeping_bytes": int(self._slot_doc.size * 16),
            "dense_backend": self.dense_backend,
            "dense_model": self.dense_model,
            "vector_bytes": vector_bytes,
            "vector_file_bytes": file_bytes,
            "lexical_bytes": self._lex.nbytes if self._lex is not None else 0,
            "lexical_backend": "BM25Index",
            "tokenizer": self.tokenizer.backend,
            "chunk_mode": self.chunker.mode,
            "persistent": self.persistent,
            "max_ram_mb": self.max_ram_mb,
            "scan_block": getattr(self._vectors, "block", None),
            "path": str(self.path) if self.path else None,
            "timings": dict(self.timings),
        }
