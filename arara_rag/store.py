"""Out-of-core storage: memory-mapped vectors and a SQLite catalog.

An index has to survive being bigger than RAM, and it has to survive being
edited. Both are handled here:

* vectors live in a memory-mapped ``.npy`` file, scanned in blocks, so the
  resident set is bounded by the block size rather than the corpus size;
* documents, chunk offsets and metadata live in SQLite, so filtering, paging
  and point reads never materialise the corpus;
* deletions are tombstones with a free list, because a RAG index is read far
  more often than it is written.

The in-memory equivalents implement the same interface so that ``Arara()``
without a path keeps working without touching the disk.
"""

from __future__ import annotations

import json
import mmap as _mmap
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Iterable, Sequence

import numpy as np

# float32, not float16. numpy has no BLAS path for float16, so a float16 store
# forces a widening pass on every query -- 65 ms at 50k documents against 7 ms
# once the matmul can read float32 pages directly. The file is twice as large
# and the queries are an order of magnitude faster; that is the right trade.
VECTOR_DTYPE = np.float32
_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Rows of the vector matrix touched per scan step. The scan must never hold more
# than this much of the corpus resident, or "out of core" is only a claim about
# the file format. 16k rows of 384 float32 is ~25 MB.
DEFAULT_SCAN_BLOCK = 16384
_PAGE = 4096


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------
class FilterError(ValueError):
    """Raised for an unsupported or malformed ``where`` clause."""


def compile_filter(where: dict[str, Any] | None, alias: str = "d") -> tuple[str, list[Any]]:
    """Compile a MongoDB-like filter into a SQL predicate over ``d.meta``.

    Supported per field: ``$eq`` (bare value), ``$ne``, ``$gt``, ``$gte``,
    ``$lt``, ``$lte``, ``$in``, ``$nin``, ``$exists``, ``$contains``,
    ``$startswith``, ``$endswith``. Top level supports ``$and`` / ``$or``.

    Field names are validated and values are always bound as parameters, so a
    filter cannot inject SQL.
    """
    if not where:
        return "1=1", []

    clauses: list[str] = []
    params: list[Any] = []

    def json_path(field: str) -> str:
        if not _FIELD.match(field.replace(".", "_")) or field.startswith("$"):
            raise FilterError(f"invalid metadata field: {field!r}")
        return f"json_extract({alias}.meta, '$.{field}')"

    def field_clause(field: str, cond: Any) -> str:
        expr = json_path(field)
        if isinstance(cond, dict):
            parts = []
            for op, val in cond.items():
                if op == "$eq":
                    parts.append(f"{expr} = ?")
                elif op == "$ne":
                    parts.append(f"({expr} IS NULL OR {expr} <> ?)")
                elif op in ("$gt", "$gte", "$lt", "$lte"):
                    sqlop = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}[op]
                    parts.append(f"{expr} {sqlop} ?")
                elif op == "$in":
                    vals = list(val)
                    if not vals:
                        parts.append("0=1")
                        continue
                    parts.append(f"{expr} IN ({','.join('?' * len(vals))})")
                    params.extend(vals)
                    continue
                elif op == "$nin":
                    vals = list(val)
                    if vals:
                        parts.append(f"({expr} IS NULL OR {expr} NOT IN ({','.join('?' * len(vals))}))")
                        params.extend(vals)
                    continue
                elif op == "$exists":
                    parts.append(f"{expr} IS NOT NULL" if val else f"{expr} IS NULL")
                    continue
                elif op == "$contains":
                    parts.append(f"{expr} LIKE ?")
                    params.append(f"%{val}%")
                    continue
                elif op == "$startswith":
                    parts.append(f"{expr} LIKE ?")
                    params.append(f"{val}%")
                    continue
                elif op == "$endswith":
                    parts.append(f"{expr} LIKE ?")
                    params.append(f"%{val}")
                    continue
                else:
                    raise FilterError(f"unsupported operator: {op!r}")
                params.append(val)
            return "(" + " AND ".join(parts) + ")" if parts else "1=1"
        parts = [f"{expr} = ?"]
        params.append(cond)
        return "(" + " AND ".join(parts) + ")"

    for key, cond in where.items():
        if key == "$and":
            sub = [compile_filter(c, alias) for c in cond]
            clauses.append("(" + " AND ".join(s[0] for s in sub) + ")")
            for s in sub:
                params.extend(s[1])
        elif key == "$or":
            sub = [compile_filter(c, alias) for c in cond]
            clauses.append("(" + " OR ".join(s[0] for s in sub) + ")")
            for s in sub:
                params.extend(s[1])
        elif key.startswith("$"):
            raise FilterError(f"unsupported top-level operator: {key!r}")
        else:
            clauses.append(field_clause(key, cond))

    return ("(" + " AND ".join(clauses) + ")" if clauses else "1=1"), params


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    slot: int
    start: int
    end: int


# ---------------------------------------------------------------------------
# vector storage
# ---------------------------------------------------------------------------
class VectorStore:
    """Growable, memory-mapped, slot-addressed matrix of float16 vectors.

    ``slot`` is a stable row index. Deleted slots are reused through a free
    list, so the file grows with the live corpus rather than with write volume.
    """

    def __init__(self, directory: str | Path, dim: int | None = None, capacity: int = 8192,
                 block: int = DEFAULT_SCAN_BLOCK, release_pages: bool = True):
        self.block = max(256, int(block))
        # Dropping pages after each scan step is what keeps resident memory flat
        # in corpus size; without it a full scan pulls the whole matrix in.
        self.release_pages = bool(release_pages)
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "vectors.npy"
        self.meta_path = self.dir / "vectors.json"
        if self.path.exists():
            self._matrix = np.load(self.path, mmap_mode="r+")
            self.dim = int(self._matrix.shape[1])
            # Sequential readahead would pull in far more than the block we ask
            # for; this scan is block-at-a-time and gains nothing from it.
            self.capacity = int(self._matrix.shape[0])
            self._advise(_mmap.MADV_RANDOM, 0, self.capacity)
            meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}
            self._n = int(meta.get("n", self._matrix.shape[0]))
            self._free: list[int] = [int(x) for x in meta.get("free", [])]
            if dim is not None and int(dim) != self.dim:
                raise ValueError(f"index was built with dim={self.dim}, got dim={dim}")
        else:
            if dim is None:
                raise ValueError("dim is required when creating a new vector store")
            self.dim = int(dim)
            self._n = 0
            self._free = []
            self.capacity = max(int(capacity), 1)
            self._matrix = np.lib.format.open_memmap(
                self.path, mode="w+", dtype=VECTOR_DTYPE, shape=(self.capacity, self.dim)
            )

    # -- geometry -----------------------------------------------------------
    def __len__(self) -> int:
        return self._n

    @property
    def nbytes(self) -> int:
        return int(self._n * self.dim * np.dtype(VECTOR_DTYPE).itemsize)

    @property
    def file_bytes(self) -> int:
        return int(self.capacity * self.dim * np.dtype(VECTOR_DTYPE).itemsize)

    def _advise(self, option: int, first_row: int, last_row: int) -> None:
        """madvise a row range, page-aligned. Silent where unsupported."""
        handle = getattr(self._matrix, "_mmap", None)
        if handle is None or not hasattr(handle, "madvise"):
            return
        row_bytes = self.dim * np.dtype(VECTOR_DTYPE).itemsize
        lo = (first_row * row_bytes) & ~(_PAGE - 1)
        hi = min((last_row * row_bytes + _PAGE - 1) & ~(_PAGE - 1),
                 self.capacity * row_bytes)
        if hi <= lo:
            return
        try:
            handle.madvise(option, lo, hi - lo)
        except (OSError, ValueError):    # pragma: no cover - platform dependent
            pass

    def _grow(self, needed: int) -> None:
        if needed <= self.capacity:
            return
        new_cap = max(needed, self.capacity * 2)
        tmp = self.dir / "vectors.grow.npy"
        grown = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=VECTOR_DTYPE, shape=(new_cap, self.dim)
        )
        grown[: self.capacity] = self._matrix[: self.capacity]
        grown.flush()
        del self._matrix
        tmp.replace(self.path)
        self._matrix = np.load(self.path, mmap_mode="r+")
        self.capacity = new_cap

    # -- writes -------------------------------------------------------------
    def append(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=VECTOR_DTYPE)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) vectors, got {vectors.shape}")
        n = vectors.shape[0]
        slots: list[int] = []
        while self._free and len(slots) < n:
            slots.append(self._free.pop())
        remaining = n - len(slots)
        if remaining:
            self._grow(self._n + remaining)
            start = self._n
            slots.extend(range(start, start + remaining))
            self._n += remaining
        order = np.asarray(slots, dtype=np.int64)
        self._matrix[order] = vectors
        self._matrix.flush()
        return order

    def overwrite(self, slots: Sequence[int], vectors: np.ndarray) -> None:
        slots = np.asarray(slots, dtype=np.int64)
        self._matrix[slots] = np.asarray(vectors, dtype=VECTOR_DTYPE)
        self._matrix.flush()

    def release(self, slots: Iterable[int]) -> None:
        """Return slots to the free list (tombstone the rows)."""
        slots = [int(s) for s in slots]
        if not slots:
            return
        self._free.extend(slots)
        self._matrix[np.asarray(slots, dtype=np.int64)] = 0
        self._matrix.flush()

    # -- reads --------------------------------------------------------------
    def gather(self, slots: Sequence[int]) -> np.ndarray:
        return np.asarray(self._matrix[np.asarray(slots, dtype=np.int64)], dtype=np.float32)

    def gather_live(self, allowed: np.ndarray | None = None, block: int = 262144) -> np.ndarray:
        """All live vectors as one float32 array. Only for small indexes."""
        idx = np.arange(self._n) if allowed is None else allowed
        out = np.empty((idx.size, self.dim), dtype=np.float32)
        for start in range(0, idx.size, block):
            chunk = idx[start : start + block]
            out[start : start + block] = self._matrix[chunk]
        return out

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        allowed: np.ndarray | None = None,
        block: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-k cosine search, optionally restricted to ``allowed`` slots.

        A selective filter gathers its rows directly; a permissive one masks
        during a streaming scan. Which is cheaper depends on selectivity, so
        both paths exist.
        """
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        block = int(block or self.block)
        if allowed is not None and allowed.size == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

        if allowed is not None and (allowed >= self._n).any():
            allowed = allowed[allowed < self._n]
            if allowed.size == 0:
                return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

        # Gathering a few rows directly is cheaper than masking the whole scan,
        # but only while "a few" fits in the resident budget. Anything larger
        # goes through the streaming path, which never holds more than a block.
        selective = allowed is not None and allowed.size < min(
            0.25 * max(self._n, 1), 2 * block
        )
        if selective:
            sims = self.gather(allowed) @ q
            k = min(top_k, sims.shape[0])
            part = np.argpartition(-sims, k - 1)[:k] if k < sims.shape[0] else np.arange(sims.shape[0])
            order = np.argsort(-sims[part], kind="stable")
            return sims[part][order], allowed[part][order].astype(np.int64)

        mask = None
        if allowed is not None:
            mask = np.zeros(self._n, dtype=bool)
            mask[allowed] = True

        best_s = np.empty(0, dtype=np.float32)
        best_i = np.empty(0, dtype=np.int64)
        n = self._n
        for start in range(0, n, block):
            stop = min(start + block, n)
            sims = self._matrix[start:stop] @ q
            idx = np.arange(start, stop, dtype=np.int64)
            if mask is not None:
                keep = mask[start:stop]
                sims, idx = sims[keep], idx[keep]
            if sims.size == 0:
                continue
            k = min(top_k, sims.shape[0])
            part = np.argpartition(-sims, k - 1)[:k] if k < sims.shape[0] else np.arange(sims.shape[0])
            cand_s, cand_i = sims[part], idx[part]
            if best_s.size:
                cand_s = np.concatenate([best_s, cand_s])
                cand_i = np.concatenate([best_i, cand_i])
            order = np.argsort(-cand_s, kind="stable")[:top_k]
            best_s, best_i = cand_s[order], cand_i[order]
            if self.release_pages:
                self._advise(_mmap.MADV_DONTNEED, start, stop)
        return best_s, best_i

    def flush(self) -> None:
        if hasattr(self._matrix, "flush"):
            self._matrix.flush()
        self.meta_path.write_text(
            json.dumps({"n": self._n, "free": self._free, "dim": self.dim})
        )

    def close(self) -> None:
        self.flush()
        del self._matrix


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------
class Catalog:
    """SQLite catalog of documents, metadata, chunk offsets and tombstones."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "catalog.sqlite"
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
                id      TEXT PRIMARY KEY,
                text    TEXT NOT NULL,
                meta    TEXT NOT NULL DEFAULT '{}',
                deleted INTEGER NOT NULL DEFAULT 0,
                updated REAL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id   TEXT NOT NULL,
                slot     INTEGER NOT NULL,
                start    INTEGER NOT NULL,
                end      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_slot ON chunks(slot);
            CREATE INDEX IF NOT EXISTS idx_docs_deleted ON docs(deleted);
            """
        )
        self._conn.commit()

    # -- documents ----------------------------------------------------------
    def put_document(self, doc_id: str, text: str, metadata: dict | None) -> None:
        self._conn.execute(
            "INSERT INTO docs(id, text, meta, deleted, updated) VALUES(?,?,?,0,strftime('%s','now')) "
            "ON CONFLICT(id) DO UPDATE SET text=excluded.text, meta=excluded.meta, "
            "deleted=0, updated=excluded.updated",
            (doc_id, text, json.dumps(metadata or {}, ensure_ascii=False)),
        )

    def add_chunks(self, records: Sequence[ChunkRecord]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks(chunk_id, doc_id, slot, start, end) VALUES(?,?,?,?,?)",
            [(r.chunk_id, r.doc_id, r.slot, r.start, r.end) for r in records],
        )

    def chunks_of(self, doc_id: str) -> list[ChunkRecord]:
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, slot, start, end FROM chunks WHERE doc_id=?", (doc_id,)
        ).fetchall()
        return [ChunkRecord(*r) for r in rows]

    def delete_document(self, doc_id: str) -> list[int]:
        slots = [r[0] for r in self._conn.execute(
            "SELECT slot FROM chunks WHERE doc_id=?", (doc_id,)
        ).fetchall()]
        self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self._conn.execute("DELETE FROM docs WHERE id=?", (doc_id,))
        return slots

    def get_document(self, doc_id: str) -> tuple[str, dict] | None:
        row = self._conn.execute("SELECT text, meta FROM docs WHERE id=?", (doc_id,)).fetchone()
        if row is None:
            return None
        return row[0], json.loads(row[1])

    def text_of(self, doc_id: str) -> str:
        row = self._conn.execute("SELECT text FROM docs WHERE id=?", (doc_id,)).fetchone()
        if row is None:
            raise KeyError(doc_id)
        return row[0]

    def has_document(self, doc_id: str) -> bool:
        return self._conn.execute("SELECT 1 FROM docs WHERE id=?", (doc_id,)).fetchone() is not None

    def document_ids(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT id FROM docs ORDER BY rowid")]

    def live_slots(self) -> np.ndarray:
        rows = self._conn.execute("SELECT slot FROM chunks").fetchall()
        return np.asarray([r[0] for r in rows], dtype=np.int64)

    def vector_dim(self) -> int:
        """Dimension recorded in the vector file, for reopening."""
        meta = self.dir / "vectors.json"
        if meta.exists():
            dim = json.loads(meta.read_text()).get("dim")
            if dim:
                return int(dim)
        matrix = self.dir / "vectors.npy"
        if matrix.exists():
            return int(np.load(matrix, mmap_mode="r").shape[1])
        return 0

    def max_slot(self) -> int:
        row = self._conn.execute("SELECT MAX(slot) FROM chunks").fetchone()
        return int(row[0]) if row and row[0] is not None else -1

    def max_doc_rowid(self) -> int:
        row = self._conn.execute("SELECT MAX(rowid) FROM docs").fetchone()
        return int(row[0]) if row and row[0] is not None else -1

    def slot_rows(self) -> Iterator[ChunkRecord]:
        """Stream the chunk table: a million rows must not become a million
        Python objects held at once while reopening an index."""
        cursor = self._conn.execute(
            "SELECT chunk_id, doc_id, slot, start, end FROM chunks ORDER BY slot"
        )
        for row in cursor:
            yield ChunkRecord(*row)

    def document_rowids(self) -> Iterator[tuple[int, str]]:
        """``(rowid, doc_id)`` for every document, in rowid order.

        Same order as :meth:`document_ids`, but the rowid is carried along so
        the caller can index documents without building a dict of every id.
        """
        for rowid, doc_id in self._conn.execute("SELECT rowid, id FROM docs ORDER BY rowid"):
            yield int(rowid), doc_id

    def slot_rows_indexed(self) -> Iterator[tuple[ChunkRecord, int]]:
        """Like :meth:`slot_rows`, plus each chunk's document rowid.

        Resolving the rowid in SQL keeps reopening O(chunks) with no in-Python
        ``doc_id -> position`` dict for the whole corpus. Ordering by slot uses
        the slot index, so there is no large sort either.
        """
        cursor = self._conn.execute(
            "SELECT c.chunk_id, c.doc_id, c.slot, c.start, c.end, d.rowid "
            "FROM chunks c JOIN docs d ON d.id = c.doc_id "
            "ORDER BY c.slot"
        )
        for chunk_id, doc_id, slot, start, end, doc_rowid in cursor:
            yield ChunkRecord(chunk_id, doc_id, slot, start, end), int(doc_rowid)

    def replace_chunks(self, records: Sequence[ChunkRecord]) -> None:
        self._conn.execute("DELETE FROM chunks")
        self.add_chunks(records)

    def filter_slots(self, where: dict | None) -> np.ndarray | None:
        """Slots whose document matches ``where``, or ``None`` for 'no filter'."""
        if not where:
            return None
        predicate, params = compile_filter(where)
        sql = (
            "SELECT c.slot FROM chunks c JOIN docs d ON d.id = c.doc_id "
            f"WHERE {predicate}"
        )
        rows = self._conn.execute(sql, params).fetchall()
        return np.asarray([r[0] for r in rows], dtype=np.int64)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()


# ---------------------------------------------------------------------------
# in-memory equivalents
# ---------------------------------------------------------------------------
class MemVectorStore:
    """Same interface and slot semantics as :class:`VectorStore`, held in RAM."""

    def __init__(self, dim: int, capacity: int = 8192):
        self.dim = int(dim)
        self._matrix = np.zeros((max(int(capacity), 1), self.dim), dtype=VECTOR_DTYPE)
        self._matrix32: np.ndarray | None = None
        self._n = 0
        self._free: list[int] = []
        self._block = 8192

    def __len__(self) -> int:
        return self._n

    @property
    def nbytes(self) -> int:
        return int(self._n * self.dim * np.dtype(VECTOR_DTYPE).itemsize)

    @property
    def file_bytes(self) -> int:
        return int(self._matrix.shape[0] * self.dim * np.dtype(VECTOR_DTYPE).itemsize)

    def _grow(self, needed: int) -> None:
        if needed <= self._matrix.shape[0]:
            return
        new_cap = max(needed, self._matrix.shape[0] * 2)
        grown = np.zeros((new_cap, self.dim), dtype=VECTOR_DTYPE)
        grown[: self._matrix.shape[0]] = self._matrix
        self._matrix = grown
        self._matrix32 = None

    def append(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=VECTOR_DTYPE)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) vectors, got {vectors.shape}")
        n = vectors.shape[0]
        slots: list[int] = []
        while self._free and len(slots) < n:
            slots.append(self._free.pop())
        remaining = n - len(slots)
        if remaining:
            self._grow(self._n + remaining)
            start = self._n
            slots.extend(range(start, start + remaining))
            self._n += remaining
        order = np.asarray(slots, dtype=np.int64)
        self._matrix[order] = vectors
        self._matrix32 = None
        return order

    def overwrite(self, slots, vectors) -> None:
        self._matrix[np.asarray(slots, dtype=np.int64)] = np.asarray(vectors, dtype=VECTOR_DTYPE)
        self._matrix32 = None

    def release(self, slots) -> None:
        slots = [int(s) for s in slots]
        if not slots:
            return
        self._free.extend(slots)
        self._matrix[np.asarray(slots, dtype=np.int64)] = 0
        self._matrix32 = None

    def gather(self, slots) -> np.ndarray:
        return np.asarray(self._matrix[np.asarray(slots, dtype=np.int64)], dtype=np.float32)

    def search(self, query, top_k, allowed=None, block=65536):
        if self._n == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        sims = self._matrix[: self._n] @ q
        idx_all = np.arange(self._n, dtype=np.int64)
        if allowed is not None:
            if allowed.size == 0:
                return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
            mask = np.zeros(self._n, dtype=bool)
            mask[allowed[allowed < self._n]] = True
            keep = mask
            sims, idx_all = sims[keep], idx_all[keep]
        k = min(top_k, sims.shape[0])
        if k == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        part = np.argpartition(-sims, k - 1)[:k] if k < sims.shape[0] else np.arange(sims.shape[0])
        order = np.argsort(-sims[part], kind="stable")
        return sims[part][order].astype(np.float32), idx_all[part][order].astype(np.int64)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class MemCatalog:
    """Same interface as :class:`Catalog`, backed by dicts."""

    def __init__(self):
        self._docs: dict[str, tuple[str, dict]] = {}
        self._chunks: dict[str, list[ChunkRecord]] = {}
        self._slot_doc: dict[int, str] = {}

    def put_document(self, doc_id, text, metadata) -> None:
        self._docs[doc_id] = (text, dict(metadata or {}))

    def add_chunks(self, records: Sequence[ChunkRecord]) -> None:
        for r in records:
            self._chunks.setdefault(r.doc_id, []).append(r)
            self._slot_doc[r.slot] = r.doc_id

    def chunks_of(self, doc_id: str) -> list[ChunkRecord]:
        return list(self._chunks.get(doc_id, []))

    def delete_document(self, doc_id: str) -> list[int]:
        records = self._chunks.pop(doc_id, [])
        for r in records:
            self._slot_doc.pop(r.slot, None)
        self._docs.pop(doc_id, None)
        return [r.slot for r in records]

    def get_document(self, doc_id: str):
        return self._docs.get(doc_id)

    def text_of(self, doc_id: str) -> str:
        return self._docs[doc_id][0]

    def has_document(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def document_ids(self) -> list[str]:
        return list(self._docs)

    def live_slots(self) -> np.ndarray:
        return np.asarray(sorted(self._slot_doc), dtype=np.int64)

    def vector_dim(self) -> int:
        """Dimension recorded in the vector file, for reopening."""
        meta = self.dir / "vectors.json"
        if meta.exists():
            dim = json.loads(meta.read_text()).get("dim")
            if dim:
                return int(dim)
        matrix = self.dir / "vectors.npy"
        if matrix.exists():
            return int(np.load(matrix, mmap_mode="r").shape[1])
        return 0

    def max_slot(self) -> int:
        return max(self._slot_doc, default=-1)

    def slot_rows(self) -> list[ChunkRecord]:
        out: list[ChunkRecord] = []
        for recs in self._chunks.values():
            out.extend(recs)
        return sorted(out, key=lambda r: r.slot)

    def replace_chunks(self, records: Sequence[ChunkRecord]) -> None:
        self._chunks.clear()
        self._slot_doc.clear()
        self.add_chunks(records)

    def filter_slots(self, where: dict | None) -> np.ndarray | None:
        if not where:
            return None
        matched = {d for d, (_, meta) in self._docs.items() if matches(meta, where)}
        return np.asarray(
            sorted(s for s, d in self._slot_doc.items() if d in matched), dtype=np.int64
        )

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# python-side filter evaluation (in-memory catalog)
# ---------------------------------------------------------------------------
def _cmp(actual, expected, op: str) -> bool:
    if actual is None:
        return op == "$ne"
    try:
        if op == "$eq":
            return actual == expected
        if op == "$ne":
            return actual != expected
        if op == "$gt":
            return actual > expected
        if op == "$gte":
            return actual >= expected
        if op == "$lt":
            return actual < expected
        if op == "$lte":
            return actual <= expected
    except TypeError:
        return False
    return False


def matches(metadata: dict, where: dict) -> bool:
    """Evaluate a filter against one metadata dict, mirroring :func:`compile_filter`."""
    for key, cond in where.items():
        if key == "$and":
            if not all(matches(metadata, c) for c in cond):
                return False
            continue
        if key == "$or":
            if not any(matches(metadata, c) for c in cond):
                return False
            continue
        actual = metadata.get(key)
        if isinstance(cond, dict):
            for op, val in cond.items():
                if op == "$exists":
                    if bool(actual is not None) != bool(val):
                        return False
                elif op == "$in":
                    if actual not in val:
                        return False
                elif op == "$nin":
                    if actual in val:
                        return False
                elif op == "$contains":
                    if not isinstance(actual, str) or str(val) not in actual:
                        return False
                elif op == "$startswith":
                    if not isinstance(actual, str) or not actual.startswith(str(val)):
                        return False
                elif op == "$endswith":
                    if not isinstance(actual, str) or not actual.endswith(str(val)):
                        return False
                elif not _cmp(actual, val, op):
                    return False
        elif actual != cond:
            return False
    return True
