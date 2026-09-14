"""Core data types for arara-rag."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A contiguous span of a source document.

    ``text`` is guaranteed to be an exact substring of the source document at
    ``[start:end]``. This invariant is what makes chunk-level citations
    possible and is enforced by the test suite.
    """

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int


@dataclass
class Hit:
    """A search result.

    Score semantics depend on the retrieval mode. Scores are only comparable
    within a single call to :meth:`Arara.search`.
    """

    doc_id: str
    score: float
    chunk_id: str | None = None
    text: str | None = None
    start: int | None = None
    end: int | None = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        preview = (self.text or "")[:60].replace("\n", " ")
        return f"Hit(doc_id={self.doc_id!r}, score={self.score:.4f}, chunk_id={self.chunk_id!r}, text={preview!r})"
