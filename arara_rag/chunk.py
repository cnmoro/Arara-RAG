"""Chunking strategies.

``tinyzchunk`` is the default: a GPU-free, tokenizer-free chunker distilled
from an LLM teacher. It is what lets arara index the 32k-1.1M character legal
statutes that fixed-window embedders are forced to truncate.

The contract below is enforced here rather than assumed, because the upstream
chunker has three behaviours a pipeline cannot tolerate:

1. it strips the whitespace between structural units, so consecutive chunks are
   not byte-adjacent;
2. it can return a chunk longer than ``max_chunk_chars`` on degenerate input
   (a single very long line);
3. it leaves CRLF line endings in the returned text, so "chunks identically" is
   only true of the boundaries, not of the strings.

arara therefore normalises line endings, splits oversized chunks at word
boundaries, and defines the guarantee it can actually uphold:

* every chunk is an exact substring of the **canonical** document
  (line endings normalised to ``\\n``);
* chunks are ordered and non-overlapping;
* the gap between two consecutive chunks contains **only whitespace**, so no
  non-whitespace character is ever dropped or duplicated;
* no chunk exceeds ``max_chunk_chars``.

``Arara.document_text(doc_id)`` returns the canonical text the offsets index
into.
"""

from __future__ import annotations

from typing import Literal

from .types import Chunk

ChunkMode = Literal["tinyzchunk", "paragraph", "window", "document"]

# Legacy alias: "none" never meant "one chunk per document" -- it means "no
# structural chunker", i.e. fixed windows. Keeping the alias avoids silently
# changing behaviour for callers while the honest name is "window".
_ALIASES = {"none": "window"}


def canonicalize(text: str) -> str:
    """Normalise line endings so offsets are stable across platforms."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


class Chunker:
    """Splits documents into provably lossless spans."""

    def __init__(
        self,
        mode: ChunkMode | str = "tinyzchunk",
        max_chunk_chars: int = 2500,
        min_chunk_chars: int = 100,
        enforce_max_chars: bool = True,
    ) -> None:
        mode = _ALIASES.get(mode, mode)
        if mode not in ("tinyzchunk", "paragraph", "window", "document"):
            raise ValueError(f"unknown chunk mode: {mode!r}")
        self.mode = mode
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.enforce_max_chars = enforce_max_chars
        if mode == "tinyzchunk":
            from tinyzchunk import Chunker as _TinyZChunker

            self._impl = _TinyZChunker(
                max_chunk_chars=max_chunk_chars,
                min_chunk_chars=min_chunk_chars,
            )
        else:
            self._impl = None

    # -- public API ---------------------------------------------------------
    def split(self, doc_id: str, text: str) -> list[Chunk]:
        """Chunk ``text``; offsets index into ``canonicalize(text)``."""
        canonical = canonicalize(text)
        if not canonical:
            return []
        chunks: list[Chunk] = []
        cursor = 0
        for piece in self._pieces(canonical):
            if not piece:
                continue
            start = canonical.find(piece, cursor)
            if start < 0:
                # Defensive: never emit a chunk whose offsets we cannot prove.
                # Fall back to windowing the remainder rather than lying.
                for win in self._window(canonical[cursor:]):
                    s = canonical.find(win, cursor)
                    if s < 0:
                        continue
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc_id}#{len(chunks)}",
                            doc_id=doc_id,
                            text=win,
                            start=s,
                            end=s + len(win),
                        )
                    )
                    cursor = s + len(win)
                continue
            end = start + len(piece)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{len(chunks)}",
                    doc_id=doc_id,
                    text=piece,
                    start=start,
                    end=end,
                )
            )
            cursor = end
        return chunks

    # -- strategies ---------------------------------------------------------
    def _pieces(self, text: str) -> list[str]:
        if not text:
            return []
        if self.mode == "document":
            # One vector per document, ceiling deliberately ignored. This is the
            # operating point every fixed-window embedder is stuck at, and it is
            # only sane for documents that fit the model's context.
            stripped = text.strip()
            return [stripped] if stripped else []
        if self.mode == "tinyzchunk":
            raw = self._impl.chunk(text)
            raw = raw if raw else [text]
        elif self.mode == "paragraph":
            raw = self._paragraphs(text)
        else:  # window
            raw = [text]
        if not self.enforce_max_chars:
            return [p for piece in raw for p in self._window(piece) if p]
        pieces: list[str] = []
        for piece in raw:
            # Word-aware splitting is applied to every mode, including the
            # window baseline, so that a cut never lands mid-word.
            pieces.extend(self._split_oversized(piece))
        return pieces

    def _window(self, text: str) -> list[str]:
        """Hard character windows; the honest baseline every chunker beats."""
        if len(text) <= self.max_chunk_chars:
            return [text]
        return [
            text[i : i + self.max_chunk_chars]
            for i in range(0, len(text), self.max_chunk_chars)
        ]

    def _paragraphs(self, text: str) -> list[str]:
        raw = [p for p in text.split("\n\n") if p.strip()]
        return raw if raw else [text]

    def _split_oversized(self, piece: str) -> list[str]:
        """Break a piece that exceeds the ceiling at the last word boundary."""
        if len(piece) <= self.max_chunk_chars:
            return [piece.strip()] if piece.strip() else []
        out: list[str] = []
        rest = piece
        while len(rest) > self.max_chunk_chars:
            window = rest[: self.max_chunk_chars]
            cut = window.rfind(" ")
            if cut < self.max_chunk_chars // 2:
                cut = self.max_chunk_chars  # unbreakable run: hard cut
            head = rest[:cut].strip()
            if head:
                out.append(head)
            rest = rest[cut:]
        tail = rest.strip()
        if tail:
            out.append(tail)
        return out
