"""Tokenization and normalization for Brazilian Portuguese and English.

The stack deliberately uses *one* normalizer everywhere so that lexical and
dense retrieval see the same token stream. When CXM25 is installed we delegate
to its Portuguese Snowball stemmer and stopword list; otherwise we fall back to
a dependency-free accent-folding tokenizer so the package degrades gracefully
instead of failing at import time.
"""

from __future__ import annotations

import re
import unicodedata

try:  # pragma: no cover - exercised by whichever branch is installed
    from cxm25.stats import tokenize_doc as _cxm25_tokenize_doc
    from cxm25.textnorm import Normalizer as _Cxm25Normalizer

    _HAVE_CXM25 = True
except ImportError:  # pragma: no cover
    _HAVE_CXM25 = False

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


class Tokenizer:
    """Language-aware tokenizer with a shared surface for the whole stack."""

    def __init__(self, lang: str = "pt") -> None:
        self.lang = lang
        self._norm = _Cxm25Normalizer(lang=lang) if _HAVE_CXM25 else None

    @property
    def backend(self) -> str:
        return "cxm25" if self._norm is not None else "fallback"

    @property
    def normalizer(self):
        """The underlying CXM25 normalizer, or ``None`` when unavailable."""
        return self._norm

    def terms(self, text: str) -> list[str]:
        """Stemmed content terms used for lexical scoring."""
        if self._norm is not None:
            return list(self._norm(text))
        return [_strip_accents(t.lower()) for t in _WORD.findall(text)]

    def doc(self, text: str):
        """Full tokenization tuple used by the CXM25 scorer.

        Returns ``None`` when CXM25 is unavailable -- the CXM25 rerank stage is
        optional and callers must treat it as such.
        """
        if _HAVE_CXM25:
            return _cxm25_tokenize_doc(self._norm, text)
        return None


DEFAULT_TOKENIZER = Tokenizer("pt")
