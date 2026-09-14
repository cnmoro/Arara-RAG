"""arara-rag -- a Portuguese-first, CPU-only RAG retrieval stack.

Everything runs on CPU with numpy. No PyTorch, no ONNX Runtime, no FAISS.

    >>> from arara_rag import Arara
    >>> a = Arara()
    >>> a.add_documents({"doc": "texto ..."})
    >>> hits = a.search("pergunta", top_k=5)
"""

from .chunk import Chunker
from .dense import DEFAULT_DENSE_MODEL, DenseEncoder, DenseIndex
from .fuse import rank_from_scores, reciprocal_rank_fusion
from .lexical import BM25Index, CXM25Scorer
from .pipeline import Arara
from .text import Tokenizer
from .types import Chunk, Hit

__version__ = "0.1.0"

__all__ = [
    "Arara",
    "BM25Index",
    "CXM25Scorer",
    "Chunk",
    "Chunker",
    "DEFAULT_DENSE_MODEL",
    "DenseEncoder",
    "DenseIndex",
    "Hit",
    "Tokenizer",
    "rank_from_scores",
    "reciprocal_rank_fusion",
    "__version__",
]
