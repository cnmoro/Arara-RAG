"""arara-rag -- a Portuguese-first, CPU-only RAG retrieval stack.

Everything runs on CPU with numpy. No PyTorch, no ONNX Runtime, no FAISS.

    >>> from arara_rag import Arara
    >>> a = Arara()
    >>> a.add_documents({"doc": "texto ..."})
    >>> hits = a.search("pergunta", top_k=5)
"""

from .chunk import Chunker
from .dense import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_USE_MODEL,
    ENCODER_BACKENDS,
    DenseEncoder,
    DenseIndex,
    StaticDenseEncoder,
    USEDenseEncoder,
    build_encoder,
)
from .fuse import rank_from_scores, reciprocal_rank_fusion
from .lexical import BM25Index, CXM25Scorer
from .pipeline import Arara
from .text import Tokenizer
from .types import Chunk, Hit

__version__ = "0.2.0"

__all__ = [
    "Arara",
    "BM25Index",
    "CXM25Scorer",
    "Chunk",
    "Chunker",
    "DEFAULT_DENSE_MODEL",
    "DEFAULT_USE_MODEL",
    "DenseEncoder",
    "DenseIndex",
    "ENCODER_BACKENDS",
    "Hit",
    "StaticDenseEncoder",
    "Tokenizer",
    "USEDenseEncoder",
    "build_encoder",
    "rank_from_scores",
    "reciprocal_rank_fusion",
    "__version__",
]
