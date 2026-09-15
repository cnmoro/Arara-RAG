"""arara-rag -- a Portuguese-first, CPU-only RAG retrieval stack.

Everything runs on CPU with numpy. No PyTorch, no ONNX Runtime, no FAISS on the
query path.

    >>> from arara_rag import Arara
    >>> a = Arara(path="./index")            # out-of-core, persistent
    >>> a.add_documents({"doc": "texto ..."}, metadata={"ano": 2024})
    >>> hits = a.search("pergunta", top_k=5, where={"ano": {"$gte": 2020}})
"""

from .chunk import Chunker
from .dense import DEFAULT_DENSE_MODEL, DenseEncoder, DenseIndex
from .fuse import rank_from_scores, reciprocal_rank_fusion
from .lexical import BM25Index, CXM25Scorer
from .pipeline import Arara
from .text import Tokenizer
from .types import Chunk, Hit

__version__ = "0.3.0"

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
