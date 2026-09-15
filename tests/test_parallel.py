"""Indexing must be deterministic in the number of workers.

Chunking and embedding run across processes to make index build fast. That is
only acceptable if the result is byte-identical to the sequential path, so these
tests compare the two directly rather than trusting the pool.
"""

from __future__ import annotations

import numpy as np
import pytest

from arara_rag import Arara
from arara_rag.pipeline import PARALLEL_MIN_DOCS

SENTENCES = [
    "A alíquota do imposto de renda é progressiva e varia conforme a faixa.",
    "O licenciamento ambiental é um instrumento preventivo da política nacional.",
    "A CONTRATADA obriga-se a manter sigilo sobre as informações a que tiver acesso.",
    "O prazo para interposição de recurso é de quinze dias úteis contados da intimação.",
    "Compete ao órgão ambiental estadual o licenciamento de impacto regional.",
]


def make_docs(n: int) -> dict[str, str]:
    rng = np.random.default_rng(0)
    out = {}
    for i in range(n):
        body = " ".join(
            str(SENTENCES[j]) for j in rng.integers(0, len(SENTENCES), size=int(rng.integers(2, 6)))
        )
        out[f"doc{i:04d}"] = f"Documento {i}.\n\n{body}"
    return out


def assert_same_index(a: Arara, b: Arara) -> None:
    assert a.n_chunks == b.n_chunks
    assert len(a) == len(b)
    assert a._doc_ids == b._doc_ids
    assert np.array_equal(np.asarray(a._slot_doc), np.asarray(b._slot_doc))
    assert np.array_equal(np.asarray(a._slot_start), np.asarray(b._slot_start))
    assert np.array_equal(np.asarray(a._slot_end), np.asarray(b._slot_end))
    for query in ["imposto de renda", "licenciamento ambiental", "prazo de recurso"]:
        for mode in ("dense", "lexical", "hybrid"):
            assert [h.doc_id for h in a.search(query, top_k=10, mode=mode)] == [
                h.doc_id for h in b.search(query, top_k=10, mode=mode)
            ], f"{query} / {mode}"


@pytest.mark.parametrize("persistent", [False, True])
def test_workers_produce_the_same_index(tmp_path, persistent: bool) -> None:
    docs = make_docs(PARALLEL_MIN_DOCS * 2)
    seq = Arara(workers=1, path=(tmp_path / "seq") if persistent else None)
    par = Arara(workers=4, path=(tmp_path / "par") if persistent else None)
    seq.add_documents(docs)
    par.add_documents(docs)
    seq.finalize()
    par.finalize()
    assert_same_index(seq, par)
    assert par.stats()["timings"]["workers"] == 4.0
    assert seq.stats()["timings"]["workers"] == 1.0
    seq.close()
    par.close()


def test_small_batches_stay_sequential() -> None:
    """Below the threshold a pool costs more than it saves."""
    a = Arara()
    assert a._effective_workers(4) == 1
    assert a._effective_workers(PARALLEL_MIN_DOCS + 1) >= 1


def test_workers_one_is_forced_sequential() -> None:
    a = Arara(workers=1)
    assert a._effective_workers(10_000) == 1


def test_explicit_worker_count_is_honoured() -> None:
    a = Arara(workers=3)
    assert a._effective_workers(10_000) == 3


def test_incremental_adds_match_a_single_batch(tmp_path) -> None:
    """Parallel batches must not change slot assignment across calls."""
    docs = make_docs(PARALLEL_MIN_DOCS + 20)
    half = len(docs) // 2
    first = dict(list(docs.items())[:half])
    second = dict(list(docs.items())[half:])

    batched = Arara(workers=4)
    batched.add_documents(docs)
    batched.finalize()

    split = Arara(workers=4)
    split.add_documents(first)
    split.add_documents(second)
    split.finalize()

    assert_same_index(batched, split)
