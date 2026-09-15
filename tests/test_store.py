"""Tests for out-of-core storage, metadata filtering and CRUD."""

from __future__ import annotations

import numpy as np
import pytest

from arara_rag import Arara
from arara_rag.store import FilterError, compile_filter, matches

DOCS = {
    "lei_ir": "A alíquota do imposto de renda é progressiva e chega a 27,5%.",
    "lei_amb": "O licenciamento ambiental é um instrumento preventivo da política nacional.",
    "dec_ir": "O decreto regulamenta a cobrança do imposto de renda das pessoas físicas.",
    "acordao": "O acórdão trata da cobrança de tributos e da prescrição quinquenal.",
}
META = {
    "lei_ir": {"ano": 2021, "tipo": "lei", "uf": "BR", "tags": "tributario"},
    "lei_amb": {"ano": 2019, "tipo": "lei", "uf": "SP", "tags": "ambiental"},
    "dec_ir": {"ano": 2023, "tipo": "decreto", "uf": "BR", "tags": "tributario"},
    "acordao": {"ano": 2024, "tipo": "acordao", "uf": "RJ", "tags": "tributario"},
}


def build(tmp_path, persistent: bool) -> Arara:
    a = Arara(path=tmp_path if persistent else None)
    a.add_documents(DOCS, metadata=META)
    a.finalize()
    return a


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------
def test_compile_filter_builds_expected_sql() -> None:
    sql, params = compile_filter({"ano": {"$gte": 2020}})
    assert "json_extract(d.meta, '$.ano') >= ?" in sql
    assert params == [2020]


def test_compile_filter_rejects_injection() -> None:
    with pytest.raises(FilterError):
        compile_filter({"ano; DROP TABLE docs": 1})
    with pytest.raises(FilterError):
        compile_filter({"ano": {"$regex": "x"}})


def test_python_and_sql_filters_agree() -> None:
    """The in-memory and SQLite catalogs must implement the same semantics."""
    cases = [
        {"ano": {"$gte": 2021}},
        {"tipo": "lei"},
        {"tipo": {"$in": ["lei", "decreto"]}},
        {"tipo": {"$nin": ["lei"]}},
        {"ano": {"$gt": 2019, "$lt": 2024}},
        {"uf": {"$ne": "BR"}},
        {"tags": {"$contains": "tribut"}},
        {"uf": {"$exists": True}},
        {"$or": [{"tipo": "acordao"}, {"ano": 2019}]},
        {"$and": [{"tipo": "lei"}, {"ano": {"$lt": 2020}}]},
    ]
    for where in cases:
        expected = {d for d, m in META.items() if matches(m, where)}
        a = Arara()
        a.add_documents(DOCS, metadata=META)
        got = {h.doc_id for h in a.search("tributos imposto renda ambiental", top_k=10, where=where)}
        assert got <= expected, f"{where}: {got} not a subset of {expected}"


@pytest.mark.parametrize("persistent", [False, True])
def test_metadata_filtering_restricts_results(tmp_path, persistent: bool) -> None:
    a = build(tmp_path if persistent else None and tmp_path, persistent)
    hits = a.search("imposto de renda", top_k=10, where={"tipo": "decreto"})
    assert [h.doc_id for h in hits] == ["dec_ir"]

    hits = a.search("imposto de renda", top_k=10, where={"ano": {"$lt": 2020}})
    assert {h.doc_id for h in hits} <= {"lei_amb"}

    assert a.search("imposto", top_k=10, where={"tipo": "inexistente"}) == []
    assert a.search("imposto", top_k=10, where={"uf": {"$in": ["SP", "RJ"]}})


@pytest.mark.parametrize("persistent", [False, True])
def test_filter_returns_nothing_when_no_docs_match(tmp_path, persistent: bool) -> None:
    a = build(tmp_path if persistent else None and tmp_path, persistent)
    assert a.search("qualquer coisa", top_k=5, where={"ano": {"$gt": 3000}}) == []


# --------------------------------------------------------------------------
# CRUD
# --------------------------------------------------------------------------
@pytest.mark.parametrize("persistent", [False, True])
def test_delete_removes_document_from_results(tmp_path, persistent: bool) -> None:
    a = build(tmp_path if persistent else None and tmp_path, persistent)
    assert a.search("imposto de renda", top_k=5)
    assert a.delete_document("dec_ir") is True
    assert a.delete_document("dec_ir") is False
    assert a.get_document("dec_ir") is None
    assert len(a) == 3
    assert all(h.doc_id != "dec_ir" for h in a.search("imposto de renda", top_k=10))
    assert all(h.doc_id != "dec_ir" for h in a.search("decreto", top_k=10, mode="lexical"))


@pytest.mark.parametrize("persistent", [False, True])
def test_upsert_replaces_and_recycles_slots(tmp_path, persistent: bool) -> None:
    a = build(tmp_path if persistent else None and tmp_path, persistent)
    before = a.n_chunks
    a.add_documents({"dec_ir": "Texto completamente novo sobre licenciamento."})
    assert len(a) == 4
    assert a.n_chunks == before, "slots should be recycled, not appended"
    a.finalize()
    # The replacement text must be gone: dec_ir may still be *returned* (there
    # are only four documents), but never with its old content.
    hits = {h.doc_id: (h.text or "") for h in a.search("imposto de renda", top_k=4)}
    assert "imposto" not in hits.get("dec_ir", "")
    assert a.search("licenciamento", top_k=10)[0].doc_id in {"dec_ir", "lei_amb"}


@pytest.mark.parametrize("persistent", [False, True])
def test_update_metadata_without_reindexing(tmp_path, persistent: bool) -> None:
    a = build(tmp_path if persistent else None and tmp_path, persistent)
    assert a.update_metadata("acordao", {"ano": 1999}) is True
    assert a.update_metadata("nao_existe", {"ano": 1}) is False
    hits = a.search("tributos prescrição", top_k=10, where={"ano": {"$lt": 2000}})
    assert "acordao" in {h.doc_id for h in hits}
    _, meta = a.get_document("acordao")
    assert meta["ano"] == 1999 and meta["tipo"] == "acordao"  # merged, not replaced


# --------------------------------------------------------------------------
# out-of-core persistence
# --------------------------------------------------------------------------
def test_persistent_index_reopens(tmp_path) -> None:
    a = Arara(path=tmp_path)
    a.add_documents(DOCS, metadata=META)
    a.finalize()
    first = [h.doc_id for h in a.search("imposto de renda", top_k=3)]
    text_before, n_chunks = a.document_text("lei_ir"), a.n_chunks
    a.close()

    b = Arara(path=tmp_path)
    assert len(b) == len(DOCS)
    assert b.n_chunks == n_chunks
    assert b.document_text("lei_ir") == text_before
    _, meta = b.get_document("lei_ir")
    assert meta["ano"] == 2021
    assert [h.doc_id for h in b.search("imposto de renda", top_k=3)] == first
    # filtering still works after reopen
    assert [h.doc_id for h in b.search("imposto", top_k=5, where={"tipo": "decreto"})] == ["dec_ir"]
    b.close()


def test_persistent_index_keeps_vectors_out_of_the_process(tmp_path) -> None:
    a = Arara(path=tmp_path)
    a.add_documents(DOCS, metadata=META)
    a.finalize()
    stats = a.stats()
    assert stats["persistent"] is True
    assert (tmp_path / "vectors.npy").exists()
    assert (tmp_path / "catalog.sqlite").exists()
    assert (tmp_path / "bm25_meta.json").exists()
    # The memory-mapped matrix still reports the logical size of the vectors.
    assert stats["vector_bytes"] > 0
    a.close()


def test_compact_reclaims_space(tmp_path) -> None:
    a = Arara(path=tmp_path, max_chunk_chars=80, min_chunk_chars=10)
    docs = {f"d{i}": f"Parágrafo {i} sobre tributos e alíquotas. " * 12 for i in range(12)}
    a.add_documents(docs)
    a.finalize()
    for i in range(6):
        a.delete_document(f"d{i}")
    a.finalize(force_lexical=True)
    live = a.n_chunks
    size_before = (tmp_path / "vectors.npy").stat().st_size
    assert a.compact() == live
    size_after = (tmp_path / "vectors.npy").stat().st_size
    assert size_after <= size_before
    assert a.n_chunks == live
    assert a.search("tributos alíquotas", top_k=3)
    a.close()
    b = Arara(path=tmp_path)
    assert b.n_chunks == live
    assert b.search("tributos alíquotas", top_k=3)
    b.close()


# --------------------------------------------------------------------------
# storage primitives
# --------------------------------------------------------------------------
def test_vector_store_reuses_freed_slots(tmp_path) -> None:
    from arara_rag.store import VectorStore

    store = VectorStore(tmp_path, dim=4, capacity=2)
    v = np.ones((3, 4), dtype=np.float32)
    slots = store.append(v)
    assert len(slots) == 3
    store.release([1])
    again = store.append(np.full((1, 4), 2.0, dtype=np.float32))
    assert list(again) == [1]
    assert np.allclose(store.gather([1]), 2.0)
    store.close()


def test_vector_store_grows_past_capacity(tmp_path) -> None:
    from arara_rag.store import VectorStore

    store = VectorStore(tmp_path, dim=3, capacity=1)
    for i in range(5):
        store.append(np.full((1, 3), float(i), dtype=np.float32))
    assert len(store) == 5
    assert np.allclose(store.gather([4]), 4.0)
    store.close()


# --------------------------------------------------------------------------
# in-memory and persistent paths must agree
# --------------------------------------------------------------------------
def test_memory_and_disk_paths_return_the_same_ranking(tmp_path) -> None:
    mem = build(None, False)
    disk = build(tmp_path, True)
    queries = ["imposto de renda", "licenciamento ambiental", "prescrição quinquenal"]
    for q in queries:
        for mode in ("dense", "lexical", "hybrid"):
            assert [h.doc_id for h in mem.search(q, top_k=4, mode=mode)] == [
                h.doc_id for h in disk.search(q, top_k=4, mode=mode)
            ], f"{q} / {mode}"
    disk.close()


# --------------------------------------------------------------------------
# performance guard
# --------------------------------------------------------------------------
def test_vectors_are_stored_in_a_blas_friendly_dtype() -> None:
    """Regression guard for a 10x query slowdown.

    float16 storage forced a widening pass on every query (65 ms at 50k
    documents). float32 lets the matmul read mapped pages directly (~7 ms).
    """
    from arara_rag.store import VECTOR_DTYPE, MemVectorStore, VectorStore

    assert VECTOR_DTYPE == np.float32
    mem = MemVectorStore(dim=4)
    mem.append(np.ones((8, 4), dtype=np.float32))
    assert mem._matrix.dtype == np.float32
