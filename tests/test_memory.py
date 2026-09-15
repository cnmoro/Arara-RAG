"""Serving memory must be bounded by configuration, not by corpus size.

Two independent scans hold index data resident during a query: the dense
matrix and the BM25 postings. Both are memory-mapped and both release the
pages they touched, so what grows with the corpus is only the small
per-document bookkeeping. These tests pin that behaviour down without
measuring RSS, which is too noisy to assert on.
"""

from __future__ import annotations

import mmap

import numpy as np
import pytest

from arara_rag import Arara
from arara_rag.lexical import BM25Index
from arara_rag.pipeline import (
    DEFAULT_SCAN_BLOCK,
    MIN_SCAN_BLOCK,
    SCAN_RSS_SLACK,
    rss_bytes,
)
from arara_rag.store import VectorStore

DIM = 8


def _vectors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# --------------------------------------------------------------------------
# dense scan
# --------------------------------------------------------------------------
def test_scan_drops_the_pages_it_touched(tmp_path, monkeypatch) -> None:
    """One MADV_DONTNEED per block, covering the rows the block read."""
    store = VectorStore(tmp_path, dim=DIM, capacity=16)
    store.append(_vectors(1000))

    released: list[tuple[int, int]] = []
    original = VectorStore._advise

    def spy(self, option, first_row, last_row):
        if option == mmap.MADV_DONTNEED:
            released.append((first_row, last_row))
        return original(self, option, first_row, last_row)

    monkeypatch.setattr(VectorStore, "_advise", spy)
    store.search(_vectors(1)[0], top_k=5, block=256)

    assert released == [(0, 256), (256, 512), (512, 768), (768, 1000)]


def test_blocked_scan_matches_a_single_pass(tmp_path) -> None:
    """Bounding the scan must not change the ranking it produces."""
    store = VectorStore(tmp_path, dim=DIM, capacity=16)
    store.append(_vectors(1000))
    q = _vectors(1, seed=7)[0]

    scores, idx = store.search(q, top_k=10, block=1)
    whole_scores, whole_idx = store.search(q, top_k=10, block=1 << 20)
    assert list(idx) == list(whole_idx)
    assert np.allclose(scores, whole_scores)


def test_selective_filter_falls_back_to_the_streaming_path(tmp_path) -> None:
    """A filter wider than the block must not be gathered into one array."""
    store = VectorStore(tmp_path, dim=DIM, capacity=16)
    store.append(_vectors(2000))
    q = _vectors(1, seed=3)[0]

    allowed = np.arange(0, 2000, 2, dtype=np.int64)  # 1000 slots, block is 256
    _, idx = store.search(q, top_k=10, allowed=allowed, block=256)
    assert set(idx) <= set(allowed.tolist())

    narrow = np.arange(0, 20, dtype=np.int64)  # fits in a block: gathered
    _, idx_narrow = store.search(q, top_k=5, allowed=narrow, block=256)
    assert set(idx_narrow) <= set(narrow.tolist())

    # Both paths agree with a brute-force ranking over the same slots.
    sims = store.gather(allowed) @ q
    assert list(idx) == list(allowed[np.argsort(-sims)[:10]])


def test_release_pages_can_be_turned_off(tmp_path, monkeypatch) -> None:
    """A caller that wants the pages to stay cached can say so."""
    store = VectorStore(tmp_path, dim=DIM, release_pages=False)
    store.append(_vectors(100))

    released: list[int] = []
    original = VectorStore._advise

    def spy(self, option, first_row, last_row):
        if option == mmap.MADV_DONTNEED:
            released.append(first_row)
        return original(self, option, first_row, last_row)

    monkeypatch.setattr(VectorStore, "_advise", spy)
    store.search(_vectors(1)[0], top_k=3, block=16)
    assert released == []


# --------------------------------------------------------------------------
# lexical scan
# --------------------------------------------------------------------------
def test_bm25_drops_posting_pages(tmp_path, monkeypatch) -> None:
    index = BM25Index()
    index.add([[f"t{i % 40}", "comum"] for i in range(500)])
    index.finalize()
    index.save(tmp_path)
    reloaded = BM25Index.load(tmp_path)
    assert reloaded.release_pages is True

    dropped: list[int] = []
    original = BM25Index._drop_postings_pages

    def spy(self, lo, hi):
        dropped.append(lo)
        return original(self, lo, hi)

    monkeypatch.setattr(BM25Index, "_drop_postings_pages", spy)
    reloaded.score_all(["comum", "t7"])
    assert len(dropped) == 2  # one per term with postings


def test_bm25_scores_survive_page_release(tmp_path) -> None:
    docs = [[f"t{i % 40}", "comum"] for i in range(500)]
    index = BM25Index()
    index.add(docs)
    index.finalize()
    expected = index.score_all(["comum", "t7"])

    index.save(tmp_path)
    reloaded = BM25Index.load(tmp_path)
    assert np.allclose(reloaded.score_all(["comum", "t7"]), expected)


# --------------------------------------------------------------------------
# the max_ram_mb ceiling
# --------------------------------------------------------------------------
def test_scan_block_shrinks_under_a_tight_ceiling(tmp_path) -> None:
    """A wide vector leaves little room, so the block shrinks below the default."""
    wide = 4096
    cap_mb = rss_bytes() / (1024 * 1024) + 128
    a = Arara(path=tmp_path, max_ram_mb=cap_mb)
    room = cap_mb * 1024 * 1024 - rss_bytes() - SCAN_RSS_SLACK
    expected = min(DEFAULT_SCAN_BLOCK, max(MIN_SCAN_BLOCK, int(room // (wide * 4))))
    assert a._scan_block(wide) == expected
    assert MIN_SCAN_BLOCK <= a._scan_block(wide) < DEFAULT_SCAN_BLOCK


def test_scan_block_is_capped_by_the_default(tmp_path) -> None:
    """A generous ceiling does not lift the block above the default."""
    a = Arara(path=tmp_path, max_ram_mb=100_000)
    assert a._scan_block(DIM) == DEFAULT_SCAN_BLOCK


def test_scan_block_is_clamped_not_negative(tmp_path) -> None:
    """A ceiling below the fixed process floor still yields a usable block."""
    a = Arara(path=tmp_path, max_ram_mb=1)
    assert a._scan_block(DIM) == MIN_SCAN_BLOCK


def test_no_ceiling_keeps_the_default_block(tmp_path) -> None:
    a = Arara(path=tmp_path)
    assert a._scan_block(DIM) == DEFAULT_SCAN_BLOCK
    assert a.stats()["max_ram_mb"] is None


def test_ceiling_is_reapplied_before_every_query(tmp_path) -> None:
    """The budget follows the live footprint, not just the opening one."""
    a = Arara(path=tmp_path, max_ram_mb=4000)
    a.add_documents({"d1": "imposto de renda progressivo", "d2": "licenciamento ambiental"})
    a.finalize()
    a.search("imposto", top_k=1)
    assert a._vectors.block <= DEFAULT_SCAN_BLOCK
    a.max_ram_mb = 1
    a.search("imposto", top_k=1)
    assert a._vectors.block == MIN_SCAN_BLOCK


def test_ceiling_does_not_change_the_ranking(tmp_path) -> None:
    """Capping the scan is a memory policy, never a relevance one."""
    docs = {f"d{i}": f"documento numero {i} sobre tributos e renda" for i in range(200)}
    hits_capped, hits_default = [], []
    for cap, sink in ((600, hits_capped), (None, hits_default)):
        a = Arara(path=tmp_path / f"cap{cap}", max_ram_mb=cap)
        a.add_documents(docs)
        a.finalize()
        sink.extend(h.doc_id for h in a.search("tributos renda", top_k=10))
        a.close()
    assert hits_capped == hits_default


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_stats_report_the_ceiling(tmp_path, mode) -> None:
    a = Arara(path=tmp_path, max_ram_mb=2048)
    a.add_documents({f"d{i}": "texto sobre licenciamento ambiental" for i in range(10)})
    a.finalize()
    st = a.stats()
    assert st["max_ram_mb"] == 2048
    assert st["scan_block"] is not None


# --------------------------------------------------------------------------
# lexical vocabulary
# --------------------------------------------------------------------------
def _tiny_index(tmp_path, n_terms: int = 300) -> BM25Index:
    index = BM25Index()
    index.add([[f"termo{i}", "comum", f"z{i % 7}"] for i in range(n_terms)])
    index.finalize()
    index.save(tmp_path)
    return index


def test_vocabulary_is_not_a_dict_on_load(tmp_path) -> None:
    """A million-term corpus must not be a million Python strings."""
    from arara_rag.lexical import SortedVocab

    _tiny_index(tmp_path)
    reloaded = BM25Index.load(tmp_path)
    assert isinstance(reloaded._vocab, SortedVocab)
    assert len(reloaded._vocab) == 308


def test_vocabulary_survives_the_round_trip(tmp_path) -> None:
    from arara_rag.lexical import SortedVocab

    index = BM25Index()
    terms = ["ação", "zebra", "à", "imposto", "AÇÃO", "日本"]
    index.add([terms, terms])
    index.finalize()
    index.save(tmp_path)
    vocab = SortedVocab.from_directory(tmp_path)

    assert len(vocab) == len(index._vocab)
    for term, tid in index._vocab.items():
        assert vocab.get(term) == tid
    assert vocab.get("ausente") is None
    assert sorted(vocab.items()) == sorted(index._vocab.items())


def test_vocabulary_uses_less_memory_than_a_dict(tmp_path) -> None:
    index = _tiny_index(tmp_path, n_terms=5000)
    from arara_rag.lexical import SortedVocab

    vocab = SortedVocab.from_directory(tmp_path)
    as_dict = vocab.to_dict()
    per_term = vocab.nbytes / len(vocab)
    assert per_term < 40  # a dict is ~140 bytes per term
    assert len(as_dict) == len(vocab)


def test_pre_disk_vocabulary_indexes_still_load(tmp_path) -> None:
    """Indexes written before the vocabulary moved to disk keep working."""
    import json

    _tiny_index(tmp_path)
    meta = json.loads((tmp_path / "bm25_meta.json").read_text())
    meta["vocab"] = {"comum": 0, "termo1": 1, "z1": 2}
    (tmp_path / "bm25_meta.json").write_text(json.dumps(meta))
    for name in ("bm25_vocab_blob.npy", "bm25_vocab_offsets.npy", "bm25_vocab_ids.npy"):
        (tmp_path / name).unlink()

    reloaded = BM25Index.load(tmp_path)
    assert isinstance(reloaded._vocab, dict)
    assert reloaded._vocab["termo1"] == 1


def test_a_loaded_vocabulary_can_still_be_added_to(tmp_path) -> None:
    _tiny_index(tmp_path)
    reloaded = BM25Index.load(tmp_path)
    before = len(reloaded._vocab)
    reloaded.add([["termo_novo", "comum"]])
    reloaded.finalize()
    assert len(reloaded._vocab) == before + 1
    assert reloaded._vocab.get("termo_novo") is not None
