"""Dense encoder backends.

The stack ships one default (a Model2Vec lookup table) and one opt-in
alternative (nanoE5.c, a 4-bit multilingual-e5-small in C). They must present
the same interface, and the asymmetric one must not have its query path routed
through the document path -- that silently costs several points of retrieval
quality and is invisible in the output.
"""

from __future__ import annotations

import numpy as np
import pytest

from arara_rag import ENCODER_BACKENDS, Arara, DenseEncoder, build_encoder

DOCS = {
    "trib": "A alíquota do imposto de renda é progressiva e chega a 27,5%.",
    "amb": "O licenciamento ambiental é um instrumento preventivo.",
    "cul": "Bolo de chocolate com cobertura de morangos frescos.",
    "jur": "O acórdão trata da prescrição quinquenal da cobrança de tributos.",
}


def test_registry_and_unknown_backend() -> None:
    assert set(ENCODER_BACKENDS) == {"static", "nanoe5"}
    assert isinstance(build_encoder("static"), DenseEncoder)
    with pytest.raises(ValueError, match="unknown dense backend"):
        build_encoder("does-not-exist")


def test_static_encoder_is_symmetric() -> None:
    enc = DenseEncoder()
    assert enc.backend == "static"
    q = enc.encode_query(["imposto de renda"])
    d = enc.encode(["imposto de renda"])
    assert q.shape == d.shape == (1, 384)
    assert np.allclose(q, d), "a symmetric model must encode both sides the same"


def test_both_backends_share_an_interface() -> None:
    for name, cls in ENCODER_BACKENDS.items():
        if name == "nanoe5":
            pytest.importorskip("nanoe5")
        enc = cls()
        assert enc.dim == 384
        vecs = enc.encode(["um texto qualquer"])
        assert vecs.shape == (1, 384)
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
        q = enc.encode_query(["um texto qualquer"])
        assert q.shape == (1, 384)


def test_nanoe5_is_asymmetric() -> None:
    """E5 prefixes queries and passages differently; the paths must differ."""
    pytest.importorskip("nanoe5")
    enc = build_encoder("nanoe5")
    text = "qual a aliquota do imposto de renda"
    assert not np.allclose(enc.encode([text]), enc.encode_query([text]), atol=1e-3)


def test_arara_accepts_the_nanoe5_backend() -> None:
    pytest.importorskip("nanoe5")
    a = Arara(dense_backend="nanoe5")
    a.add_documents(DOCS)
    a.finalize()
    assert a.stats()["dense_backend"] == "nanoe5"
    assert a._vectors.gather([0]).shape[1] == 384
    hits = a.search("qual a aliquota do imposto de renda?", top_k=3, mode="dense")
    assert hits and hits[0].doc_id == "trib"
    for h in hits:
        assert a.resolve(h) == h.text


def test_backends_produce_different_rankings_but_the_same_index_shape() -> None:
    """Switching backend changes quality, not the index structure."""
    pytest.importorskip("nanoe5")
    static = Arara(dense_backend="static")
    nano = Arara(dense_backend="nanoe5")
    for a in (static, nano):
        a.add_documents(DOCS)
        a.finalize()
    assert static.n_chunks == nano.n_chunks
    assert static._doc_ids == nano._doc_ids
    assert np.array_equal(np.asarray(static._slot_start), np.asarray(nano._slot_start))
    # lexical ignores the encoder entirely
    assert [h.doc_id for h in static.search("imposto de renda", top_k=4, mode="lexical")] == [
        h.doc_id for h in nano.search("imposto de renda", top_k=4, mode="lexical")
    ]
